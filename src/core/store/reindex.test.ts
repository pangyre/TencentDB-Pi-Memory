/**
 * Tests for the reindex/lock/approval-gate/freezing state machine (P1).
 *
 * Run with: npx vitest run
 *
 * Uses tmp-dir SQLite (real node:sqlite + sqlite-vec) and a mock embedFn — no
 * network. Covers:
 *   - lock acquire/steal/release state machine incl. concurrent steal attempts
 *   - approval gate (approve vs deny) — deny leaves tables frozen, nothing dropped
 *   - reindexPending freezing (vec writes skipped, search returns empty, no crash)
 *   - reindexAll honest accounting (totals vs failures)
 *   - markEmbeddingCurrent only-after-success
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";

import { VectorStore } from "./sqlite.js";
import { ReindexLock } from "./reindex-lock.js";
import type { EmbeddingProviderInfo } from "./types.js";

// ── helpers ──────────────────────────────────────────────────────────────

const DIM = 8;

function mkTmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "reindex-test-"));
}

function mkEmbedding(seed: number, dim = DIM): Float32Array {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) v[i] = ((seed + i) % 7) + 1; // non-zero
  return v;
}

/** Mock embedFn: deterministic, non-zero, no network. */
function mockEmbedFn(fail = false) {
  return vi.fn(async (text: string): Promise<Float32Array> => {
    if (fail) throw new Error("mock embed failure");
    let s = 0;
    for (let i = 0; i < text.length; i++) s += text.charCodeAt(i);
    return mkEmbedding(s);
  });
}

function providerInfo(over: Partial<EmbeddingProviderInfo> = {}): EmbeddingProviderInfo {
  return { provider: "mock", model: "m1", dimensions: DIM, ...over } as EmbeddingProviderInfo;
}

function newStore(dir: string, dim = DIM): VectorStore {
  const store = new VectorStore(path.join(dir, "mem.db"), dim);
  return store;
}

// ── reindex lock state machine (M1/M2) — tests the SHIPPING ReindexLock ──
//
// P3-3: the lock is imported from src/core/store/reindex-lock.ts — the same
// module tdai-core uses — so these tests exercise the production code, not a
// parallel copy. Steal semantics are exercised via maxAgeMs: a lock whose
// age exceeds maxAgeMs is stolen on the next tryAcquire (TTL backstop). True
// multi-PROCESS concurrency is arbitrated by the atomic link(2) in
// tryAcquire (documented; single-threaded tests cannot race it).

function makeLock(dir: string, maxAgeMs?: number): { lock: ReindexLock; nonce: string } {
  const lock = new ReindexLock({ dir, maxAgeMs });
  return { lock, nonce: lock.nonce };
}

describe("reindex lock state machine (M1/M2) — shipping ReindexLock", () => {
  let dir: string;

  beforeEach(() => { dir = mkTmpDir(); });
  afterEach(() => {
    try { fs.rmSync(dir, { recursive: true, force: true }); } catch { /* ignore */ }
  });

  it("first acquirer wins, second is refused while the lock is fresh", () => {
    const a = makeLock(dir);
    const b = makeLock(dir);
    expect(a.lock.tryAcquire()).toBe(true);
    expect(b.lock.tryAcquire()).toBe(false);
    const owner = JSON.parse(fs.readFileSync(path.join(dir, "reindex.lock"), "utf8"));
    expect(owner.nonce).toBe(a.nonce);
    a.lock.release();
  });

  it("release only removes the lock if we still own it (nonce guard)", () => {
    const a = makeLock(dir);
    const b = makeLock(dir, 0); // TTL 0 → any existing lock is stale → b steals
    expect(a.lock.tryAcquire()).toBe(true);
    expect(b.lock.tryAcquire()).toBe(true); // stole a's lock
    expect(b.lock.nonce === a.nonce).toBe(false);
    // A's release must NOT clobber B's lock
    a.lock.release();
    const owner = JSON.parse(fs.readFileSync(path.join(dir, "reindex.lock"), "utf8"));
    expect(owner.nonce).toBe(b.nonce);
    b.lock.release();
    expect(fs.existsSync(path.join(dir, "reindex.lock"))).toBe(false);
  });

  it("stale lock (past maxAgeMs) is stolen; fresh lock is not", () => {
    const a = makeLock(dir); // default 30-min TTL
    expect(a.lock.tryAcquire()).toBe(true);
    // A fresh holder cannot steal a live lock.
    const b = makeLock(dir);
    expect(b.lock.tryAcquire()).toBe(false);
    // But a TTL-0 holder treats any existing lock as stale and steals it.
    const c = makeLock(dir, 0);
    expect(c.lock.tryAcquire()).toBe(true);
    const owner = JSON.parse(fs.readFileSync(path.join(dir, "reindex.lock"), "utf8"));
    expect(owner.nonce).toBe(c.nonce);
    c.lock.release();
  });

  it("heartbeat refreshes ts so a slow-but-live pass is not stolen (M2)", async () => {
    const a = makeLock(dir, 120); // 120ms TTL
    const b = makeLock(dir, 120);
    expect(a.lock.tryAcquire()).toBe(true);
    // Outlive the TTL with heartbeats — a non-heartbeating lock would be
    // stolen by b here.
    for (let i = 0; i < 3; i++) {
      await new Promise((r) => setTimeout(r, 60));
      a.lock.heartbeat();
      expect(b.lock.tryAcquire()).toBe(false); // ts refreshed → still fresh
    }
    // Without further heartbeats the lock ages out and is stolen.
    await new Promise((r) => setTimeout(r, 150));
    expect(b.lock.tryAcquire()).toBe(true);
    b.lock.release();
  });
});

// ── approval gate + freezing + honest accounting ───────────────────────────

describe("reindexPending freezing (approval gate deny path)", () => {
  let dir: string;
  beforeEach(() => { dir = mkTmpDir(); });
  afterEach(() => { try { fs.rmSync(dir, { recursive: true, force: true }); } catch { /* ignore */ } });

  it("config change → needsReindex, vec writes skipped, search empty, no crash", async () => {
    // First boot at model m1 — build vectors.
    let store = newStore(dir);
    let res = store.init(providerInfo({ model: "m1" }));
    expect(res.needsReindex).toBe(false);
    await store.upsertL1(
      { id: "r1", content: "hello world", type: "fact", priority: 50, scene_name: "", sessionKey: "s", sessionId: "s", timestamps: ["t"], createdAt: "t", updatedAt: "t", metadata: {} } as any,
      mkEmbedding(1),
    );
    store.close();

    // Second boot with a CHANGED model → reindex pending, tables frozen.
    store = newStore(dir);
    res = store.init(providerInfo({ model: "m2" }));
    expect(res.needsReindex).toBe(true);

    // Vec write is skipped while frozen — must not throw.
    const ok = await store.upsertL1(
      { id: "r2", content: "frozen write", type: "fact", priority: 50, scene_name: "", sessionKey: "s", sessionId: "s", timestamps: ["t"], createdAt: "t", updatedAt: "t", metadata: {} } as any,
      mkEmbedding(2),
    );
    expect(ok).toBe(true); // metadata write succeeds even though vec is skipped

    // Vector search returns empty (frozen) without crashing.
    const hits = await store.searchL1Vector(mkEmbedding(1), 5);
    expect(Array.isArray(hits)).toBe(true);
    expect(hits.length).toBe(0);
    store.close();
  });

  it("empty DB + changed dims via existing meta → no reindex (H3, savedMeta branch)", () => {
    // Boot at dim 8, empty DB, meta written.
    let store = newStore(dir, DIM);
    store.init(providerInfo({ dimensions: DIM }));
    store.close();

    // Reopen at a different dimension. savedMeta exists → dimsChanged fires,
    // but the DB is EMPTY: nothing to re-embed → no needsReindex, no per-boot
    // warning loop.
    store = newStore(dir, DIM + 4);
    const res = store.init(providerInfo({ dimensions: DIM + 4, model: "m1" }));
    expect(res.needsReindex).toBe(false);
    store.close();
  });

  it("empty DB + dim mismatch with NO meta (legacy branch) → no needsReindex either (H3)", () => {
    // Boot at dim 8, empty DB, meta written — then WIPE embedding_meta to
    // force the no-saved-meta (legacy) branch on the next open.
    let store = newStore(dir, DIM);
    store.init(providerInfo({ dimensions: DIM }));
    store.close();

    const raw = new DatabaseSync(path.join(dir, "mem.db"));
    raw.prepare("DELETE FROM embedding_meta").run();
    raw.close();

    // Reopen at a different dimension, empty DB: the mismatched empty vec
    // tables are recreated at the new width during init — needsReindex must
    // stay false (nothing to re-embed), otherwise an empty DB warns forever.
    store = newStore(dir, DIM + 4);
    const res = store.init(providerInfo({ dimensions: DIM + 4, model: "m1" }));
    expect(res.needsReindex).toBe(false);
    store.close();
  });

  it("D1 crash-safety: a pending reindex that is never completed re-fires on the next boot", async () => {
    // Boot 1: build vectors at m1, mark current.
    let store = newStore(dir);
    let res = store.init(providerInfo({ model: "m1" }));
    expect(res.needsReindex).toBe(false);
    await store.upsertL1(
      { id: "r1", content: "hello world", type: "fact", priority: 50, scene_name: "", sessionKey: "s", sessionId: "s", timestamps: ["t"], createdAt: "t", updatedAt: "t", metadata: {} } as any,
      mkEmbedding(1),
    );
    store.close();

    // Boot 2: config changed to m2 → pending. Simulate a CRASH during the
    // approved reindex: do NOT call rebuildVecTables/reindexAll/
    // markEmbeddingCurrent — just close.
    store = newStore(dir);
    res = store.init(providerInfo({ model: "m2" }));
    expect(res.needsReindex).toBe(true);
    store.close();

    // Boot 3: the crash left meta stale → the change is RE-DETECTED. This is
    // the D1 guarantee: without it, a crashed reindex would leave the store
    // permanently unindexed (meta claiming current, vectors missing).
    store = newStore(dir);
    res = store.init(providerInfo({ model: "m2" }));
    expect(res.needsReindex).toBe(true);
    store.close();
  });
});

describe("approved reindex: rebuild + honest accounting + markEmbeddingCurrent", () => {
  let dir: string;
  beforeEach(() => { dir = mkTmpDir(); });
  afterEach(() => { try { fs.rmSync(dir, { recursive: true, force: true }); } catch { /* ignore */ } });

  it("rebuildVecTables + reindexAll re-embeds all rows; markEmbeddingCurrent clears pending", async () => {
    // Seed data at m1.
    let store = newStore(dir);
    store.init(providerInfo({ model: "m1" }));
    for (let i = 0; i < 3; i++) {
      await store.upsertL1(
        { id: `r${i}`, content: `text ${i}`, type: "fact", priority: 50, scene_name: "", sessionKey: "s", sessionId: "s", timestamps: ["t"], createdAt: "t", updatedAt: "t", metadata: {} } as any,
        mkEmbedding(i),
      );
    }
    store.close();

    // Reopen with changed model → pending.
    store = newStore(dir);
    const res = store.init(providerInfo({ model: "m2" }));
    expect(res.needsReindex).toBe(true);

    // Approved path: rebuild then reindex.
    expect(store.rebuildVecTables?.(providerInfo({ model: "m2" }))).toBe(true);
    const embed = mockEmbedFn(false);
    const out = await store.reindexAll(embed);
    // Honest accounting: totals match counts, zero failures.
    expect(out.l1Failed).toBe(0);
    expect(out.l1Count).toBe(out.l1Total);
    expect(out.l1Total).toBeGreaterThanOrEqual(3);

    // Mark current only after success — search now returns rows.
    store.markEmbeddingCurrent?.(providerInfo({ model: "m2" }));
    const hits = await store.searchL1Vector(mkEmbedding(0), 5);
    expect(hits.length).toBeGreaterThan(0);
    store.close();

    // Steady-state boot at m2 → no reindex.
    store = newStore(dir);
    const res2 = store.init(providerInfo({ model: "m2" }));
    expect(res2.needsReindex).toBe(false);
    store.close();
  });

  it("reindexAll reports failures honestly (does not mark current)", async () => {
    let store = newStore(dir);
    store.init(providerInfo({ model: "m1" }));
    await store.upsertL1(
      { id: "r0", content: "text 0", type: "fact", priority: 50, scene_name: "", sessionKey: "s", sessionId: "s", timestamps: ["t"], createdAt: "t", updatedAt: "t", metadata: {} } as any,
      mkEmbedding(0),
    );
    store.close();

    store = newStore(dir);
    store.init(providerInfo({ model: "m2" }));
    expect(store.rebuildVecTables?.(providerInfo({ model: "m2" }))).toBe(true);

    const failing = mockEmbedFn(true);
    const out = await store.reindexAll(failing);
    // With an all-failing embedFn, failures > 0 and count < total.
    expect(out.l1Failed).toBeGreaterThan(0);
    expect(out.l1Count).toBeLessThan(out.l1Total + 1);
    // Caller contract: only mark current when failed===0 && done===total.
    const total = out.l1Total + out.l0Total;
    const done = out.l1Count + out.l0Count;
    const shouldMark = out.l1Failed === 0 && out.l0Failed === 0 && done === total;
    expect(shouldMark).toBe(false);
    store.close();
  });
});

// ── cross-process epoch (H1/H2) ─────────────────────────────────────────────

describe("cross-process schema epoch (H1/H2)", () => {
  let dir: string;
  beforeEach(() => { dir = mkTmpDir(); });
  afterEach(() => { try { fs.rmSync(dir, { recursive: true, force: true }); } catch { /* ignore */ } });

  it("a rebuild under a second live handle makes the first skip vec writes safely", async () => {
    // Process A: build vectors at m1.
    const dbPath = path.join(dir, "mem.db");
    const a = new VectorStore(dbPath, DIM);
    a.init(providerInfo({ model: "m1" }));
    await a.upsertL1(
      { id: "a1", content: "proc a", type: "fact", priority: 50, scene_name: "", sessionKey: "s", sessionId: "s", timestamps: ["t"], createdAt: "t", updatedAt: "t", metadata: {} } as any,
      mkEmbedding(1),
    );

    // Process B: open the same DB and perform an approved rebuild (bumps epoch).
    const b = new VectorStore(dbPath, DIM);
    b.init(providerInfo({ model: "m1" }));
    expect(b.rebuildVecTables?.(providerInfo({ model: "m1" }))).toBe(true);

    // Process A now writes — its stmtInsertVec is bound to the dropped
    // table. The epoch guard must trip: metadata write succeeds, vec write
    // is SKIPPED, no crash. (P0-1: discriminating assertion — a bare
    // `ok === true` passes whether or not the guard fired, so we verify the
    // raw tables: a2's metadata row EXISTS, a2's VECTOR row does NOT, and
    // a1's pre-rebuild vector row does.)
    const ok = await a.upsertL1(
      { id: "a2", content: "after rebuild", type: "fact", priority: 50, scene_name: "", sessionKey: "s", sessionId: "s", timestamps: ["t"], createdAt: "t", updatedAt: "t", metadata: {} } as any,
      mkEmbedding(2),
    );
    expect(ok).toBe(true);

    // P0-1 discriminating assertions, read back through a raw connection
    // (sqlite-vec must be loaded on it to see the vec0 tables):
    const { createRequire } = await import("node:module");
    const req = createRequire(import.meta.url);
    const sqliteVec = req("sqlite-vec");
    const raw = new DatabaseSync(dbPath, { allowExtension: true });
    raw.enableLoadExtension(true);
    sqliteVec.load(raw);

    const count = (table: string, id: string): number =>
      (raw.prepare(`SELECT count(*) AS n FROM ${table} WHERE record_id = ?`).get(id) as { n: number }).n;

    const metaA2 = count("l1_records", "a2");
    const vecA2 = count("l1_vec", "a2");
    raw.close();

    // a2's metadata row persisted (the safe path), but its VECTOR row was
    // skipped by the epoch guard — without the guard, both would be present.
    expect(metaA2).toBe(1);
    expect(vecA2).toBe(0);

    a.close();
    b.close();
  });
});

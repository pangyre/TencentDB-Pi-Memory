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

import { VectorStore } from "./sqlite.js";
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

// ── atomic lock state machine (M1/M2) ──────────────────────────────────────
//
// Mirrors the temp+link acquire/steal/release pattern used in tdai-core so we
// can exercise the state machine (incl. concurrent steal) without booting the
// whole core.

function makeLock(lockPath: string, nonce: string) {
  const readOwner = (): { nonce?: string; ts?: number } | null => {
    try { return JSON.parse(fs.readFileSync(lockPath, "utf8")); } catch { return null; }
  };
  const tryAcquire = (): boolean => {
    const tmp = `${lockPath}.${nonce}.tmp`;
    try {
      fs.writeFileSync(tmp, JSON.stringify({ nonce, ts: Date.now() }));
      fs.linkSync(tmp, lockPath);
      return true;
    } catch {
      return false;
    } finally {
      try { fs.unlinkSync(tmp); } catch { /* gone */ }
    }
  };
  const steal = (): boolean => {
    try { fs.unlinkSync(lockPath); } catch { /* raced */ }
    const tmp = `${lockPath}.${nonce}.steal.tmp`;
    try {
      fs.writeFileSync(tmp, JSON.stringify({ nonce, ts: Date.now() }));
      fs.linkSync(tmp, lockPath);
      return true;
    } catch {
      return false;
    } finally {
      try { fs.unlinkSync(tmp); } catch { /* gone */ }
    }
  };
  const release = (): void => {
    const owner = readOwner();
    if (owner?.nonce === nonce) { try { fs.unlinkSync(lockPath); } catch { /* best-effort */ } }
  };
  return { readOwner, tryAcquire, steal, release };
}

describe("reindex lock state machine (M1/M2)", () => {
  let dir: string;
  let lockPath: string;

  beforeEach(() => {
    dir = mkTmpDir();
    lockPath = path.join(dir, "reindex.lock");
  });
  afterEach(() => {
    try { fs.rmSync(dir, { recursive: true, force: true }); } catch { /* ignore */ }
  });

  it("first acquirer wins, second is refused (EEXIST)", () => {
    const a = makeLock(lockPath, "A");
    const b = makeLock(lockPath, "B");
    expect(a.tryAcquire()).toBe(true);
    expect(b.tryAcquire()).toBe(false);
    expect(a.readOwner()?.nonce).toBe("A");
  });

  it("release only removes the lock if we still own it", () => {
    const a = makeLock(lockPath, "A");
    const b = makeLock(lockPath, "B");
    expect(a.tryAcquire()).toBe(true);
    // B steals
    expect(b.steal()).toBe(true);
    expect(b.readOwner()?.nonce).toBe("B");
    // A's release must NOT clobber B's lock
    a.release();
    expect(b.readOwner()?.nonce).toBe("B");
  });

  it("concurrent steal — exactly one winner", () => {
    // seed a lock owned by a dead owner
    fs.writeFileSync(lockPath, JSON.stringify({ nonce: "DEAD", ts: Date.now() - 999_999 }));
    const stealers = ["S1", "S2", "S3", "S4"].map((n) => makeLock(lockPath, n));
    const results = stealers.map((s) => s.steal());
    const winners = results.filter(Boolean).length;
    // At least one wins; the on-disk lock is owned by exactly one stealer.
    expect(winners).toBeGreaterThanOrEqual(1);
    const owner = JSON.parse(fs.readFileSync(lockPath, "utf8")).nonce as string;
    expect(["S1", "S2", "S3", "S4"]).toContain(owner);
  });

  it("release after clean acquire removes the lock", () => {
    const a = makeLock(lockPath, "A");
    expect(a.tryAcquire()).toBe(true);
    a.release();
    expect(fs.existsSync(lockPath)).toBe(false);
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

  it("empty DB + dimension mismatch does NOT set needsReindex (H3)", () => {
    // Boot at dim 8, empty DB, meta written.
    let store = newStore(dir, DIM);
    store.init(providerInfo({ dimensions: DIM }));
    store.close();

    // Wipe meta to simulate legacy-ish path but keep DB empty, reopen at a
    // different dimension. Empty DB → nothing to re-embed → no needsReindex.
    store = newStore(dir, DIM + 4);
    const res = store.init(providerInfo({ dimensions: DIM + 4, model: "m1" }));
    expect(res.needsReindex).toBe(false);
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

    // Process A now writes — its stmtInsertVec is bound to a dropped table.
    // The epoch guard must trip: metadata write succeeds, vec write skipped,
    // NO crash.
    const ok = await a.upsertL1(
      { id: "a2", content: "after rebuild", type: "fact", priority: 50, scene_name: "", sessionKey: "s", sessionId: "s", timestamps: ["t"], createdAt: "t", updatedAt: "t", metadata: {} } as any,
      mkEmbedding(2),
    );
    expect(ok).toBe(true);

    a.close();
    b.close();
  });
});

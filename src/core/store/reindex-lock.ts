/**
 * ReindexLock — single-flight cross-process lock for vector-table rebuilds.
 *
 * Several pi processes (one per project directory) share one global SQLite
 * store. The destructive part of a reindex (DROP + recreate of the vec0
 * tables + full re-embed) must run in exactly one process. This lock provides
 * that single-flight guarantee across processes.
 *
 * Design (after hostile review rounds 2–4):
 *
 * - **Atomic acquisition.** A unique temp file is written, then link(2)'d onto
 *   the lock path. link(2) is atomic and refuses to clobber an existing file
 *   (EEXIST), so exactly one contender can ever win — there is no
 *   unlink-then-create window where two processes both believe they hold the
 *   lock.
 * - **Owner nonce.** Every holder embeds a random nonce; release and steal
 *   verification compare nonces so a process never clobbers a successor's
 *   lock, and a stealer can prove its own takeover took.
 * - **TTL backstop.** A lock older than `maxAgeMs` (default 30 min — a reindex
 *   takes minutes) is stolen regardless of liveness. This covers pid-recycle
 *   (process.kill(pid, 0) succeeding for an unrelated process) and
 *   cross-host/NFS pid-namespace cases where liveness checks lie. A reindex
 *   that legitimately outlives the TTL will have its lock stolen — callers
 *   re-run it; the rebuild is idempotent.
 *
 * Note: pid liveness is NOT used as a steal criterion for the same reason —
 * it is unreliable across pid recycling and pid namespaces. The TTL is the
 * authority; the nonce is the identity.
 */

import fs from "node:fs";
import path from "node:path";

/** Default: a lock older than 30 minutes is stale and may be stolen. */
const DEFAULT_MAX_AGE_MS = 30 * 60 * 1000;

interface LockPayload {
  nonce: string;
  pid: number;
  ts: number;
}

export interface ReindexLockOptions {
  /** Directory the lock file lives in (the shared store's data dir). */
  dir: string;
  /** Lock file name (default: "reindex.lock"). */
  name?: string;
  /** Max lock age before it may be stolen (default: 30 min). */
  maxAgeMs?: number;
  /** debug sink (optional) */
  debug?: (msg: string) => void;
}

export class ReindexLock {
  private readonly lockPath: string;
  private readonly maxAgeMs: number;
  private readonly debug?: (msg: string) => void;
  private readonly nonce: string;
  private held = false;

  constructor(opts: ReindexLockOptions) {
    this.lockPath = path.join(opts.dir, opts.name ?? "reindex.lock");
    this.maxAgeMs = opts.maxAgeMs ?? DEFAULT_MAX_AGE_MS;
    this.debug = opts.debug;
    // Nonce identifies THIS holder instance, even across pid reuse.
    this.nonce = `${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }

  get held(): boolean {
    return this.held;
  }

  private readOwner(): LockPayload | null {
    try {
      return JSON.parse(fs.readFileSync(this.lockPath, "utf8")) as LockPayload;
    } catch {
      return null; // vanished between operations — treat as free
    }
  }

  /** Atomic create-or-fail: write a unique temp file, link(2) onto the lock. */
  private createLockFile(): boolean {
    const tmp = `${this.lockPath}.${this.nonce}.tmp`;
    try {
      fs.writeFileSync(tmp, JSON.stringify({ nonce: this.nonce, pid: process.pid, ts: Date.now() }));
      // link(2) is atomic and fails EEXIST if the lock already exists —
      // exactly one contender can ever create the lock.
      fs.linkSync(tmp, this.lockPath);
      this.held = true;
      return true;
    } catch {
      return false;
    } finally {
      try { fs.unlinkSync(tmp); } catch { /* temp already gone */ }
    }
  }

  /**
   * Steal a stale lock. Replaces the file, then verifies OUR nonce is on
   * disk — if a faster process replaced ours between write and read, it won
   * and we back off. (The replacement itself is not atomic, but the link-based
   * tryAcquire arbitration in the winner's path means at most one process
   * ends up owning the lock; nonce verification keeps our bookkeeping honest.)
   */
  private stealStale(): boolean {
    try {
      const fd = fs.openSync(this.lockPath, "w");
      fs.writeSync(fd, JSON.stringify({ nonce: this.nonce, pid: process.pid, ts: Date.now() }));
      fs.closeSync(fd);
    } catch {
      return false;
    }
    const owner = this.readOwner();
    if (owner?.nonce !== this.nonce) return false;
    this.held = true;
    return true;
  }

  /**
   * Acquire the lock. Returns false (never throws) when another live holder
   * owns a non-stale lock — the caller should skip and let that holder
   * finish.
   */
  tryAcquire(): boolean {
    try {
      fs.mkdirSync(path.dirname(this.lockPath), { recursive: true });
    } catch { /* ignore */ }

    if (this.createLockFile()) return true;

    // Lock exists — steal it if stale.
    const owner = this.readOwner();
    const age = owner?.ts ? Date.now() - owner.ts : Number.POSITIVE_INFINITY;
    if (age > this.maxAgeMs) {
      this.debug?.(`reindex lock stale (age ${Math.round(age / 60000)}min) — stealing`);
      return this.stealStale();
    }
    return false;
  }

  /**
   * Refresh the lock's timestamp (heartbeat). Call periodically during a long
   * reindex so a slow-but-alive pass is not stolen by the TTL backstop.
   */
  heartbeat(): void {
    if (!this.held) return;
    try {
      const owner = this.readOwner();
      if (owner?.nonce !== this.nonce) return; // lost the lock — do not clobber
      const fd = fs.openSync(this.lockPath, "w");
      fs.writeSync(fd, JSON.stringify({ nonce: this.nonce, pid: process.pid, ts: Date.now() }));
      fs.closeSync(fd);
    } catch { /* best-effort */ }
  }

  /**
   * Release the lock — but only if WE still own it (nonce check). Never
   * clobbers a successor's lock. Idempotent, never throws.
   */
  release(): void {
    if (!this.held) return;
    try {
      const owner = this.readOwner();
      if (owner?.nonce === this.nonce) fs.unlinkSync(this.lockPath);
    } catch { /* best-effort */ }
    this.held = false;
  }
}

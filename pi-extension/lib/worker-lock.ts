/**
 * PipelineWorkerLock — ensures only ONE pi process runs the L1/L2/L3
 * extraction pipelines against the shared global store.
 *
 * Lock file contains JSON { pid, ts }. A lock is stale when its owner pid is
 * dead (ESRCH) or its heartbeat timestamp is older than 120s. The holder
 * refreshes the heartbeat every 30s.
 */

import fs from "node:fs";
import path from "node:path";
import type { Logger } from "../../src/core/types.js";

const HEARTBEAT_INTERVAL_MS = 30_000;
const STALE_AFTER_MS = 120_000;

interface LockPayload {
  pid: number;
  ts: number;
}

export class PipelineWorkerLock {
  private readonly lockPath: string;
  private readonly logger: Logger;
  private owned = false;
  private heartbeatTimer: ReturnType<typeof setInterval> | undefined;

  constructor(lockPath: string, logger: Logger) {
    this.lockPath = lockPath;
    this.logger = logger;
  }

  get held(): boolean {
    return this.owned;
  }

  /** Attempt to acquire the lock. Steals stale locks. Never throws. */
  tryAcquire(): boolean {
    try {
      fs.mkdirSync(path.dirname(this.lockPath), { recursive: true });
    } catch {
      // ignore
    }

    if (this.writeLockExclusive()) {
      this.onAcquired("acquired");
      return true;
    }

    // Lock exists — check staleness.
    if (this.isExistingLockStale()) {
      try {
        fs.unlinkSync(this.lockPath);
      } catch {
        // another process may have raced us; fall through to one retry
      }
      if (this.writeLockExclusive()) {
        this.onAcquired("stole stale lock");
        return true;
      }
    }

    return false;
  }

  /** Release the lock if we own it. Idempotent, never throws. */
  release(): void {
    if (this.heartbeatTimer !== undefined) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = undefined;
    }
    if (!this.owned) return;
    try {
      const payload = this.readPayload();
      if (payload && payload.pid === process.pid) {
        fs.unlinkSync(this.lockPath);
      }
    } catch {
      // best-effort
    }
    this.owned = false;
    this.logger.debug?.(`pipeline lock released: ${this.lockPath}`);
  }

  // ── internals ──────────────────────────────────────────────────────────

  private writeLockExclusive(): boolean {
    try {
      fs.writeFileSync(
        this.lockPath,
        JSON.stringify({ pid: process.pid, ts: Date.now() } satisfies LockPayload),
        { flag: "wx" },
      );
      return true;
    } catch {
      return false;
    }
  }

  private onAcquired(how: string): void {
    this.owned = true;
    this.logger.debug?.(`pipeline lock ${how}: ${this.lockPath} (pid=${process.pid})`);
    this.heartbeatTimer = setInterval(() => {
      try {
        fs.writeFileSync(
          this.lockPath,
          JSON.stringify({ pid: process.pid, ts: Date.now() } satisfies LockPayload),
        );
      } catch {
        // best-effort heartbeat
      }
    }, HEARTBEAT_INTERVAL_MS);
    this.heartbeatTimer.unref?.();
  }

  private readPayload(): LockPayload | undefined {
    try {
      const raw = fs.readFileSync(this.lockPath, "utf8");
      const parsed = JSON.parse(raw) as Partial<LockPayload>;
      if (typeof parsed.pid !== "number" || typeof parsed.ts !== "number") return undefined;
      return { pid: parsed.pid, ts: parsed.ts };
    } catch {
      return undefined;
    }
  }

  private isExistingLockStale(): boolean {
    const payload = this.readPayload();
    if (!payload) return true; // unreadable/corrupt → stale

    if (Date.now() - payload.ts > STALE_AFTER_MS) return true;

    try {
      process.kill(payload.pid, 0);
      return false; // signal succeeded → owner alive
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code;
      // EPERM means the pid exists but belongs to another user → alive.
      return code === "ESRCH";
    }
  }
}

/**
 * TdaiCore — Host-neutral facade for TDAI memory capabilities.
 *
 * This is the single entry point that both OpenClaw and Hermes/Gateway call
 * to perform recall, capture, search, and pipeline management. It depends
 * only on abstract interfaces (HostAdapter, LLMRunner), never on a specific host.
 *
 * Usage:
 *   // OpenClaw path (in-process)
 *   const adapter = new OpenClawHostAdapter({ api, pluginDataDir, config });
 *   const core = new TdaiCore({ hostAdapter: adapter, config: parsedCfg });
 *   await core.initialize();
 *   const recall = await core.handleBeforeRecall("user query", "session-1");
 *
 *   // Gateway path (HTTP)
 *   const adapter = new StandaloneHostAdapter({ ... });
 *   const core = new TdaiCore({ hostAdapter: adapter, config: parsedCfg });
 *   await core.initialize();
 *   // HTTP handler calls core.handleBeforeRecall / core.handleTurnCommitted / etc.
 */

import fs from "node:fs";
import path from "node:path";
import type {
  HostAdapter,
  Logger,
  LLMRunnerFactory,
  RecallResult,
  CaptureResult,
  CompletedTurn,
  MemorySearchParams,
  ConversationSearchParams,
} from "./types.js";
import type { MemoryTdaiConfig } from "../config.js";
import type { IMemoryStore } from "./store/types.js";
import type { EmbeddingService } from "./store/embedding.js";
import { performAutoRecall } from "./hooks/auto-recall.js";
import { performAutoCapture } from "./hooks/auto-capture.js";
import { executeMemorySearch, formatSearchResponse } from "./tools/memory-search.js";
import { executeConversationSearch, formatConversationSearchResponse } from "./tools/conversation-search.js";
import {
  initDataDirectories,
  initStores,
  resetStores,
  createPipelineManager,
  createL1Runner,
  createPersister,
  createL2Runner,
  createL3Runner,
} from "../utils/pipeline-factory.js";
import { MemoryPipelineManager } from "../utils/pipeline-manager.js";
import { CheckpointManager } from "../utils/checkpoint.js";
import { SessionFilter } from "../utils/session-filter.js";
import { StandaloneLLMRunnerFactory } from "../adapters/standalone/llm-runner.js";

const TAG = "[memory-tdai] [core]";

// ============================
// Constructor options
// ============================

export interface TdaiCoreOptions {
  /** Host adapter providing runtime context, logger, and LLM runner factory. */
  hostAdapter: HostAdapter;
  /** Parsed TDAI memory configuration. */
  config: MemoryTdaiConfig;
  /** Session filter for excluding internal/benchmark sessions. */
  sessionFilter?: SessionFilter;
  /** Plugin instance ID for metric reporting. */
  instanceId?: string;
}

// ============================
// TdaiCore
// ============================

export class TdaiCore {
  private hostAdapter: HostAdapter;
  private cfg: MemoryTdaiConfig;
  private logger: Logger;
  private dataDir: string;
  private runnerFactory: LLMRunnerFactory;
  private sessionFilter: SessionFilter;
  private instanceId?: string;

  // Lazy-initialized resources
  private vectorStore?: IMemoryStore;
  private embeddingService?: EmbeddingService;
  private scheduler?: MemoryPipelineManager;
  /**
   * Promise gate for the one-shot scheduler-start sequence.
   *
   * ``ensureSchedulerStarted`` reads a checkpoint file (async) and then
   * calls ``scheduler.start(restoredStates)``.  Under the Gateway, several
   * HTTP requests can reach ``handleTurnCommitted`` concurrently and all
   * race into that function.  Using a plain boolean flag is unsafe: the
   * first caller flips the flag to ``true`` *before* the await completes,
   * so subsequent callers slip past the check and touch the scheduler
   * before ``start()`` has actually run — which makes ``start()``'s
   * ``sessionStates.set(key, restored)`` later clobber the state that
   * those concurrent captures already incremented.
   *
   * Storing the in-flight promise lets every concurrent caller ``await``
   * the same start sequence.  Once it resolves the promise is kept as a
   * sentinel so subsequent calls are a single already-resolved await
   * (effectively a no-op).
   */
  private schedulerStartPromise?: Promise<void>;
  private storeReady?: Promise<void>;

  /**
   * In-flight fire-and-forget background tasks started by
   * ``handleTurnCommitted`` (currently: deferred L0 embedding for
   * SQLite-style stores — see auto-capture.ts path A).
   *
   * ``destroy()`` awaits all pending entries (with a hard timeout)
   * before closing ``vectorStore`` / ``embeddingService`` so that a
   * late ``updateL0Embedding`` cannot land on an already-closed
   * database connection.
   *
   * Each task registers itself on creation and removes itself in its
   * own ``finally`` handler, so the set stays bounded by the number
   * of currently-running background tasks.
   */
  private readonly bgTasks = new Set<Promise<void>>();

  constructor(opts: TdaiCoreOptions) {
    this.hostAdapter = opts.hostAdapter;
    this.cfg = opts.config;
    this.logger = opts.hostAdapter.getLogger();
    this.dataDir = opts.hostAdapter.getRuntimeContext().dataDir;
    this.runnerFactory = opts.hostAdapter.getLLMRunnerFactory();
    this.sessionFilter = opts.sessionFilter ?? new SessionFilter([]);
    this.instanceId = opts.instanceId;
  }

  // ============================
  // Lifecycle
  // ============================

  /**
   * Initialize data directories, storage, and pipeline scheduler.
   * Must be called once before any other methods.
   */
  async initialize(): Promise<void> {
    this.logger.debug?.(`${TAG} Initializing TDAI Core: dataDir=${this.dataDir}`);
    initDataDirectories(this.dataDir);

    // Initialize stores (async)
    this.storeReady = this.initStores();

    // Create pipeline manager (sync — does not need store)
    if (this.cfg.extraction.enabled) {
      this.scheduler = createPipelineManager(this.cfg, this.logger, this.sessionFilter);
      // Wire runners after store is ready (or after store init fails — runners
      // still work in degraded mode with JSONL fallback and no embedding)
      this.storeReady
        .then(() => this.wirePipelineRunners())
        .catch((err) => {
          this.logger.error(`${TAG} Store init failed; wiring pipeline runners in degraded mode: ${err instanceof Error ? err.message : String(err)}`);
          this.wirePipelineRunners();
        });
    }

    this.logger.debug?.(`${TAG} TDAI Core initialized`);
  }

  /**
   * Destroy all resources. Call on shutdown.
   */
  async destroy(): Promise<void> {
    this.logger.debug?.(`${TAG} Destroying TDAI Core...`);

    // Wait for store init to complete before tearing down
    await this.storeReady?.catch(() => {});

    if (this.scheduler && this.schedulerStartPromise) {
      await this.scheduler.destroy();
      this.schedulerStartPromise = undefined;
      this.logger.debug?.(`${TAG} Scheduler destroyed`);
    }

    // Drain fire-and-forget background tasks started by auto-capture
    // (currently: deferred L0 embedding writes).  We must wait for
    // them here — BEFORE closing vectorStore / embeddingService —
    // otherwise a late updateL0Embedding lands on an already-closed
    // DB connection and either throws "database is not open" or
    // (worse) corrupts state.  A hard timeout keeps destroy bounded
    // when a background task is stuck on a hung embed HTTP call.
    if (this.bgTasks.size > 0) {
      const pending = [...this.bgTasks];
      this.logger.debug?.(
        `${TAG} Draining ${pending.length} background task(s) before closing stores...`,
      );
      const BG_DRAIN_TIMEOUT_MS = 5_000;
      let drainTimeoutId: ReturnType<typeof setTimeout> | undefined;
      try {
        await Promise.race([
          Promise.allSettled(pending).then(() => undefined),
          new Promise<never>((_, reject) => {
            drainTimeoutId = setTimeout(
              () => reject(new Error("bgTasks drain timeout")),
              BG_DRAIN_TIMEOUT_MS,
            );
          }),
        ]);
        this.logger.debug?.(`${TAG} Background tasks drained`);
      } catch (err) {
        this.logger.warn(
          `${TAG} Background-task drain timed out (${BG_DRAIN_TIMEOUT_MS}ms): ` +
          `${err instanceof Error ? err.message : String(err)}. ` +
          `Closing stores anyway — residual writes may surface as warnings.`,
        );
      } finally {
        if (drainTimeoutId !== undefined) clearTimeout(drainTimeoutId);
      }
    }

    if (this.vectorStore) {
      this.vectorStore.close();
      this.vectorStore = undefined;
      this.logger.debug?.(`${TAG} VectorStore closed`);
    }

    if (this.embeddingService?.close) {
      try {
        await this.embeddingService.close();
      } catch (err) {
        this.logger.warn(`${TAG} EmbeddingService close error: ${err instanceof Error ? err.message : String(err)}`);
      }
      this.embeddingService = undefined;
    }

    resetStores(this.dataDir);
    this.logger.debug?.(`${TAG} TDAI Core destroyed`);
  }

  // ============================
  // Core capabilities
  // ============================

  /**
   * Handle recall (memory retrieval) before an LLM turn.
   * Maps to: OpenClaw `before_prompt_build` / Hermes `prefetch()`.
   */
  async handleBeforeRecall(userText: string, sessionKey: string): Promise<RecallResult> {
    await this.storeReady?.catch(() => {});

    const result = await performAutoRecall({
      userText,
      actorId: "default_user",
      sessionKey,
      cfg: this.cfg,
      pluginDataDir: this.dataDir,
      logger: this.logger,
      vectorStore: this.vectorStore,
      embeddingService: this.embeddingService,
    });

    return result ?? {};
  }

  /**
   * Handle turn commitment (conversation capture + pipeline trigger).
   * Maps to: OpenClaw `agent_end` / Hermes `sync_turn()`.
   */
  async handleTurnCommitted(turn: CompletedTurn): Promise<CaptureResult> {
    await this.storeReady?.catch(() => {});
    await this.ensureSchedulerStarted();

    return performAutoCapture({
      messages: turn.messages,
      sessionKey: turn.sessionKey,
      sessionId: turn.sessionId,
      cfg: this.cfg,
      pluginDataDir: this.dataDir,
      logger: this.logger,
      scheduler: this.scheduler,
      originalUserText: turn.userText,
      originalUserMessageCount: turn.originalUserMessageCount,
      pluginStartTimestamp: turn.startedAt ?? Date.now(),
      vectorStore: this.vectorStore,
      embeddingService: this.embeddingService,
      bgTaskRegistry: this.bgTasks,
    });
  }

  /**
   * Search L1 structured memories.
   * Maps to: `tdai_memory_search` tool.
   */
  async searchMemories(params: MemorySearchParams): Promise<{ text: string; total: number; strategy: string }> {
    const result = await executeMemorySearch({
      query: params.query,
      limit: params.limit ?? 5,
      type: params.type,
      scene: params.scene,
      vectorStore: this.vectorStore,
      embeddingService: this.embeddingService,
      logger: this.logger,
    });

    return {
      text: formatSearchResponse(result),
      total: result.total,
      strategy: result.strategy,
    };
  }

  /**
   * Search L0 raw conversations.
   * Maps to: `tdai_conversation_search` tool.
   */
  async searchConversations(params: ConversationSearchParams): Promise<{ text: string; total: number }> {
    const result = await executeConversationSearch({
      query: params.query,
      limit: params.limit ?? 5,
      sessionKey: params.sessionKey,
      vectorStore: this.vectorStore,
      embeddingService: this.embeddingService,
      logger: this.logger,
    });

    return {
      text: formatConversationSearchResponse(result),
      total: result.total,
    };
  }

  /**
   * Handle end-of-conversation for a single session.
   *
   * ⚠️ Read this if you are editing the method:
   *
   * There are two distinct shutdown-ish events, and they must **NOT**
   * share an implementation:
   *
   *   - **`gateway_stop` (OpenClaw / process exit)**
   *     The host is going away.  Tear everything down — scheduler,
   *     VectorStore, EmbeddingService, caches.  That is
   *     {@link destroy}, not this method.
   *
   *   - **`on_session_end` (Hermes) / `POST /session/end` (Gateway)**
   *     One conversation ended while the process keeps serving other
   *     concurrent sessions.  **Only** this session's buffered work
   *     should be flushed; every other session's timers, buffers,
   *     pipeline state, and the shared scheduler itself MUST remain
   *     untouched.  That is this method.
   *
   * Historically this method did ``scheduler.destroy() +
   * createPipelineManager()``, which conflated the two semantics and
   * wiped concurrent sessions' in-memory state on every ``/session/end``
   * call.  That bug is covered by the concurrency test
   * ``P0-1: handleSessionEnd must be scoped to its session``.
   *
   * @param sessionKey  Session whose buffered work should be flushed.
   *                    Unknown keys are tolerated as a no-op so callers
   *                    don't have to pre-check whether the session was
   *                    already evicted or never produced a capture.
   */
  async handleSessionEnd(sessionKey: string): Promise<void> {
    if (!sessionKey) return;
    await this.storeReady?.catch(() => {});
    if (!this.scheduler) return;
    await this.scheduler.flushSession(sessionKey);
  }

  // ============================
  // Accessors (for migration bridge)
  // ============================

  /** Get the LLM runner factory (for creating host-neutral LLM runners). */
  getLLMRunnerFactory(): LLMRunnerFactory {
    return this.runnerFactory;
  }

  /** Get the shared VectorStore (may be undefined if init failed). */
  getVectorStore(): IMemoryStore | undefined {
    return this.vectorStore;
  }

  /** Get the shared EmbeddingService (may be undefined if not configured). */
  getEmbeddingService(): EmbeddingService | undefined {
    return this.embeddingService;
  }

  /** Get the pipeline scheduler (may be undefined if extraction disabled). */
  getScheduler(): MemoryPipelineManager | undefined {
    return this.scheduler;
  }

  /** Whether the scheduler has been started (or is currently starting). */
  isSchedulerStarted(): boolean {
    return this.schedulerStartPromise !== undefined;
  }

  /** Set the instance ID for metrics (may be resolved asynchronously). */
  setInstanceId(id: string): void {
    this.instanceId = id;
    if (this.scheduler) {
      this.scheduler.instanceId = id;
    }
  }

  // ============================
  // Internal helpers
  // ============================

  private async initStores(): Promise<void> {
    try {
      const stores = await initStores(this.cfg, this.dataDir, this.logger);
      this.vectorStore = stores.vectorStore;
      this.embeddingService = stores.embeddingService;
      this.logger.debug?.(`${TAG} Stores initialized: backend=${this.cfg.storeBackend}, embedding=${this.cfg.embedding.provider}`);

      // Embedding config changed (provider / model / dimensions) — the store
      // detected the change and set reindexPending: the vector tables are
      // FROZEN (intact on disk, vecTablesReady=false — nothing dropped, no
      // destructive work done). Reindex in the background so semantic search
      // has vectors. Only fires when the persisted embedding_meta differs
      // from current config — steady-state boots skip this entirely.
      //
      // Defect guards (all five):
      // D1 crash-safety — embedding_meta is NOT updated by initSchema when a
      //    rebuild is pending (see sqlite.ts); we write it only after a fully
      //    successful pass via markEmbeddingCurrent(). A crash mid-reindex
      //    leaves meta stale → next boot re-detects and retries.
      // D2 honest counts — reindexAll returns failed counts AND totals; we
      //    only mark current when failed===0 AND counts===totals (an empty
      //    candidate set is not success).
      // D3 readiness — local (node-llama-cpp) providers need warmup before
      //    embed() will succeed; we call startWarmup() and wait for isReady()
      //    instead of firing embed() blind.
      // D4 concurrency — a lock file (dataDir/reindex.lock) ensures only one
      //    pi process runs the reindex when several boot against the shared
      //    global store; the loser skips and lets the winner mark current.
      // D5 approval — reindexing is destructive (drop + full re-embed of
      //    every stored text). A config change may be accidental (typo,
      //    experiment); it must NOT silently trigger it. Opt-in via
      //    reindex.approveChanges; unapproved changes leave the index
      //    frozen and warn on every boot until a human decides.
      if (stores.needsReindex && this.vectorStore && this.embeddingService) {
        if (!this.cfg.reindex.approveChanges) {
          // D5 approval gate: the vector tables are frozen (intact, but
          // stale-width/stale-model — vector search is OFF while pending).
          // Nothing has been destroyed: the drop happens only inside the
          // approved rebuild. Keyword/FTS recall is unaffected. Old vectors
          // survive on disk, so reverting the config restores the old index
          // with zero loss.
          this.logger?.warn(
            `${TAG} Embedding config change detected (${stores.reindexReason ?? "provider/model/dimensions"}) ` +
              `but reindex is NOT auto-approved. Vector tables are frozen (intact, not rebuilt) ` +
              `and vector search is disabled until you approve. ` +
              `To approve, set "reindex": { "approveChanges": true } in the memory config and restart, ` +
              `or run the explicit reindex tool/script (scripts/reindex-now.ts). ` +
              `Reverting the embedding config restores the previous index untouched. ` +
              `Keyword/FTS recall is unaffected.`,
          );
          return;
        }

        const vs = this.vectorStore;
        const es = this.embeddingService;
        const reason = stores.reindexReason ?? "embedding config changed";
        this.logger?.info(
          `${TAG} ${reason}; reindex APPROVED (reindex.approveChanges=true); ` +
            `rebuilding vector tables and re-embedding all stored texts in the background ` +
            `(L1 + L0 through ${this.cfg.embedding.provider}/${this.cfg.embedding.model ?? es.getProviderInfo().model})...`,
        );

        // Deliberately not awaited: boot must not block on a full re-embed.
        // Keyword/FTS recall remains available while the background pass runs.
        void (async () => {
          const lockPath = path.join(this.dataDir, "reindex.lock");

          // D4: single-flight reindex across pi processes sharing the global
          // store. Atomic acquisition: create a UNIQUE temp file, then link()
          // it onto the lock path — link(2) is atomic and fails EEXIST if the
          // lock exists, so two processes can never both hold it (no
          // unlink-then-create TOCTOU). Steal rules: the lock owner embeds a
          // random nonce; on takeover we write our own and re-verify ours is
          // the one on disk. A lock older than 30 minutes is stolen outright
          // (a reindex takes minutes; 30 min covers pid-recycle and NFS
          // pid-namespace cases where liveness checks are unreliable).
          const LOCK_STALE_MS = 30 * 60 * 1000;
          const myNonce = `${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2)}`;

          const readLockOwner = (): { nonce?: string; ts?: number } | null => {
            try {
              return JSON.parse(fs.readFileSync(lockPath, "utf8")) as { nonce?: string; ts?: number };
            } catch {
              return null; // vanished between operations — treat as free
            }
          };

          const tryAcquire = (): boolean => {
            const tmpPath = `${lockPath}.${process.pid}.${Math.random().toString(36).slice(2)}.tmp`;
            try {
              fs.writeFileSync(tmpPath, JSON.stringify({ nonce: myNonce, pid: process.pid, ts: Date.now() }));
              // Atomic create-or-fail: link(2) refuses to clobber an existing
              // lock, so exactly one contender wins.
              fs.linkSync(tmpPath, lockPath);
              return true;
            } catch (err) {
              const code = (err as NodeJS.ErrnoException).code;
              if (code !== "EEXIST") {
                this.logger?.warn?.(
                  `${TAG} Reindex lock acquisition error (${code ?? "unknown"}): treating as not-acquired.`,
                );
              }
              return false;
            } finally {
              try { fs.unlinkSync(tmpPath); } catch { /* temp already gone */ }
            }
          };

          const stealStaleLock = (): boolean => {
            // Replace whatever is on disk with our lock, then verify OUR
            // nonce survived — if a faster process replaced ours, it won.
            try {
              const fd = fs.openSync(lockPath, "w");
              fs.writeSync(fd, JSON.stringify({ nonce: myNonce, pid: process.pid, ts: Date.now() }));
              fs.closeSync(fd);
            } catch {
              return false;
            }
            const now = readLockOwner();
            return now?.nonce === myNonce;
          };

          const acquireLock = (): boolean => {
            try {
              fs.mkdirSync(path.dirname(lockPath), { recursive: true });
            } catch { /* ignore */ }

            if (tryAcquire()) return true;

            // Lock exists — check staleness.
            const owner = readLockOwner();
            const age = owner?.ts ? Date.now() - owner.ts : Infinity;
            if (age > LOCK_STALE_MS) {
              this.logger?.info?.(
                `${TAG} Reindex lock stale (age ${Math.round(age / 60000)}min) — stealing.`,
              );
              return stealStaleLock();
            }
            return false; // held by a live recent owner — skip
          };

          const releaseLock = (): void => {
            // Only remove the lock if WE still own it (nonce check) — never
            // clobber a successor's lock.
            try {
              const owner = readLockOwner();
              if (owner?.nonce === myNonce) fs.unlinkSync(lockPath);
            } catch { /* best-effort */ }
          };

          if (!acquireLock()) {
            this.logger?.info?.(
              `${TAG} Reindex lock held by another live process — skipping (they will mark embedding current).`,
            );
            return;
          }

          try {
            // D3: local providers must be warm before embed() works.
            if (!es.isReady()) {
              this.logger?.info?.(`${TAG} Waiting for embedding service readiness before reindex...`);
              es.startWarmup();
              const readyAt = Date.now() + 5 * 60 * 1000; // 5 min for model download+load
              while (!es.isReady()) {
                if (Date.now() > readyAt) {
                  this.logger?.warn?.(
                    `${TAG} Embedding service not ready after 5min — aborting reindex. ` +
                      `Tables left frozen (nothing destroyed); meta stays stale; next boot will retry.`,
                  );
                  return;
                }
                await new Promise((r) => setTimeout(r, 1000));
              }
            }

            // The destructive step happens HERE — inside the approved,
            // locked, readied path: drop stale-width tables, recreate at the
            // configured dimensions, re-prepare vec statements.
            if (!vs.rebuildVecTables?.(es.getProviderInfo())) {
              this.logger?.warn?.(
                `${TAG} Vector table rebuild refused/failed — aborting reindex. ` +
                  `Tables left frozen (nothing destroyed); meta stays stale; next boot will retry.`,
              );
              return;
            }

            const result = await vs.reindexAll(
              async (text) => es.embed(text),
              (succeeded, failed, totalCount, layer) => {
                if (succeeded === 1 || succeeded % 500 === 0 || succeeded + failed === totalCount) {
                  this.logger?.info?.(
                    `${TAG} reindex progress: ${layer} ${succeeded} ok / ${failed} failed / ${totalCount} total`,
                  );
                }
              },
            );

            // D2 + H1: mark current only on zero failures AND full coverage —
            // an empty candidate set is not success (a silently broken text
            // query must not read as "index built").
            const total = result.l1Total + result.l0Total;
            const done = result.l1Count + result.l0Count;
            if (result.l1Failed === 0 && result.l0Failed === 0 && done === total) {
              // D1: only now is the store fully indexed at the new config.
              vs.markEmbeddingCurrent?.(es.getProviderInfo());
              this.logger?.info?.(
                `${TAG} Background reindex complete: L1=${result.l1Count}/${result.l1Total}, L0=${result.l0Count}/${result.l0Total}; embedding marked current.`,
              );
            } else {
              // Honest failure report; meta stays stale → next boot retries.
              this.logger?.warn?.(
                `${TAG} Reindex incomplete (L1 ${result.l1Count}/${result.l1Total} done, ${result.l1Failed} failed; ` +
                  `L0 ${result.l0Count}/${result.l0Total} done, ${result.l0Failed} failed). ` +
                  `Embedding meta left stale so the next boot retries. Keyword recall unaffected.`,
              );
            }
          } catch (err) {
            this.logger?.warn(
              `${TAG} Background reindex failed (non-fatal; keyword recall unaffected): ` +
                `${err instanceof Error ? err.message : String(err)}`,
            );
          } finally {
            releaseLock();
          }
        })();
      }
    } catch (err) {
      this.logger.warn(
        `${TAG} Store init failed; recall/dedup degraded: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }

  private wirePipelineRunners(): void {
    if (!this.scheduler) return;

    // Determine whether to use standalone LLM runner for extraction.
    // Priority: cfg.llm.enabled (explicit override) > hostType detection.
    const useStandaloneRunner = this.cfg.llm.enabled || this.hostAdapter.hostType !== "openclaw";

    const openclawConfig = (!useStandaloneRunner && this.hostAdapter.hostType === "openclaw")
      ? (this.hostAdapter as { getOpenClawConfig?(): unknown }).getOpenClawConfig?.()
      : undefined;

    // When standalone runner is active, create LLM runners from the factory.
    // If cfg.llm is configured AND we're in OpenClaw mode, build a dedicated
    // StandaloneLLMRunnerFactory from cfg.llm to override the host runner.
    let runnerFactory = this.runnerFactory;
    if (useStandaloneRunner && this.cfg.llm.enabled && this.hostAdapter.hostType === "openclaw") {
      runnerFactory = new StandaloneLLMRunnerFactory({
        config: {
          baseUrl: this.cfg.llm.baseUrl,
          apiKey: this.cfg.llm.apiKey,
          model: this.cfg.llm.model,
          maxTokens: this.cfg.llm.maxTokens,
          timeoutMs: this.cfg.llm.timeoutMs,
          disableThinking: this.cfg.llm.disableThinking,
        },
        logger: this.logger,
      });
      this.logger.debug?.(`${TAG} Using standalone LLM override: model=${this.cfg.llm.model}, baseUrl=${this.cfg.llm.baseUrl}`);
    }

    const l1LlmRunner = useStandaloneRunner
      ? runnerFactory.createRunner({ enableTools: false })
      : undefined;
    const l2l3LlmRunner = useStandaloneRunner
      ? runnerFactory.createRunner({ enableTools: true })
      : undefined;

    // L1 runner
    this.scheduler.setL1Runner(createL1Runner({
      pluginDataDir: this.dataDir,
      cfg: this.cfg,
      openclawConfig,
      vectorStore: this.vectorStore,
      embeddingService: this.embeddingService,
      logger: this.logger,
      getInstanceId: () => this.instanceId,
      llmRunner: l1LlmRunner,
    }));

    // Persister
    this.scheduler.setPersister(createPersister(this.dataDir, this.logger));

    // L2 runner
    this.scheduler.setL2Runner(async (sessionKey: string, cursor?: string) => {
      const l2Runner = createL2Runner({
        pluginDataDir: this.dataDir,
        cfg: this.cfg,
        openclawConfig,
        vectorStore: this.vectorStore,
        logger: this.logger,
        instanceId: this.instanceId,
        llmRunner: l2l3LlmRunner,
      });
      return l2Runner(sessionKey, cursor);
    });

    // L3 runner
    this.scheduler.setL3Runner(async () => {
      const l3Runner = createL3Runner({
        pluginDataDir: this.dataDir,
        cfg: this.cfg,
        openclawConfig,
        vectorStore: this.vectorStore,
        logger: this.logger,
        instanceId: this.instanceId,
        llmRunner: l2l3LlmRunner,
      });
      await l3Runner();
    });

    this.logger.debug?.(`${TAG} Pipeline runners wired`);
  }

  private ensureSchedulerStarted(): Promise<void> {
    // Fast path: already started (or starting) — every concurrent caller
    // awaits the same in-flight promise.  The promise is kept around as a
    // permanently-resolved sentinel after success so subsequent calls
    // collapse into a cheap already-resolved await.
    if (this.schedulerStartPromise) return this.schedulerStartPromise;
    if (!this.scheduler) return Promise.resolve();

    // Capture scheduler locally so TypeScript narrows inside the closure
    // even after ``this.scheduler`` is re-assigned by handleSessionEnd.
    const scheduler = this.scheduler;
    this.schedulerStartPromise = (async () => {
      try {
        const checkpoint = new CheckpointManager(this.dataDir, this.logger);
        const cp = await checkpoint.read();
        scheduler.start(checkpoint.getAllPipelineStates(cp));
        this.logger.debug?.(`${TAG} Scheduler started`);
      } catch (err) {
        this.logger.error(`${TAG} Failed to restore checkpoint: ${err instanceof Error ? err.message : String(err)}`);
        scheduler.start({});
      }
    })();

    // If the start sequence itself rejects we clear the gate so the next
    // caller can retry; on success we keep the resolved promise so it
    // short-circuits permanently.
    this.schedulerStartPromise.catch(() => {
      this.schedulerStartPromise = undefined;
    });

    return this.schedulerStartPromise;
  }
}

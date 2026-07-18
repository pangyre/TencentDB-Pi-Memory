/**
 * PiOffload — symbolic short-term memory (context offload) for pi.
 *
 * Wires the ported offload pipelines onto pi's event model:
 *   - tool_result         → buffer tool pairs; threshold-flush L1 symbolization
 *   - before_agent_start  → cache prompt; trigger L1.5 task-boundary judgment
 *   - context             → assembleContext(): L3 compression + MMD injection
 *   - session_shutdown    → final L1 flush + state save + timer disposal
 *
 * One instance per pi process/session. All storage lives under
 * `<dataDir>/offload/<agentId>/`.
 */

import { join } from "node:path";
import type { MemoryTdaiConfig } from "../../src/config.js";
import type { Logger } from "../../src/core/types.js";
import type { PluginConfig, ToolPair } from "../../src/offload/types.js";
import { OffloadStateManager } from "../../src/offload/state-manager.js";
import { LocalLlmClient } from "../../src/offload/local-llm/index.js";
import { shouldForceL1 } from "../../src/offload/hooks/llm-output.js";
import { configureTokenTracker, tiktokenCount } from "../../src/offload/context-token-tracker.js";
import { nowChinaISO } from "../../src/offload/time-utils.js";
import { PLUGIN_DEFAULTS } from "../../src/offload/types.js";
import { assembleContext } from "./assemble.js";
import { flushL1, judgeL15, simpleHash, L2Scheduler } from "./support.js";

export interface PiOffloadInitOptions {
  cfg: MemoryTdaiConfig;
  /** Extension data dir (~/.pi/agent/memory-tdai); offload lives in a subdir. */
  dataDir: string;
  agentId: string;
  sessionId: string;
  logger: Logger;
}

/** Convert pi tool-result content blocks into a storable result value. */
function contentToStorable(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((block: any) => {
        if (block?.type === "text" && typeof block.text === "string") return block.text;
        if (block?.type === "image") return "[image]";
        try {
          return JSON.stringify(block);
        } catch {
          return String(block);
        }
      })
      .join("\n");
  }
  try {
    return JSON.stringify(content ?? "");
  } catch {
    return String(content);
  }
}

export class PiOffload {
  private mgr: OffloadStateManager | undefined;
  private client: LocalLlmClient | null = null;
  private l2: L2Scheduler | undefined;
  private pCfg: Partial<PluginConfig> = {};
  private logger: Logger | undefined;
  private ready: Promise<void> | undefined;
  private disposed = false;
  private flushing = false;
  private sysTokensCache: { len: number; tokens: number } | null = null;

  get active(): boolean {
    return this.mgr !== undefined && !this.disposed;
  }

  get llmEnabled(): boolean {
    return this.client !== null;
  }

  init(opts: PiOffloadInitOptions): void {
    const { cfg, dataDir, agentId, sessionId, logger } = opts;
    this.logger = logger;

    const o = cfg.offload;
    this.pCfg = {
      model: o.model,
      temperature: o.temperature,
      forceTriggerThreshold: o.forceTriggerThreshold,
      defaultContextWindow: o.defaultContextWindow,
      maxPairsPerBatch: o.maxPairsPerBatch,
      l2NullThreshold: o.l2NullThreshold,
      l2TimeoutSeconds: o.l2TimeoutSeconds,
      mildOffloadRatio: o.mildOffloadRatio,
      aggressiveCompressRatio: o.aggressiveCompressRatio,
      mmdMaxTokenRatio: o.mmdMaxTokenRatio,
    };

    configureTokenTracker(PLUGIN_DEFAULTS.l3TiktokenEncoding);

    if (cfg.llm.enabled) {
      this.client = new LocalLlmClient(
        {
          baseUrl: cfg.llm.baseUrl,
          apiKey: cfg.llm.apiKey,
          model: o.model ?? cfg.llm.model,
          temperature: o.temperature,
          timeoutMs: cfg.llm.timeoutMs,
          disableThinking: o.disableThinking,
        },
        logger,
      );
    } else {
      logger.info(
        "[pi-offload] llm.enabled is false — L1/L1.5/L2 disabled; only threshold-based context compression is active",
      );
    }

    const mgr = new OffloadStateManager();
    this.ready = mgr
      .init(join(dataDir, "offload"), agentId, sessionId)
      .then(() => {
        mgr.setLastSessionKey(`pi:${agentId}:${sessionId}`);
        this.mgr = mgr;
        if (this.client) {
          this.l2 = new L2Scheduler(() => this.mgr, this.client, this.pCfg, logger);
        }
        logger.info(
          `[pi-offload] ready: agent=${agentId} session=${sessionId} llm=${this.client ? "on" : "off"} ` +
            `mild=${this.pCfg.mildOffloadRatio} aggressive=${this.pCfg.aggressiveCompressRatio}`,
        );
      })
      .catch((err) => {
        logger.error(
          `[pi-offload] init failed: ${err instanceof Error ? err.message : String(err)} — offload disabled`,
        );
      });
  }

  /** pi `tool_result` handler: buffer the tool pair, maybe flush L1. */
  async onToolResult(event: {
    toolName: string;
    toolCallId: string;
    input: Record<string, unknown>;
    content: unknown;
    isError?: boolean;
  }): Promise<void> {
    await this.ready;
    const mgr = this.mgr;
    if (!mgr || this.disposed) return;
    try {
      const pair: ToolPair = {
        toolName: event.toolName,
        toolCallId: event.toolCallId,
        params: event.input ?? {},
        result: contentToStorable(event.content),
        error: event.isError ? "error" : undefined,
        timestamp: nowChinaISO(),
      };
      mgr.addToolPair(pair);
      this.logger?.debug?.(
        `[pi-offload] buffered ${event.toolName} (${event.toolCallId}), pending=${mgr.getPendingCount()}`,
      );

      if (this.client && !this.flushing && shouldForceL1(mgr, this.pCfg)) {
        this.flushing = true;
        flushL1(mgr, this.client, this.logger!, "force_threshold")
          .then(() => this.l2?.kick("post_l1"))
          .catch((err) => this.logger?.warn(`[pi-offload] L1 flush failed: ${err}`))
          .finally(() => {
            this.flushing = false;
          });
      }
    } catch (err) {
      this.logger?.warn(`[pi-offload] onToolResult error: ${err}`);
    }
  }

  /** pi `before_agent_start` handler: cache prompt + trigger L1.5. */
  async onBeforeAgentStart(prompt: string): Promise<void> {
    await this.ready;
    const mgr = this.mgr;
    if (!mgr || this.disposed) return;
    if (typeof prompt !== "string" || prompt.length === 0) return;
    mgr.cachedUserPrompt = prompt;

    if (!this.client) return;
    const promptHash = simpleHash(prompt);
    if (promptHash === mgr.lastL15PromptHash) return;
    mgr.lastL15PromptHash = promptHash;
    judgeL15(mgr, this.client, this.logger!, this.l2, () => this.disposed).catch((err) => {
      this.logger?.warn(`[pi-offload] L1.5 failed: ${err}`);
    });
  }

  /**
   * pi `context` handler: run assembly. Returns replacement messages when
   * anything changed, otherwise undefined (pass-through).
   */
  async onContext(
    messages: any[],
    ctx: { model?: unknown; getSystemPrompt?: () => string },
  ): Promise<{ messages: any[] } | undefined> {
    await this.ready;
    const mgr = this.mgr;
    if (!mgr || this.disposed) return undefined;
    try {
      const contextWindow =
        (ctx.model as { contextWindow?: number } | undefined)?.contextWindow ??
        this.pCfg.defaultContextWindow ??
        PLUGIN_DEFAULTS.defaultContextWindow;

      let systemTokens = 0;
      try {
        const sys = ctx.getSystemPrompt?.() ?? "";
        if (sys.length > 0) {
          if (this.sysTokensCache && this.sysTokensCache.len === sys.length) {
            systemTokens = this.sysTokensCache.tokens;
          } else {
            systemTokens = tiktokenCount(sys);
            this.sysTokensCache = { len: sys.length, tokens: systemTokens };
          }
          mgr.cachedSystemPromptTokens = systemTokens;
        }
      } catch {
        /* fall back to ratio-based estimate inside assemble */
      }

      const result = await assembleContext(messages, {
        mgr,
        pCfg: this.pCfg,
        logger: this.logger!,
        contextWindow,
        systemTokens,
        prompt: mgr.cachedUserPrompt,
      });
      return result.changed ? { messages: result.messages } : undefined;
    } catch (err) {
      this.logger?.warn(`[pi-offload] onContext error: ${err}`);
      return undefined;
    }
  }

  /** Final flush + cleanup. Idempotent. */
  async shutdown(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    this.l2?.dispose();
    this.l2 = undefined;
    try {
      await this.ready;
      const mgr = this.mgr;
      if (mgr) {
        if (this.client && mgr.hasPending()) {
          const FLUSH_TIMEOUT_MS = 5_000;
          let timeoutId: ReturnType<typeof setTimeout> | undefined;
          try {
            await Promise.race([
              flushL1(mgr, this.client, this.logger!, "shutdown"),
              new Promise<never>((_, reject) => {
                timeoutId = setTimeout(() => reject(new Error("shutdown flush timeout")), FLUSH_TIMEOUT_MS);
              }),
            ]);
          } catch (err) {
            this.logger?.warn(`[pi-offload] shutdown flush aborted: ${err}`);
          } finally {
            if (timeoutId !== undefined) clearTimeout(timeoutId);
          }
        }
        await mgr.save().catch(() => {});
      }
    } catch {
      /* best-effort */
    }
    this.mgr = undefined;
    this.logger?.debug?.("[pi-offload] shutdown complete");
  }
}

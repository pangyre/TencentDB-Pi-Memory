/**
 * pi extension: TencentDB-Agent-Memory (Phase 1 — layered long-term memory).
 *
 * - Auto-recall: relevant L1 memories + L3 persona injected into the per-turn
 *   system prompt (not persisted to the session transcript).
 * - Auto-capture: user/assistant messages recorded to L0 on agent_end;
 *   L1/L2/L3 pipelines run in the single process holding the worker lock.
 * - Tools: tdai_memory_search, tdai_conversation_search.
 * - Command: /memory [status|search <query>]
 *
 * One global store (~/.pi/agent/memory-tdai) shared across all pi agents;
 * agent identity is baked into session keys: `pi:<agentId>:<sessionId>`.
 */

import path from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import { TdaiCore } from "../src/core/tdai-core.js";
import { StandaloneHostAdapter } from "../src/adapters/standalone/host-adapter.js";
import { SessionFilter } from "../src/utils/session-filter.js";
import { initTimeModule } from "../src/utils/time.js";
import type { Logger } from "../src/core/types.js";

import { createLogger } from "./lib/logger.js";
import {
  getDataDir,
  loadMemoryConfig,
  loadProjectOverlay,
  deriveAgentId,
  buildSessionKey,
} from "./lib/config.js";
import { PipelineWorkerLock } from "./lib/worker-lock.js";
import { PiOffload } from "./offload/index.js";

const extensionLoadTs = Date.now();

export default function (pi: ExtensionAPI) {
  let core: TdaiCore | undefined;
  let coreReady: Promise<void> | undefined;
  let lock: PipelineWorkerLock | undefined;
  let offload: PiOffload | undefined;
  let logger: Logger = createLogger(getDataDir());
  let agentId = "unknown";
  let sessionKey = "pi:unknown:ephemeral";
  let captureEnabled = false;
  let recallEnabled = false;
  let extractionActive = false;
  let warmupTriggered = false;
  let initialized = false;

  const safeGetSessionId = (ctx: ExtensionContext): string => {
    try {
      return ctx.sessionManager.getSessionId() || "ephemeral";
    } catch {
      return "ephemeral";
    }
  };

  const initMemory = (cwd: string, sessionId: string): void => {
    if (initialized) return;
    try {
      const dataDir = getDataDir();
      logger = createLogger(dataDir);
      const cfg = loadMemoryConfig(dataDir, logger);
      const overlay = loadProjectOverlay(cwd, logger);

      if (overlay.capture === false) cfg.capture.enabled = false;
      if (overlay.recall === false) cfg.recall.enabled = false;

      agentId = deriveAgentId(cwd, overlay);
      sessionKey = buildSessionKey(agentId, sessionId);

      if (cfg.extraction.enabled && !cfg.llm.enabled) {
        cfg.extraction.enabled = false;
        logger.info(
          "extraction disabled: no LLM endpoint configured (set llm.enabled + baseUrl/apiKey/model in config.json)",
        );
      }

      if (cfg.extraction.enabled) {
        lock = new PipelineWorkerLock(path.join(dataDir, "pipeline-worker.lock"), logger);
        if (!lock.tryAcquire()) {
          cfg.extraction.enabled = false;
          logger.info(
            "another pi process holds the pipeline lock — this process is capture/recall only",
          );
        }
      }

      extractionActive = cfg.extraction.enabled;
      captureEnabled = cfg.capture.enabled;
      recallEnabled = cfg.recall.enabled;

      initTimeModule({ timezone: cfg.timezone }, logger);

      const hostAdapter = new StandaloneHostAdapter({
        dataDir,
        llmConfig: {
          baseUrl: cfg.llm.baseUrl,
          apiKey: cfg.llm.apiKey,
          model: cfg.llm.model,
          maxTokens: cfg.llm.maxTokens,
          timeoutMs: cfg.llm.timeoutMs,
          disableThinking: cfg.llm.disableThinking,
        },
        logger,
        defaultUserId: "default_user",
        platform: "pi",
      });

      core = new TdaiCore({
        hostAdapter,
        config: cfg,
        sessionFilter: new SessionFilter(cfg.capture.excludeAgents),
      });
      coreReady = core.initialize().catch((err) => {
        logger.error(`core init failed: ${err instanceof Error ? err.message : String(err)}`);
      });

      if (cfg.offload.enabled) {
        offload = new PiOffload();
        offload.init({ cfg, dataDir, agentId, sessionId, logger });
      }

      initialized = true;
      logger.info(
        `memory ready: agent=${agentId} session=${sessionKey} ` +
          `capture=${captureEnabled} recall=${recallEnabled} extraction=${extractionActive} ` +
          `offload=${cfg.offload.enabled}`,
      );
    } catch (err) {
      logger.error(
        `memory init failed: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}`,
      );
      core = undefined;
    }
  };

  // ── Lifecycle ────────────────────────────────────────────────────────────

  pi.on("session_start", async (_event, ctx) => {
    const sessionId = safeGetSessionId(ctx);
    if (!initialized) {
      initMemory(ctx.cwd, sessionId);
    } else {
      sessionKey = buildSessionKey(agentId, sessionId);
    }
  });

  pi.on("before_agent_start", async (event, _ctx) => {
    offload?.onBeforeAgentStart(event.prompt).catch(() => {});

    if (!core || !recallEnabled) return;

    if (!warmupTriggered) {
      warmupTriggered = true;
      try {
        core.getEmbeddingService()?.startWarmup();
      } catch {
        // warmup is best-effort
      }
    }

    try {
      await coreReady;
      const res = await core.handleBeforeRecall(event.prompt, sessionKey);
      const parts = [res.appendSystemContext, res.prependContext].filter(
        (p): p is string => typeof p === "string" && p.length > 0,
      );
      if (parts.length === 0) return;
      logger.debug?.(
        `recall injected ${parts.join("").length} chars (strategy=${res.recallStrategy ?? "n/a"})`,
      );
      return { systemPrompt: event.systemPrompt + "\n\n" + parts.join("\n\n") };
    } catch (err) {
      logger.warn(`recall failed: ${err instanceof Error ? err.message : String(err)}`);
      return;
    }
  });

  pi.on("agent_end", async (event, ctx) => {
    if (!core || !captureEnabled) return;
    try {
      await coreReady;
      // userText is only used upstream to repair recall-polluted user messages;
      // we inject into the system prompt instead, so pi transcripts are always clean.
      const result = await core.handleTurnCommitted({
        userText: "",
        assistantText: "",
        messages: (event.messages ?? []) as unknown[],
        sessionKey,
        sessionId: safeGetSessionId(ctx),
        startedAt: extensionLoadTs,
      });
      logger.debug?.(
        `captured l0=${result.l0RecordedCount} vectors=${result.l0VectorsWritten} ` +
          `scheduler=${result.schedulerNotified}`,
      );
    } catch (err) {
      logger.warn(`capture failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  });

  pi.on("tool_result", async (event, _ctx) => {
    if (!offload) return;
    await offload.onToolResult({
      toolName: event.toolName,
      toolCallId: event.toolCallId,
      input: (event.input ?? {}) as Record<string, unknown>,
      content: event.content,
      isError: event.isError,
    });
  });

  pi.on("context", async (event, ctx) => {
    if (!offload) return;
    return offload.onContext(event.messages as any[], {
      model: (ctx as { model?: unknown }).model,
      getSystemPrompt: () => ctx.getSystemPrompt(),
    });
  });

  pi.on("session_shutdown", async (_event, _ctx) => {
    try {
      await offload?.shutdown();
    } catch {
      // best-effort
    }
    offload = undefined;
    try {
      await core?.handleSessionEnd(sessionKey);
    } catch {
      // flush is best-effort
    }
    try {
      await core?.destroy();
    } catch (err) {
      logger.warn(`shutdown error: ${err instanceof Error ? err.message : String(err)}`);
    }
    core = undefined;
    initialized = false;
    lock?.release();
  });

  // ── Tools ────────────────────────────────────────────────────────────────

  pi.registerTool({
    name: "tdai_memory_search",
    label: "Memory Search",
    description:
      "Search the user's long-term structured memories (preferences, instructions, past events) " +
      "across all pi agents. Returns memory records ranked by relevance.",
    parameters: Type.Object({
      query: Type.String({ description: "What to recall about the user" }),
      limit: Type.Optional(Type.Number({ description: "Max results (default 5, max 20)" })),
      type: Type.Optional(
        Type.String({ description: "Filter by memory type: persona | episodic | instruction" }),
      ),
      scene: Type.Optional(Type.String({ description: "Filter by scene name" })),
    }),
    async execute(_toolCallId, params) {
      if (!core) {
        return {
          content: [{ type: "text" as const, text: "Memory system not initialized." }],
          details: { count: 0, strategy: "none" },
        };
      }
      try {
        await coreReady;
        const limit = Math.min(Math.max(Number(params.limit) || 5, 1), 20);
        const r = await core.searchMemories({
          query: params.query,
          limit,
          type: params.type,
          scene: params.scene,
        });
        return {
          content: [{ type: "text" as const, text: r.text }],
          details: { count: r.total, strategy: r.strategy },
        };
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.warn(`tdai_memory_search failed: ${msg}`);
        return {
          content: [{ type: "text" as const, text: `Memory search failed: ${msg}` }],
          details: { count: 0, strategy: "error" },
          isError: true,
        };
      }
    },
  });

  pi.registerTool({
    name: "tdai_conversation_search",
    label: "Conversation Search",
    description:
      "Search raw past conversation history (exact dialogue) across all pi agents. " +
      "Use when tdai_memory_search (structured memories) lacks the detail you need.",
    parameters: Type.Object({
      query: Type.String({ description: "What conversation content to find" }),
      limit: Type.Optional(Type.Number({ description: "Max results (default 5, max 20)" })),
      session_key: Type.Optional(
        Type.String({ description: "Restrict results to one session key" }),
      ),
    }),
    async execute(_toolCallId, params) {
      if (!core) {
        return {
          content: [{ type: "text" as const, text: "Memory system not initialized." }],
          details: { count: 0 },
        };
      }
      try {
        await coreReady;
        const limit = Math.min(Math.max(Number(params.limit) || 5, 1), 20);
        const r = await core.searchConversations({
          query: params.query,
          limit,
          sessionKey: params.session_key,
        });
        return {
          content: [{ type: "text" as const, text: r.text }],
          details: { count: r.total },
        };
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.warn(`tdai_conversation_search failed: ${msg}`);
        return {
          content: [{ type: "text" as const, text: `Conversation search failed: ${msg}` }],
          details: { count: 0 },
          isError: true,
        };
      }
    },
  });

  // ── Command ──────────────────────────────────────────────────────────────

  pi.registerCommand("memory", {
    description: "Memory status and search (usage: /memory [status|search <query>])",
    handler: async (args, ctx) => {
      const a = (args ?? "").trim();
      if (a.startsWith("search ")) {
        const query = a.slice(7).trim();
        if (!core) {
          ctx.ui.notify("memory not initialized", "error");
          return;
        }
        try {
          await coreReady;
          const r = await core.searchMemories({ query, limit: 5 });
          ctx.ui.notify(r.text.length > 800 ? r.text.slice(0, 800) + "…" : r.text, "info");
        } catch (err) {
          ctx.ui.notify(
            `search failed: ${err instanceof Error ? err.message : String(err)}`,
            "error",
          );
        }
        return;
      }
      ctx.ui.notify(
        `memory-tdai — agent=${agentId} key=${sessionKey} capture=${captureEnabled} ` +
          `recall=${recallEnabled} extraction=${extractionActive ? "worker" : "off"} ` +
          `offload=${offload?.active ? (offload.llmEnabled ? "on" : "compress-only") : "off"} ` +
          `data=${getDataDir()}`,
        "info",
      );
    },
  });
}

# Contract — pi extension for TencentDB-Agent-Memory (Phase 1: long-term memory)

## Goal
A pi coding-agent extension that gives pi persistent layered memory by reusing this
repo's host-neutral core (`TdaiCore`). New code lives ONLY under `pi-extension/`.
No files outside `pi-extension/` may be created or modified.

## Environment & conventions
- Repo root: `/home/nick/TencentDB-Agent-Memory`, `"type": "module"`, Node >= 22.
- Runtime: pi loads the extension's TypeScript directly (jiti). No build step.
- All imports of repo core code from `pi-extension/*.ts` use relative paths into
  `../src/...` **with `.js` extensions** (ESM style used throughout the repo),
  e.g. `import { TdaiCore } from "../src/core/tdai-core.js";`
  From `pi-extension/lib/*.ts` it is `../../src/...`.
- Pi API imports: `import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";`
  and `import { Type } from "typebox";` — both are provided by pi at runtime.
- Node builtins use `node:` prefix. Strict TypeScript. Never throw out of event
  handlers/tools — wrap in try/catch and log via the Logger.
- The extension must never block or crash the agent: every failure degrades to
  "no memory this turn" with a logged warning.

## Pinned upstream APIs (do not re-derive; these are exact)

### `../src/core/tdai-core.js`
```ts
class TdaiCore {
  constructor(opts: { hostAdapter: HostAdapter; config: MemoryTdaiConfig; sessionFilter?: SessionFilter; instanceId?: string });
  initialize(): Promise<void>;
  destroy(): Promise<void>;
  handleBeforeRecall(userText: string, sessionKey: string): Promise<RecallResult>;
  handleTurnCommitted(turn: CompletedTurn): Promise<CaptureResult>;
  searchMemories(p: { query: string; limit?: number; type?: string; scene?: string }): Promise<{ text: string; total: number; strategy: string }>;
  searchConversations(p: { query: string; limit?: number; sessionKey?: string }): Promise<{ text: string; total: number }>;
  handleSessionEnd(sessionKey: string): Promise<void>;
  getEmbeddingService(): { startWarmup(): void; isReady(): boolean } | undefined;
}
```

### `../src/core/types.js`
```ts
interface RecallResult { prependContext?: string; appendSystemContext?: string; recalledL1Memories?: Array<{content:string;score:number;type:string}>; recalledL3Persona?: string|null; recallStrategy?: string; }
interface CompletedTurn { userText: string; assistantText: string; messages: unknown[]; sessionKey: string; sessionId?: string; startedAt?: number; originalUserMessageCount?: number; }
interface CaptureResult { l0RecordedCount: number; schedulerNotified: boolean; l0VectorsWritten: number; filteredMessages: Array<{role:string;content:string;timestamp:number}>; }
interface Logger { debug?(msg: string): void; info(msg: string): void; warn(msg: string): void; error(msg: string): void; }
```
`CompletedTurn.messages` accepts objects shaped `{ role: "user"|"assistant", content: string | Array<{type:"text",text:string}|...>, timestamp?: number }` — pi's session messages already match; pass `event.messages` through unmodified.

### `../src/adapters/standalone/host-adapter.js`
```ts
class StandaloneHostAdapter {
  constructor(opts: { dataDir: string; llmConfig: StandaloneLLMConfig; logger: Logger; defaultUserId?: string; platform?: string });
}
// StandaloneLLMConfig: { baseUrl: string; apiKey: string; model: string; maxTokens?: number; timeoutMs?: number; disableThinking?: false|string }
```

### `../src/config.js`
```ts
function parseConfig(raw: Record<string, unknown> | undefined): MemoryTdaiConfig;
// Fields used by the extension:
// cfg.capture.enabled: boolean         cfg.capture.excludeAgents: string[]
// cfg.recall.enabled: boolean          cfg.extraction.enabled: boolean
// cfg.llm: { enabled: boolean; baseUrl: string; apiKey: string; model: string; maxTokens: number; timeoutMs: number; disableThinking: false|string }
// cfg.timezone?: string
```
`parseConfig(undefined)` returns full defaults — always safe.

### `../src/utils/session-filter.js`
```ts
class SessionFilter { constructor(excludePatterns: string[]); }
```

### `../src/utils/time.js`
```ts
function initTimeModule(opts: { timezone?: string }, logger: Logger): void; // call once before core.initialize()
```

### pi ExtensionAPI surface used (from `@earendil-works/pi-coding-agent`)
```ts
export default function (pi: ExtensionAPI) { ... }   // sync factory
pi.on("session_start", async (event, ctx) => {...});          // event.reason: "startup"|"reload"|"new"|"resume"|"fork"
pi.on("before_agent_start", async (event, ctx) => {           // event.prompt: string; event.systemPrompt: string
  return { systemPrompt: event.systemPrompt + "..." };        // per-turn, NOT persisted to session
});
pi.on("agent_end", async (event, ctx) => {...});              // event.messages: messages from this run
pi.on("session_shutdown", async (event, ctx) => {...});
pi.registerTool({ name, label, description, parameters: Type.Object({...}),
  async execute(toolCallId, params, signal, onUpdate, ctx) {
    return { content: [{ type: "text" as const, text: "..." }], details: {} };
  }});
pi.registerCommand("memory", { description: "...", handler: async (args: string, ctx) => {...} });
// ctx.sessionManager.getSessionId(): string   ctx.cwd: string
// ctx.ui.notify(text: string, "info"|"warning"|"error"): void
```

## Cross-file design decisions (pinned)
- **Data dir (global store):** `path.join(os.homedir(), ".pi", "agent", "memory-tdai")`.
- **Global config:** JSON at `<dataDir>/config.json`, parsed with `parseConfig`. Missing/invalid file → `parseConfig(undefined)` + warn.
- **Project overlay:** JSON at `<cwd>/.pi/memory-tdai.json`, shape `{ agentName?: string; capture?: boolean; recall?: boolean }`. Missing → `{}`.
- **Agent identity:** `agentId = sanitize(overlay.agentName)` if set, else `sanitize(basename(cwd)) + "-" + sha256(cwd).slice(0,6)`. `sanitize` = lowercase, replace `[^a-z0-9-]` runs with `-`, trim `-`.
- **Session key:** `` `pi:${agentId}:${sessionId}` `` where sessionId = `ctx.sessionManager.getSessionId()` or `"ephemeral"`.
- **Memory injection goes into the SYSTEM PROMPT only** (per-turn chaining via `before_agent_start` return). Never mutate/prepend user messages — pi does not persist per-turn system prompts, so no transcript-stripping is needed.
- **Pipeline worker lock:** only one pi process runs L1/L2/L3 extraction. Lock file `<dataDir>/pipeline-worker.lock` (JSON `{pid: number, ts: number}`). Non-holders force `extraction.enabled = false` (capture/recall still on). Stale = owning pid dead OR ts older than 120s. Heartbeat rewrites the file every 30s.
- **Extraction also forced off when `cfg.llm.enabled === false`** (no LLM endpoint configured → L1/L2/L3 cannot run; log info once).
- **Logger:** `createLogger(dataDir)` from `lib/logger.ts`; used everywhere; also passed to StandaloneHostAdapter and initTimeModule.

## Files & ownership (dep graph)
1. `pi-extension/lib/logger.ts` — SPRINT-001 (no deps)
2. `pi-extension/lib/config.ts` — SPRINT-002 (no deps)
3. `pi-extension/lib/worker-lock.ts` — SPRINT-003 (no deps)
4. `pi-extension/index.ts` — SPRINT-004 (deps: 1,2,3)
5. `pi-extension/tsconfig.json` — SPRINT-005 (no deps)
6. `pi-extension/README.md` — SPRINT-006 (deps: 4)

## Exported seams (must match exactly)
```ts
// lib/logger.ts
export function createLogger(dataDir: string): Logger;   // Logger from ../../src/core/types.js
// lib/config.ts
export interface ProjectOverlay { agentName?: string; capture?: boolean; recall?: boolean; }
export function getDataDir(): string;
export function loadMemoryConfig(dataDir: string, logger: Logger): MemoryTdaiConfig;
export function loadProjectOverlay(cwd: string, logger: Logger): ProjectOverlay;
export function deriveAgentId(cwd: string, overlay: ProjectOverlay): string;
export function buildSessionKey(agentId: string, sessionId: string): string;
// lib/worker-lock.ts
export class PipelineWorkerLock {
  constructor(lockPath: string, logger: Logger);
  tryAcquire(): boolean;   // atomic create; steals stale locks; starts 30s heartbeat on success; unref() the timer
  release(): void;         // stop heartbeat, unlink only if we own it; idempotent
  get held(): boolean;
}
```

## Gate
`npx tsc --noEmit -p pi-extension/tsconfig.json` must pass from the repo root.

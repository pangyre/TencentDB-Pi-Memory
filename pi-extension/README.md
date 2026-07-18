# pi extension — TencentDB-Agent-Memory

A [pi coding agent](https://github.com/badlogic/pi-mono) extension that gives pi
persistent, layered long-term memory, powered by this repo's host-neutral core
(`TdaiCore`): L0 conversation records → L1 structured memories → L2 scenes →
L3 persona, with local SQLite + vector/FTS search.

Includes both memory halves:

- **Long-term memory** (Phase 1): auto-recall, auto-capture, memory tools, `/memory` command.
- **Symbolic short-term memory / context offload** (Phase 2): heavy tool results
  are symbolized into compact summaries + mermaid task graphs, and the live
  context is compressed non-destructively before every LLM call.

## How it works

| Feature | Mechanism |
|---|---|
| Auto-recall | On each prompt, relevant L1 memories + L3 persona are appended to the **per-turn system prompt** (`before_agent_start`). Never persisted into the session transcript — no cleanup hacks needed. |
| Auto-capture | On `agent_end`, user/assistant messages are recorded to L0 and the L1/L2/L3 extraction pipelines are scheduled. |
| `tdai_memory_search` | LLM-callable tool: search structured memories (`persona` / `episodic` / `instruction`), optional scene filter. |
| `tdai_conversation_search` | LLM-callable tool: search raw conversation history across **all** pi agents. |
| `/memory` | `/memory` or `/memory status` shows agent id, session key, feature flags. `/memory search <query>` runs a manual memory search. |
| Context offload (opt-in) | On `tool_result`, tool call/result pairs are buffered and symbolized by a local LLM (L1) into `offload.jsonl` + raw ref files; an L1.5 judge tracks task boundaries and an L2 pipeline maintains mermaid task graphs. On every `context` event the live message list is compressed: offloaded tool results are swapped for summaries (score cascade), old task ranges are head-deleted with their mermaid graph injected instead, and an emergency cutter guards the context window. All non-destructive — the session transcript on disk is never modified. |

## Multi-agent model

All pi agents (one per project directory) share **one global store** at
`~/.pi/agent/memory-tdai/`:

- Each project gets a stable **agent id** (`<dirname>-<hash>` or an explicit
  `agentName`) baked into its session keys: `pi:<agentId>:<sessionId>`.
- **L3 persona is user-level** — synthesized from all agents, injected everywhere.
- **L0 conversations are agent-scoped but globally searchable** — a triaging /
  personal-assistant agent can search what any specialist agent discussed.
- A **pipeline worker lock** (`pipeline-worker.lock`, heartbeat + stale
  detection) ensures only one pi process runs LLM extraction at a time; other
  concurrent processes stay capture/recall-only. The SQLite store runs in WAL
  mode, so concurrent multi-process access is safe.

## Install

The repo is a **pi package**: `package.json` carries a `pi` manifest
(`"pi": { "extensions": ["./pi-extension/index.ts"] }`), so pi discovers the
extension automatically from any of the install methods below. Pi loads the
TypeScript directly — no build step.

### Option A — install from git (recommended for sharing)

```bash
pi install git:github.com/nicwn/TencentDB-Pi-Memory
# or pinned: pi install git:github.com/nicwn/TencentDB-Pi-Memory@<tag-or-commit>
```

pi clones the repo under `~/.pi/agent/git/` and runs `npm install` for you.
Use `-l` to install into the project's `.pi/settings.json` instead of the
global settings, or try it once without installing:

```bash
pi -e git:github.com/nicwn/TencentDB-Pi-Memory
```

### Option B — install from a local checkout

```bash
cd /path/to/TencentDB-Agent-Memory && npm install   # once
pi install /path/to/TencentDB-Agent-Memory
```

### Option C — register the extension file manually

Add the entry file to `~/.pi/agent/settings.json` (after `npm install` in the
repo):

```json
{
  "extensions": ["/path/to/TencentDB-Agent-Memory/pi-extension/index.ts"]
}
```

Then start a new pi session (or `/reload`). `/memory` should show status.

> Note: the repo's npm `postinstall` hook only applies an OpenClaw runtime
> patch when OpenClaw is present; on a pi-only machine it logs a skip and does
> nothing.

## Configuration

### Global: `~/.pi/agent/memory-tdai/config.json`

Same schema as the OpenClaw plugin's `pluginConfig` (see repo README). Everything
works zero-config (keyword-only search, no extraction). To enable the full
pipeline, configure the LLM and embedding endpoints:

```json
{
  "llm": {
    "enabled": true,
    "baseUrl": "http://localhost:11434/v1",
    "apiKey": "sk-...",
    "model": "qwen3:8b"
  },
  "embedding": {
    "provider": "openai",
    "baseUrl": "http://localhost:11434/v1",
    "apiKey": "sk-...",
    "model": "bge-m3",
    "dimensions": 1024
  }
}
```

- `llm.enabled: false` (default) → L1/L2/L3 extraction is disabled; capture and
  keyword recall still work.
- `embedding.provider: "none"` (default) → FTS keyword search only, no vectors.

To enable context offload add:

```json
{
  "offload": {
    "enabled": true,
    "mildOffloadRatio": 0.5,
    "aggressiveCompressRatio": 0.85
  }
}
```

- Offload symbolization (L1/L1.5/L2) uses the same `llm` endpoint; set
  `offload.model` to override the model for offload tasks only.
- With `llm.enabled: false`, offload runs in **compress-only** mode: no
  summaries/task graphs, but threshold-based context compression still protects
  the context window.
- Offload state lives under `~/.pi/agent/memory-tdai/offload/<agentId>/`
  (`offload-<session>.jsonl`, `refs/`, `mmds/`, `state.json`).

### Per-project overlay: `<project>/.pi/memory-tdai.json`

```json
{
  "agentName": "tax-agent",
  "capture": true,
  "recall": true
}
```

- `agentName` — stable, human-chosen agent identity (recommended for your
  specialist agents; otherwise a dirname+hash id is derived).
- `capture` / `recall` — set `false` to opt this project out.

## Troubleshooting

- Log file: `~/.pi/agent/memory-tdai/logs/pi-extension.log`
- `PI_MEMORY_DEBUG=1 pi` echoes debug logs to stderr.
- `/memory` shows `extraction=off` when `llm.enabled` is false **or** another pi
  process holds the pipeline worker lock.
- Typecheck (dev): `npx tsc --noEmit -p pi-extension/tsconfig.json` from the repo
  root — expect zero errors in `pi-extension/` (a handful of pre-existing,
  typecheck-only errors surface in `src/`).

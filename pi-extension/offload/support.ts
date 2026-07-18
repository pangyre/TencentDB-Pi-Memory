/**
 * Offload pipeline runners for the pi port — L1 symbolization, L1.5 task
 * judgment, and L2 mermaid graph construction, all against LocalLlmClient.
 *
 * Ported from src/offload/index.ts (OpenClaw glue), minus: backend mode,
 * opik tracing, heartbeat filtering (pi has no heartbeat cron), and the
 * multi-session registry (pi runs one session per process).
 */

import { join } from "node:path";
import { mkdir, writeFile } from "node:fs/promises";
import type { OffloadStateManager } from "../../src/offload/state-manager.js";
import { LocalLlmClient } from "../../src/offload/local-llm/index.js";
import type { L1Request, L15Request, L2Request } from "../../src/offload/backend-client.js";
import type { OffloadEntry, PluginConfig, PluginLogger, ToolPair } from "../../src/offload/types.js";
import {
  appendOffloadEntries,
  readAllOffloadEntries,
  rewriteAllOffloadEntries,
  writeRefMd,
  sanitizeText,
  listMmds,
  readMmd,
  writeMmd,
  patchMmd,
} from "../../src/offload/storage.js";
import { checkL2Trigger, backfillNodeIds } from "../../src/offload/pipelines/l2-mermaid.js";
import { handleTaskTransition, normalizeJudgment } from "../../src/offload/hooks/before-agent-start.js";
import { parseMmdMeta } from "../../src/offload/mmd-meta.js";
import { nowChinaISO } from "../../src/offload/time-utils.js";

const MAX_L1_CHUNK_RETRIES = 3;
const L1_BATCH_SIZE = 5;
const L2_BATCH_SIZE = 30;
const L15_RETRY_DELAY_MS = 3000;

// ─── Context builders (ported from src/offload/index.ts) ───────────────────

export function simpleHash(str: string): number {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash + str.charCodeAt(i)) | 0;
  }
  return hash;
}

export function msgFingerprint(msg: any): number {
  const role = msg.role ?? msg.message?.role ?? msg.type ?? "";
  let content = "";
  const raw = msg.type === "message" ? msg.message?.content : msg.content;
  if (typeof raw === "string") content = raw.slice(0, 200);
  else if (Array.isArray(raw)) content = JSON.stringify(raw).slice(0, 200);
  return simpleHash(`${role}:${content}`);
}

export function extractLatestTurn(currentPrompt: string | null): string | null {
  if (!currentPrompt) return null;
  return `[User]: ${String(currentPrompt).slice(0, 500)}`;
}

function extractMsgText(msg: any): string {
  const content = msg.content ?? msg.message?.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((c: any) => c.type === "text" && typeof c.text === "string")
      .map((c: any) => c.text)
      .join(" ");
  }
  return "";
}

function normalizePromptForCompare(text: string | null): string {
  return String(text ?? "").replace(/\s+/g, " ").trim();
}

/** Recent history as user/assistant turn pairs (up to 5 turns). */
export function extractRecentHistory(
  messages: any[],
  currentPrompt: string | null = null,
  maxAssistantPerUser = 3,
): string | null {
  const normalizedCurrent = normalizePromptForCompare(currentPrompt);
  const turns: Array<{ user: string; assistants: string[] }> = [];
  let currentTurn: { user: string; assistants: string[] } | null = null;

  for (const msg of messages) {
    if (msg._mmdContextMessage || msg._mmdInjection) continue;
    const role = msg.role ?? msg.message?.role ?? msg.type;

    if (role === "user") {
      let text = extractMsgText(msg);
      if (!text || text.length <= 5) continue;
      text = text.slice(0, 400);
      if (normalizedCurrent) {
        const normalizedText = normalizePromptForCompare(text);
        if (
          normalizedText === normalizedCurrent ||
          normalizedText.startsWith(normalizedCurrent) ||
          normalizedCurrent.startsWith(normalizedText)
        ) {
          continue;
        }
      }
      currentTurn = { user: text, assistants: [] };
      turns.push(currentTurn);
    } else if (role === "assistant" && currentTurn) {
      if (currentTurn.assistants.length >= maxAssistantPerUser) continue;
      const directText = extractMsgText(msg);
      if (!directText || directText.length <= 10) continue;
      currentTurn.assistants.push(directText.slice(0, 400));
    }
  }

  const recentTurns = turns.slice(-5);
  const parts: string[] = [];
  for (const turn of recentTurns) {
    parts.push(`[User]: ${turn.user}`);
    for (const a of turn.assistants) parts.push(`[Assistant]: ${a}`);
  }
  return parts.length > 0 ? parts.join("\n") : null;
}

function buildL1RecentContext(mgr: OffloadStateManager): string {
  const rawPrompt = mgr.cachedUserPrompt;
  const currentLine =
    typeof rawPrompt === "string" && rawPrompt.trim()
      ? `[User]: ${rawPrompt.slice(0, 500)}`
      : mgr.cachedLatestTurnMessages || "(none)";
  const historyBlock = mgr.cachedRecentHistory || "(none)";
  return `## current msg:\n${currentLine}\n\n## history msg:\n${historyBlock}`;
}

function buildL15RecentContext(mgr: OffloadStateManager): string {
  const rawPrompt = mgr.cachedUserPrompt;
  const currentLine =
    typeof rawPrompt === "string" && rawPrompt.trim()
      ? `[User]: ${rawPrompt.slice(0, 500)}`
      : mgr.cachedLatestTurnMessages || "(none)";
  const historyBlock = mgr.cachedRecentHistory || "(none)";
  return `历史消息，可作为参考：\n${historyBlock}\n\n最新user message：\n${currentLine}`;
}

// ─── L1: tool-pair symbolization ────────────────────────────────────────────

export async function flushL1(
  mgr: OffloadStateManager,
  client: LocalLlmClient,
  logger: PluginLogger,
  triggerSource: string,
  maxCount?: number,
): Promise<void> {
  if (!mgr.hasPending()) return;

  const release = await mgr.acquireL1Lock();
  try {
    const pendingCount = mgr.getPendingCount();
    const takeCount = maxCount != null ? Math.min(maxCount, pendingCount) : pendingCount;
    const pairs = mgr.takePending(takeCount);
    if (pairs.length === 0) return;

    // L1.1: write raw ref MD files (full tool results, for later recovery)
    const refByToolCallId = new Map<string, string>();
    for (const p of pairs) {
      try {
        const resultStr =
          typeof p.result === "string"
            ? sanitizeText(p.result)
            : sanitizeText(JSON.stringify(p.result, null, 2));
        const content = `**Tool:** ${p.toolName}\n**Call ID:** ${p.toolCallId}\n\n**Result:**\n\`\`\`\n${resultStr}\n\`\`\``;
        const refPath = await writeRefMd(mgr.ctx, p.timestamp, p.toolName, content);
        refByToolCallId.set(p.toolCallId, refPath);
      } catch (err) {
        logger.error(`[pi-offload] L1.1 ref write error (${p.toolCallId}): ${err}`);
      }
    }

    const batches: ToolPair[][] = [];
    for (let i = 0; i < pairs.length; i += L1_BATCH_SIZE) {
      batches.push(pairs.slice(i, i + L1_BATCH_SIZE));
    }
    logger.debug?.(
      `[pi-offload] L1 (${triggerSource}): ${pairs.length} pairs → ${batches.length} batch(es)`,
    );

    const recentMessages = buildL1RecentContext(mgr);

    for (const chunk of batches) {
      const chunkKey = chunk[0].toolCallId;
      const prevFails = mgr._l1ChunkFailCounts.get(chunkKey) ?? 0;

      try {
        const req: L1Request = {
          recentMessages,
          toolPairs: chunk.map((p) => ({
            toolName: p.toolName,
            toolCallId: p.toolCallId,
            params: typeof p.params === "string" ? sanitizeText(p.params) : p.params,
            result: typeof p.result === "string" ? sanitizeText(p.result as string) : p.result,
            timestamp: p.timestamp,
          })),
        };
        const resp = await client.l1Summarize(req);

        mgr._l1ChunkFailCounts.delete(chunkKey);
        if (resp.entries && resp.entries.length > 0) {
          for (const entry of resp.entries) {
            if (!entry.result_ref && refByToolCallId.has(entry.tool_call_id)) {
              entry.result_ref = refByToolCallId.get(entry.tool_call_id)!;
            }
          }
          await appendOffloadEntries(mgr.ctx, resp.entries, undefined, logger);
          mgr.entryCounter += resp.entries.length;
          logger.debug?.(
            `[pi-offload] L1 batch OK: ${resp.entries.length} entries from ${chunk.length} pairs (counter=${mgr.entryCounter})`,
          );
        }
      } catch (err) {
        const newFails = prevFails + 1;
        logger.warn(
          `[pi-offload] L1 batch FAILED (${chunkKey}, attempt ${newFails}/${MAX_L1_CHUNK_RETRIES}): ${err}`,
        );

        if (newFails >= MAX_L1_CHUNK_RETRIES) {
          mgr._l1ChunkFailCounts.delete(chunkKey);
          const fallbackEntries: OffloadEntry[] = [];
          for (const p of chunk) {
            const resultStr = typeof p.result === "string" ? p.result : JSON.stringify(p.result ?? "");
            const truncResult = resultStr.length > 300 ? resultStr.slice(0, 297) + "..." : resultStr;
            const truncParams =
              typeof p.params === "string"
                ? p.params.length > 200
                  ? p.params.slice(0, 197) + "..."
                  : p.params
                : JSON.stringify(p.params ?? "").slice(0, 200);
            fallbackEntries.push({
              timestamp: p.timestamp,
              node_id: null,
              tool_call: `${p.toolName}(${truncParams})`,
              summary: `[L1 degraded] ${p.toolName}: ${truncResult}`,
              result_ref: refByToolCallId.get(p.toolCallId) ?? "",
              tool_call_id: p.toolCallId,
              score: 0,
            });
          }
          await appendOffloadEntries(mgr.ctx, fallbackEntries, undefined, logger);
          mgr.entryCounter += fallbackEntries.length;
          logger.debug?.(`[pi-offload] L1 fallback: wrote ${fallbackEntries.length} degraded entries`);
        } else {
          mgr._l1ChunkFailCounts.set(chunkKey, newFails);
          for (const p of chunk) {
            mgr.processedToolCallIds.delete(p.toolCallId);
            mgr.pendingToolPairs.push(p as ToolPair & { _sessionId?: string | null });
          }
          logger.debug?.(
            `[pi-offload] L1 batch: re-enqueued ${chunk.length} pairs (retry ${newFails}/${MAX_L1_CHUNK_RETRIES})`,
          );
        }
      }
    }
  } finally {
    release();
  }
}

// ─── L2: mermaid task-graph construction ────────────────────────────────────

export async function runL2(
  mgr: OffloadStateManager,
  client: LocalLlmClient,
  logger: PluginLogger,
  entriesByMmd: Map<string, OffloadEntry[]>,
  triggerSource: string,
): Promise<void> {
  try {
    for (const [mmdFile, mmdEntries] of entriesByMmd) {
      const taskLabel = mmdFile.replace(/^\d+-/, "").replace(/\.mmd$/, "") || "unnamed-task";
      const prefixMatch = mmdFile.match(/^(\d+)-/);
      const mmdPrefix = prefixMatch ? prefixMatch[1] : "000";

      const batches: OffloadEntry[][] = [];
      for (let i = 0; i < mmdEntries.length; i += L2_BATCH_SIZE) {
        batches.push(mmdEntries.slice(i, i + L2_BATCH_SIZE));
      }
      logger.debug?.(
        `[pi-offload] L2 (${triggerSource}): mmd=${mmdFile}, ${mmdEntries.length} entries → ${batches.length} batch(es)`,
      );

      for (let bIdx = 0; bIdx < batches.length; bIdx++) {
        const batch = batches[bIdx];
        const batchWaitIds = new Set(batch.map((e) => e.tool_call_id));
        const existingMmd = await readMmd(mgr.ctx, mmdFile);

        const req: L2Request = {
          existingMmd,
          newEntries: batch.map((e) => ({
            tool_call_id: e.tool_call_id,
            tool_call: e.tool_call,
            summary: e.summary,
            timestamp: e.timestamp,
          })),
          recentHistory: mgr.cachedRecentHistory || null,
          currentTurn: mgr.cachedLatestTurnMessages || null,
          taskLabel,
          mmdPrefix,
          mmdCharCount: existingMmd ? existingMmd.length : 0,
        };

        // Mark batch entries as "wait" before the LLM call
        const allEntries = await readAllOffloadEntries(mgr.ctx);
        let changed = false;
        for (const entry of allEntries) {
          if (batchWaitIds.has(entry.tool_call_id) && entry.node_id === null) {
            entry.node_id = "wait";
            changed = true;
          }
        }
        if (changed) await rewriteAllOffloadEntries(mgr.ctx, allEntries);
        if (bIdx === 0) {
          mgr.setLastL2TriggerTime(nowChinaISO());
          await mgr.save();
        }

        try {
          const resp = await client.l2Generate(req);

          if (!resp.fileAction) {
            logger.warn(
              `[pi-offload] L2 [${mmdFile}] batch ${bIdx + 1}/${batches.length}: degraded response, fallback backfill`,
            );
            await backfillNodeIds(mgr.ctx, resp.nodeMapping ?? {}, batchWaitIds, logger, {
              mmdFallbackText: existingMmd ?? "",
              mmdPrefix,
            });
            continue;
          }

          if (resp.fileAction === "replace" && resp.replaceBlocks && resp.replaceBlocks.length > 0) {
            const patchOk = await patchMmd(mgr.ctx, mmdFile, resp.replaceBlocks);
            if (!patchOk && resp.mmdContent) {
              await writeMmd(mgr.ctx, mmdFile, resp.mmdContent);
            }
          } else if (resp.mmdContent) {
            await writeMmd(mgr.ctx, mmdFile, resp.mmdContent);
          }

          const mmdAfterWrite = await readMmd(mgr.ctx, mmdFile);
          const mmdForBackfill =
            typeof mmdAfterWrite === "string" && mmdAfterWrite.trim().length > 0
              ? mmdAfterWrite
              : typeof existingMmd === "string" && existingMmd.trim().length > 0
                ? existingMmd
                : "";
          await backfillNodeIds(mgr.ctx, resp.nodeMapping ?? {}, batchWaitIds, logger, {
            mmdFallbackText: mmdForBackfill,
            mmdPrefix,
          });

          logger.debug?.(
            `[pi-offload] L2 [${mmdFile}] batch ${bIdx + 1}/${batches.length}: applied, action=${resp.fileAction}`,
          );
        } catch (err) {
          logger.error(`[pi-offload] L2 [${mmdFile}] batch ${bIdx + 1}/${batches.length} failed: ${err}`);
        }
      }
    }
  } catch (err) {
    logger.error(`[pi-offload] L2 failed: ${err}`);
  }
}

/** L2 scheduler: threshold + timeout driven, single-flight, with a poll timer. */
export class L2Scheduler {
  private running = false;
  private pollTimer: ReturnType<typeof setInterval> | undefined;
  private disposed = false;

  constructor(
    private readonly getMgr: () => OffloadStateManager | undefined,
    private readonly client: LocalLlmClient,
    private readonly pCfg: Partial<PluginConfig>,
    private readonly logger: PluginLogger,
  ) {
    this.pollTimer = setInterval(() => {
      this.kick("poll").catch(() => {});
    }, 60_000);
    this.pollTimer.unref?.();
  }

  async kick(source: string): Promise<void> {
    if (this.running || this.disposed) return;
    const mgr = this.getMgr();
    if (!mgr) return;
    this.running = true;
    try {
      const { shouldTrigger, reason, entriesByMmd } = await checkL2Trigger(mgr, this.pCfg, this.logger);
      if (!shouldTrigger) return;
      const total = Array.from(entriesByMmd.values()).reduce((s, a) => s + a.length, 0);
      this.logger.debug?.(`[pi-offload] L2 triggered (${source}): ${reason}, ${total} entries`);
      await runL2(mgr, this.client, this.logger, entriesByMmd, source);
    } catch (err) {
      this.logger.error(`[pi-offload] L2 trigger error: ${err}`);
    } finally {
      this.running = false;
    }
  }

  dispose(): void {
    this.disposed = true;
    if (this.pollTimer !== undefined) clearInterval(this.pollTimer);
    this.pollTimer = undefined;
  }
}

// ─── L1.5: task boundary judgment ───────────────────────────────────────────

async function l15FailSafe(mgr: OffloadStateManager, logger: PluginLogger, startIndex: number): Promise<void> {
  mgr.setActiveMmd(null, null);
  mgr.pushBoundary({ startIndex, result: "short", targetMmd: null });
  await mgr.save();
  mgr.setMmdInjectionReady(false);
  mgr.l15Settled = true;
  logger.warn(`[pi-offload] L1.5 fail-safe: settled (boundary short @${startIndex})`);
}

async function attemptL15(
  mgr: OffloadStateManager,
  client: LocalLlmClient,
  logger: PluginLogger,
  startIndex: number,
  l2: L2Scheduler | undefined,
): Promise<boolean> {
  try {
    const allMmds = await listMmds(mgr.ctx);
    const availableMmds = allMmds.slice(-10);
    const mmdMetas: L15Request["availableMmdMetas"] = [];
    for (const mmdFile of availableMmds) {
      try {
        const content = await readMmd(mgr.ctx, mmdFile);
        if (content) {
          mmdMetas.push(parseMmdMeta(mmdFile, join(mgr.ctx.mmdsDir, mmdFile), content));
        }
      } catch {
        /* skip */
      }
    }
    const currentMmdFilename = mgr.getActiveMmdFile();
    let currentMmd: L15Request["currentMmd"] = null;
    if (currentMmdFilename) {
      const content = await readMmd(mgr.ctx, currentMmdFilename);
      if (content) {
        currentMmd = {
          filename: currentMmdFilename,
          content,
          path: join(mgr.ctx.mmdsDir, currentMmdFilename),
        };
      }
    }
    const recentMessages = buildL15RecentContext(mgr);

    mgr.setMmdInjectionReady(false);
    const resp = await client.l15Judge({ recentMessages, currentMmd, availableMmdMetas: mmdMetas });

    const judgment = normalizeJudgment(resp as unknown as Record<string, unknown>);
    if (!judgment) {
      logger.warn("[pi-offload] L1.5: all-null response (LLM unavailable)");
      return false;
    }

    logger.debug?.(
      `[pi-offload] L1.5: completed=${judgment.taskCompleted}, continuation=${judgment.isContinuation}, ` +
        `longTask=${judgment.isLongTask}, label=${judgment.newTaskLabel ?? "none"}`,
    );

    const prevMmdFile = currentMmdFilename;
    await handleTaskTransition(mgr, judgment, logger);

    const newMmdFile = mgr.getActiveMmdFile();
    const mmdSwitched = prevMmdFile && newMmdFile !== prevMmdFile;
    if (mmdSwitched && l2) {
      // Flush residual null entries belonging to the OLD mmd (fire-and-forget)
      const flushStartIndex = startIndex;
      const flushPrevMmd = prevMmdFile!;
      (async () => {
        try {
          const allEntries = await readAllOffloadEntries(mgr.ctx);
          const residualEntries: OffloadEntry[] = [];
          for (let idx = 0; idx < allEntries.length && idx < flushStartIndex; idx++) {
            const e = allEntries[idx];
            if (e.node_id === null || e.node_id === "wait") residualEntries.push(e);
          }
          if (residualEntries.length === 0) return;
          const residualByMmd = new Map<string, OffloadEntry[]>();
          residualByMmd.set(flushPrevMmd, residualEntries);
          logger.debug?.(
            `[pi-offload] L1.5 task-switch flush: ${residualEntries.length} residual entries for old mmd=${flushPrevMmd}`,
          );
          await runL2(mgr, client, logger, residualByMmd, "task_switch_flush");
        } catch (flushErr) {
          logger.warn(`[pi-offload] L1.5 task-switch flush failed: ${flushErr}`);
        }
      })().catch(() => {});
    }

    const activeMmdFile = mgr.getActiveMmdFile();
    if (activeMmdFile) {
      mgr.pushBoundary({ startIndex, result: "long", targetMmd: activeMmdFile });
    } else {
      mgr.pushBoundary({ startIndex, result: "short", targetMmd: null });
    }

    await mgr.save();
    mgr.setMmdInjectionReady(true);
    mgr.l15Settled = true;
    logger.debug?.("[pi-offload] L1.5: settled, MMD injection ready");
    return true;
  } catch (err) {
    logger.warn(`[pi-offload] L1.5 attempt failed: ${err}`);
    return false;
  }
}

/** Judge task boundary for a new user prompt. Pre-flushes existing pairs first. */
export async function judgeL15(
  mgr: OffloadStateManager,
  client: LocalLlmClient,
  logger: PluginLogger,
  l2: L2Scheduler | undefined,
  isDisposed: () => boolean,
): Promise<void> {
  mgr.l15Settled = false;

  const snapshotCount = mgr.getPendingCount();
  if (snapshotCount > 0) {
    try {
      await flushL1(mgr, client, logger, "l15_pre_flush", snapshotCount);
    } catch (err) {
      logger.warn(`[pi-offload] L1.5 pre-flush failed: ${err}`);
    }
  }

  const startIndex = mgr.entryCounter;
  logger.debug?.(`[pi-offload] L1.5 boundary startIndex=${startIndex} (pre-flushed=${snapshotCount})`);

  if (await attemptL15(mgr, client, logger, startIndex, l2)) return;

  const retry = async () => {
    await new Promise((r) => setTimeout(r, L15_RETRY_DELAY_MS));
    if (isDisposed() || mgr.l15Settled) return;
    logger.debug?.("[pi-offload] L1.5 retrying... (1/1)");
    if (await attemptL15(mgr, client, logger, startIndex, l2)) return;
    logger.warn("[pi-offload] L1.5 FAILED after 1 retry, activating fail-safe");
    await l15FailSafe(mgr, logger, startIndex);
  };
  retry().catch(() => {});
}

/**
 * Context assembly for the pi port — the L3 heart of symbolic short-term memory.
 *
 * Runs on pi's `context` event (fires before EVERY LLM call, including between
 * tool calls, with a deep copy of the messages). Ported from
 * OffloadContextEngine.assemble() in src/offload/index.ts, minus opik tracing,
 * backend reporting, patch-effectiveness checks, and L4 injection.
 *
 * Stages:
 *   1. FP-BOUNDARY-DELETE  — fast head-splice using the recorded aggressive boundary
 *   2. Fast-path re-apply  — re-replace confirmed offloads / drop deleted messages
 *   3. Token estimation    — fast char-based estimate, tiktoken only near thresholds
 *   4. AGGRESSIVE          — tail-accumulate (first run) or cascade head-delete,
 *                            plus history-MMD injection for deleted ranges
 *   5. MILD                — score-cascade replacement of tool results with summaries
 *   6. EMERGENCY           — hard delete until below target
 *   7. Active-MMD inject   — current task mermaid graph as in-context orientation
 */

import type { OffloadStateManager } from "../../src/offload/state-manager.js";
import type { OffloadEntry, PluginConfig, PluginLogger } from "../../src/offload/types.js";
import { PLUGIN_DEFAULTS } from "../../src/offload/types.js";
import {
  compressByScoreCascade,
  aggressiveCompressUntilBelowThreshold,
  emergencyCompress,
  buildHistoryMmdInjection,
  removeExistingMmdInjections,
  EMERGENCY_MIN_MESSAGES_TO_KEEP,
} from "../../src/offload/hooks/llm-input-l3.js";
import {
  normalizeToolCallIdForLookup,
  getOffloadEntry,
  populateOffloadLookupMap,
  isToolResultMessage,
  extractToolCallId,
  isOnlyToolUseAssistant,
  extractAllToolUseIds,
  isAssistantMessageWithToolUse,
  replaceWithSummary,
  replaceAssistantToolUseWithSummary,
  compressNonCurrentToolUseBlocks,
  getCurrentTaskNodeIds,
} from "../../src/offload/l3-helpers.js";
import { readOffloadEntries, markOffloadStatus, readMmd } from "../../src/offload/storage.js";
import {
  buildTiktokenContextSnapshot,
  tiktokenCount,
  jsonReplacer,
} from "../../src/offload/context-token-tracker.js";
import { fastEstimateMessages } from "../../src/offload/fast-token-estimate.js";
import { createL3TokenCounter } from "../../src/offload/l3-token-counter.js";
import {
  findActiveMmdInsertionPoint,
  findHistoryMmdInsertionPoint,
} from "../../src/offload/mmd-injector.js";
import { msgFingerprint, extractLatestTurn, extractRecentHistory } from "./support.js";

export interface AssembleOptions {
  mgr: OffloadStateManager;
  pCfg: Partial<PluginConfig>;
  logger: PluginLogger;
  /** Active model's context window (tokens). */
  contextWindow: number;
  /** Measured/estimated system prompt tokens. */
  systemTokens: number;
  /** Current turn's user prompt (null when unknown). */
  prompt: string | null;
}

export interface AssembleResult {
  messages: any[];
  changed: boolean;
}

export async function assembleContext(messages: any[], opts: AssembleOptions): Promise<AssembleResult> {
  const { mgr, pCfg, logger, contextWindow, systemTokens, prompt } = opts;
  const workMessages = [...messages];
  const startMs = Date.now();
  const originalCount = workMessages.length;

  // Cache context for L1/L1.5/L2 prompts
  if (typeof prompt === "string" && prompt.length > 0) {
    mgr.cachedUserPrompt = prompt;
  }
  if (workMessages.length > 0) {
    mgr.cachedLatestTurnMessages = extractLatestTurn(prompt);
    mgr.cachedRecentHistory = extractRecentHistory(workMessages, prompt);
  }

  let changed = false;

  try {
    const effectiveBudget = contextWindow;
    const mildRatio = pCfg.mildOffloadRatio ?? PLUGIN_DEFAULTS.mildOffloadRatio;
    const aggressiveRatio = pCfg.aggressiveCompressRatio ?? PLUGIN_DEFAULTS.aggressiveCompressRatio;
    const mildThreshold = Math.floor(effectiveBudget * mildRatio);
    const aggressiveThreshold = Math.floor(effectiveBudget * aggressiveRatio);
    const systemTokensEstimate =
      systemTokens > 0
        ? systemTokens
        : Math.floor(
            effectiveBudget * (pCfg.defaultSystemOverheadRatio ?? PLUGIN_DEFAULTS.defaultSystemOverheadRatio),
          );
    const precomputed = { systemTokens: systemTokensEstimate, userPromptTokens: 0 };

    // ── 1. FP-BOUNDARY-DELETE ────────────────────────────────────────────────
    const boundary = mgr._lastAggressiveBoundary;
    let fpBoundaryDeleted = 0;
    if (
      boundary &&
      prompt &&
      prompt.length > 0 &&
      workMessages.length > boundary.originalIndex &&
      boundary.originalIndex > 0
    ) {
      const candidateMsg = workMessages[boundary.originalIndex];
      if (msgFingerprint(candidateMsg) === boundary.fingerprint) {
        let headDeleteEnd = boundary.originalIndex;
        while (headDeleteEnd < workMessages.length && isToolResultMessage(workMessages[headDeleteEnd])) {
          headDeleteEnd++;
        }
        if (headDeleteEnd > 0 && headDeleteEnd < workMessages.length) {
          const lastDeleted = workMessages[headDeleteEnd - 1];
          if (isAssistantMessageWithToolUse(lastDeleted)) {
            while (headDeleteEnd < workMessages.length && isToolResultMessage(workMessages[headDeleteEnd])) {
              headDeleteEnd++;
            }
          }
        }
        if (headDeleteEnd > 0 && headDeleteEnd < workMessages.length) {
          workMessages.splice(0, headDeleteEnd);
          fpBoundaryDeleted = headDeleteEnd;
          changed = true;
          logger.debug?.(
            `[pi-offload] assemble FP-BOUNDARY-DELETE: spliced ${headDeleteEnd} old msgs (now=${workMessages.length})`,
          );
        }
      } else {
        logger.debug?.(`[pi-offload] assemble FP-BOUNDARY-DELETE: fingerprint mismatch, cleared`);
        mgr._lastAggressiveBoundary = null;
      }
    }

    // ── 2. Fast-path re-apply (confirmed replacements + deleted drops) ──────
    const hasConfirmed = mgr.confirmedOffloadIds?.size > 0;
    const hasDeleted = mgr.deletedOffloadIds?.size > 0;
    let offloadEntries: OffloadEntry[] | null = null;
    let offloadMap: Map<string, OffloadEntry> | null = null;
    let fpReplaced = 0;
    let fpDeleted = 0;

    if (hasConfirmed || hasDeleted) {
      offloadEntries = await readOffloadEntries(mgr.ctx);
      offloadMap = new Map();
      populateOffloadLookupMap(offloadMap, offloadEntries);
      mgr.setCachedOffloadMap(offloadMap);

      const indicesToDelete: number[] = [];
      for (let i = 0; i < workMessages.length; i++) {
        const msg = workMessages[i];
        const tid = extractToolCallId(msg);
        const tidNorm = tid ? normalizeToolCallIdForLookup(tid) : null;
        if (
          tid &&
          hasDeleted &&
          (mgr.deletedOffloadIds.has(tid) || (tidNorm && mgr.deletedOffloadIds.has(tidNorm)))
        ) {
          indicesToDelete.push(i);
          fpDeleted++;
          continue;
        }
        if (hasDeleted && isOnlyToolUseAssistant(msg)) {
          const tuIds = extractAllToolUseIds(msg);
          if (
            tuIds.length > 0 &&
            tuIds.every(
              (id) =>
                mgr.deletedOffloadIds.has(id) ||
                mgr.deletedOffloadIds.has(normalizeToolCallIdForLookup(id)),
            )
          ) {
            indicesToDelete.push(i);
            fpDeleted++;
            continue;
          }
        }
        // Strip deleted tool_use blocks from mixed assistant messages to avoid
        // orphaned tool_use without a matching tool_result.
        if (hasDeleted && isAssistantMessageWithToolUse(msg) && !isOnlyToolUseAssistant(msg)) {
          const content = msg.type === "message" ? msg.message?.content : msg.content;
          if (Array.isArray(content)) {
            for (let j = content.length - 1; j >= 0; j--) {
              const block = content[j] as any;
              if ((block.type === "tool_use" || block.type === "toolCall") && block.id) {
                const blockIdNorm = normalizeToolCallIdForLookup(block.id);
                if (mgr.deletedOffloadIds.has(block.id) || mgr.deletedOffloadIds.has(blockIdNorm)) {
                  content.splice(j, 1);
                  changed = true;
                }
              }
            }
          }
        }
        if (msg._offloaded) continue;
        if (
          tid &&
          hasConfirmed &&
          (mgr.confirmedOffloadIds.has(tid) || (tidNorm && mgr.confirmedOffloadIds.has(tidNorm)))
        ) {
          const entry = getOffloadEntry(offloadMap, tid);
          if (entry && isToolResultMessage(msg)) {
            replaceWithSummary(msg, entry);
            msg._offloaded = true;
            fpReplaced++;
            changed = true;
          }
        }
        if (isOnlyToolUseAssistant(msg)) {
          const tuIds = extractAllToolUseIds(msg);
          const allConfirmed =
            tuIds.length > 0 &&
            tuIds.every(
              (id) =>
                mgr.confirmedOffloadIds.has(id) ||
                mgr.confirmedOffloadIds.has(normalizeToolCallIdForLookup(id)),
            );
          if (allConfirmed) {
            const tuEntries = tuIds
              .map((id) => getOffloadEntry(offloadMap!, id))
              .filter(Boolean) as OffloadEntry[];
            if (tuEntries.length === tuIds.length) {
              replaceAssistantToolUseWithSummary(msg, tuEntries);
              msg._offloaded = true;
              changed = true;
            }
          }
        } else if (isAssistantMessageWithToolUse(msg)) {
          compressNonCurrentToolUseBlocks(msg, offloadMap, new Set(), mgr.confirmedOffloadIds);
        }
      }
      if (indicesToDelete.length > 0) {
        for (let k = indicesToDelete.length - 1; k >= 0; k--) workMessages.splice(indicesToDelete[k], 1);
        changed = true;
      }
    }

    // ── 3. Token estimation ─────────────────────────────────────────────────
    const fastEst =
      fastEstimateMessages(workMessages) +
      systemTokensEstimate +
      (prompt ? Math.ceil(prompt.length / 4) : 0);
    const FAST_EST_SAFETY_MARGIN = 0.85;

    let workingTokens: number;
    let usedFastPath = false;

    const boundaryCache = mgr._lastAggressiveBoundary;
    const BOUNDARY_NEW_MSG_TOLERANCE = 20;
    if (
      fpBoundaryDeleted > 0 &&
      boundaryCache &&
      workMessages.length <= boundaryCache.keptMsgCount + BOUNDARY_NEW_MSG_TOLERANCE &&
      boundaryCache.remainingTokens < aggressiveThreshold
    ) {
      const newMsgCount = Math.max(0, workMessages.length - boundaryCache.keptMsgCount);
      const newMsgTokens =
        newMsgCount > 0
          ? fastEstimateMessages(workMessages.slice(workMessages.length - newMsgCount)) +
            (prompt ? Math.ceil(prompt.length / 4) : 0)
          : prompt
            ? Math.ceil(prompt.length / 4)
            : 0;
      const incrementalEst = boundaryCache.remainingTokens + newMsgTokens;
      if (incrementalEst < aggressiveThreshold) {
        workingTokens = incrementalEst;
        usedFastPath = true;
      } else {
        const snap = buildTiktokenContextSnapshot("assemble", workMessages, null, prompt ?? null, precomputed);
        workingTokens = snap.totalTokens;
      }
    } else if (fastEst < aggressiveThreshold * FAST_EST_SAFETY_MARGIN) {
      workingTokens = fastEst;
      usedFastPath = true;
    } else if (!mgr._lastAggressiveBoundary && prompt && prompt.length > 0) {
      // TAIL-ACCUMULATE below will do its own precise tail counting.
      workingTokens = fastEst;
    } else {
      const snap = buildTiktokenContextSnapshot("assemble", workMessages, null, prompt ?? null, precomputed);
      workingTokens = snap.totalTokens;
    }
    logger.debug?.(
      `[pi-offload] assemble tokens≈${workingTokens} (sys≈${systemTokensEstimate}, fast=${usedFastPath}), ` +
        `budget=${effectiveBudget}, mild@${mildThreshold}, aggressive@${aggressiveThreshold}, msgs=${workMessages.length}`,
    );

    // ── 4. AGGRESSIVE ───────────────────────────────────────────────────────
    if (workingTokens >= aggressiveThreshold) {
      const TAIL_ACCUM_TARGET_RATIO = 0.6;
      const tailAccumTarget = Math.floor(effectiveBudget * TAIL_ACCUM_TARGET_RATIO) - systemTokensEstimate;

      if (!mgr._lastAggressiveBoundary && workMessages.length > 0 && prompt && prompt.length > 0) {
        // First run: accumulate tokens from the tail, discard the head wholesale.
        let accum = 0;
        let keepFrom = 0;
        for (let i = workMessages.length - 1; i >= 0; i--) {
          const msgTokens = tiktokenCount(JSON.stringify(workMessages[i], jsonReplacer));
          if (accum + msgTokens > tailAccumTarget) {
            keepFrom = i + 1;
            break;
          }
          accum += msgTokens;
        }
        while (keepFrom < workMessages.length && isToolResultMessage(workMessages[keepFrom])) {
          accum += tiktokenCount(JSON.stringify(workMessages[keepFrom], jsonReplacer));
          keepFrom++;
        }
        if (keepFrom > 0 && keepFrom < workMessages.length) {
          const lastDeleted = workMessages[keepFrom - 1];
          if (isAssistantMessageWithToolUse(lastDeleted)) {
            while (keepFrom < workMessages.length && isToolResultMessage(workMessages[keepFrom])) {
              accum += tiktokenCount(JSON.stringify(workMessages[keepFrom], jsonReplacer));
              keepFrom++;
            }
          }
        }
        // Keep at least the last user message
        for (let u = workMessages.length - 1; u >= keepFrom; u--) {
          const role = workMessages[u].role ?? workMessages[u].message?.role ?? workMessages[u].type;
          if (role === "user" || role === "human") break;
          if (u === keepFrom) {
            for (let u2 = keepFrom - 1; u2 >= 0; u2--) {
              const r2 = workMessages[u2].role ?? workMessages[u2].message?.role ?? workMessages[u2].type;
              if (r2 === "user" || r2 === "human") {
                keepFrom = u2;
                break;
              }
            }
          }
        }
        const MIN_KEEP = 10;
        if (workMessages.length - keepFrom < MIN_KEEP) {
          keepFrom = Math.max(0, workMessages.length - MIN_KEEP);
        }
        if (keepFrom > 0 && keepFrom < workMessages.length) {
          const tailDeletedIds: string[] = [];
          for (let d = 0; d < keepFrom; d++) {
            const msg = workMessages[d];
            const tid =
              extractToolCallId(msg) ?? (isOnlyToolUseAssistant(msg) ? extractAllToolUseIds(msg)[0] : null);
            if (tid) tailDeletedIds.push(tid);
          }
          workMessages.splice(0, keepFrom);
          workingTokens = accum + systemTokensEstimate;
          changed = true;
          logger.info(
            `[pi-offload] assemble TAIL-ACCUMULATE: kept ${workMessages.length} msgs, deleted ${keepFrom}, tokens≈${workingTokens}`,
          );
          if (tailDeletedIds.length > 0) {
            const statusUpdates = new Map<string, string | boolean>();
            for (const id of tailDeletedIds) {
              statusUpdates.set(id, "deleted");
              mgr.confirmedOffloadIds.add(id);
              mgr.deletedOffloadIds.add(id);
            }
            markOffloadStatus(mgr.ctx, statusUpdates).catch(() => {});
          }
          recordBoundary(mgr, messages, workMessages, workingTokens, logger);

          // History MMD injection for the deleted range
          if (tailDeletedIds.length > 0) {
            if (!offloadEntries) {
              offloadEntries = await readOffloadEntries(mgr.ctx);
              offloadMap = new Map();
              populateOffloadLookupMap(offloadMap, offloadEntries);
            }
            const countTokens = createL3TokenCounter(pCfg, logger);
            const mmdInj = await buildHistoryMmdInjection(
              tailDeletedIds,
              offloadMap!,
              offloadEntries,
              mgr,
              logger,
              countTokens,
              effectiveBudget,
              pCfg,
            );
            if (mmdInj.injectedMessages.length > 0) {
              removeExistingMmdInjections(workMessages);
              const histInsertIdx = findHistoryMmdInsertionPoint(workMessages);
              workMessages.splice(histInsertIdx, 0, ...mmdInj.injectedMessages);
              workingTokens += mmdInj.totalMmdTokens;
            }
          }
        }
      } else {
        // Standard aggressive cascade
        if (!offloadEntries) {
          offloadEntries = await readOffloadEntries(mgr.ctx);
          offloadMap = new Map();
          populateOffloadLookupMap(offloadMap!, offloadEntries);
        }
        const countTokens = createL3TokenCounter(pCfg, logger);
        const aggressiveDeleteRatio =
          (pCfg as any).aggressiveDeleteRatio ?? PLUGIN_DEFAULTS.aggressiveDeleteRatio;
        const currentTaskNodeIds = await getCurrentTaskNodeIds(mgr);
        const AGGRESSIVE_TARGET_RATIO = 0.85;
        const aggressiveTargetForMsgs = Math.max(
          0,
          Math.floor(aggressiveThreshold * AGGRESSIVE_TARGET_RATIO) - systemTokensEstimate,
        );
        const result = await aggressiveCompressUntilBelowThreshold(
          workMessages,
          offloadMap!,
          currentTaskNodeIds,
          aggressiveDeleteRatio,
          mgr,
          logger,
          aggressiveTargetForMsgs,
          countTokens,
          null,
          prompt ?? null,
        );
        workingTokens = result.remainingTokens + systemTokensEstimate;
        if (result.deletedCount > 0) changed = true;
        logger.debug?.(
          `[pi-offload] assemble AGGRESSIVE: rounds=${result.rounds}, deleted=${result.deletedCount}, remaining≈${workingTokens}`,
        );
        if (result.deletedCount > 0 && workMessages.length > 0 && prompt && prompt.length > 0) {
          recordBoundary(mgr, messages, workMessages, workingTokens, logger);
        }
        if (result.allDeletedToolCallIds.length > 0) {
          const statusUpdates = new Map<string, string | boolean>();
          for (const id of result.allDeletedToolCallIds) {
            statusUpdates.set(id, "deleted");
            mgr.confirmedOffloadIds.add(id);
            mgr.deletedOffloadIds.add(id);
          }
          markOffloadStatus(mgr.ctx, statusUpdates).catch(() => {});
          const mmdInj = await buildHistoryMmdInjection(
            result.allDeletedToolCallIds,
            offloadMap!,
            offloadEntries,
            mgr,
            logger,
            countTokens,
            effectiveBudget,
            pCfg,
          );
          if (mmdInj.injectedMessages.length > 0) {
            removeExistingMmdInjections(workMessages);
            const histInsertIdx = findHistoryMmdInsertionPoint(workMessages);
            workMessages.splice(histInsertIdx, 0, ...mmdInj.injectedMessages);
            workingTokens += mmdInj.totalMmdTokens;
          }
        }
        if (result.stalledByUserMsg && workingTokens >= aggressiveThreshold) {
          logger.warn(`[pi-offload] assemble AGGRESSIVE stalled, forcing emergency fallback`);
          mgr._forceEmergencyNext = true;
        }
      }
    }

    // ── 5. MILD (score cascade) ─────────────────────────────────────────────
    if (workingTokens >= mildThreshold) {
      if (!offloadEntries) {
        offloadEntries = await readOffloadEntries(mgr.ctx);
        offloadMap = new Map();
        populateOffloadLookupMap(offloadMap!, offloadEntries);
      }
      const currentTaskNodeIds = await getCurrentTaskNodeIds(mgr);
      const mildScanRatio = (pCfg as any).mildOffloadScanRatio ?? PLUGIN_DEFAULTS.mildOffloadScanRatio;
      const cascadeResult = compressByScoreCascade(
        workMessages,
        offloadMap!,
        currentTaskNodeIds,
        mildScanRatio,
        logger,
      );
      if (cascadeResult.replacedCount > 0) {
        changed = true;
        for (const id of cascadeResult.replacedToolCallIds) mgr.confirmedOffloadIds.add(id);
        const mildUpdates = new Map<string, string | boolean>();
        for (const id of cascadeResult.replacedToolCallIds) mildUpdates.set(id, true);
        markOffloadStatus(mgr.ctx, mildUpdates).catch(() => {});
        logger.debug?.(
          `[pi-offload] assemble MILD: replaced=${cascadeResult.replacedCount}, finalThreshold=${cascadeResult.finalThreshold}`,
        );
      }
    }

    // ── 6. EMERGENCY ────────────────────────────────────────────────────────
    const emergencyRatio = pCfg.emergencyCompressRatio ?? PLUGIN_DEFAULTS.emergencyCompressRatio;
    const emergencyTargetRatio = pCfg.emergencyTargetRatio ?? PLUGIN_DEFAULTS.emergencyTargetRatio;
    const emergencyThreshold = Math.floor(effectiveBudget * emergencyRatio);
    const emergencyTarget = Math.floor(effectiveBudget * emergencyTargetRatio);
    const forceEmergency = mgr._forceEmergencyNext === true;
    if (forceEmergency) mgr._forceEmergencyNext = false;
    if (
      (workingTokens >= emergencyThreshold || forceEmergency) &&
      workMessages.length > EMERGENCY_MIN_MESSAGES_TO_KEEP
    ) {
      logger.warn(
        `[pi-offload] assemble EMERGENCY: tokens≈${workingTokens} >= ${emergencyThreshold} (force=${forceEmergency})`,
      );
      const countTokensEmg = createL3TokenCounter(pCfg, logger);
      const emResult = emergencyCompress(
        workMessages,
        emergencyTarget - systemTokensEstimate,
        countTokensEmg,
        null,
        prompt ?? null,
        logger,
      );
      workingTokens = emResult.remainingTokens + systemTokensEstimate;
      if (emResult.deletedCount > 0) changed = true;
      if (emResult.deletedToolCallIds.length > 0) {
        const emUpdates = new Map<string, string | boolean>();
        for (const id of emResult.deletedToolCallIds) {
          emUpdates.set(id, "deleted");
          mgr.confirmedOffloadIds.add(id);
          mgr.deletedOffloadIds.add(id);
        }
        markOffloadStatus(mgr.ctx, emUpdates).catch(() => {});
      }
      if (emResult.deletedCount > 0 && workMessages.length > 0 && prompt && prompt.length > 0) {
        recordBoundary(mgr, messages, workMessages, workingTokens, logger);
      }
    }

    // ── 7. Active MMD injection ─────────────────────────────────────────────
    try {
      const activeMmdFile = mgr.l15Settled ? mgr.getActiveMmdFile() : null;
      if (activeMmdFile) {
        const mmdContent = await readMmd(mgr.ctx, activeMmdFile);
        if (mmdContent) {
          let taskGoal = "";
          const metaMatch = mmdContent.match(/^%%\{\s*(.*?)\s*\}%%/);
          if (metaMatch) {
            try {
              const meta = JSON.parse(`{${metaMatch[1]}}`);
              taskGoal = meta.taskGoal || "";
            } catch {
              /* ignore */
            }
          }
          const mmdText = [
            `<current_task_context>`,
            `【当前活跃任务的mermaid流程图】这是你最近正在执行的任务的阶段性记录（此条下方的tool use未被汇总，进程可能有延迟，仅供参考）。`,
            taskGoal ? `**任务目标:** ${taskGoal}` : "",
            `**任务文件:** ${activeMmdFile}`,
            "```mermaid",
            mmdContent,
            "```",
            `标记为 "doing" 的节点是近期焦点（注：可能有延迟，下方的tool use未被统计，仅供参考），"done" 的已完成。请参考此保持方向感，避免重复已完成的工作。`,
            `</current_task_context>`,
          ]
            .filter((line) => line !== "")
            .join("\n");

          const existingIdx = workMessages.findIndex((m: any) => m._mmdContextMessage === "active");
          const newMsg = {
            role: "user",
            content: [{ type: "text", text: mmdText }],
            timestamp: Date.now(),
            _mmdContextMessage: "active",
          };
          if (existingIdx >= 0) {
            workMessages[existingIdx] = newMsg;
          } else {
            const insertIdx = findActiveMmdInsertionPoint(workMessages);
            workMessages.splice(insertIdx, 0, newMsg);
          }
          changed = true;
        }
      }
    } catch (err) {
      logger.warn(`[pi-offload] assemble active MMD error: ${err}`);
    }

    logger.debug?.(
      `[pi-offload] assemble END: ${originalCount}→${workMessages.length} msgs, tokens≈${workingTokens}, ` +
        `fpReplaced=${fpReplaced}, fpDeleted=${fpDeleted}, changed=${changed}, duration=${Date.now() - startMs}ms`,
    );
    return { messages: workMessages, changed };
  } catch (err) {
    logger.error(
      `[pi-offload] assemble failed, passing through unchanged: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}`,
    );
    return { messages, changed: false };
  }
}

/** Record the aggressive/emergency boundary for next-turn FP-BOUNDARY-DELETE. */
function recordBoundary(
  mgr: OffloadStateManager,
  originalMessages: any[],
  workMessages: any[],
  remainingTokens: number,
  logger: PluginLogger,
): void {
  if (workMessages.length === 0) {
    mgr._lastAggressiveBoundary = null;
    return;
  }
  const boundaryFp = msgFingerprint(workMessages[0]);
  let boundaryOrigIdx = -1;
  for (let bi = 0; bi < originalMessages.length; bi++) {
    if (msgFingerprint(originalMessages[bi]) === boundaryFp) {
      if (bi + 1 < originalMessages.length && workMessages.length > 1) {
        if (msgFingerprint(originalMessages[bi + 1]) === msgFingerprint(workMessages[1])) {
          boundaryOrigIdx = bi;
          break;
        }
      } else {
        boundaryOrigIdx = bi;
        break;
      }
    }
  }
  if (boundaryOrigIdx >= 0) {
    mgr._lastAggressiveBoundary = {
      originalIndex: boundaryOrigIdx,
      fingerprint: boundaryFp,
      keptMsgCount: workMessages.length,
      remainingTokens,
    };
    logger.debug?.(
      `[pi-offload] boundary recorded: idx=${boundaryOrigIdx}, kept=${workMessages.length}, tokens≈${remainingTokens}`,
    );
  } else {
    mgr._lastAggressiveBoundary = null;
    logger.debug?.(`[pi-offload] boundary: not found in original msgs, cleared`);
  }
}

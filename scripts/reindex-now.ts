/**
 * Reindex tool: drop + rebuild the vector tables at the current dimensions
 * and re-embed all stored texts (L1 + L0) through the configured embedding
 * provider.
 *
 * Running this script IS the explicit approval for the destructive rebuild
 * (see the reindex.approveChanges gate in tdai-core). Intended for use when
 * the approval-gate warning reports a pending reindex.
 *
 * Tolerates the pi extension being live: the SQLite store runs in WAL mode,
 * so concurrent multi-process access is supported. Keyword/FTS recall is
 * unaffected; vector search gains rows as this pass commits them.
 *
 * Usage: bun scripts/reindex-now.ts   (run from the repo root)
 */
import os from "node:os";
import path from "node:path";
import { loadMemoryConfig } from "./pi-extension/lib/config.ts";
import { initStores } from "./src/utils/pipeline-factory.ts";
import type { PipelineLogger } from "./src/utils/pipeline-factory.ts";

const logger: PipelineLogger = {
  debug: (...a: unknown[]) => console.log("[debug]", ...a),
  info: (...a: unknown[]) => console.log("[info]", ...a),
  warn: (...a: unknown[]) => console.log("[warn]", ...a),
  error: (...a: unknown[]) => console.log("[error]", ...a),
};

const dataDir = path.join(os.homedir(), ".pi", "agent", "memory-tdai");
const cfg = loadMemoryConfig(dataDir, logger);

console.log(`dataDir: ${dataDir}`);
console.log(`embedding provider: ${cfg.embedding.provider} model: ${cfg.embedding.model ?? "(default)"}`);

const stores = await initStores(cfg, dataDir, logger);
const vs = stores.vectorStore;
const es = stores.embeddingService;

if (!vs || !es) {
  console.error("Store or embedding service unavailable — cannot reindex.");
  process.exit(1);
}

console.log(`Store ready. needsReindex flag: ${stores.needsReindex} (${stores.reindexReason ?? "none"})`);

// D3 readiness (M4): local (node-llama-cpp) providers must be warm before
// embed() succeeds. The sanctioned approval tool MUST apply the same wait as
// the background reindex path — otherwise for local providers this script
// drops the tables then embed-fails-empty. Warm up and wait for isReady()
// BEFORE the destructive rebuild.
if (!es.isReady()) {
  console.log("Waiting for embedding service readiness before reindex...");
  es.startWarmup();
  const readyAt = Date.now() + 5 * 60 * 1000; // 5 min for model download+load
  while (!es.isReady()) {
    if (Date.now() > readyAt) {
      console.error("Embedding service not ready after 5min — aborting reindex (tables untouched).");
      process.exit(1);
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
}
// Remote providers are stateless HTTP; a warmup embed is a cheap no-op there.
try {
  await es.embed("warmup");
} catch {
  /* best-effort */
}

console.log("Starting rebuild + full reindex (drop/verify tables, then L1 then L0)...");
const started = Date.now();

// Rebuild the vector tables at the current dimensions — this is the
// destructive step, and running this script IS the explicit approval for it.
if (!vs.rebuildVecTables?.(es.getProviderInfo())) {
  console.error("Vector table rebuild failed — aborting.");
  process.exit(1);
}
console.log(`Vector tables rebuilt at ${cfg.embedding.dimensions} dims.`);

const result = await vs.reindexAll(
  async (text: string) => {
    const emb = await es.embed(text);
    return emb;
  },
  (succeeded: number, failed: number, total: number, layer: "L1" | "L0") => {
    if (succeeded === 1 || succeeded % 250 === 0 || succeeded + failed === total) {
      const pct = (((succeeded + failed) / total) * 100).toFixed(0);
      console.log(`  ${layer}: ${succeeded} ok / ${failed} failed / ${total} total (${pct}%)`);
    }
  },
);
const elapsed = ((Date.now() - started) / 1000).toFixed(1);
console.log(`Reindex finished in ${elapsed}s: L1=${result.l1Count}/${result.l1Total} (${result.l1Failed} failed), L0=${result.l0Count}/${result.l0Total} (${result.l0Failed} failed)`);

// Running this script IS the explicit approval — on a fully successful pass
// (zero failures AND full coverage), mark the embedding provider as current
// so steady-state boots stop re-detecting the change.
const total = result.l1Total + result.l0Total;
const done = result.l1Count + result.l0Count;
if (result.l1Failed === 0 && result.l0Failed === 0 && done === total) {
  vs.markEmbeddingCurrent?.(es.getProviderInfo());
  console.log("Marked embedding config as current (full coverage, no failures).");
} else {
  console.log("NOT marked current: incomplete coverage or failures present. Meta stays stale so the next boot/run retries.");
}
process.exit(result.l1Failed === 0 && result.l0Failed === 0 && done === total ? 0 : 1);

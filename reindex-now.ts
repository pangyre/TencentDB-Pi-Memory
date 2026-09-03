/**
 * One-off: force a full re-embed of all stored texts (L1 + L0) through the
 * configured embedding provider (oMLX / Qwen3-Embedding-0.6B).
 *
 * Safe to run while the pi extension is live: the SQLite store runs in WAL
 * mode, so concurrent multi-process access is supported. Keyword/FTS recall
 * is unaffected; vector search gains rows as this pass commits them.
 *
 * Mirrors the repo's own seed-runtime store construction.
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

// Warm up the remote embedding service (stateless HTTP, so mostly a no-op).
try {
  await es.embed("warmup");
} catch {
  /* best-effort */
}

console.log("Starting full reindex (L1 then L0)...");
const started = Date.now();
const result = await vs.reindexAll(
  async (text: string) => {
    const emb = await es.embed(text);
    return emb;
  },
  (done: number, total: number, layer: "L1" | "L0") => {
    if (done === 1 || done % 250 === 0 || done === total) {
      const pct = ((done / total) * 100).toFixed(0);
      console.log(`  ${layer}: ${done}/${total} (${pct}%)`);
    }
  },
);
const elapsed = ((Date.now() - started) / 1000).toFixed(1);
console.log(`Reindex complete in ${elapsed}s: L1=${result.l1Count}, L0=${result.l0Count}`);
process.exit(0);

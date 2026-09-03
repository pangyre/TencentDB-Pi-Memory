import os from "node:os";
import path from "node:path";
import { loadMemoryConfig } from "./pi-extension/lib/config.ts";
import { initStores } from "./src/utils/pipeline-factory.ts";

const logger = { debug() {}, info() {}, warn() {}, error() {} } as never;
const dataDir = path.join(os.homedir(), ".pi", "agent", "memory-tdai");
const cfg = loadMemoryConfig(dataDir, logger);
const stores = await initStores(cfg, dataDir, logger);
const vs = stores.vectorStore!;
const es = stores.embeddingService!;

const q = await es.embed("user is male, he/him, plain language, no PC speak");
const res = await vs.searchL1Vector(q, 5, "");
console.log("vec search hits:", res?.length ?? 0);
for (const r of (res ?? []).slice(0, 5)) {
  const txt = String((r as any).content ?? (r as any).record_id ?? "");
  console.log(" -", txt.slice(0, 110));
}
process.exit(0);

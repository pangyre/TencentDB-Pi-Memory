/**
 * Runner for reindex-now.ts — loads the TypeScript through jiti (the same TS
 * runtime pi uses for its extensions), since the repo's source imports .ts
 * modules via .js specifiers (NodeNext) which plain node cannot resolve.
 *
 * Usage (repo root): node scripts/reindex-now.mjs
 * Debug:             TDAI_REINDEX_DEBUG=1 node scripts/reindex-now.mjs
 */
import { createJiti } from "jiti/static";
import url from "node:url";
import path from "node:path";

const here = path.dirname(url.fileURLToPath(import.meta.url));
const jiti = createJiti(import.meta.url, { interopDefault: true });
await jiti.import(path.join(here, "reindex-now.ts"));

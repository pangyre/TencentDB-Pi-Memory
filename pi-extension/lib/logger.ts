/**
 * Simple file-backed logger for the pi memory extension.
 *
 * - All levels append to `<dataDir>/logs/pi-extension.log` (best-effort).
 * - warn/error echo to stderr only on a TTY (or when PI_MEMORY_DEBUG is set), so
 *   headless pi runs (print mode, roborev, RPC consumers) stay clean.
 * - debug echoes only when PI_MEMORY_DEBUG is set.
 * - Never throws.
 */

import fs from "node:fs";
import path from "node:path";
import type { Logger } from "../../src/core/types.js";

export function createLogger(dataDir: string): Logger {
  const logDir = path.join(dataDir, "logs");
  const logFile = path.join(logDir, "pi-extension.log");
  let dirReady = false;

  const writeLine = (level: string, msg: string): void => {
    try {
      if (!dirReady) {
        fs.mkdirSync(logDir, { recursive: true });
        dirReady = true;
      }
      fs.appendFileSync(logFile, `${new Date().toISOString()} [${level}] ${msg}\n`);
    } catch {
      // best-effort only
    }
  };

  const echoAllowed = (): boolean =>
    Boolean(process.env.PI_MEMORY_DEBUG) || Boolean(process.stderr.isTTY);

  return {
    debug(msg: string): void {
      writeLine("DEBUG", msg);
      if (process.env.PI_MEMORY_DEBUG) {
        console.error(`[memory-tdai] ${msg}`);
      }
    },
    info(msg: string): void {
      writeLine("INFO", msg);
    },
    warn(msg: string): void {
      writeLine("WARN", msg);
      if (echoAllowed()) console.error(`[memory-tdai] warn: ${msg}`);
    },
    error(msg: string): void {
      writeLine("ERROR", msg);
      if (echoAllowed()) console.error(`[memory-tdai] error: ${msg}`);
    },
  };
}

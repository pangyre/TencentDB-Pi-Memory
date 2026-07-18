/**
 * Config loading + agent identity for the pi memory extension.
 *
 * - Global config: ~/.pi/agent/memory-tdai/config.json (parseConfig schema)
 * - Project overlay: <cwd>/.pi/memory-tdai.json  { agentName?, capture?, recall? }
 * - Agent id: stable per project dir, baked into session keys `pi:<agentId>:<sessionId>`.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createHash } from "node:crypto";
import { parseConfig } from "../../src/config.js";
import type { MemoryTdaiConfig } from "../../src/config.js";
import type { Logger } from "../../src/core/types.js";

export interface ProjectOverlay {
  agentName?: string;
  capture?: boolean;
  recall?: boolean;
}

/** Global data directory shared by all pi agents. */
export function getDataDir(): string {
  return path.join(os.homedir(), ".pi", "agent", "memory-tdai");
}

/** Load and parse the global config; always returns a valid config. */
export function loadMemoryConfig(dataDir: string, logger: Logger): MemoryTdaiConfig {
  const configPath = path.join(dataDir, "config.json");
  let raw: string;
  try {
    raw = fs.readFileSync(configPath, "utf8");
  } catch {
    logger.info(`no config.json at ${configPath}, using defaults`);
    return parseConfig(undefined);
  }
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    return parseConfig(parsed);
  } catch (err) {
    logger.warn(
      `invalid config.json (${err instanceof Error ? err.message : String(err)}), using defaults`,
    );
    return parseConfig(undefined);
  }
}

/** Load the per-project overlay; tolerant of missing/invalid files. */
export function loadProjectOverlay(cwd: string, logger: Logger): ProjectOverlay {
  const overlayPath = path.join(cwd, ".pi", "memory-tdai.json");
  let raw: string;
  try {
    raw = fs.readFileSync(overlayPath, "utf8");
  } catch {
    return {};
  }
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const overlay: ProjectOverlay = {};
    if (typeof parsed.agentName === "string") overlay.agentName = parsed.agentName;
    if (typeof parsed.capture === "boolean") overlay.capture = parsed.capture;
    if (typeof parsed.recall === "boolean") overlay.recall = parsed.recall;
    return overlay;
  } catch (err) {
    logger.warn(
      `invalid .pi/memory-tdai.json (${err instanceof Error ? err.message : String(err)}), ignoring`,
    );
    return {};
  }
}

function sanitize(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Stable agent identity: explicit overlay name, else dir basename + cwd hash. */
export function deriveAgentId(cwd: string, overlay: ProjectOverlay): string {
  if (overlay.agentName) {
    const named = sanitize(overlay.agentName);
    if (named) return named;
  }
  const base = sanitize(path.basename(cwd)) || "agent";
  const hash = createHash("sha256").update(cwd).digest("hex").slice(0, 6);
  return `${base}-${hash}`;
}

export function buildSessionKey(agentId: string, sessionId: string): string {
  return `pi:${agentId}:${sessionId}`;
}

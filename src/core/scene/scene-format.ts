/**
 * Scene Block file format: parse and format the META-delimited Markdown files.
 */

export interface SceneBlockMeta {
  created: string;
  updated: string;
  summary: string;
  heat: number;
  /** Project scope tag. Empty/absent = visible everywhere; "global" = explicit everywhere;
   *  otherwise the project's agentId (cwd-derived, e.g. "tuesday", "k0", "karma-come-lately").
   *  Added 2026-09-06 (rules audit §7 A.5) for per-project scene-navigation scoping. */
  project?: string;
}

export interface SceneBlock {
  filename: string;
  meta: SceneBlockMeta;
  content: string;
}

const META_START = "-----META-START-----";
const META_END = "-----META-END-----";

/**
 * Parse a Scene Block file into structured data.
 */
export function parseSceneBlock(raw: string, filename: string): SceneBlock {
  const startIdx = raw.indexOf(META_START);
  const endIdx = raw.indexOf(META_END);

  if (startIdx === -1 || endIdx === -1) {
    // No META section — treat entire file as content
    return {
      filename,
      meta: { created: "", updated: "", summary: "", heat: 0 },
      content: raw.trim(),
    };
  }

  const metaBlock = raw.slice(startIdx + META_START.length, endIdx).trim();
  const content = raw.slice(endIdx + META_END.length).trim();

  const meta: SceneBlockMeta = {
    created: extractMetaField(metaBlock, "created"),
    updated: extractMetaField(metaBlock, "updated"),
    summary: extractMetaField(metaBlock, "summary"),
    heat: parseInt(extractMetaField(metaBlock, "heat"), 10) || 0,
    project: extractMetaField(metaBlock, "project") || undefined,
  };

  return { filename, meta, content };
}

/**
 * Format a Scene Block back into file content.
 */
export function formatSceneBlock(meta: SceneBlockMeta, content: string): string {
  return `${formatMeta(meta)}\n\n${content}`;
}

/**
 * Format the META section.
 */
export function formatMeta(meta: SceneBlockMeta): string {
  const lines = [
    META_START,
    `created: ${meta.created}`,
    `updated: ${meta.updated}`,
    `summary: ${meta.summary}`,
    `heat: ${meta.heat}`,
  ];
  if (meta.project) lines.push(`project: ${meta.project}`);
  lines.push(META_END);
  return lines.join("\n");
}

function extractMetaField(metaBlock: string, field: string): string {
  const re = new RegExp(`^${field}:\\s*(.*)$`, "m");
  const m = metaBlock.match(re);
  return m ? m[1]!.trim() : "";
}

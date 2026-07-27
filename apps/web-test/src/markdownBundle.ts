const BUNDLE_MEDIA_TYPE = "application/vnd.rag-kb.markdown-bundle+zip";
const UTF8_FLAG = 0x0800;
const DOS_DATE_1980_01_01 = 0x0021;

interface ZipMember {
  name: Uint8Array;
  content: Uint8Array;
  crc32: number;
  offset: number;
}

export interface MarkdownFolderSelection {
  files: File[];
  entrypoints: string[];
}

export function inspectMarkdownFolder(files: File[]): MarkdownFolderSelection {
  const normalized = normalizedFolderFiles(files);
  return {
    files,
    entrypoints: normalized
      .filter(({ path }) => path.toLowerCase().endsWith(".md"))
      .map(({ path }) => path)
      .sort(),
  };
}

export async function buildMarkdownBundle(
  files: File[],
  entrypoint: string,
): Promise<File> {
  const normalized = normalizedFolderFiles(files);
  const selected = normalized.find(({ path }) => path === entrypoint);
  if (!selected) throw new Error("The selected Markdown entrypoint is unavailable.");

  const localMedia = referencedLocalMediaPaths(
    await selected.file.text(),
    entrypoint,
  );
  const available = new Set(normalized.map(({ path }) => path));
  const missing = [...localMedia].filter((path) => !available.has(path));
  if (missing.length > 0) {
    throw new Error(
      `A local Markdown image is missing from the selected folder: ${missing[0]}`,
    );
  }
  const accepted = normalized.filter(({ path }) =>
    path === entrypoint || localMedia.has(path)
  );
  const encoder = new TextEncoder();
  const members: ZipMember[] = [];
  const manifest = encoder.encode(JSON.stringify({
    version: 1,
    entrypoint,
  }));
  members.push({
    name: encoder.encode("manifest.json"),
    content: manifest,
    crc32: crc32(manifest),
    offset: 0,
  });
  for (const item of accepted) {
    const content = new Uint8Array(await item.file.arrayBuffer());
    members.push({
      name: encoder.encode(item.path),
      content,
      crc32: crc32(content),
      offset: 0,
    });
  }
  members.sort((left, right) =>
    new TextDecoder().decode(left.name).localeCompare(
      new TextDecoder().decode(right.name),
    )
  );

  const localParts: Uint8Array[] = [];
  let offset = 0;
  for (const member of members) {
    member.offset = offset;
    const header = new Uint8Array(30 + member.name.length);
    const view = new DataView(header.buffer);
    view.setUint32(0, 0x04034b50, true);
    view.setUint16(4, 20, true);
    view.setUint16(6, UTF8_FLAG, true);
    view.setUint16(8, 0, true);
    view.setUint16(10, 0, true);
    view.setUint16(12, DOS_DATE_1980_01_01, true);
    view.setUint32(14, member.crc32, true);
    view.setUint32(18, member.content.length, true);
    view.setUint32(22, member.content.length, true);
    view.setUint16(26, member.name.length, true);
    header.set(member.name, 30);
    localParts.push(header, member.content);
    offset += header.length + member.content.length;
  }

  const centralOffset = offset;
  const centralParts: Uint8Array[] = [];
  for (const member of members) {
    const header = new Uint8Array(46 + member.name.length);
    const view = new DataView(header.buffer);
    view.setUint32(0, 0x02014b50, true);
    view.setUint16(4, 0x0314, true);
    view.setUint16(6, 20, true);
    view.setUint16(8, UTF8_FLAG, true);
    view.setUint16(10, 0, true);
    view.setUint16(12, 0, true);
    view.setUint16(14, DOS_DATE_1980_01_01, true);
    view.setUint32(16, member.crc32, true);
    view.setUint32(20, member.content.length, true);
    view.setUint32(24, member.content.length, true);
    view.setUint16(28, member.name.length, true);
    view.setUint32(38, 0o100600 << 16, true);
    view.setUint32(42, member.offset, true);
    header.set(member.name, 46);
    centralParts.push(header);
    offset += header.length;
  }
  const centralSize = offset - centralOffset;
  const end = new Uint8Array(22);
  const endView = new DataView(end.buffer);
  endView.setUint32(0, 0x06054b50, true);
  endView.setUint16(8, members.length, true);
  endView.setUint16(10, members.length, true);
  endView.setUint32(12, centralSize, true);
  endView.setUint32(16, centralOffset, true);

  const name = entrypoint.split("/").pop()?.replace(/\.md$/i, ".mdz")
    ?? "document.mdz";
  const archive = concatenate([...localParts, ...centralParts, end]);
  return new File([archive.buffer], name, {
    type: BUNDLE_MEDIA_TYPE,
  });
}

export function referencedLocalMediaPaths(
  markdown: string,
  entrypoint: string,
): Set<string> {
  const references = new Set<string>();
  const withoutCode = markdown
    .replace(/^( {0,3})(`{3,}|~{3,})[^\n]*\n[\s\S]*?^\1\2[ \t]*$/gm, "")
    .replace(/`[^`\n]*`/g, "");
  const definitions = new Map<string, string>();
  for (const match of withoutCode.matchAll(
    /^[ \t]{0,3}\[([^\]]+)\]:[ \t]*(?:<([^>\n]+)>|([^\s]+))/gm,
  )) {
    definitions.set(normalizeReferenceLabel(match[1]), match[2] || match[3]);
  }
  for (const match of withoutCode.matchAll(
    /!\[([^\]]*)\]\[([^\]]*)\]/g,
  )) {
    const label = normalizeReferenceLabel(match[2] || match[1]);
    const reference = definitions.get(label);
    if (reference) references.add(reference);
  }
  for (const match of withoutCode.matchAll(
    /!\[[^\]]*\]\([ \t]*(?:<([^>\n]+)>|([^\s)]+))/g,
  )) {
    references.add(match[1] || match[2]);
  }
  for (const match of withoutCode.matchAll(
    /<img\b[^>]*\bsrc[ \t]*=[ \t]*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))[^>]*>/gi,
  )) {
    references.add(match[1] || match[2] || match[3]);
  }

  const resolved = new Set<string>();
  for (const reference of references) {
    const path = resolveLocalMediaPath(reference, entrypoint);
    if (path) resolved.add(path);
  }
  return resolved;
}

function resolveLocalMediaPath(
  reference: string,
  entrypoint: string,
): string | null {
  const value = reference.trim();
  if (
    !value
    || value.startsWith("//")
    || /^(?:data|https?):/i.test(value)
    || /^[a-z][a-z0-9+.-]*:/i.test(value)
  ) {
    return null;
  }
  let decoded: string;
  try {
    decoded = decodeURIComponent(value.split(/[?#]/, 1)[0]);
  } catch {
    throw new Error(`A local Markdown image path is invalid: ${value}`);
  }
  const base = entrypoint.split("/").slice(0, -1);
  const segments = decoded.startsWith("/")
    ? []
    : [...base];
  for (const segment of decoded.split("/")) {
    if (!segment || segment === ".") continue;
    if (segment === "..") {
      if (segments.length === 0) {
        throw new Error(`A local Markdown image escapes the selected folder: ${value}`);
      }
      segments.pop();
      continue;
    }
    segments.push(segment);
  }
  return segments.join("/");
}

function normalizeReferenceLabel(value: string): string {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
}

function concatenate(parts: Uint8Array[]): Uint8Array<ArrayBuffer> {
  const result = new Uint8Array(
    parts.reduce((total, part) => total + part.length, 0),
  );
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.length;
  }
  return result;
}

function normalizedFolderFiles(files: File[]): Array<{ file: File; path: string }> {
  const raw = files.map((file) =>
    (file.webkitRelativePath || file.name).replaceAll("\\", "/")
  );
  const first = raw[0]?.split("/")[0];
  const stripRoot = Boolean(
    first && raw.every((path) => path.startsWith(`${first}/`)),
  );
  return files.map((file, index) => {
    const path = stripRoot ? raw[index].slice(first!.length + 1) : raw[index];
    if (!path || path.startsWith("/") || path.split("/").includes("..")) {
      throw new Error("The selected folder contains an unsafe path.");
    }
    return { file, path };
  });
}

function crc32(content: Uint8Array): number {
  let value = 0xffffffff;
  for (const byte of content) {
    value ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      value = (value >>> 1) ^ ((value & 1) ? 0xedb88320 : 0);
    }
  }
  return (value ^ 0xffffffff) >>> 0;
}

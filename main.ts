// mend — one patch tool for every edit format an AI model writes: unified diffs,
// SEARCH/REPLACE blocks, V4A patches and JSON edit calls. It reads a patch, applies
// every edit to the tree of text files under the working directory, and reports what
// changed; if any edit fails, nothing is written. The behaviour is SPEC.md, section
// by section. Node 22, the standard library alone.
//
// In a trial the grader runs this file with MAKEFUN_ENV_URL set; the last lines then
// hand over to runner.ts, which the kit ships and which stays as it is.

import { mkdirSync, readFileSync, rmSync, statSync, writeFileSync, writeSync } from 'node:fs';
import { dirname } from 'node:path';

/** An exit-2 outcome: the single line to print (a usage, no-patch, format or parse error). */
class ExitLine {
  text: string;
  constructor(text: string) {
    this.text = text;
  }
}

/** One edit failed: its code and the path the failure concerns. */
class EditFault {
  code: string;
  path: string;
  constructor(code: string, path: string) {
    this.code = code;
    this.path = path;
  }
}

const OPTS = { strict: false };
const FORMAT_NAMES = ['auto', 'unified', 'search-replace', 'v4a', 'json'];

/** The patch's lines (SPEC.md section 3): split at \n, one trailing \r removed, no empty piece after a final \n. */
function patchLinesOf(text: string): string[] {
  if (text === '') return [];
  const parts = text.split('\n');
  if (text.endsWith('\n')) parts.pop();
  return parts.map((l) => (l.endsWith('\r') ? l.slice(0, -1) : l));
}

/** A safe path (section 3): relative, no backslash, no empty, dot or dot-dot segment. */
function pathSafe(p: string): boolean {
  if (p === '' || p.startsWith('/') || p.includes('\\')) return false;
  return p.split('/').every((seg) => seg !== '' && seg !== '.' && seg !== '..');
}

function guardPath(p: string): void {
  if (!pathSafe(p)) throw new EditFault('unsafe', p);
}

/** A file as lines: its own line break, and whether it ends with one (section 3). */
interface FileModel {
  lines: string[];
  br: string;
  tail: boolean;
}

function modelOf(text: string): FileModel {
  if (text === '') return { lines: [], br: '\n', tail: false };
  const at = text.indexOf('\n');
  const br = at > 0 && text.charAt(at - 1) === '\r' ? '\r\n' : '\n';
  const parts = text.split(br);
  const tail = text.endsWith(br);
  if (tail) parts.pop();
  return { lines: parts, br, tail };
}

function textOf(m: FileModel): string {
  if (m.lines.length === 0) return '';
  return m.lines.join(m.br) + (m.tail ? m.br : '');
}

/**
 * The tree as the edits leave it. `before` caches each file's bytes on disk; `now`
 * holds the state the edits so far have made (null: the file is not there). A failed
 * edit is rolled back by restoring a snapshot of `now` and of the move list.
 */
class Tree {
  private disk = new Map<string, string | null>();
  now = new Map<string, string | null>();
  moves: Array<[string, string]> = [];
  before(p: string): string | null {
    if (!this.disk.has(p)) {
      let v: string | null = null;
      try {
        if (statSync(p).isFile()) v = readFileSync(p, 'utf8');
      } catch {
        v = null;
      }
      this.disk.set(p, v);
    }
    return this.disk.get(p)!;
  }
  get(p: string): string | null {
    return this.now.has(p) ? this.now.get(p)! : this.before(p);
  }
  put(p: string, v: string | null): void {
    this.now.set(p, v);
  }
}

type EditFn = (tree: Tree) => void;

// Matching lines (section 5): three tiers of sameness, and writing the new lines.

type Tier = 0 | 1 | 2;

function atTier(line: string, tier: Tier): string {
  if (tier === 0) return line;
  const noTrail = line.replace(/[ \t]+$/, '');
  return tier === 1 ? noTrail : noTrail.replace(/^[ \t]+/, '');
}

/** Every start position, at or after `from`, where `pat` matches `lines` at `tier`; positions may overlap. */
function occurrences(lines: string[], pat: string[], tier: Tier, from: number): number[] {
  const want = pat.map((l) => atTier(l, tier));
  const hits: number[] = [];
  for (let i = Math.max(0, from); i + pat.length <= lines.length; i++) {
    let ok = true;
    for (let j = 0; j < pat.length; j++) {
      if (atTier(lines[i + j]!, tier) !== want[j]) {
        ok = false;
        break;
      }
    }
    if (ok) hits.push(i);
  }
  return hits;
}

/** The first tier at which `pat` occurs at or after `from`, with its positions; null when it occurs at none. */
function findMatch(lines: string[], pat: string[], from: number): { tier: Tier; hits: number[] } | null {
  const tiers: Tier[] = OPTS.strict ? [0] : [0, 1, 2];
  for (const t of tiers) {
    const hits = occurrences(lines, pat, t, from);
    if (hits.length > 0) return { tier: t, hits };
  }
  return null;
}

interface Span {
  text: string;
  ctx: boolean;
}

function spansOf(body: Array<{ kind: string; text: string }>): { oldSpan: Span[]; newSpan: Span[] } {
  const oldSpan: Span[] = [];
  const newSpan: Span[] = [];
  for (const b of body) {
    if (b.kind === 'ctx') {
      oldSpan.push({ text: b.text, ctx: true });
      newSpan.push({ text: b.text, ctx: true });
    } else if (b.kind === 'del') oldSpan.push({ text: b.text, ctx: false });
    else if (b.kind === 'add') newSpan.push({ text: b.text, ctx: false });
  }
  return { oldSpan, newSpan };
}

/**
 * Replace the matched lines at `at` by the new span, section 5's way: a context line
 * is written as the file's own matched line, and at tier 2 every other new line is
 * re-indented from the pattern's first line to the file's first matched line.
 */
function splice(lines: string[], at: number, oldSpan: Span[], newSpan: Span[], tier: Tier): string[] {
  const ctxFile: string[] = [];
  oldSpan.forEach((s, j) => {
    if (s.ctx) ctxFile.push(lines[at + j]!);
  });
  let lead = '';
  let patLead = '';
  if (tier === 2 && oldSpan.length > 0) {
    lead = /^[ \t]*/.exec(lines[at] ?? '')![0];
    patLead = /^[ \t]*/.exec(oldSpan[0]!.text)![0];
  }
  let ci = 0;
  const out: string[] = [];
  for (const s of newSpan) {
    if (s.ctx) {
      out.push(ctxFile[ci++]!);
      continue;
    }
    let t = s.text;
    if (tier === 2 && t !== '' && t.trim() !== '' && t.startsWith(patLead)) t = lead + t.slice(patLead.length);
    out.push(t);
  }
  return [...lines.slice(0, at), ...out, ...lines.slice(at + oldSpan.length)];
}

// JSON edits (section 9): one array of edit objects, or one object a line.

function jsonObjectEdit(v: unknown): EditFn | null {
  if (typeof v !== 'object' || v === null || Array.isArray(v)) return null;
  const o = v as Record<string, unknown>;
  const p = o.path;
  if (typeof p !== 'string') return null;
  if (o.command === 'create') {
    if (typeof o.file_text !== 'string') return null;
    const text = o.file_text;
    return (tree) => {
      guardPath(p);
      if (tree.get(p) !== null) throw new EditFault('exists', p);
      tree.put(p, text);
    };
  }
  if (o.command === 'str_replace') {
    if (typeof o.old_str !== 'string' || o.old_str === '') return null;
    if (o.new_str !== undefined && typeof o.new_str !== 'string') return null;
    const oldStr = o.old_str;
    const newStr = (o.new_str as string | undefined) ?? '';
    return (tree) => {
      guardPath(p);
      const before = tree.get(p);
      if (before === null) throw new EditFault('missing', p);
      // In a CRLF file the text and both strings are read with every \r\n as \n,
      // and the result is written back with \r\n.
      const crlf = modelOf(before).br === '\r\n';
      const flat = (s: string): string => (crlf ? s.split('\r\n').join('\n') : s);
      const hay = flat(before);
      const needle = flat(oldStr);
      let count = 0;
      for (let k = hay.indexOf(needle); k >= 0; k = hay.indexOf(needle, k + 1)) count += 1;
      if (count === 0) throw new EditFault('no-match', p);
      if (count > 1) throw new EditFault('ambiguous', p);
      const at = hay.indexOf(needle);
      const joined = hay.slice(0, at) + flat(newStr) + hay.slice(at + needle.length);
      tree.put(p, crlf ? joined.split('\n').join('\r\n') : joined);
    };
  }
  if (o.command === 'insert') {
    if (typeof o.insert_line !== 'number' || !Number.isInteger(o.insert_line)) return null;
    if (typeof o.new_str !== 'string') return null;
    const at = o.insert_line;
    const add = patchLinesOf(o.new_str);
    return (tree) => {
      guardPath(p);
      const before = tree.get(p);
      if (before === null) throw new EditFault('missing', p);
      const m = modelOf(before);
      if (at < 0 || at > m.lines.length) throw new EditFault('no-match', p);
      m.lines.splice(at, 0, ...add);
      tree.put(p, textOf(m));
    };
  }
  return null;
}

function readJson(text: string, lines: string[]): EditFn[] {
  const first = lines.findIndex((l) => l.trim() !== '');
  if (first < 0) return [];
  const edits: EditFn[] = [];
  if (lines[first]!.trimStart().startsWith('[')) {
    // Array form: any fault is a parse error at the line where the array starts.
    let v: unknown;
    try {
      v = JSON.parse(text);
    } catch {
      throw new ExitLine(`error: parse ${first + 1}`);
    }
    if (!Array.isArray(v)) throw new ExitLine(`error: parse ${first + 1}`);
    for (const item of v) {
      const e = jsonObjectEdit(item);
      if (e === null) throw new ExitLine(`error: parse ${first + 1}`);
      edits.push(e);
    }
    return edits;
  }
  lines.forEach((l, i) => {
    if (l.trim() === '') return;
    let e: EditFn | null = null;
    try {
      e = jsonObjectEdit(JSON.parse(l));
    } catch {
      e = null;
    }
    if (e === null) throw new ExitLine(`error: parse ${i + 1}`);
    edits.push(e);
  });
  return edits;
}

// SEARCH/REPLACE blocks (section 7).

const SR_OPEN = '<<<<<<< SEARCH';
const SR_MID = '=======';
const SR_END = '>>>>>>> REPLACE';

function searchReplaceEdit(path: string, oldLines: string[], newLines: string[]): EditFn {
  return (tree) => {
    guardPath(path);
    const before = tree.get(path);
    if (oldLines.length === 0) {
      // An empty old side creates the file with the new lines.
      if (before !== null) throw new EditFault('exists', path);
      tree.put(path, textOf({ lines: [...newLines], br: '\n', tail: newLines.length > 0 }));
      return;
    }
    if (before === null) throw new EditFault('missing', path);
    const m = modelOf(before);
    const found = findMatch(m.lines, oldLines, 0);
    if (found === null) throw new EditFault('no-match', path);
    if (found.hits.length > 1) throw new EditFault('ambiguous', path);
    const oldSpan: Span[] = oldLines.map((t) => ({ text: t, ctx: false }));
    const newSpan: Span[] = newLines.map((t) => ({ text: t, ctx: false }));
    m.lines = splice(m.lines, found.hits[0]!, oldSpan, newSpan, found.tier);
    tree.put(path, textOf(m));
  };
}

function readSearchReplace(lines: string[]): EditFn[] {
  const edits: EditFn[] = [];
  let prevPath: string | null = null;
  let scanFrom = 0;
  for (;;) {
    let s = -1;
    for (let k = scanFrom; k < lines.length; k++) {
      if (lines[k]!.trimEnd() === SR_OPEN) {
        s = k;
        break;
      }
    }
    if (s < 0) break;
    let mid = -1;
    for (let k = s + 1; k < lines.length; k++) {
      if (lines[k]!.trimEnd() === SR_MID) {
        mid = k;
        break;
      }
    }
    if (mid < 0) throw new ExitLine(`error: parse ${s + 1}`);
    let r = -1;
    for (let k = mid + 1; k < lines.length; k++) {
      if (lines[k]!.trimEnd() === SR_END) {
        r = k;
        break;
      }
    }
    if (r < 0) throw new ExitLine(`error: parse ${s + 1}`);
    // The path is the nearest line before the block, after the previous block, that
    // is not blank and does not open a code fence; failing that, the previous path.
    let path: string | null = null;
    for (let j = s - 1; j >= scanFrom; j--) {
      const t = lines[j]!.trim();
      if (t === '' || t.startsWith('```')) continue;
      path = t;
      break;
    }
    if (path === null) {
      if (prevPath === null) throw new ExitLine(`error: parse ${s + 1}`);
      path = prevPath;
    }
    edits.push(searchReplaceEdit(path, lines.slice(s + 1, mid), lines.slice(mid + 1, r)));
    prevPath = path;
    scanFrom = r + 1;
  }
  return edits;
}

// Unified diffs (section 6).

interface BodyLine {
  kind: 'ctx' | 'del' | 'add' | 'slash';
  text: string;
  blank: boolean;
}

interface Hunk {
  a: number;
  body: BodyLine[];
}

interface Section {
  pairOld: string | null;
  pairNew: string | null;
  hasPair: boolean;
  renameFrom: string | null;
  renameTo: string | null;
  gitOld: string | null;
  gitNew: string | null;
  create: boolean;
  del: boolean;
  hunks: Hunk[];
}

function newSection(): Section {
  return { pairOld: null, pairNew: null, hasPair: false, renameFrom: null, renameTo: null, gitOld: null, gitNew: null, create: false, del: false, hunks: [] };
}

/** The paths of a `diff --git a/<X> b/<Y>` line, or null when the line is not of that form. */
function diffGitPaths(line: string): [string, string] | null {
  const rest = line.slice('diff --git '.length);
  if (!rest.startsWith('a/')) return null;
  const cut = rest.indexOf(' b/', 2);
  if (cut < 0) return null;
  const x = rest.slice(2, cut);
  const y = rest.slice(cut + 3);
  if (x === '' || y === '') return null;
  return [x, y];
}

/** A ---/+++ path: up to the first tab, spaces removed, /dev/null is no file, a leading a/ or b/ removed. */
function headerPath(raw: string): string | null {
  const p = raw.split('\t')[0]!.trim();
  if (p === '/dev/null') return null;
  return p.startsWith('a/') || p.startsWith('b/') ? p.slice(2) : p;
}

/** Whether a `\` line follows the hunk's last context or + line in its body. */
function slashAfterLastWrite(body: BodyLine[]): boolean {
  let last = -1;
  body.forEach((b, i) => {
    if (b.kind === 'ctx' || b.kind === 'add') last = i;
  });
  return last >= 0 && body[last + 1]?.kind === 'slash';
}

function applyHunk(m: FileModel, h: Hunk, d: number, path: string): void {
  const { oldSpan, newSpan } = spansOf(h.body);
  if (oldSpan.length === 0) {
    // No old lines: the new lines go after the first A + d lines of the file.
    const at = Math.min(h.a + d, m.lines.length);
    const atEnd = at === m.lines.length;
    const written = newSpan.map((s) => s.text);
    m.lines = [...m.lines.slice(0, at), ...written, ...m.lines.slice(at)];
    if (atEnd) m.tail = !slashAfterLastWrite(h.body);
    return;
  }
  const found = findMatch(m.lines, oldSpan.map((s) => s.text), 0);
  if (found === null) throw new EditFault('no-match', path);
  // Of the occurrences, the one whose position is nearest to A - 1 + d; ties go earlier.
  const want = h.a - 1 + d;
  let at = found.hits[0]!;
  for (const hit of found.hits) {
    if (Math.abs(hit - want) < Math.abs(at - want)) at = hit;
  }
  const touchesEnd = at + oldSpan.length === m.lines.length;
  m.lines = splice(m.lines, at, oldSpan, newSpan, found.tier);
  if (touchesEnd) m.tail = !slashAfterLastWrite(h.body);
}

function unifiedEdit(s: Section): EditFn {
  let rawOld: string | null;
  let rawNew: string | null;
  if (s.hasPair) {
    rawOld = s.pairOld;
    rawNew = s.pairNew;
  } else if (s.renameFrom !== null || s.renameTo !== null) {
    rawOld = s.renameFrom;
    rawNew = s.renameTo;
  } else {
    rawOld = s.gitOld;
    rawNew = s.gitNew;
  }
  return (tree) => {
    if (rawOld !== null) guardPath(rawOld);
    if (rawNew !== null) guardPath(rawNew);
    const oldP = s.create ? null : rawOld;
    const newP = s.del ? null : rawNew;
    if (oldP === null) {
      // A creation: the file is the + lines of all the section's hunks, in order.
      const target = newP;
      if (target === null) throw new EditFault('unsafe', '');
      if (tree.get(target) !== null) throw new EditFault('exists', target);
      const entries = s.hunks.flatMap((h) => h.body);
      const plus = entries.filter((b) => b.kind === 'add').map((b) => b.text);
      let lastAdd = -1;
      entries.forEach((b, i) => {
        if (b.kind === 'add') lastAdd = i;
      });
      const tail = !(lastAdd >= 0 && entries[lastAdd + 1]?.kind === 'slash');
      tree.put(target, textOf({ lines: plus, br: '\n', tail }));
      return;
    }
    if (newP === null) {
      // A deletion: the file goes; its hunks are not read.
      if (tree.get(oldP) === null) throw new EditFault('missing', oldP);
      tree.put(oldP, null);
      return;
    }
    const before = tree.get(oldP);
    if (before === null) throw new EditFault('missing', oldP);
    if (newP !== oldP && tree.get(newP) !== null) throw new EditFault('exists', newP);
    const m = modelOf(before);
    let d = 0;
    for (const h of s.hunks) {
      applyHunk(m, h, d, oldP);
      const { oldSpan, newSpan } = spansOf(h.body);
      d += newSpan.length - oldSpan.length;
    }
    const out = textOf(m);
    if (newP === oldP) {
      tree.put(oldP, out);
    } else {
      tree.put(newP, out);
      tree.put(oldP, null);
      tree.moves.push([oldP, newP]);
    }
  };
}

function readUnified(lines: string[]): EditFn[] {
  const sections: Section[] = [];
  let sec: Section | null = null;
  let i = 0;
  const n = lines.length;
  while (i < n) {
    const line = lines[i]!;
    if (line.startsWith('diff --git ')) {
      const xy = diffGitPaths(line);
      if (xy === null) throw new ExitLine(`error: parse ${i + 1}`);
      sec = newSection();
      sec.gitOld = xy[0];
      sec.gitNew = xy[1];
      sections.push(sec);
      i += 1;
      continue;
    }
    if (line.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')) {
      if (sec !== null && !sec.hasPair && sec.hunks.length === 0 && sec.gitOld !== null) {
        // The header pair of the section the diff --git line opened.
        sec.pairOld = headerPath(line.slice(4));
        sec.pairNew = headerPath(lines[i + 1]!.slice(4));
        sec.hasPair = true;
      } else {
        sec = newSection();
        sec.pairOld = headerPath(line.slice(4));
        sec.pairNew = headerPath(lines[i + 1]!.slice(4));
        sec.hasPair = true;
        sections.push(sec);
      }
      i += 2;
      continue;
    }
    if (line.startsWith('--- ') && sec !== null && !sec.hasPair && sec.hunks.length === 0) {
      // A --- line inside a section's header with no +++ after it.
      throw new ExitLine(`error: parse ${i + 2}`);
    }
    if (line.startsWith('@@')) {
      if (sec === null) throw new ExitLine(`error: parse ${i + 1}`);
      const hm = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/.exec(line);
      if (hm === null) throw new ExitLine(`error: parse ${i + 1}`);
      const hunk: Hunk = { a: parseInt(hm[1]!, 10), body: [] };
      i += 1;
      while (i < n) {
        const b = lines[i]!;
        if (b.startsWith('diff --git ')) break;
        if (b.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')) break;
        if (b === '') hunk.body.push({ kind: 'ctx', text: '', blank: true });
        else if (b.startsWith(' ')) hunk.body.push({ kind: 'ctx', text: b.slice(1), blank: false });
        else if (b.startsWith('-')) hunk.body.push({ kind: 'del', text: b.slice(1), blank: false });
        else if (b.startsWith('+')) hunk.body.push({ kind: 'add', text: b.slice(1), blank: false });
        else if (b.startsWith('\\')) hunk.body.push({ kind: 'slash', text: b.slice(1), blank: false });
        else break;
        i += 1;
      }
      // Empty lines at the end of a body are not part of it.
      while (hunk.body.length > 0 && hunk.body[hunk.body.length - 1]!.blank) hunk.body.pop();
      sec.hunks.push(hunk);
      continue;
    }
    if (sec !== null && sec.hunks.length === 0) {
      // Header lines before the first hunk; anything unlisted here is ignored.
      if (line.startsWith('rename from ')) sec.renameFrom = line.slice('rename from '.length).trim();
      else if (line.startsWith('rename to ')) sec.renameTo = line.slice('rename to '.length).trim();
      else if (line.startsWith('new file mode')) sec.create = true;
      else if (line.startsWith('deleted file mode')) sec.del = true;
      i += 1;
      continue;
    }
    i += 1; // Prose outside bodies and headers is ignored.
  }
  return sections.map((s) => unifiedEdit(s));
}

// V4A patches (section 8).

interface VChunk {
  anchor: string | null;
  eof: boolean;
  body: Array<{ kind: 'ctx' | 'del' | 'add'; text: string }>;
}

function v4aAdd(path: string, fileLines: string[]): EditFn {
  return (tree) => {
    guardPath(path);
    if (tree.get(path) !== null) throw new EditFault('exists', path);
    tree.put(path, textOf({ lines: [...fileLines], br: '\n', tail: fileLines.length > 0 }));
  };
}

function v4aDelete(path: string): EditFn {
  return (tree) => {
    guardPath(path);
    if (tree.get(path) === null) throw new EditFault('missing', path);
    tree.put(path, null);
  };
}

function v4aUpdate(path: string, secLine: number, raw: Array<{ text: string; no: number }>): EditFn {
  // Blank lines at the end of the section are not part of it (a blank line is one
  // that is empty or only whitespace).
  const body = [...raw];
  while (body.length > 0 && body[body.length - 1]!.text.trim() === '') body.pop();
  let moveTo: string | null = null;
  const chunks: VChunk[] = [];
  let cur: VChunk | null = null;
  let started = false;
  for (const r of body) {
    const l = r.text;
    if (!started && l === '') continue; // spacing before the section's first line
    if (!started && l.startsWith('*** Move to: ')) {
      moveTo = l.slice('*** Move to: '.length).trim();
      started = true;
      continue;
    }
    started = true;
    if (l.startsWith('@@')) {
      const anchor = l.slice(2).trim();
      cur = { anchor: anchor === '' ? null : anchor, eof: false, body: [] };
      chunks.push(cur);
      continue;
    }
    if (l.trim() === '*** End of File') {
      if (cur === null) throw new ExitLine(`error: parse ${r.no}`);
      cur.eof = true;
      cur = null;
      continue;
    }
    const c = l.charAt(0);
    if (l === '' || c === ' ' || c === '-' || c === '+') {
      if (cur === null) {
        // A body line opens the first chunk; a later chunk must open at an @@ line.
        if (chunks.length > 0) throw new ExitLine(`error: parse ${r.no}`);
        cur = { anchor: null, eof: false, body: [] };
        chunks.push(cur);
      }
      cur.body.push({ kind: l === '' || c === ' ' ? 'ctx' : c === '-' ? 'del' : 'add', text: l === '' ? '' : l.slice(1) });
      continue;
    }
    throw new ExitLine(`error: parse ${r.no}`);
  }
  if (chunks.length === 0 && moveTo === null) throw new ExitLine(`error: parse ${secLine}`);
  return (tree) => {
    guardPath(path);
    if (moveTo !== null) guardPath(moveTo);
    const before = tree.get(path);
    if (before === null) throw new EditFault('missing', path);
    if (moveTo !== null && tree.get(moveTo) !== null) throw new EditFault('exists', moveTo);
    const m = modelOf(before);
    let cursor = 0;
    for (const ch of chunks) {
      if (ch.anchor !== null) {
        // The anchor is found first, at the first tier where a line at or after
        // the cursor equals it; the cursor moves past that line.
        const tiers: Tier[] = OPTS.strict ? [0] : [0, 1, 2];
        let atIdx = -1;
        for (const t of tiers) {
          const hits = occurrences(m.lines, [ch.anchor], t, cursor);
          if (hits.length > 0) {
            atIdx = hits[0]!;
            break;
          }
        }
        if (atIdx < 0) throw new EditFault('no-match', path);
        cursor = atIdx + 1;
      }
      const { oldSpan, newSpan } = spansOf(ch.body);
      if (oldSpan.length === 0) {
        const at = ch.eof ? m.lines.length : Math.min(cursor, m.lines.length);
        const written = newSpan.map((s) => s.text);
        m.lines = [...m.lines.slice(0, at), ...written, ...m.lines.slice(at)];
        cursor = at + written.length;
        continue;
      }
      // The first occurrence at or after the cursor, at the first tier with one;
      // an end-of-file chunk takes only an occurrence that ends at the last line.
      const tiers: Tier[] = OPTS.strict ? [0] : [0, 1, 2];
      let placed: { tier: Tier; at: number } | null = null;
      for (const t of tiers) {
        const hits = occurrences(m.lines, oldSpan.map((s) => s.text), t, cursor).filter((h) => !ch.eof || h + oldSpan.length === m.lines.length);
        if (hits.length > 0) {
          placed = { tier: t, at: hits[0]! };
          break;
        }
      }
      if (placed === null) throw new EditFault('no-match', path);
      m.lines = splice(m.lines, placed.at, oldSpan, newSpan, placed.tier);
      cursor = placed.at + newSpan.length;
    }
    const out = textOf(m);
    if (moveTo !== null) {
      tree.put(moveTo, out);
      tree.put(path, null);
      tree.moves.push([path, moveTo]);
    } else {
      tree.put(path, out);
    }
  };
}

function readV4a(lines: string[]): EditFn[] {
  const first = lines.findIndex((l) => l.trim() !== '');
  if (first < 0) throw new ExitLine('error: parse 1');
  if (lines[first]!.trim() !== '*** Begin Patch') throw new ExitLine(`error: parse ${first + 1}`);
  let last = lines.length - 1;
  while (last > first && lines[last]!.trim() === '') last -= 1;
  if (last === first || lines[last]!.trim() !== '*** End Patch') throw new ExitLine(`error: parse ${lines.length + 1}`);
  const secKind = (l: string): 'add' | 'del' | 'update' | null =>
    l.startsWith('*** Add File: ') ? 'add' : l.startsWith('*** Delete File: ') ? 'del' : l.startsWith('*** Update File: ') ? 'update' : null;
  const edits: EditFn[] = [];
  let i = first + 1;
  while (i < last) {
    const line = lines[i]!;
    if (line.trim() === '') {
      i += 1;
      continue;
    }
    const kind = secKind(line);
    if (kind === null) throw new ExitLine(`error: parse ${i + 1}`);
    const prefix = kind === 'add' ? '*** Add File: ' : kind === 'del' ? '*** Delete File: ' : '*** Update File: ';
    const path = line.slice(prefix.length).trim();
    const secLine = i + 1;
    i += 1;
    const raw: Array<{ text: string; no: number }> = [];
    while (i < last && secKind(lines[i]!) === null) {
      raw.push({ text: lines[i]!, no: i + 1 });
      i += 1;
    }
    if (kind === 'add') {
      // Blank lines at the end of the section are not part of it.
      const kept = [...raw];
      while (kept.length > 0 && kept[kept.length - 1]!.text.trim() === '') kept.pop();
      const fileLines: string[] = [];
      for (const r of kept) {
        if (r.text.startsWith('+')) {
          fileLines.push(r.text.slice(1));
        } else if (r.text === '') {
          fileLines.push('');
        } else {
          throw new ExitLine(`error: parse ${r.no}`);
        }
      }
      edits.push(v4aAdd(path, fileLines));
    } else if (kind === 'del') {
      for (const r of raw) {
        if (r.text.trim() !== '') throw new ExitLine(`error: parse ${r.no}`);
      }
      edits.push(v4aDelete(path));
    } else {
      edits.push(v4aUpdate(path, secLine, raw));
    }
  }
  return edits;
}

// Detecting the format (section 4), in its order: v4a, json, search-replace, unified.

function detectFormat(lines: string[]): string | null {
  const first = lines.findIndex((l) => l.trim() !== '');
  if (first < 0) return null;
  if (lines[first]!.trim() === '*** Begin Patch') return 'v4a';
  const c = lines[first]!.trimStart().charAt(0);
  if (c === '[' || c === '{') return 'json';
  if (lines.some((l) => l.trimEnd() === SR_OPEN)) return 'search-replace';
  if (lines.some((l, i) => l.startsWith('diff --git ') || (l.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')))) return 'unified';
  return null;
}

// The command line (sections 1, 2 and 10).

function runCli(args: string[]): { code: number; out: string[] } {
  let format: string | null = null;
  let checkOnly = false;
  let strict = false;
  const pos: string[] = [];
  let ended = false;
  for (const a of args) {
    if (!ended && a === '--') {
      ended = true;
      continue;
    }
    if (!ended && a.startsWith('-') && a !== '-') {
      if (a.startsWith('--format=') && format === null) {
        const v = a.slice('--format='.length);
        if (!FORMAT_NAMES.includes(v)) return { code: 2, out: ['error: usage'] };
        format = v;
      } else if (a === '--check' && !checkOnly) {
        checkOnly = true;
      } else if (a === '--strict' && !strict) {
        strict = true;
      } else {
        return { code: 2, out: ['error: usage'] };
      }
      continue;
    }
    pos.push(a);
  }
  if (pos.length > 1) return { code: 2, out: ['error: usage'] };
  OPTS.strict = strict;
  const file = pos.length === 0 || pos[0] === '-' ? null : pos[0]!;
  let text: string;
  if (file === null) {
    try {
      text = readFileSync(0, 'utf8');
    } catch {
      text = '';
    }
  } else {
    try {
      text = readFileSync(file, 'utf8');
    } catch {
      return { code: 2, out: ['error: no-patch'] };
    }
  }
  const lines = patchLinesOf(text);
  const kind = format === null || format === 'auto' ? detectFormat(lines) : format;
  if (kind === null) return { code: 2, out: ['error: format'] };
  let edits: EditFn[];
  try {
    edits =
      kind === 'json'
        ? readJson(text, lines)
        : kind === 'search-replace'
          ? readSearchReplace(lines)
          : kind === 'v4a'
            ? readV4a(lines)
            : readUnified(lines);
    if (edits.length === 0) throw new ExitLine('error: parse 1');
  } catch (e) {
    if (e instanceof ExitLine) return { code: 2, out: [e.text] };
    throw e;
  }
  // The edits run in the patch's order, each on the tree the earlier ones left; a
  // failed edit changes nothing and the ones after it still run.
  const tree = new Tree();
  const fails: EditFault[] = [];
  for (const edit of edits) {
    const snap = new Map(tree.now);
    const moveCount = tree.moves.length;
    try {
      edit(tree);
    } catch (e) {
      if (!(e instanceof EditFault)) throw e;
      tree.now = snap;
      tree.moves.length = moveCount;
      fails.push(e);
    }
  }
  if (fails.length > 0) {
    return { code: 1, out: [...fails.map((f) => `error: ${f.code} ${f.path}`), `failed: ${fails.length} of ${edits.length} edits`] };
  }
  // Report every path whose state changed, in byte order of the paths; a move that
  // leaves its old path gone and its new path present is one R line, sorted by the new path.
  const after = (p: string): string | null => (tree.now.has(p) ? tree.now.get(p)! : tree.before(p));
  const rPairs: Array<[string, string]> = [];
  const usedOld = new Set<string>();
  const usedNew = new Set<string>();
  for (const [o, nw] of tree.moves) {
    if (usedOld.has(o) || usedNew.has(nw)) continue;
    if (tree.before(o) !== null && after(o) === null && tree.before(nw) === null && after(nw) !== null) {
      rPairs.push([o, nw]);
      usedOld.add(o);
      usedNew.add(nw);
    }
  }
  const rows: Array<{ key: string; text: string }> = [];
  for (const p of tree.now.keys()) {
    if (usedOld.has(p) || usedNew.has(p)) continue;
    const was = tree.before(p);
    const now = tree.now.get(p)!;
    if (was === now) continue;
    rows.push({ key: p, text: `${was === null ? 'A' : now === null ? 'D' : 'M'} ${p}` });
  }
  for (const [o, nw] of rPairs) rows.push({ key: nw, text: `R ${o} -> ${nw}` });
  rows.sort((x, y) => Buffer.compare(Buffer.from(x.key, 'utf8'), Buffer.from(y.key, 'utf8')));
  if (!checkOnly) {
    for (const p of tree.now.keys()) {
      const was = tree.before(p);
      const now = tree.now.get(p)!;
      if (was === now) continue;
      if (now === null) rmSync(p, { force: true });
      else {
        mkdirSync(dirname(p), { recursive: true });
        writeFileSync(p, now);
      }
    }
  }
  const out = rows.map((r) => r.text);
  out.push(`ok: ${rows.length} ${rows.length === 1 ? 'file' : 'files'}`);
  return { code: 0, out };
}

const trialEnv = process.env.MAKEFUN_ENV_URL;
if (trialEnv !== undefined && trialEnv !== '') {
  // A trial: the kit's runner reads the checks from the env and runs this file on each one, without this variable.
  const { runTrial } = await import('./runner.ts');
  await runTrial(trialEnv);
} else {
  const r = runCli(process.argv.slice(2));
  writeSync(1, r.out.map((l) => `${l}\n`).join(''));
  process.exitCode = r.code;
}

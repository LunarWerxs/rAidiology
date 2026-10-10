// On-device de-identification for a donated scan: decide which series may go, and rebuild each
// accepted picture as a brand-new DICOM file that holds only an allow-list of technical tags.
//
// The rules are those in docs/DONATED_SCANS.md ("What leaves the browser" and "The build"). Nothing
// of an original file is copied across: every kept value is re-encoded, the pixels are decoded and
// written fresh, and the UIDs are new. No DOM is used, and nothing runs at import time, so the same
// module runs in a browser and in Node (the check is tools/deid_check.mjs).
//
// What the panel calls (files are { name, bytes: Uint8Array }):
//
//   judgeFiles(files, { now, onProgress })  -> { series: [Series], ignored }
//       Groups the files into series and decides each one. A Series is
//       { id, files, count, accepted, reason, modality, bodyPart, ageYears, description }.
//       `reason` is plain English for the donor when accepted is false (otherwise null). `bodyPart`
//       is the allow-list word that matched. `ageYears` is the whole-year age at the study that
//       would be written (capped at 90). `description` is for the screen only and is never sent.
//       `ignored` counts the files that are not DICOM at all. Every picture of a series goes through
//       the text finder here. `now` is a Date (default: today).
//   newUidMap()                             -> a map to share across one donation's series, so the
//       same original study, series, frame of reference or image always gets the same new UID.
//   deidentifySeries(series, { uidMap, seriesIndex, onProgress })
//       -> { index, code, modality, bodyPart, ageYears, tags, preview, images }
//       for an accepted series. `images` are { name: 's<index>/i<n>.dcm', bytes, sha256, rows, cols }
//       in instance order. `tags` is the list of kept tag keys (8 hex digits, upper case, group 0002
//       left out), for the manifest and the consent record. `preview` is [{ key, name, value }] of
//       the first picture, for the list of tags and values the donor reads before sending.
//   buildManifest(list)                     -> the manifest.json object { terms: 1, series: [...] }
//       from the results of deidentifySeries.
//   fingerprint(pixelBytes), seriesCode(sha256s)   the contract's SHA-256 values.
//   readAgeProof(bytes, now)                -> { ok, years, atStudy } or { ok: false, why }
//       Reads ONLY birth date, age and the study dates, straight from the raw bytes. It returns whole
//       years and never keeps or returns a date.
//   findText(raw, rows, cols, invert)       -> boolean, the text finder on one picture.
//   matchBodyPart(fields)                   -> { word } or { refuse }, the body-part rule.
//
// dicom.js is unchanged: it still never puts a name, ID or birth date into its element map.

import { readDataset, describe, decodeFrame, text, isDicom, IMPLICIT_VR } from './dicom.js';

export const TERMS_VERSION = 1;
const IMPLEMENTATION_UID = '2.25.240710231002871236541907316538274113';
const IMPLEMENTATION_NAME = 'RAIDIOLOGY1';
const EXPLICIT_LE = '1.2.840.10008.1.2.1';

// ---- Picture types ---------------------------------------------------------------------------

const ALLOWED_SOP = new Map([
  ['1.2.840.10008.5.1.4.1.1.2', 'CT'],
  ['1.2.840.10008.5.1.4.1.1.4', 'MR'],
  ['1.2.840.10008.5.1.4.1.1.1', 'CR'],
  ['1.2.840.10008.5.1.4.1.1.1.1', 'DX'],
  ['1.2.840.10008.5.1.4.1.1.1.1.1', 'DX'],
]);

const SOP_KINDS = [
  ['1.2.840.10008.5.1.4.1.1.2.1', 'an enhanced multi-frame CT'],
  ['1.2.840.10008.5.1.4.1.1.4.1', 'an enhanced multi-frame MR'],
  ['1.2.840.10008.5.1.4.1.1.6', 'an ultrasound'],
  ['1.2.840.10008.5.1.4.1.1.3', 'an ultrasound'],
  ['1.2.840.10008.5.1.4.1.1.7', 'a secondary capture (a picture made from another picture)'],
  ['1.2.840.10008.5.1.4.1.1.77', 'a photograph'],
  ['1.2.840.10008.5.1.4.1.1.12', 'an angiography or fluoroscopy'],
  ['1.2.840.10008.5.1.4.1.1.20', 'a nuclear medicine'],
  ['1.2.840.10008.5.1.4.1.1.128', 'a PET'],
  ['1.2.840.10008.5.1.4.1.1.88', 'a written report'],
  ['1.2.840.10008.5.1.4.1.1.104', 'a document'],
];

function sopKind(uid) {
  for (const [prefix, name] of SOP_KINDS) {
    if (uid === prefix || uid.startsWith(prefix + '.')) return name;
  }
  return 'this kind of';
}

// ---- Body part -------------------------------------------------------------------------------

const REFUSE_WORDS = ['HEAD', 'BRAIN', 'SKULL', 'FACE', 'FACIAL', 'ORBIT', 'SINUS', 'NECK', 'CERVICAL',
  'DENTAL', 'TEETH', 'TOOTH', 'JAW', 'MANDIBLE', 'MAXILLA', 'TMJ', 'PAROTID', 'THYROID',
  'TEMPORAL BONE', 'PITUITARY', 'CAROTID', 'WILLIS', 'STROKE', 'AIRWAY', 'WHOLE BODY', 'WHOLEBODY',
  'H&N'];
// These are short enough to sit inside ordinary words (THORACIC SPINE holds "C SPINE", CARDIAC holds
// "IAC"), so they only count as a word of their own.
const REFUSE_TOKENS = [/(^|[^A-Z])C[- ]?SPINE/, /(^|[^A-Z])IACS?($|[^A-Z])/, /(^|[^A-Z])WB($|[^A-Z])/];
const ALLOW_WORDS = ['CHEST', 'THORAX', 'LUNG', 'ABDOMEN', 'ABD', 'PELVIS', 'LUMBAR', 'L-SPINE', 'LSPINE',
  'THORACIC SPINE', 'T-SPINE', 'TSPINE', 'HIP', 'KNEE', 'SHOULDER', 'ELBOW', 'WRIST', 'HAND', 'FINGER',
  'FOOT', 'ANKLE', 'TOE', 'FEMUR', 'TIBIA', 'FIBULA', 'HUMERUS', 'FOREARM', 'RIB', 'SACRUM',
  'EXTREMITY'];
const SHORT_ALLOW = new Set(['ABD', 'HIP', 'RIB', 'TOE']);

function hasWord(s, w) {
  return new RegExp('(^|[^A-Z])' + w + '(S)?($|[^A-Z])').test(s);
}

// fields: the upper-cased texts of BodyPartExamined, StudyDescription, SeriesDescription and
// ProtocolName, in that order. Refused when any of them names a forbidden part. Otherwise the first
// allow-list word found is returned, written so it is a valid code string (L-SPINE becomes LSPINE).
export function matchBodyPart(fields) {
  const all = fields.map((f) => (f || '').toUpperCase());
  for (const s of all) {
    for (const w of REFUSE_WORDS) if (s.includes(w)) return { refuse: `it mentions "${w}"` };
    for (const re of REFUSE_TOKENS) if (re.test(s)) return { refuse: 'it mentions the neck or the whole body' };
  }
  for (const s of all) {
    for (const w of ALLOW_WORDS) {
      if (SHORT_ALLOW.has(w) ? hasWord(s, w) : s.includes(w)) return { word: w.replace(/[^A-Z0-9 ]/g, '') };
    }
  }
  return { refuse: null };
}

// ---- Age proof, read from the raw bytes ------------------------------------------------------

function a(bytes, from, n) {
  let s = '';
  for (let i = from; i < from + n && i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return s;
}

const LONG = new Set(['OB', 'OW', 'OF', 'OL', 'OD', 'OV', 'SQ', 'UT', 'UN', 'UC', 'UR']);

// Past an undefined-length value (a sequence): its items, whose own contents may be undefined too.
function skipUndefined(view, bytes, p, little, explicit, depth) {
  const end = bytes.length;
  while (p + 8 <= end) {
    const g = view.getUint16(p, little);
    const e = view.getUint16(p + 2, little);
    const len = view.getUint32(p + 4, little);
    p += 8;
    if (g === 0xfffe && e === 0xe0dd) return p;
    if (g !== 0xfffe) return end;
    if (len !== 0xffffffff) { p += len; continue; }
    // An item of undefined length: elements until its item delimiter.
    while (p + 8 <= end) {
      const g2 = view.getUint16(p, little);
      const e2 = view.getUint16(p + 2, little);
      if (g2 === 0xfffe && e2 === 0xe00d) { p += 8; break; }
      let l2;
      if (explicit) {
        if (LONG.has(a(bytes, p + 4, 2))) { l2 = view.getUint32(p + 8, little); p += 12; }
        else { l2 = view.getUint16(p + 6, little); p += 8; }
      } else { l2 = view.getUint32(p + 4, little); p += 8; }
      p = l2 === 0xffffffff ? (depth > 8 ? end : skipUndefined(view, bytes, p, little, explicit, depth + 1)) : p + l2;
    }
  }
  return end;
}

function validDate(s) {
  const m = /^(\d{4})(\d{2})(\d{2})$/.exec(s.trim());
  if (!m) return null;
  const y = +m[1], mo = +m[2], d = +m[3];
  if (y < 1800 || mo < 1 || mo > 12 || d < 1 || d > 31) return null;
  return [y, mo, d];
}

function wholeYears(from, to) {
  return to[0] - from[0] - ((to[1] < from[1] || (to[1] === from[1] && to[2] < from[2])) ? 1 : 0);
}

// Returns whole years only. The dates are held in local variables for the length of the call.
//   years   = age today, and atStudy = age at the study (what gets written).
// A birth date wins over Patient's Age; when both are present the younger answer counts.
// Fails closed: with neither, or without a study date for a bare age, it says why it cannot prove it.
export function readAgeProof(bytes, now = new Date()) {
  if (!isDicom(bytes)) return { ok: false, why: 'not a DICOM file' };
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const { p, little, explicit } = transferSyntax(bytes, view);
  const { birth, age, dates } = readAgeTags(bytes, view, p, little, explicit);
  const { nowYears, atStudy } = ageAnswer(birth, age, dates, now);
  if (nowYears === null) return { ok: false, why: 'no birth date, and no age with a study date' };
  if (atStudy < 0 || nowYears < 0) return { ok: false, why: 'the dates do not agree' };
  return { ok: true, years: nowYears, atStudy };
}

// Where the data set starts, its byte order and whether its tags carry their own value types. The
// file meta group (after "DICM") always says little-endian and explicit.
function transferSyntax(bytes, view) {
  let p = 0;
  let little = true;
  let explicit = true;
  if (a(bytes, 128, 4) === 'DICM') {
    p = 132;
    while (p + 8 <= bytes.length && view.getUint16(p, true) === 0x0002) {
      const vr = a(bytes, p + 4, 2);
      let len;
      let head;
      if (LONG.has(vr)) { len = view.getUint32(p + 8, true); head = 12; }
      else { len = view.getUint16(p + 6, true); head = 8; }
      if (view.getUint16(p + 2, true) === 0x0010) {
        const ts = a(bytes, p + head, len).replace(/\0+$/, '').trim();
        if (ts === '1.2.840.10008.1.2.2') little = false;
        if (ts === '1.2.840.10008.1.2') explicit = false;
      }
      p += head + len;
    }
  } else {
    explicit = /^[A-Z]{2}$/.test(a(bytes, 4, 2));
  }
  return { p, little, explicit };
}

// The birth date, the age and the study dates, read from the data set; the rest is skipped.
function readAgeTags(bytes, view, p, little, explicit) {
  let birth = null;
  let age = null;
  const dates = {};
  while (p + 8 <= bytes.length) {
    const g = view.getUint16(p, little);
    const e = view.getUint16(p + 2, little);
    if (g > 0x0010 || (g === 0x0010 && e > 0x1010)) break;
    let len;
    if (explicit) {
      const vr = a(bytes, p + 4, 2);
      if (LONG.has(vr)) { len = view.getUint32(p + 8, little); p += 12; }
      else { len = view.getUint16(p + 6, little); p += 8; }
    } else { len = view.getUint32(p + 4, little); p += 8; }
    if (len === 0xffffffff) { p = skipUndefined(view, bytes, p, little, explicit, 0); continue; }
    if (g === 0x0010 && e === 0x0030) birth = validDate(a(bytes, p, len).replace(/\0/g, ''));
    else if (g === 0x0010 && e === 0x1010) age = a(bytes, p, len).replace(/\0/g, '').trim();
    else if (g === 0x0008 && (e === 0x0020 || e === 0x0021 || e === 0x0022)) {
      dates[e] = validDate(a(bytes, p, len).replace(/\0/g, ''));
    }
    p += len;
  }
  return { birth, age, dates };
}

// Age today (nowYears) and age at the study (atStudy), from the birth date when there is one, and
// from Patient's Age when it is the younger answer. Null in both when neither can be proved.
function ageAnswer(birth, age, dates, now) {
  const study = dates[0x0020] || dates[0x0021] || dates[0x0022] || null;
  const today = [now.getFullYear(), now.getMonth() + 1, now.getDate()];
  let nowYears = null;
  let atStudy = null;
  if (birth) {
    nowYears = wholeYears(birth, today);
    atStudy = study ? wholeYears(birth, study) : nowYears;
  }
  const m = /^(\d{3})([DWMY])$/.exec(age || '');
  if (m && study) {
    const n = +m[1];
    const yrs = m[2] === 'Y' ? n : m[2] === 'M' ? Math.floor(n / 12) : m[2] === 'W' ? Math.floor(n / 52.18) : Math.floor(n / 365.25);
    const sinceStudy = Math.max(0, wholeYears(study, today));
    if (nowYears === null || yrs + sinceStudy < nowYears) { nowYears = yrs + sinceStudy; atStudy = yrs; }
  }
  return { nowYears, atStudy };
}

// ---- The text finder -------------------------------------------------------------------------

let stackBuf = new Int32Array(0);

function chainLength(glyphs, vertical) {
  // Glyphs sorted along the line. A chain continues while the next glyph sits on the same line,
  // is about the same size, and follows within a couple of glyph heights.
  const along = vertical ? 1 : 0;       // index of the along-line coordinate pair in each glyph
  const chains = [];
  const list = glyphs.slice().sort((p, q) => (vertical ? p.y0 - q.y0 : p.x0 - q.x0));
  let best = 0;
  for (const g of list) {
    const size = vertical ? g.w : g.h;
    const across = vertical ? (g.x0 + g.x1) / 2 : (g.y0 + g.y1) / 2;
    const start = vertical ? g.y0 : g.x0;
    let hit = null;
    for (const c of chains) {
      const gap = start - c.end;
      const sameSize = size <= 2 * c.size && size * 2 >= c.size;
      if (sameSize && gap >= -1 && gap <= 2.5 * Math.max(size, c.size) &&
          Math.abs(across - c.across) <= 0.6 * Math.max(size, c.size)) { hit = c; break; }
    }
    const end = vertical ? g.y1 : g.x1;
    if (hit) {
      hit.end = Math.max(hit.end, end); hit.n += 1; hit.across = across; hit.size = Math.max(hit.size, size);
      if (hit.n > best) best = hit.n;
    } else {
      chains.push({ end, size, across, n: 1 });
      if (best < 1) best = 1;
    }
  }
  void along;
  return best;
}

let seenBuf = new Uint8Array(0);

// A cheap look at the border bands of one picture (where scanners print text) for printed letters in
// a line. `raw` holds the stored values; `invert` is true for MONOCHROME1, where the displayed bright
// is the stored low. Version 2: printed text is flat (every pixel of a letter holds one value) and
// stands on background, while bone and vessels are noisy and sit inside soft tissue. So a letter is a
// glyph-sized 8-connected shape of exactly equal value whose outside edge is mostly much darker than
// it. Four or more such letters of one value in a row, across or down, count as text; a single marker
// letter makes a line of one and passes.
export function findText(raw, rows, cols, invert) {
  const n = rows * cols;
  if (raw.length < n || rows < 16 || cols < 16) return false;
  let min = raw[0];
  let max = raw[0];
  for (let i = 1; i < n; i++) {
    const v = raw[i];
    if (v < min) min = v;
    else if (v > max) max = v;
  }
  const range = max - min;
  if (!(range > 0)) return false;
  const gap = 0.05 * range;               // how much darker the ground just outside a letter must be
  const sign = invert ? -1 : 1;          // displayed brightness = sign * stored value
  const byValue = borderLetters(raw, rows, cols, sign, gap);
  if (!byValue) return true;
  for (const list of byValue.values()) {
    if (list.length >= 4 && (chainLength(list, false) >= 4 || chainLength(list, true) >= 4)) return true;
  }
  return false;
}

// Every letter-sized shape on the border bands, grouped by its value. Null when there are more than
// 2000 of them, which is too much flat detail to read, so the caller fails closed.
function borderLetters(raw, rows, cols, sign, gap) {
  const n = rows * cols;
  const bandH = Math.max(8, Math.ceil(rows * 0.16));
  const bandW = Math.max(8, Math.ceil(cols * 0.16));
  if (seenBuf.length < n) seenBuf = new Uint8Array(n);
  const seen = seenBuf;
  seen.fill(0, 0, n);
  if (stackBuf.length < n) stackBuf = new Int32Array(n);
  const stack = stackBuf;
  const maxH = Math.max(24, Math.round(0.08 * Math.min(rows, cols)));
  const byValue = new Map();
  let count = 0;
  for (let y = 0; y < rows; y++) {
    const full = y < bandH || y >= rows - bandH;
    const row = y * cols;
    for (let x = 0; x < cols; x++) {
      if (!full && x >= bandW && x < cols - bandW) { x = cols - bandW - 1; continue; }
      const s = row + x;
      if (seen[s]) continue;
      const v = raw[s];
      const shape = floodShape(raw, seen, stack, s, rows, cols);
      if (!isLetterShape(shape, maxH)) continue;
      const votes = ringVotes(raw, stack, shape.area, v, rows, cols, gap, sign);
      if (votes.outside === 0 || (votes.dark < 0.75 * votes.outside && votes.light < 0.75 * votes.outside)) continue;
      let list = byValue.get(v);
      if (!list) byValue.set(v, (list = []));
      list.push(shape);
      if (++count > 2000) return null;
    }
  }
  return byValue;
}

// Floods the equal-valued shape that starts at pixel s, marking every pixel seen. The shape's pixels
// are left in stack[0 .. area), so every pixel is visited once.
function floodShape(raw, seen, stack, s, rows, cols) {
  const v = raw[s];
  const y = (s / cols) | 0;
  const x = s - y * cols;
  let top = 0, head = 0;
  stack[top++] = s;
  seen[s] = 1;
  let x0 = x, x1 = x, y0 = y, y1 = y;
  while (head < top) {
    const q = stack[head++];
    const qy = (q / cols) | 0;
    const qx = q - qy * cols;
    if (qx < x0) x0 = qx;
    if (qx > x1) x1 = qx;
    if (qy < y0) y0 = qy;
    if (qy > y1) y1 = qy;
    const ya = qy > 0 ? qy - 1 : 0;
    const yb = qy < rows - 1 ? qy + 1 : rows - 1;
    const xa = qx > 0 ? qx - 1 : 0;
    const xb = qx < cols - 1 ? qx + 1 : cols - 1;
    for (let yy = ya; yy <= yb; yy++) {
      for (let xx = xa, r = yy * cols + xa; xx <= xb; xx++, r++) {
        if (!seen[r] && raw[r] === v) { seen[r] = 1; stack[top++] = r; }
      }
    }
  }
  return { area: top, x0, x1, y0, y1, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

// A letter is a shape of a sensible size that is not a solid block.
function isLetterShape({ area, w, h }, maxH) {
  const small = Math.max(w, h) >= 5 && Math.min(w, h) >= 3 && w <= maxH * 1.5 && h <= maxH * 1.5 && area >= 4;
  return small && !(w >= 4 && h >= 4 && area / (w * h) > 0.9);
}

// The ring: the 4-neighbours just outside the shape must mostly be clearly darker or lighter.
function ringVotes(raw, stack, area, v, rows, cols, gap, sign) {
  const votes = { outside: 0, dark: 0, light: 0 };
  for (let i = 0; i < area; i++) {
    const q = stack[i];
    const qy = (q / cols) | 0;
    const qx = q - qy * cols;
    if (qx > 0) countEdge(votes, v, raw[q - 1], gap, sign);
    if (qx < cols - 1) countEdge(votes, v, raw[q + 1], gap, sign);
    if (qy > 0) countEdge(votes, v, raw[q - cols], gap, sign);
    if (qy < rows - 1) countEdge(votes, v, raw[q + cols], gap, sign);
  }
  return votes;
}

// Counts one neighbour of the shape: a neighbour of another value is an outside edge pixel.
function countEdge(votes, v, u, gap, sign) {
  if (u === v) return;
  votes.outside++;
  const d = sign * (v - u);
  if (d >= gap) votes.dark++;
  else if (d <= -gap) votes.light++;
}

// ---- Hashes ----------------------------------------------------------------------------------

function hex(buf) {
  const b = new Uint8Array(buf);
  let s = '';
  for (let i = 0; i < b.length; i++) s += b[i].toString(16).padStart(2, '0');
  return s;
}

export async function fingerprint(pixelBytes) {
  return hex(await crypto.subtle.digest('SHA-256', pixelBytes));
}

export async function seriesCode(sha256s) {
  const joined = sha256s.slice().sort().join('\n');
  return hex(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(joined))).slice(0, 16);
}

// ---- Judging a series ------------------------------------------------------------------------

function tick() {
  return new Promise((r) => (typeof setTimeout === 'function' ? setTimeout(r, 0) : r()));
}

function fieldsOf(ds) {
  const e = ds.elements;
  return [text(e, '00180015'), text(e, '00081030'), text(e, '0008103E'), text(e, '00181030')];
}

function refuseFile(f) {
  const { ds, d } = f;
  const e = ds.elements;
  if (!ALLOWED_SOP.has(d.sopClassUid)) {
    return `Pictures of ${sopKind(d.sopClassUid)} type are not accepted. Only CT, MR and X-ray (CR or DX) images are.`;
  }
  if (d.numberOfFrames > 1) return 'Multi-frame (enhanced) images are not accepted yet.';
  if (d.samplesPerPixel !== 1 || (d.photometric !== 'MONOCHROME1' && d.photometric !== 'MONOCHROME2')) {
    return 'Colour pictures are not accepted. Only greyscale scans are.';
  }
  const itype = text(e, '00080008').toUpperCase();
  const desc = fieldsOf(ds).join(' ').toUpperCase();
  if (itype.includes('LOCALIZER') || /LOCALI[SZ]ER|SCOUT|TOPOGRAM/.test(desc)) {
    return 'Scout or localiser pictures are not accepted.';
  }
  if (text(e, '00280301').toUpperCase() === 'YES') {
    return 'The file says the picture has text printed into it (burned-in annotation).';
  }
  const bits = d.bitsAllocated;
  const signed = d.pixelRepresentation === 1;
  const hb = d.highBit === null ? d.bitsStored - 1 : d.highBit;
  if (!(bits === 8 || bits === 16) || (signed && bits !== 16) || hb < 0 || hb >= bits || d.bitsStored > bits) {
    return 'The pixel format of these pictures is not one this page can rewrite.';
  }
  if (!d.rows || !d.cols || d.rows > 16384 || d.cols > 16384) return 'The picture size is not usable.';
  return null;
}

async function judgeSeries(series, now, onProgress) {
  const out = series;
  out.accepted = false;
  out.reason = null;
  const no = (why) => { out.reason = why; return out; };
  for (const f of series.files) {
    const why = refuseFile(f);
    if (why) return no(why);
  }
  // Body part, from all four texts of every file.
  const body = bodyPartOf(series.files);
  if (body.refuse) return no(`This series looks like a head, neck or whole-body scan (${body.refuse}). Those are never taken, so no face is sent.`);
  if (!body.word) return no('The body part of this series could not be told from the file, so it is not sent. Only chest, abdomen, pelvis, lower spine and limb scans are taken.');
  out.bodyPart = body.word;
  // Age: proven for every picture, or the series is refused.
  const age = ageOf(series.files, now);
  if (age.why) return no(age.why);
  out.ageYears = Math.min(age.atStudy, 90);
  // The text finder, on every picture.
  const textWhy = await textRefusal(series.files, onProgress);
  if (textWhy) return no(textWhy);
  out.accepted = true;
  return out;
}

// The series' body part from the distinct texts of its files: a refusal word, or the first allowed word.
function bodyPartOf(files) {
  const seen = new Set();
  let word = null;
  let first = true;
  for (const f of files) {
    const fields = fieldsOf(f.ds);
    const key = fields.join('|');
    if (seen.has(key)) continue;
    seen.add(key);
    const r = matchBodyPart(fields);
    if (r.refuse) return { refuse: r.refuse };
    if (r.word && first) word = r.word;
    if (r.word) first = false;
  }
  return { word };
}

// The youngest age and the youngest age at study across the files, or the reason it is refused.
function ageOf(files, now) {
  let years = Infinity;
  let atStudy = Infinity;
  for (const f of files) {
    const proof = readAgeProof(f.bytes, now);
    if (!proof.ok) return { why: `Your age could not be proven from the file (${proof.why}), so it is not sent.` };
    years = Math.min(years, proof.years);
    atStudy = Math.min(atStudy, proof.atStudy);
  }
  if (years < 18) return { why: 'The scan is of someone who is under 18 today. Only adults donate.' };
  if (years > 110) return { why: 'The age in the file does not look right (over 110 years), so it is not sent.' };
  return { atStudy };
}

// The reason a picture in the series has printed text (or is not an accepted picture), or null.
async function textRefusal(files, onProgress) {
  let i = 0;
  for (const f of files) {
    i++;
    const frame = await decodeFrame(f.ds, 0);
    if (frame.unsupported) return `These pictures are stored in a format this page cannot open (${frame.unsupported}).`;
    if (frame.samples !== 1) return 'Colour pictures are not accepted. Only greyscale scans are.';
    const raw = frame.raw || Float32Array.from(frame.values);
    if (findText(raw, frame.height, frame.width, frame.invert)) {
      return `Printed text was found in a picture (number ${i} of ${files.length}), so this series is not sent.`;
    }
    if (onProgress && i % 8 === 0) { onProgress(i, files.length); await tick(); }
  }
  return null;
}

export async function judgeFiles(files, { now = new Date(), onProgress } = {}) {
  const groups = new Map();
  let ignored = 0;
  for (let k = 0; k < files.length; k++) {
    const { name = String(k), bytes } = files[k];
    let ds = null;
    try { ds = readDataset(bytes); } catch { ds = null; }
    if (!ds) { ignored++; continue; }
    const d = describe(ds);
    const id = d.studyUid + '|' + d.seriesUid;
    if (!groups.has(id)) groups.set(id, []);
    groups.get(id).push({ name, bytes, ds, d, order: k });
  }
  const list = [];
  for (const [id, fs] of groups) {
    fs.sort((p, q) => ((p.d.instanceNumber ?? 1e12) - (q.d.instanceNumber ?? 1e12)) || p.order - q.order);
    const d = fs[0].d;
    list.push({
      id, files: fs, count: fs.length, accepted: false, reason: null,
      modality: ALLOWED_SOP.get(d.sopClassUid) === 'CR' || ALLOWED_SOP.get(d.sopClassUid) === 'DX'
        ? (d.modality === 'CR' || d.modality === 'DX' ? d.modality : ALLOWED_SOP.get(d.sopClassUid))
        : (ALLOWED_SOP.get(d.sopClassUid) || d.modality),
      bodyPart: null, ageYears: null,
      description: d.seriesDescription,
    });
  }
  for (const s of list) await judgeSeries(s, now, onProgress);
  return { series: list, ignored };
}

// ---- Writing the new files -------------------------------------------------------------------

export function newUidMap() {
  return { map: new Map() };
}

function randomUid() {
  const b = crypto.getRandomValues(new Uint8Array(16));
  let n = 0n;
  for (const x of b) n = (n << 8n) | BigInt(x);
  return '2.25.' + n.toString();
}

function mapUid(uidMap, original) {
  if (!original) return randomUid();
  let u = uidMap.map.get(original);
  if (!u) { u = randomUid(); uidMap.map.set(original, u); }
  return u;
}

function ascii(s, pad) {
  let n = s.length;
  const odd = n % 2;
  const out = new Uint8Array(n + odd);
  for (let i = 0; i < n; i++) out[i] = s.charCodeAt(i) & 0x7f;
  if (odd) out[n] = pad;
  return out;
}

// Every element is written with its real VR (the output is Explicit VR Little Endian), whether the
// source file was explicit or implicit. A 'UN' here would mean a kept tag missing from the
// dictionary in dicom.js, so it stops the write rather than put an untyped value in a donation.
function element(group, elem, vr, value) {
  if (!/^[A-Z]{2}$/.test(vr) || vr === 'UN') throw new Error('no real VR for tag ' + group.toString(16) + ',' + elem.toString(16));
  const long = LONG.has(vr);
  const head = long ? 12 : 8;
  const out = new Uint8Array(head + value.length);
  const dv = new DataView(out.buffer);
  dv.setUint16(0, group, true);
  dv.setUint16(2, elem, true);
  out[4] = vr.charCodeAt(0);
  out[5] = vr.charCodeAt(1);
  if (long) dv.setUint32(8, value.length, true);
  else dv.setUint16(6, value.length, true);
  out.set(value, head);
  return out;
}

function u16(v, signed) {
  const out = new Uint8Array(2);
  new DataView(out.buffer)[signed ? 'setInt16' : 'setUint16'](0, v, true);
  return out;
}

// Each kept tag: [key, VR, name]. The new UIDs, BodyPartExamined, PatientAge and the pixels are
// handled by name below; everything else is read by VR and re-encoded.
const KEPT = [
  ['00080008', 'CS', 'ImageType'], ['00080016', 'UI', 'SOPClassUID'], ['00080060', 'CS', 'Modality'],
  ['00080070', 'LO', 'Manufacturer'], ['00081090', 'LO', 'ManufacturerModelName'],
  ['00100040', 'CS', 'PatientSex'], ['00180020', 'CS', 'ScanningSequence'],
  ['00180021', 'CS', 'SequenceVariant'], ['00180023', 'CS', 'MRAcquisitionType'],
  ['00180050', 'DS', 'SliceThickness'], ['00180060', 'DS', 'KVP'], ['00180080', 'DS', 'RepetitionTime'],
  ['00180081', 'DS', 'EchoTime'], ['00180087', 'DS', 'MagneticFieldStrength'],
  ['00180088', 'DS', 'SpacingBetweenSlices'], ['00180091', 'IS', 'EchoTrainLength'],
  ['00181164', 'DS', 'ImagerPixelSpacing'], ['00181210', 'SH', 'ConvolutionKernel'],
  ['00181314', 'DS', 'FlipAngle'], ['00185101', 'CS', 'ViewPosition'],
  ['00200013', 'IS', 'InstanceNumber'], ['00200020', 'CS', 'PatientOrientation'],
  ['00200032', 'DS', 'ImagePositionPatient'], ['00200037', 'DS', 'ImageOrientationPatient'],
  ['00280002', 'US', 'SamplesPerPixel'], ['00280004', 'CS', 'PhotometricInterpretation'],
  ['00280030', 'DS', 'PixelSpacing'], ['00281040', 'CS', 'PixelIntensityRelationship'],
  ['00281041', 'SS', 'PixelIntensityRelationshipSign'], ['00281050', 'DS', 'WindowCenter'],
  ['00281051', 'DS', 'WindowWidth'], ['00281052', 'DS', 'RescaleIntercept'],
  ['00281053', 'DS', 'RescaleSlope'], ['00281054', 'LO', 'RescaleType'],
  ['00282110', 'CS', 'LossyImageCompression'],
];
// The reader's implicit-VR dictionary must agree with every VR written here, so a kept tag never
// reaches this file as 'UN'.
for (const [k, vr] of KEPT) {
  if (IMPLICIT_VR[k] !== vr) throw new Error('dicom.js IMPLICIT_VR disagrees with KEPT for ' + k);
}
const NAMES = new Map(KEPT.map(([k, , n]) => [k, n]));
for (const [k, n] of [['0008' + '0018', 'SOPInstanceUID'], ['0020000D', 'StudyInstanceUID'],
  ['0020000E', 'SeriesInstanceUID'], ['00200052', 'FrameOfReferenceUID'], ['00180015', 'BodyPartExamined'],
  ['00101010', 'PatientAge'], ['00280010', 'Rows'], ['00280011', 'Columns'], ['00280100', 'BitsAllocated'],
  ['00280101', 'BitsStored'], ['00280102', 'HighBit'], ['00280103', 'PixelRepresentation'],
  ['00280120', 'PixelPaddingValue'], ['7FE00010', 'PixelData']]) NAMES.set(k, n);

function keyParts(key) {
  return [parseInt(key.slice(0, 4), 16), parseInt(key.slice(4), 16)];
}

function readText(ds, key) {
  const t = text(ds.elements, key).replace(/[^\x20-\x7e]/g, '');
  return t;
}

function readInt(ds, key, signed) {
  const e = ds.elements.get(key);
  if (!e || e.bytes.length < 2) return null;
  const dv = new DataView(e.bytes.buffer, e.bytes.byteOffset, e.bytes.byteLength);
  return signed ? dv.getInt16(0, ds.little) : dv.getUint16(0, ds.little);
}

const HOST_LITTLE = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

async function writeOne(file, series, uidMap, seriesIndex, imageIndex) {
  const { ds, d } = file;
  const frame = await decodeFrame(ds, 0);
  if (frame.unsupported) throw new Error('cannot decode: ' + frame.unsupported);
  if (frame.samples !== 1) throw new Error('not greyscale');
  const layout = pixelLayout(d, frame);
  const kept = keptElements(ds, d, series, uidMap, layout, !!frame.raw);
  const { out, headLen } = assembleFile(d, uidMap, kept, layout);
  fillPixels(out, headLen, layout);
  const pixelBytes = out.subarray(headLen);
  const preview = kept.map(([key]) => key).filter((k) => k !== '7FE00010');
  return {
    name: `s${seriesIndex}/i${imageIndex}.dcm`, bytes: out, rows: layout.rows, cols: layout.cols,
    sha256: await fingerprint(pixelBytes), keys: [...preview, '7FE00010'],
    values: kept, ds,
  };
}

// How the pixel bytes are laid out: the bit depth and highest bit written, the mask that keeps the
// stored bits, the shift that sign-extends a signed value, and an empty pixel buffer of the right size.
function pixelLayout(d, frame) {
  const rows = frame.height;
  const cols = frame.width;
  const n = rows * cols;
  const signed = d.pixelRepresentation === 1 && !!frame.raw;
  let bitsAllocated = d.bitsAllocated > 8 ? 16 : 8;
  let bitsStored = d.bitsStored;
  let highBit = d.highBit === null ? d.bitsStored - 1 : d.highBit;
  const src = frame.raw || Uint8Array.from(frame.values, (v) => Math.max(0, Math.min(255, Math.round(v))));
  if (!frame.raw) { bitsAllocated = 8; bitsStored = 8; highBit = 7; }
  let shift = 0;
  let mask = 0xffff;
  let pixelVr;
  let pixel;
  if (signed) {
    shift = 16 - (highBit + 1);
    bitsAllocated = 16; bitsStored = 16;
    highBit = 15;
    pixelVr = 'OW';
    pixel = new Uint8Array(n * 2);
  } else {
    mask = highBit >= 15 ? 0xffff : (1 << (highBit + 1)) - 1;
    pixelVr = bitsAllocated === 8 ? 'OB' : 'OW';
    pixel = new Uint8Array(n * (bitsAllocated === 8 ? 1 : 2) + ((n * (bitsAllocated === 8 ? 1 : 2)) % 2));
  }
  return { rows, cols, n, signed, bitsAllocated, bitsStored, highBit, src, shift, mask, pixelVr, pixel };
}

function addElement(kept, key, vr, value) {
  const [g, e] = keyParts(key);
  kept.push([key, element(g, e, vr, value)]);
}

// The source's elements that are kept as they are (the KEPT list, minus the ones written below).
function copiedElements(ds) {
  const kept = [];
  for (const [key, vr] of KEPT) {
    if (key === '00080016') continue;
    if (vr === 'UI' || !ds.elements.has(key)) continue;
    if (vr === 'US' || vr === 'SS') {
      if (key === '00280002') continue;       // written below, always 1
      const v = readInt(ds, key, vr === 'SS');
      if (v !== null) addElement(kept, key, vr, u16(v, vr === 'SS'));
    } else {
      const t = readText(ds, key);
      if (t) addElement(kept, key, vr, ascii(t, 0x20));
    }
  }
  return kept;
}

// Sex is kept only when it is M, F or O; a kept sex element is rewritten with the cleaned value.
function fixSex(kept, ds) {
  const sex = readText(ds, '00100040').toUpperCase();
  const sexKeep = kept.findIndex(([k]) => k === '00100040');
  if (sexKeep >= 0 && !['M', 'F', 'O'].includes(sex)) kept.splice(sexKeep, 1);
  else if (sexKeep >= 0) kept[sexKeep] = [kept[sexKeep][0], element(0x0010, 0x0040, 'CS', ascii(sex, 0x20))];
}

// The identifiers, mapped to this export's UIDs, and the body part and age the series was accepted with.
function addIdentity(kept, ds, d, series, uidMap) {
  addElement(kept, '00080018', 'UI', ascii(mapUid(uidMap, d.sopUid), 0));
  addElement(kept, '0020000D', 'UI', ascii(mapUid(uidMap, d.studyUid), 0));
  addElement(kept, '0020000E', 'UI', ascii(mapUid(uidMap, d.seriesUid), 0));
  const frameOfRef = readText(ds, '00200052');
  if (frameOfRef) addElement(kept, '00200052', 'UI', ascii(mapUid(uidMap, frameOfRef), 0));
  if (series.bodyPart) addElement(kept, '00180015', 'CS', ascii(series.bodyPart, 0x20));
  if (series.ageYears !== null && series.ageYears !== undefined) {
    addElement(kept, '00101010', 'AS', ascii(String(Math.min(90, series.ageYears)).padStart(3, '0') + 'Y', 0x20));
  }
}

// The picture's size and bit layout.
function addGeometry(kept, layout) {
  addElement(kept, '00280010', 'US', u16(layout.rows, false));
  addElement(kept, '00280011', 'US', u16(layout.cols, false));
  addElement(kept, '00280100', 'US', u16(layout.bitsAllocated, false));
  addElement(kept, '00280101', 'US', u16(layout.bitsStored, false));
  addElement(kept, '00280102', 'US', u16(layout.highBit, false));
  addElement(kept, '00280103', 'US', u16(layout.signed ? 1 : 0, false));
}

// The pixel padding value, when the source has one and its pixels were decoded.
function addPadding(kept, ds, d, layout, hasRaw) {
  const padRaw = readInt(ds, '00280120', d.pixelRepresentation === 1);
  if (padRaw !== null && hasRaw) {
    const pv = layout.signed ? (padRaw << layout.shift) >> layout.shift : (padRaw & layout.mask);
    addElement(kept, '00280120', layout.signed ? 'SS' : 'US', u16(pv, layout.signed));
  }
}

function keptElements(ds, d, series, uidMap, layout, hasRaw) {
  const kept = copiedElements(ds);
  addElement(kept, '00080016', 'UI', ascii(d.sopClassUid, 0));
  addElement(kept, '00280002', 'US', u16(1, false));
  fixSex(kept, ds);
  addIdentity(kept, ds, d, series, uidMap);
  addGeometry(kept, layout);
  addPadding(kept, ds, d, layout, hasRaw);
  kept.sort((p, q) => (p[0] < q[0] ? -1 : 1));
  return kept;
}

// The file meta group (transfer syntax and the implementation details). Its own UID is mapped here.
function metaElements(d, uidMap) {
  const metaBody = [
    element(0x0002, 0x0001, 'OB', new Uint8Array([0, 1])),
    element(0x0002, 0x0002, 'UI', ascii(d.sopClassUid, 0)),
  ];
  const imageUid = mapUid(uidMap, d.sopUid);
  metaBody.push(element(0x0002, 0x0003, 'UI', ascii(imageUid, 0)));
  metaBody.push(element(0x0002, 0x0010, 'UI', ascii(EXPLICIT_LE, 0)));
  metaBody.push(element(0x0002, 0x0012, 'UI', ascii(IMPLEMENTATION_UID, 0)));
  metaBody.push(element(0x0002, 0x0013, 'SH', ascii(IMPLEMENTATION_NAME, 0x20)));
  return metaBody;
}

// The preamble, the file meta group, the kept elements and the pixel data header, joined into one
// buffer with room for the pixels after them. Returns the buffer and where the pixels start.
function assembleFile(d, uidMap, kept, layout) {
  const metaBody = metaElements(d, uidMap);
  const metaLen = metaBody.reduce((s, b) => s + b.length, 0);
  const groupLen = element(0x0002, 0x0000, 'UL', new Uint8Array(new Uint32Array([metaLen]).buffer));
  if (!HOST_LITTLE) new DataView(groupLen.buffer).setUint32(8, metaLen, true);
  const parts = [new Uint8Array(128), ascii('DICM', 0), groupLen, ...metaBody];
  // Elements below 7FE0 come first; the pixel data header goes last.
  for (const [, bytes] of kept) parts.push(bytes);
  const pixelHeader = element(0x7fe0, 0x0010, layout.pixelVr, new Uint8Array(0));
  new DataView(pixelHeader.buffer).setUint32(8, layout.pixel.length, true);
  parts.push(pixelHeader);
  const headLen = parts.reduce((s, b) => s + b.length, 0);
  const out = new Uint8Array(headLen + layout.pixel.length);
  let o = 0;
  for (const b of parts) { out.set(b, o); o += b.length; }
  return { out, headLen };
}

// Pixel values: the decoded stored numbers, with the spare bits (or the sign) made clean.
function fillPixels(out, headLen, layout) {
  const { n, src, bitsAllocated, signed, shift, mask } = layout;
  if (bitsAllocated === 8) {
    for (let i = 0; i < n; i++) out[headLen + i] = src[i] & mask;
  } else if (HOST_LITTLE && headLen % 2 === 0) {
    if (signed) {
      const dst = new Int16Array(out.buffer, headLen, n);
      for (let i = 0; i < n; i++) dst[i] = (src[i] << shift) >> shift;
    } else {
      const dst = new Uint16Array(out.buffer, headLen, n);
      for (let i = 0; i < n; i++) dst[i] = src[i] & mask;
    }
  } else {
    const dv = new DataView(out.buffer);
    for (let i = 0; i < n; i++) {
      const v = signed ? (src[i] << shift) >> shift : (src[i] & mask);
      if (signed) dv.setInt16(headLen + i * 2, v, true);
      else dv.setUint16(headLen + i * 2, v, true);
    }
  }
}

function readBackValue(bytes, key) {
  // Walk the file we just wrote (Explicit VR Little Endian, no sequences) to show a value.
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let p = 132;
  while (p + 8 <= bytes.length) {
    const g = dv.getUint16(p, true);
    const e = dv.getUint16(p + 2, true);
    const vr = String.fromCharCode(bytes[p + 4], bytes[p + 5]);
    let len;
    let head;
    if (LONG.has(vr)) { len = dv.getUint32(p + 8, true); head = 12; } else { len = dv.getUint16(p + 6, true); head = 8; }
    const k = g.toString(16).padStart(4, '0').toUpperCase() + e.toString(16).padStart(4, '0').toUpperCase();
    if (k === key) {
      if (vr === 'US') return String(dv.getUint16(p + head, true));
      if (vr === 'SS') return String(dv.getInt16(p + head, true));
      return a(bytes, p + head, len).replace(/\0+$/, '').trim();
    }
    if (g === 0x7fe0) return '';
    p += head + len;
  }
  return '';
}

export async function deidentifySeries(series, { uidMap = newUidMap(), seriesIndex = 0, onProgress, force = false } = {}) {
  if (!series.accepted && !force) throw new Error('this series was not accepted');
  const images = [];
  const tagSet = new Set();
  for (let i = 0; i < series.files.length; i++) {
    const w = await writeOne(series.files[i], series, uidMap, seriesIndex, i);
    for (const k of w.keys) tagSet.add(k);
    images.push({ name: w.name, bytes: w.bytes, sha256: w.sha256, rows: w.rows, cols: w.cols });
    if (onProgress && i % 8 === 7) { onProgress(i + 1, series.files.length); await tick(); }
  }
  const tags = [...tagSet].sort();
  const preview = images.length
    ? tags.filter((k) => k !== '7FE00010').map((k) => ({ key: k, name: NAMES.get(k) || k, value: readBackValue(images[0].bytes, k) }))
    : [];
  return {
    index: seriesIndex, code: await seriesCode(images.map((im) => im.sha256)),
    modality: series.modality, bodyPart: series.bodyPart, ageYears: series.ageYears,
    tags, preview, images,
  };
}

export function buildManifest(list) {
  return {
    terms: TERMS_VERSION,
    series: list.map((s) => ({
      code: s.code, modality: s.modality, bodyPart: s.bodyPart,
      images: s.images.map((im) => ({ name: im.name, sha256: im.sha256 })),
      tags: s.tags,
    })),
  };
}

// Measures findText (site/js/open/deid.js) against the bar in docs/DONATED_SCANS.md, "The text finder".
//   node tools/textfinder_eval.mjs            full run (exit 0 only when the bar is met)
//   node tools/textfinder_eval.mjs --quick    fewer scans, for iterating (never exits 0)
//   node tools/textfinder_eval.mjs --clean-only | --text-only
//
// Reads ONLY LIDC-IDRI-0001..0040, the ten LUNGx CT-Training-* series and NIH ChestX-ray14 pictures
// named in train_val_list.txt. The held-out LIDC-IDRI-05xx patients and the MR disc are never opened.
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { readDataset, decodeFrame } from '../site/js/open/dicom.js';
import { findText } from '../site/js/open/deid.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.resolve(here, '../../Datasets/public');
const args = new Set(process.argv.slice(2));
const QUICK = args.has('--quick');

// ---- The synthetic text generator (written before tuning; nothing below the finder changes it) ----

// A 5 x 7 scanner-style bitmap font. Each glyph is seven rows of five characters.
const FONT = {
  A: ['01110', '10001', '10001', '11111', '10001', '10001', '10001'],
  B: ['11110', '10001', '10001', '11110', '10001', '10001', '11110'],
  C: ['01110', '10001', '10000', '10000', '10000', '10001', '01110'],
  D: ['11110', '10001', '10001', '10001', '10001', '10001', '11110'],
  E: ['11111', '10000', '10000', '11110', '10000', '10000', '11111'],
  F: ['11111', '10000', '10000', '11110', '10000', '10000', '10000'],
  G: ['01110', '10001', '10000', '10111', '10001', '10001', '01111'],
  H: ['10001', '10001', '10001', '11111', '10001', '10001', '10001'],
  I: ['01110', '00100', '00100', '00100', '00100', '00100', '01110'],
  J: ['00111', '00010', '00010', '00010', '00010', '10010', '01100'],
  K: ['10001', '10010', '10100', '11000', '10100', '10010', '10001'],
  L: ['10000', '10000', '10000', '10000', '10000', '10000', '11111'],
  M: ['10001', '11011', '10101', '10101', '10001', '10001', '10001'],
  N: ['10001', '11001', '10101', '10011', '10001', '10001', '10001'],
  O: ['01110', '10001', '10001', '10001', '10001', '10001', '01110'],
  P: ['11110', '10001', '10001', '11110', '10000', '10000', '10000'],
  Q: ['01110', '10001', '10001', '10001', '10101', '10010', '01101'],
  R: ['11110', '10001', '10001', '11110', '10100', '10010', '10001'],
  S: ['01111', '10000', '10000', '01110', '00001', '00001', '11110'],
  T: ['11111', '00100', '00100', '00100', '00100', '00100', '00100'],
  U: ['10001', '10001', '10001', '10001', '10001', '10001', '01110'],
  V: ['10001', '10001', '10001', '10001', '10001', '01010', '00100'],
  W: ['10001', '10001', '10001', '10101', '10101', '11011', '10001'],
  X: ['10001', '10001', '01010', '00100', '01010', '10001', '10001'],
  Y: ['10001', '10001', '01010', '00100', '00100', '00100', '00100'],
  Z: ['11111', '00001', '00010', '00100', '01000', '10000', '11111'],
  0: ['01110', '10001', '10011', '10101', '11001', '10001', '01110'],
  1: ['00100', '01100', '00100', '00100', '00100', '00100', '01110'],
  2: ['01110', '10001', '00001', '00010', '00100', '01000', '11111'],
  3: ['11110', '00001', '00001', '01110', '00001', '00001', '11110'],
  4: ['00010', '00110', '01010', '10010', '11111', '00010', '00010'],
  5: ['11111', '10000', '11110', '00001', '00001', '10001', '01110'],
  6: ['00110', '01000', '10000', '11110', '10001', '10001', '01110'],
  7: ['11111', '00001', '00010', '00100', '01000', '01000', '01000'],
  8: ['01110', '10001', '10001', '01110', '10001', '10001', '01110'],
  9: ['01110', '10001', '10001', '01111', '00001', '00010', '01100'],
  '.': ['00000', '00000', '00000', '00000', '00000', '01100', '01100'],
  '/': ['00001', '00001', '00010', '00100', '01000', '10000', '10000'],
  '-': ['00000', '00000', '00000', '11111', '00000', '00000', '00000'],
  ':': ['00000', '01100', '01100', '00000', '01100', '01100', '00000'],
  ' ': ['00000', '00000', '00000', '00000', '00000', '00000', '00000'],
};

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SURNAMES = ['SMITH', 'JOHNSON', 'WILLIAMS', 'BROWN', 'JONES', 'GARCIA', 'MILLER', 'DAVIS', 'MARTINEZ', 'ANDERSON', 'TAYLOR', 'MOORE', 'JACKSON', 'LEE', 'WHITE', 'HARRIS', 'KOWALSKI', 'NGUYEN', 'MUELLER', 'ROSSI', 'TANAKA', 'OKAFOR', 'HOFFMANN', 'SVENSSON'];
const GIVEN = ['JOHN', 'MARY', 'ANNA', 'PETER', 'LINDA', 'DAVID', 'SUSAN', 'KARL', 'MEI', 'OMAR', 'EVA', 'LUIS', 'IVAN', 'SARA', 'JAMES'];

function makeText(rnd) {
  const pick = (a) => a[Math.floor(rnd() * a.length)];
  const num = (d) => { let s = ''; for (let i = 0; i < d; i++) s += Math.floor(rnd() * 10); return s; };
  const yr = 1990 + Math.floor(rnd() * 36);
  const mo = String(1 + Math.floor(rnd() * 12)).padStart(2, '0');
  const dy = String(1 + Math.floor(rnd() * 28)).padStart(2, '0');
  let s;
  switch (Math.floor(rnd() * 8)) {
    case 0: s = pick(SURNAMES); break;
    case 1: s = `${pick(SURNAMES)} ${pick(GIVEN)}`; break;
    case 2: s = `${yr}-${mo}-${dy}`; break;
    case 3: s = `${mo}/${dy}/${yr}`; break;
    case 4: s = `ID ${num(4 + Math.floor(rnd() * 6))}`; break;
    case 5: s = `MRN ${num(6 + Math.floor(rnd() * 4))}`; break;
    case 6: s = `DOB ${dy}.${mo}.${yr}`; break;
    default: s = `${pick(SURNAMES)} ${num(3 + Math.floor(rnd() * 5))}`; break;
  }
  s = s.slice(0, 30);
  return s;
}

const alnum = (s) => (s.match(/[A-Z0-9]/g) || []).length;

// Renders a string to a boolean bitmap w x h (nearest-neighbour scale of the 5 x 7 font).
function renderText(s, h) {
  const gw = Math.max(1, Math.round((5 * h) / 7));
  const adv = gw + Math.max(1, Math.round(h / 7));
  const w = s.length * adv - (adv - gw);
  const bits = new Uint8Array(w * h);
  for (let c = 0; c < s.length; c++) {
    const g = FONT[s[c]] || FONT[' '];
    for (let y = 0; y < h; y++) {
      const row = g[Math.min(6, Math.floor((y * 7) / h))];
      for (let x = 0; x < gw; x++) {
        if (row[Math.min(4, Math.floor((x * 5) / gw))] === '1') bits[y * w + c * adv + x] = 1;
      }
    }
  }
  return { bits, w, h, adv };
}

function rotate90(r) {            // reads bottom to top
  const bits = new Uint8Array(r.w * r.h);
  for (let y = 0; y < r.h; y++) for (let x = 0; x < r.w; x++) if (r.bits[y * r.w + x]) bits[(r.w - 1 - x) * r.h + y] = 1;
  return { bits, w: r.h, h: r.w, adv: r.adv };
}

const PLACES = ['tl', 'tr', 'bl', 'br', 'top', 'bottom', 'left', 'right', 'leftV', 'rightV'];

// Draws one text line on a copy of the picture. Returns { raw, desc } or null when no line fits.
function drawText(pic, rnd, variant) {
  const { rows, cols, invert } = pic;
  const bandH = Math.max(8, Math.ceil(rows * 0.16));
  const bandW = Math.max(8, Math.ceil(cols * 0.16));
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < pic.raw.length; i++) { const v = pic.raw[i]; if (v < min) min = v; if (v > max) max = v; }
  const range = max - min;
  const place = PLACES[Math.floor(rnd() * PLACES.length)];
  const vertical = place === 'leftV' || place === 'rightV';
  const h = 7 + Math.floor(rnd() * 18);
  let s = makeText(rnd);
  while (alnum(s) < 4) s = makeText(rnd);
  let r = renderText(s, h);
  // The line must sit wholly inside the outer sixth: shorten it to fit, keeping at least four characters.
  const room = place === 'top' || place === 'bottom' || place === 'tl' || place === 'tr' || place === 'bl' || place === 'br' ? cols - 8 : place === 'left' || place === 'right' ? bandW - 6 : rows - 8;
  while (s.length > 4 && (vertical ? s.length * r.adv : r.w) > room) { s = s.slice(0, -1); r = renderText(s, h); }
  if (alnum(s) < 4) return null;
  if (vertical) r = rotate90(r);
  const thick = vertical ? r.w : r.h;
  const lenAlong = vertical ? r.h : r.w;
  if (vertical ? r.w > bandW - 6 || r.h > rows - 8 : r.h > bandH - 6 || r.w > (place === 'left' || place === 'right' ? bandW - 6 : cols - 8)) return null;
  const m = 3 + Math.floor(rnd() * 5);
  let x0, y0;
  switch (place) {
    case 'tl': x0 = m; y0 = m; break;
    case 'tr': x0 = cols - m - r.w; y0 = m; break;
    case 'bl': x0 = m; y0 = rows - m - r.h; break;
    case 'br': x0 = cols - m - r.w; y0 = rows - m - r.h; break;
    case 'top': x0 = Math.floor((cols - r.w) / 2); y0 = m; break;
    case 'bottom': x0 = Math.floor((cols - r.w) / 2); y0 = rows - m - r.h; break;
    case 'left': x0 = m; y0 = Math.floor(bandH + rnd() * Math.max(1, rows - 2 * bandH - r.h)); break;
    case 'right': x0 = cols - m - r.w; y0 = Math.floor(bandH + rnd() * Math.max(1, rows - 2 * bandH - r.h)); break;
    case 'leftV': x0 = m; y0 = Math.floor(rnd() * Math.max(1, rows - r.h - 2 * m)) + m; break;
    default: x0 = cols - m - r.w; y0 = Math.floor(rnd() * Math.max(1, rows - r.h - 2 * m)) + m; break;
  }
  if (x0 < 1 || y0 < 1 || x0 + r.w > cols - 1 || y0 + r.h > rows - 1) return null;
  // Variant 0 is drawn at the picture's maximum, variant 1 at 70% of its range (displayed brightness).
  const level = variant === 0 ? 1 : 0.7;
  const val = invert ? Math.round(max - level * range) : Math.round(min + level * range);
  const out = pic.raw.slice();
  let seen = 0, total = 0;
  for (let y = 0; y < r.h; y++) {
    for (let x = 0; x < r.w; x++) {
      if (!r.bits[y * r.w + x]) continue;
      const i = (y0 + y) * cols + x0 + x;
      total++;
      const was = out[i];
      if (Math.abs(val - was) >= 0.15 * range) seen++;
      out[i] = val;
    }
  }
  // A line nobody could see (drawn over a surface as bright as the ink) is not a test of the finder.
  if (total === 0 || seen < 0.6 * total) return { skipped: true };
  void thick; void lenAlong;
  return { raw: out, desc: `${place} h${h} "${s}" at ${level * 100}%` };
}

// ---- Backgrounds -----------------------------------------------------------------------------

function pngGrey(buf) {                 // first channel of an 8-bit PNG
  if (buf.readUInt32BE(0) !== 0x89504e47) throw new Error('not a PNG');
  let p = 8, w = 0, h = 0, depth = 8, ctype = 0;
  const idat = [];
  while (p < buf.length) {
    const len = buf.readUInt32BE(p);
    const type = buf.toString('latin1', p + 4, p + 8);
    const d = buf.subarray(p + 8, p + 8 + len);
    if (type === 'IHDR') { w = d.readUInt32BE(0); h = d.readUInt32BE(4); depth = d[8]; ctype = d[9]; }
    else if (type === 'IDAT') idat.push(d);
    else if (type === 'IEND') break;
    p += 12 + len;
  }
  if (depth !== 8 || ![0, 2, 4, 6].includes(ctype)) throw new Error('unsupported PNG');
  const bpp = { 0: 1, 2: 3, 4: 2, 6: 4 }[ctype];
  const data = zlib.inflateSync(Buffer.concat(idat));
  const stride = w * bpp;
  const cur = new Uint8Array(stride), prev = new Uint8Array(stride);
  const out = new Uint16Array(w * h);
  let q = 0;
  for (let y = 0; y < h; y++) {
    const f = data[q++];
    for (let i = 0; i < stride; i++) {
      const x = data[q++];
      const a = i >= bpp ? cur[i - bpp] : 0, b = prev[i], c = i >= bpp ? prev[i - bpp] : 0;
      let v;
      if (f === 0) v = x;
      else if (f === 1) v = x + a;
      else if (f === 2) v = x + b;
      else if (f === 3) v = x + ((a + b) >> 1);
      else { const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c); v = x + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c); }
      cur[i] = v & 255;
    }
    for (let i = 0; i < w; i++) out[y * w + i] = cur[i * bpp];
    prev.set(cur);
  }
  return { raw: out, rows: h, cols: w };
}

function gauss(rnd) { return Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd()); }

// An MR-like picture from a CT slice: a soft-tissue window, the body scaled to 12 bits, speckle noise
// everywhere (the air as well), and a dark noisy background.
function mrLike(pic, rnd) {
  const { raw, rows, cols } = pic;
  const sorted = Float64Array.from({ length: Math.ceil(raw.length / 16) }, (_, i) => raw[i * 16]).sort();
  const lo = sorted[Math.floor(sorted.length * 0.5)];
  const hi = sorted[Math.floor(sorted.length * 0.985)];
  const span = Math.max(1, hi - lo);
  const out = new Uint16Array(raw.length);
  const noise = 8 + rnd() * 30;
  for (let i = 0; i < raw.length; i++) {
    let v = Math.min(1, Math.max(0, (raw[i] - lo) / span)) * 900 + Math.abs(gauss(rnd)) * noise + (rnd() < 0.5 ? gauss(rnd) * noise * 0.5 : 0);
    out[i] = Math.max(0, Math.min(4095, Math.round(v)));
  }
  return { raw: out, rows, cols, invert: false };
}

function invertPic(pic) {
  let max = -Infinity, min = Infinity;
  for (let i = 0; i < pic.raw.length; i++) { const v = pic.raw[i]; if (v > max) max = v; if (v < min) min = v; }
  const out = new pic.raw.constructor(pic.raw.length);
  for (let i = 0; i < out.length; i++) out[i] = max + min - pic.raw[i];
  return { raw: out, rows: pic.rows, cols: pic.cols, invert: true };
}

// ---- Clean CT --------------------------------------------------------------------------------

function seriesDirs() {
  const dirs = [];
  const walk = (d, depth) => {
    let ents;
    try { ents = fs.readdirSync(d, { withFileTypes: true }); } catch { return; }
    const dcm = ents.filter((e) => e.isFile() && /\.dcm$/i.test(e.name));
    if (dcm.length) { dirs.push({ dir: d, files: dcm.map((e) => path.join(d, e.name)).sort() }); return; }
    if (depth < 4) for (const e of ents) if (e.isDirectory()) walk(path.join(d, e.name), depth + 1);
  };
  const nLidc = QUICK ? 6 : 40;
  for (let i = 1; i <= nLidc; i++) walk(path.join(DATA, 'lidc-idri', `LIDC-IDRI-${String(i).padStart(4, '0')}`), 0);
  const sp = path.join(DATA, 'spie-aapm');
  for (const e of fs.readdirSync(sp, { withFileTypes: true })) if (e.isDirectory() && /^CT-Training-/.test(e.name)) walk(path.join(sp, e.name), 0);
  return dirs;
}

const pool = [];        // CT bases for the text test (512 x 512 mostly)
const timing = [];      // 512 x 512 pictures for the speed test

async function cleanRun() {
  const series = seriesDirs();
  let pics = 0, fired = 0, refused = 0, skipped = 0;
  const bad = [];
  const poolEvery = QUICK ? 40 : 30;
  const t0 = Date.now();
  for (const s of series) {
    let seriesFires = 0, first = null;
    for (let k = 0; k < s.files.length; k++) {
      let fr;
      try { fr = await decodeFrame(readDataset(new Uint8Array(fs.readFileSync(s.files[k]))), 0); } catch { skipped++; continue; }
      if (!fr || !fr.raw || fr.samples > 1 || !fr.height) { skipped++; continue; }
      pics++;
      if (findText(fr.raw, fr.height, fr.width, !!fr.invert)) { fired++; seriesFires++; if (!first) first = path.basename(s.files[k]); }
      if ((pics % poolEvery) === 0 && pool.length < 450) pool.push({ raw: fr.raw.slice(), rows: fr.height, cols: fr.width, invert: !!fr.invert, signedOffset: 0 });
      if (fr.height === 512 && fr.width === 512 && (pics % 97) === 0 && timing.length < 120) timing.push({ raw: fr.raw, rows: 512, cols: 512, invert: !!fr.invert });
    }
    if (seriesFires) { refused++; bad.push(`${path.relative(DATA, s.dir).replace(/\\/g, '/').split('/').slice(-3).join('/')}: ${seriesFires} of ${s.files.length} pictures fire, first ${first}`); }
  }
  console.log(`CLEAN CT: ${series.length} series, ${pics} pictures read (${skipped} unreadable or colour, left out) in ${((Date.now() - t0) / 1000).toFixed(0)} s.`);
  console.log(`CLEAN CT: the finder fired on ${fired} of ${pics} pictures (${((100 * fired) / Math.max(1, pics)).toFixed(3)}%).`);
  console.log(`CLEAN CT: ${refused} of ${series.length} series would be refused.`);
  for (const b of bad.slice(0, 12)) console.log(`  refused: ${b}`);
  return { series: series.length, refused, pics, fired };
}

// ---- Text test -------------------------------------------------------------------------------

function nihBases(count) {
  const list = fs.readFileSync(path.join(DATA, 'nih-chestxray14/train_val_list.txt'), 'utf8').split(/\r?\n/).filter((l) => /\.png$/.test(l));
  const step = Math.max(1, Math.floor(list.length / count));
  const out = [];
  for (let i = 0; i < list.length && out.length < count; i += step) out.push(list[i]);
  return out;
}

function textRun() {
  const rnd = mulberry32(20261003);
  const stats = {};
  const misses = [];
  const bump = (name, hit, skip) => {
    const s = stats[name] || (stats[name] = { n: 0, hit: 0, skip: 0 });
    if (skip) s.skip++; else { s.n++; if (hit) s.hit++; }
  };
  const test = (name, pic, variant) => {
    let d = drawText(pic, rnd, variant);
    for (let t = 0; t < 4 && (!d || d.skipped); t++) d = drawText(pic, rnd, variant);
    if (!d || d.skipped) { bump(name, false, true); return; }
    const hit = findText(d.raw, pic.rows, pic.cols, pic.invert);
    bump(name, hit, false);
    if (!hit && misses.length < 15) misses.push(`${name} ${pic.invert ? '(inverted) ' : ''}${d.desc}`);
  };
  const nCt = pool.length;
  let nPng = 0;
  for (const b of pool) {
    for (let v = 0; v < 2; v++) test('CT', b, v);
    const mr = mrLike(b, rnd);
    for (let v = 0; v < 2; v++) test('MR-like (from CT)', mr, v);
    if ((nPng++ % 4) === 0) { const iv = invertPic(b); test('CT, MONOCHROME1', iv, 0); test('CT, MONOCHROME1', iv, 1); }
  }
  const names = nihBases(Math.max(200, nCt));
  let k = 0;
  for (const nm of names) {
    let pic;
    try { pic = pngGrey(fs.readFileSync(path.join(DATA, 'nih-chestxray14/images', nm))); } catch { continue; }
    pic.invert = false;
    const inv = (k++ % 4) === 3;
    const p = inv ? invertPic(pic) : pic;
    for (let v = 0; v < 2; v++) test(inv ? 'X-ray, MONOCHROME1' : 'X-ray', p, v);
  }
  let n = 0, hit = 0, skip = 0;
  for (const [name, s] of Object.entries(stats)) {
    n += s.n; hit += s.hit; skip += s.skip;
    console.log(`TEXT ${name}: caught ${s.hit} of ${s.n} lines (${((100 * s.hit) / Math.max(1, s.n)).toFixed(2)}%), ${s.skip} drawn too faint to see and left out.`);
  }
  console.log(`TEXT total: caught ${hit} of ${n} lines (${((100 * hit) / Math.max(1, n)).toFixed(2)}%); ${skip} faint ones left out.`);
  for (const m of misses) console.log(`  missed: ${m}`);
  return { n, hit };
}

// ---- Speed -----------------------------------------------------------------------------------

function speedRun() {
  if (!timing.length) { console.log('SPEED: no 512 x 512 pictures to time.'); return Infinity; }
  for (const p of timing.slice(0, 20)) findText(p.raw, p.rows, p.cols, p.invert);    // warm up
  const t0 = process.hrtime.bigint();
  let runs = 0;
  for (let rep = 0; rep < 3; rep++) for (const p of timing) { findText(p.raw, p.rows, p.cols, p.invert); runs++; }
  const ms = Number(process.hrtime.bigint() - t0) / 1e6 / runs;
  console.log(`SPEED: ${ms.toFixed(2)} ms per 512 x 512 picture (${runs} runs).`);
  return ms;
}

// ---- Run -------------------------------------------------------------------------------------

const clean = args.has('--text-only') ? null : await cleanRun();
if (args.has('--clean-only')) process.exit(0);
if (args.has('--text-only')) {
  // The text test needs the CT bases; take a few pictures from the first LUNGx and LIDC series.
  const series = seriesDirs();
  for (const s of series) {
    for (let k = 0; k < s.files.length && pool.length < 300; k += 40) {
      const fr = await decodeFrame(readDataset(new Uint8Array(fs.readFileSync(s.files[k]))), 0);
      if (fr && fr.raw && fr.samples <= 1) pool.push({ raw: fr.raw.slice(), rows: fr.height, cols: fr.width, invert: !!fr.invert });
      if (fr.height === 512 && timing.length < 100) timing.push({ raw: fr.raw, rows: 512, cols: 512, invert: !!fr.invert });
    }
  }
}
const text = textRun();
const ms = speedRun();

const refusedMax = clean ? Math.floor((clean.series * 2) / 50) : 0;
const okClean = !clean || clean.refused <= refusedMax;
const okText = text.n >= 2000 && text.hit / text.n >= 0.99;
const okSpeed = ms <= 10;
console.log('');
if (clean) console.log(`Clean scans: ${clean.refused} of ${clean.series} series refused; the limit is ${refusedMax} (2 in 50). ${okClean ? 'PASS' : 'FAIL'}`);
console.log(`Text: ${text.hit} of ${text.n} lines caught (${((100 * text.hit) / Math.max(1, text.n)).toFixed(2)}%); the bar is 99% on at least 2,000. ${okText ? 'PASS' : 'FAIL'}`);
console.log(`Speed: ${ms.toFixed(2)} ms per picture; the limit is 10. ${okSpeed ? 'PASS' : 'FAIL'}`);
process.exit(!QUICK && okClean && okText && okSpeed && clean ? 0 : 1);

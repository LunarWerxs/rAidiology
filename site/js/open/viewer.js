// "Open your own scan": take whatever the visitor dropped, find the DICOM series in it, and show
// them. Everything happens in this tab. There is no fetch, no upload and no third-party code here,
// which is the whole promise of the page. The sample scan is the one thing that comes over the
// network, and it lives in sample.js so that this file can keep saying that. The other is a donation,
// which the visitor starts and confirms, and which lives in donate.js: it talks only to the donation host.

import { listen } from '../lifetime.js';
import { fromDataTransfer, fromDirectoryHandle, fromFileList, expand } from './collect.js';
import { isDicom, readDataset, describe, decodeFrame, autoWindow, rescale } from './dicom.js';
import { scanOpened } from './donate.js';
import { parse as parseReading, attach as attachReading, looksLikeFindings, decodePath } from './findings.js';

// Decoded frames kept at once, by size rather than count: a 512x512 CT slice kept as its own 16-bit
// numbers is 0.5 MB, so this holds about 190 of them, and a colour frame kept as floats far fewer.
// At least a handful are always kept, so the slice on screen and the ones fetched ahead of it fit.
const MAX_CACHED_BYTES = 96 * 1024 * 1024;
const MIN_CACHED_FRAMES = 8;
// Slices decoded ahead of the visitor, in the direction they are moving, plus one behind.
const PREFETCH_AHEAD = 3;

const el = (id) => document.getElementById(id);
const state = {
  series: [],
  problems: [],
  current: null,
  frames: new Map(),                // "seriesIndex:slice" -> decoded frame
  loading: new Map(),               // "seriesIndex:slice" -> the decode in progress, so it runs once
  generation: 0,                    // bumped whenever the frames on hand stop belonging to what is open
  direction: 1,                     // which way the visitor last stepped through the slices
  slice: 0,
  centre: 0,
  width: 1,
  zoom: 1,
  panX: 0,
  panY: 0,
  cancelled: false,
  reading: null,                    // the parsed reading file, when one has been opened
  readingSeries: -1,                // which loaded series it turned out to be about
  outlines: new Map(),              // slice index -> [structure, path] pairs, for a labeller's reading
  pointedAt: -1,                    // the outlined structure under the pointer, or -1
  keyOpen: true,                    // whether the list of outlined parts is open (it starts open)
  chosen: -1,                       // the part whose row the visitor tapped, or -1
  view: null,                       // where the last picture was drawn: { x0, y0, fit, dpr }
  showMarks: true,
};

function show(id, on) {
  el(id).hidden = !on;
}

function problem(what, why) {
  state.problems.push({ what, why });
}

function renderProblems() {
  const list = el('problems-list');
  list.replaceChildren();
  for (const p of state.problems.slice(0, 200)) {
    const li = document.createElement('li');
    const code = document.createElement('code');
    code.textContent = p.what;
    li.append(code, document.createTextNode(' ' + p.why));
    list.append(li);
  }
  if (state.problems.length > 200) {
    const li = document.createElement('li');
    li.textContent = `and ${state.problems.length - 200} more.`;
    list.append(li);
  }
  show('problems', state.problems.length > 0);
}

function progress(text, done, total) {
  el('progress-text').textContent = text;
  el('progress-fill').style.width = total ? `${Math.round((done / total) * 100)}%` : '0%';
}

// Slices come out in whatever order the disc lists them, so order them along the direction the
// scanner stepped through the body: the normal of the image plane. Failing that, by instance number.
function orderSlices(slices) {
  const first = slices[0];
  const o = first && first.info.imageOrientation;
  if (o && slices.every((s) => s.info.imagePosition)) {
    const n = [
      o[1] * o[5] - o[2] * o[4],
      o[2] * o[3] - o[0] * o[5],
      o[0] * o[4] - o[1] * o[3],
    ];
    for (const s of slices) {
      const p = s.info.imagePosition;
      s.along = p[0] * n[0] + p[1] * n[1] + p[2] * n[2];
    }
    slices.sort((a, b) => a.along - b.along);
    return 'position along the scan direction';
  }
  if (slices.every((s) => Number.isFinite(s.info.instanceNumber))) {
    slices.sort((a, b) => a.info.instanceNumber - b.info.instanceNumber);
    return 'image number';
  }
  slices.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
  return 'file name';
}

function parseBytes(bytes) {
  if (!isDicom(bytes)) return { notDicom: true };
  try {
    const ds = readDataset(bytes);
    return ds ? { ds } : { notDicom: true };
  } catch {
    return { broken: true };
  }
}

// What a file is, read from its first bytes. Tags come in ascending order and the pixels come
// last, so once the parse reaches the pixel-data tag everything describe() needs has been read. A
// file whose head stops short of that (a long header, or a head cut mid-tag) is read whole and
// parsed again, which is exactly what reading it whole at the start would have given.
async function readHeader(f) {
  const head = parseBytes(f.head);
  if (f.whole || head.notDicom || (head.ds && head.ds.pixelDataSeen)) return head;
  return parseBytes(await f.load());
}

// One slice's picture: its bytes read (or unpacked) now, parsed, and decoded.
async function decodeSlice(slice) {
  let bytes;
  try {
    bytes = await slice.load();
  } catch (err) {
    return { unreadable: String(err && err.message ? err.message : err).replace(/\.$/, '') };
  }
  const parsed = parseBytes(bytes);
  // Its head parsed when it was opened; the whole of it can still be damaged past the head.
  if (!parsed.ds) return { unreadable: 'it is damaged after its first few kilobytes' };
  try {
    return await decodeFrame(parsed.ds, 0);
  } catch {
    return { unreadable: 'its pixels are damaged' };
  }
}

function forgetFrames() {
  state.generation += 1;
  state.frames.clear();
  state.loading.clear();
}

// `credit` is set only for the sample scan: the line that says whose pictures these are, so nobody
// mistakes a stranger's scan for their own.
// Clear the last reading and set up the screen for this one. Only the sample has a credit line.
function resetForOpen(credit) {
  scanOpened(null);
  state.cancelled = false;
  state.series = [];
  state.problems = [];
  state.current = null;
  forgetFrames();
  clearReading();
  el('stage-status').hidden = true;
  el('sample-reading-about').hidden = true;     // sample.js shows it once the sample's reading is open
  setCreditLine(el('sample-credit'), credit);
  setCreditLine(el('about-credit'), credit);
  el('viewer-credit').textContent = credit ? credit.short : '';
  el('viewer-credit').hidden = !credit;
  el('outline-caption').hidden = true;
  el('sample-next').hidden = true;
  show('progress', true);
  show('series', false);
  show('viewer', false);
  show('problems', false);
  el('start-over').hidden = false;
}

function setCreditLine(line, credit) {
  line.replaceChildren();
  if (credit) {
    // What a visitor needs first; the attribution the licence asks for after it, smaller, on its
    // own line, where it is on record without being mistaken for something to understand.
    const source = document.createElement('span');
    source.className = 'source';
    source.textContent = credit.source;
    line.append(credit.text, document.createElement('br'), source);
  }
  line.hidden = !credit;
}

// How long a scan waits for its outlines before it opens as plain pictures and draws them on arrival.
const READING_WAIT_MS = 5000;
const LATE = Symbol('late');

// `readingFile` is only given for the sample: a promise for its reading file (or null), fetched at the
// same moment as the pictures. It is waited for here, before anything is shown, so the first picture
// the visitor sees already has its outlines on it. Returns { late }: the promise again when it was too slow.
async function handleFiles(entries, credit = null, readingFile = null) {
  resetForOpen(credit);

  let read;
  try {
    read = await expand(entries, (done, total, path) => {
      progress(`Reading ${done} of ${total}…`, done, total);
      return !state.cancelled;
    });
  } catch (err) {
    progress(String(err && err.message ? err.message : err), 0, 0);
    el('progress-fill').style.width = '0%';
    return;
  }
  if (state.cancelled) { show('progress', false); return; }
  for (const s of read.skipped || []) problem(s.path, s.why);

  const bySeries = await sortIntoSeries(read);
  if (!bySeries) return;
  const series = await buildSeries(bySeries, read);
  series.sort((a, b) => (a.info.seriesNumber ?? 1e9) - (b.info.seriesNumber ?? 1e9));
  state.series = series;

  let late = null;
  if (readingFile && series.length === 1) {
    progress('Outlining parts…', 0, 0);
    const got = await Promise.race([readingFile, new Promise((r) => setTimeout(r, READING_WAIT_MS, LATE))]);
    if (got === LATE) late = readingFile;
    else if (got) await loadReading(got);
    if (state.cancelled) { show('progress', false); return null; }
  }

  show('progress', false);
  renderProblems();
  renderSeries(read);
  renderSeriesMarks();
  // The sample is someone else's scan: only a visitor's own scan can be offered for donation.
  if (!credit && state.series.length) scanOpened(wholeFiles);
  // Wrapped: an async function that returns a promise hands back what it resolves to, not the promise.
  return { late };
}

// Read each file's header and group the usable pictures by series. Null when the reading was cancelled.
async function sortIntoSeries(read) {
  const bySeries = new Map();
  let checked = 0;
  for (const f of read.files) {
    checked += 1;
    if (checked % 200 === 0) {
      progress(`Looking at ${checked} of ${read.files.length} files…`, checked, read.files.length);
      await new Promise((r) => setTimeout(r, 0));
      if (state.cancelled) { show('progress', false); return null; }
    }
    let header;
    try {
      header = await readHeader(f);
    } catch (err) {
      problem(f.path, `could not be read: ${String(err && err.message ? err.message : err).replace(/\.$/, '')}.`);
      continue;
    }
    const picked = pictureOf(f, header);
    if (!picked) continue;
    const { ds, info } = picked;
    const key = info.seriesUid;
    if (!bySeries.has(key)) bySeries.set(key, { info, slices: [], transferSyntaxName: ds.transferSyntaxName });
    // The parsed header is let go here: what is kept is the description and the way back to the
    // pixels, so 500 open files cost their descriptions and not their bytes.
    bySeries.get(key).slices.push({ info, path: f.path, load: f.load });
  }
  return bySeries;
}

// The picture's header and description, or null after saying why the file was left out.
function pictureOf(f, header) {
  if (header.notDicom) { problem(f.path, 'is not a scan picture, so it was skipped.'); return null; }
  if (header.broken) { problem(f.path, 'looks like a scan picture but could not be read.'); return null; }
  const { ds } = header;
  const info = describe(ds);
  if (!info.rows || !info.cols) { problem(f.path, 'has no image in it (it may be a report or an index).'); return null; }
  return { ds, info };
}

// Order each series' slices and decode its first picture. Series that cannot be opened are reported.
async function buildSeries(bySeries, read) {
  progress('Sorting the series…', read.files.length, read.files.length);
  const series = [];
  for (const s of bySeries.values()) {
    s.orderedBy = orderSlices(s.slices);
    const frame = await decodeSlice(s.slices[0]);
    if (frame.unreadable) {
      problem(s.slices[0].path, `could not be read when its picture was wanted: ${frame.unreadable}.`);
      continue;
    }
    if (frame.unsupported) {
      problem(`${s.info.modality || 'A'} series “${s.info.seriesDescription || 'untitled'}” (${s.slices.length} images)`,
        `could not be opened: this disc stores its images as ${frame.unsupported}, which this page cannot open yet.`);
      continue;
    }
    s.firstFrame = frame;
    series.push(s);
  }
  return series;
}

// Every file of every opened series, whole, for the donation check. Read only when asked.
async function wholeFiles() {
  const files = [];
  for (const s of state.series) {
    for (const slice of s.slices) files.push({ name: slice.path, bytes: await slice.load() });
  }
  return files;
}

function renderSeries(read) {
  const cards = el('series-cards');
  cards.replaceChildren();
  const total = state.series.reduce((n, s) => n + s.slices.length, 0);
  const sets = state.series.length === 1 ? '1 set of pictures' : `${state.series.length} sets of pictures`;
  el('series-summary').textContent = state.series.length
    ? `${sets}, ${total} images, read from ${(read.bytes / 1e6).toFixed(0)} MB on this device. Nothing was uploaded.`
    : 'No images on this disc could be opened. What went wrong is listed below.';
  state.series.forEach((s, i) => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'series-card';
    const canvas = document.createElement('canvas');
    canvas.width = 220;
    canvas.height = 220;
    paint(canvas, s.firstFrame, autoWindow(s.firstFrame), 1, 0, 0);
    const title = document.createElement('div');
    title.className = 'title';
    title.textContent = `${s.info.modality || 'Scan'} · ${s.info.seriesDescription || 'untitled'}`;
    const sub = document.createElement('div');
    sub.className = 'sub';
    sub.textContent = `${s.slices.length} image${s.slices.length === 1 ? '' : 's'} · ${s.info.cols}x${s.info.rows}`
      + (s.info.acquisitionDate ? ` · ${s.info.acquisitionDate}` : '');
    card.append(canvas, title, sub);
    card.addEventListener('click', () => openSeries(i));
    cards.append(card);
  });
  // Once a scan is open the page is the scan. Leaving the "drop your scan here" box above it put
  // the pictures a screen and a half down on a phone; "Open a different scan" brings it back.
  show('intake', state.series.length === 0);
  show('series', true);
}

// A reading is written somewhere else: by the person's own AI, on their own machine, through
// rAidiology's MCP server. All this page does is recognise it, work out which of the series they
// have open it is about, and draw it. It still never uploads anything, and the file it reads holds
// no name, patient number or date of birth.

function clearReading() {
  state.reading = null;
  state.readingSeries = -1;
  state.outlines = new Map();
  state.pointedAt = -1;
  state.chosen = -1;
  el('name-tip').hidden = true;
  const out = el('reading-out');
  out.replaceChildren();
  out.hidden = true;
  el('finding').hidden = true;
  el('finding').dataset.sig = '';
  el('toggle-marks').hidden = true;
}

function readingProblem(message) {
  const out = el('reading-out');
  out.replaceChildren();
  const p = document.createElement('p');
  p.className = 'reading-bad';
  p.textContent = message;
  out.append(p);
  out.hidden = false;
  showReadingBox();
}

// The reading box stays out of sight until a reading arrives. Shown to everybody, it told visitors
// about an AI reading their scan on a page that had just said nothing here explains a scan, and
// two in five stopped to ask what it was. Somebody who has a reading file was told by the tool
// that wrote it to drop it here, and a dropped reading is recognised anywhere on the page.
function showReadingBox() {
  const box = el('reading-box');
  box.hidden = false;
  box.open = true;
}

async function isReadingFile(file) {
  if (/\.raidiology\.json(\.gz)?$/i.test(file.name)) return true;
  if (file.size > 4e6 || !/\.(json|gz)$/i.test(file.name)) return false;
  const head = new Uint8Array(await file.slice(0, 200).arrayBuffer());
  return looksLikeFindings(file.name, head);
}

/** The words over the picture while a reading is being matched to it and drawn. */
function setStageStatus(on) {
  el('stage-status').hidden = !on;
}

async function loadReading(file) {
  const viewing = state.current !== null;
  if (viewing) setStageStatus(true);
  try {
    await attachFile(file);
  } finally {
    if (viewing) setStageStatus(false);
  }
}

async function attachFile(file) {
  const series = state.series;
  if (!series.length) {
    readingProblem('Open the scan itself first, then add the reading: the reading on its own has no pictures in it.');
    return;
  }
  let doc;
  try {
    doc = await parseReading(new Uint8Array(await file.arrayBuffer()));
  } catch (err) {
    readingProblem(String(err && err.message ? err.message : err));
    return;
  }
  const { seriesIndex, outlines } = await attachReading(doc, series);
  if (state.series !== series) return;            // another scan was opened while this was read
  if (seriesIndex < 0) {
    readingProblem('That reading is about a different scan from the one open here. Open the scan it was written from, or open a different reading.');
    return;
  }
  state.reading = doc;
  state.readingSeries = seriesIndex;
  state.outlines = outlines;
  state.showMarks = true;
  el('finding').dataset.sig = '';
  forgetPointing();
  renderReading();
  renderSeriesMarks();
  if (state.current === state.readingSeries) {
    el('toggle-marks').hidden = false;
    el('toggle-marks').textContent = 'Hide the outlines';
    draw();
  }
}

/** Forget what the visitor was pointing at: a new picture, series or reading has other outlines. */
function forgetPointing() {
  state.pointedAt = -1;
  state.chosen = -1;
  el('name-tip').hidden = true;
}

function renderReading() {
  const out = el('reading-out');
  out.replaceChildren();
  const names = (state.reading.outlines && state.reading.outlines.structures) || [];
  const p = document.createElement('p');
  p.className = 'reading-note';
  p.textContent = state.outlines.size
    ? `${state.outlines.size} image${state.outlines.size === 1 ? '' : 's'} outlined, naming ${names.length} part${names.length === 1 ? '' : 's'} of the body. Tap or hover an outline on a picture to see its name.`
    : 'This file holds no outlines for the pictures open here.';
  out.append(p);
  out.hidden = false;
  showReadingBox();
}

function renderSeriesMarks() {
  const cards = el('series-cards').children;
  for (let i = 0; i < cards.length; i++) {
    const was = cards[i].querySelector('.series-flag');
    if (was) was.remove();
    if (i !== state.readingSeries) continue;
    const flag = document.createElement('div');
    flag.className = 'series-flag';
    const outlined = state.outlines.size;
    flag.textContent = `${outlined} image${outlined === 1 ? '' : 's'} outlined`;
    cards[i].append(flag);
  }
}

/** A colour for structure `s`, the same every time: hues a golden angle apart, so neighbours in
 * the list (the left and right of one organ, one rib and the next) never share a colour. */
function colourOf(s) {
  // Round five's visitors could not tell the liver, the stomach and the food pipe apart: three teals.
  // A lightness step as well as a hue step keeps more of the structures on one picture apart.
  return `hsl(${Math.round((s * 137.508) % 360)} 85% ${[60, 44, 74][s % 3]}%)`;
}

function pointsOf(text) {
  let p = pathCache.get(text);
  if (!p) { p = decodePath(text); pathCache.set(text, p); }
  return p;
}

/** Which outlined structure is at image pixel (x, y) on this picture: even-odd over every path of
 * each structure, so a hole in one stays a hole. The smallest one wins where two overlap. */
function structureAt(x, y) {
  let best = -1;
  let bestSize = Infinity;
  const inside = new Map();
  const size = new Map();
  for (const [s, text] of outlinesForSlice()) {
    const p = pointsOf(text);
    let hit = false;
    let area = 0;
    for (let i = 0, j = p.length - 2; i < p.length; j = i, i += 2) {
      const [xi, yi, xj, yj] = [p[i], p[i + 1], p[j], p[j + 1]];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
      area += xj * yi - xi * yj;
    }
    if (hit) inside.set(s, !inside.get(s));
    size.set(s, (size.get(s) || 0) + area / 2);
  }
  for (const [s, on] of inside) {
    if (on && Math.abs(size.get(s)) < bestSize) { best = s; bestSize = Math.abs(size.get(s)); }
  }
  return best;
}

// Round eight: a thumb is wider than a thin outline, and a name 14 px above a tap sat under the
// fingertip. Half a fingertip, in CSS pixels.
const FINGER_PX = 22;

/** The outlined structure nearest (x, y) within `radius` image pixels, searched outwards in rings,
 * or -1: what a tap that landed between outlines meant. */
function nearestStructure(x, y, radius) {
  const step = Math.max(1, radius / 4);
  for (let r = step; r <= radius; r += step) {
    for (let k = 0; k < 16; k++) {
      const a = (k * Math.PI) / 8;
      const s = structureAt(x + r * Math.cos(a), y + r * Math.sin(a));
      if (s >= 0) return s;
    }
  }
  return -1;
}

/** Name the structure under a point on the picture, or say nothing when there is none. A finger
 * covers what is under and beside it, so after a tap the name goes above the point. */
function pointAt(clientX, clientY, above = false) {
  const tip = el('name-tip');
  const v = state.view;
  if (!v || !state.reading || !state.reading.outlines) { tip.hidden = true; return; }
  const rect = el('canvas').getBoundingClientRect();
  const x = ((clientX - rect.left) * v.dpr - v.x0) / v.fit;
  const y = ((clientY - rect.top) * v.dpr - v.y0) / v.fit;
  let s = structureAt(x, y);
  if (s < 0 && above) s = nearestStructure(x, y, (FINGER_PX * v.dpr) / v.fit);
  if (s !== state.pointedAt) { state.pointedAt = s; state.chosen = -1; draw(); }
  if (s < 0) { tip.hidden = true; return; }
  nameTip(s, clientX - rect.left, clientY - rect.top, rect, above);
}

/** Write structure `s`'s name (and what it is) into the tip and put it at (x, y) on the picture. */
function nameTip(s, x, y, rect, above) {
  const tip = el('name-tip');
  const o = state.reading.outlines;
  const name = document.createElement('strong');
  name.textContent = o.structures[s] || `structure ${s + 1}`;
  tip.replaceChildren(name);
  if (o.about && o.about[s]) tip.append(document.createTextNode(`: ${o.about[s]}`));
  tip.hidden = false;
  placeTip(tip, x, y, rect, above);
}

/** A point inside structure `s` on this picture, near the middle of its outline, in image pixels; or
 * null when it has no outline here. Where no point of a coarse grid lies inside it (a thin or tiny
 * part), the middle of its box is used. */
function interiorPoint(s) {
  const mine = outlinesForSlice().filter(([k]) => k === s);
  if (!mine.length) return null;
  const box = [Infinity, Infinity, -Infinity, -Infinity];
  for (const [, text] of mine) {
    const p = pointsOf(text);
    for (let i = 0; i < p.length; i += 2) {
      box[0] = Math.min(box[0], p[i]); box[1] = Math.min(box[1], p[i + 1]);
      box[2] = Math.max(box[2], p[i]); box[3] = Math.max(box[3], p[i + 1]);
    }
  }
  const [x0, y0, x1, y1] = box;
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  let best = null;
  for (let i = 1; i < 16; i++) {
    for (let k = 1; k < 16; k++) {
      const x = x0 + ((x1 - x0) * i) / 16;
      const y = y0 + ((y1 - y0) * k) / 16;
      if (structureAt(x, y) !== s) continue;
      const d = (x - cx) ** 2 + (y - cy) ** 2;
      if (!best || d < best[2]) best = [x, y, d];
    }
  }
  return best ? [best[0], best[1]] : [cx, cy];
}

/** Point at structure `s` from its row in the list: outline it on the picture and put its name beside
 * a point inside it, the same way a tap on the picture does. */
function namePart(s) {
  const v = state.view;
  if (!v || !state.reading || !state.reading.outlines) return;
  const at = interiorPoint(s);
  if (!at) return;
  state.pointedAt = s;
  state.chosen = s;
  draw();
  const rect = el('canvas').getBoundingClientRect();
  nameTip(s, (at[0] * v.fit + v.x0) / v.dpr, (at[1] * v.fit + v.y0) / v.dpr, rect, false);
}

/** A tap on a row of the list: name that part on the picture, or let go of it when it already is. */
function choosePart(s) {
  if (state.chosen === s) {
    state.chosen = -1;
    state.pointedAt = -1;
    el('name-tip').hidden = true;
    draw();
    return;
  }
  namePart(s);
}

/** Put the name next to the point, on whichever side keeps all of it inside the picture. Round seven:
 * one visitor hunted for a name the picture's edge had cut off, another found it covering half the
 * picture. */
function placeTip(tip, x, y, rect, above) {
  const gap = 14;
  const w = tip.offsetWidth;
  const h = tip.offsetHeight;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi));
  const centred = clamp(x - w / 2, 4, rect.width - w - 4);
  const right = x + gap + w <= rect.width;
  const left = x - gap - w >= 0;
  let at;
  if (above) {
    // After a tap: above the point, clear of the fingertip when there is room, since the hand
    // covers the point and what is below it; below it only when the point is too near the top.
    const top = [2 * FINGER_PX, gap].map((g) => y - g - h).find((t) => t >= 4);
    at = [centred, top !== undefined ? top : clamp(y + 2 * FINGER_PX, 4, rect.height - h - 4)];
  } else if (right || left) {
    at = [right ? x + gap : x - gap - w, clamp(y + gap, 4, rect.height - h - 4)];
  } else {
    // Too wide for either side: centred over the point, so it never covers it.
    at = [centred, y - gap - h >= 4 ? y - gap - h : clamp(y + gap, 4, rect.height - h - 4)];
  }
  tip.style.left = `${at[0]}px`;
  tip.style.top = `${at[1]}px`;
}

const pathCache = new Map();               // path text -> its points, decoded once

function outlinesForSlice() {
  if (state.current !== state.readingSeries || !state.showMarks) return [];
  return state.outlines.get(state.slice) || [];
}

/** Draw a labeller's outlines over the picture, in the picture's own pixels, and return which
 * structures were drawn. Lines rather than filled shapes: the picture underneath stays readable,
 * and an outline says "the model put this structure here" without pretending to be the organ. */
function overlayOutlines(canvas, frame, fit, panX, panY, paths, dpr) {
  if (!paths.length) return [];
  const ctx = canvas.getContext('2d');
  const x0 = (canvas.width - frame.width * fit) / 2 + panX;
  const y0 = (canvas.height - frame.height * fit) / 2 + panY;
  const drawn = new Set();
  state.view = { x0, y0, fit, dpr };
  ctx.save();
  ctx.lineJoin = 'round';
  for (const [s, text] of paths) {
    const p = pointsOf(text);
    if (p.length < 4) continue;
    const pointed = s === state.pointedAt;
    ctx.strokeStyle = colourOf(s);
    ctx.lineWidth = Math.max(1, (pointed ? 3 : 1.25) * dpr);
    ctx.beginPath();
    ctx.moveTo(x0 + p[0] * fit, y0 + p[1] * fit);
    for (let i = 2; i < p.length; i += 2) ctx.lineTo(x0 + p[i] * fit, y0 + p[i + 1] * fit);
    ctx.closePath();
    ctx.stroke();
    if (pointed) {
      ctx.globalAlpha = 0.25;
      ctx.fillStyle = colourOf(s);
      ctx.fill('evenodd');
      ctx.globalAlpha = 1;
    }
    drawn.add(s);
  }
  ctx.restore();
  return [...drawn].sort((a, b) => a - b);
}

/** Under the picture: the names of the outlined parts, when there is a reading of this series. */
function describeImage(note, drawn) {
  note.hidden = !(state.current === state.readingSeries && state.reading);
  if (note.hidden) return;
  describeOutlines(note, drawn);
}

/** The list of outlined parts, open by default: a tinted row per part, each a button at least 44px
 * tall with the colour it is drawn in. It is built once for each set of parts and only has its marks
 * updated on later draws, so hovering the picture does not rebuild it or scroll it back to the top. */
function describeOutlines(note, drawn) {
  const names = state.reading.outlines.structures || [];
  note.classList.toggle('marked', drawn.length > 0);
  if (!drawn.length) {
    note.dataset.sig = '';
    note.textContent = state.showMarks ? 'Nothing is outlined on this image.'
      : 'The outlines are hidden. Press Show the outlines to see the parts.';
    return;
  }
  const sig = drawn.join(',');
  if (note.dataset.sig !== sig) {
    note.dataset.sig = sig;
    note.replaceChildren();
    const how = document.createElement('p');
    how.textContent = 'Tap a part below to find it on the picture. You can also tap or hover an outline.';
    const all = document.createElement('details');
    all.className = 'outline-all';
    all.open = state.keyOpen;
    all.addEventListener('toggle', () => { state.keyOpen = all.open; });
    const summary = document.createElement('summary');
    summary.textContent = drawn.length === 1 ? 'The one part outlined on this picture'
      : `${drawn.length} parts outlined on this picture`;
    const list = document.createElement('ul');
    list.className = 'parts';
    const about = state.reading.outlines.about;
    for (const s of drawn) {
      const item = document.createElement('li');
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'part-row';
      row.dataset.part = String(s);
      const swatch = document.createElement('span');
      swatch.className = 'outline-swatch';
      swatch.style.backgroundColor = colourOf(s);
      const label = document.createElement('span');
      label.className = 'part-name';
      label.textContent = names[s] || `structure ${s + 1}`;
      row.append(swatch, label);
      if (about && about[s]) {
        const what = document.createElement('span');
        what.className = 'part-about';
        what.textContent = about[s];
        row.append(what);
      }
      row.addEventListener('click', () => choosePart(s));
      row.addEventListener('pointerenter', (ev) => {
        if (ev.pointerType === 'touch') return;
        state.pointedAt = s;
        draw();
      });
      row.addEventListener('pointerleave', (ev) => {
        if (ev.pointerType === 'touch') return;
        state.pointedAt = state.chosen;
        draw();
      });
      item.append(row);
      list.append(item);
    }
    all.append(summary, list);
    note.append(how, all);
  }
  for (const row of note.querySelectorAll('.part-row')) {
    const s = Number(row.dataset.part);
    row.classList.toggle('is-pointed', s === state.pointedAt);
    row.setAttribute('aria-pressed', String(s === state.chosen));
  }
}

async function openSeries(index) {
  state.current = index;
  const s = state.series[index];
  // Start in the middle. The first picture of a scan is its edge, which is often a sliver of
  // nothing, and a blank-looking first picture reads as "it did not work".
  state.slice = Math.floor((s.slices.length - 1) / 2);
  state.moved = false;
  state.zoom = 1;
  state.panX = 0;
  state.panY = 0;
  state.direction = 1;
  forgetFrames();
  forgetPointing();
  el('viewer-h').textContent = `${s.info.modality || 'Scan'} · ${s.info.seriesDescription || 'untitled'}`;
  el('slice').max = String(s.slices.length - 1);
  el('slice').value = String(state.slice);
  const frame = await frameAt(index, state.slice);
  const w = (Number.isFinite(s.info.windowCentre) && Number.isFinite(s.info.windowWidth) && s.info.windowWidth > 0)
    ? { centre: s.info.windowCentre, width: s.info.windowWidth }
    : autoWindow(frame);
  state.centre = w.centre;
  state.width = w.width;
  syncWindowSliders(frame);
  const marks = el('toggle-marks');
  marks.hidden = !(state.reading && index === state.readingSeries);
  marks.textContent = state.showMarks ? 'Hide the outlines' : 'Show the outlines';
  // One way out of the viewer: back to the list when there is more than one set to choose from,
  // otherwise straight to opening a different scan (a list of one has nothing in it).
  el('back').textContent = state.series.length > 1 ? 'Back to the list' : 'Open a different scan';
  el('start-over').hidden = true;
  show('viewer', true);
  show('series', false);
  draw();
  el('viewer').scrollIntoView({ block: 'start' });
}

function frameBytes(frame) {
  return (frame.raw || frame.values).byteLength;
}

// Drop whatever is furthest from where the visitor is looking, not simply the oldest, until what is
// kept fits the budget.
function evictFrames() {
  let total = 0;
  for (const f of state.frames.values()) total += frameBytes(f);
  if (total <= MAX_CACHED_BYTES) return;
  const keys = [...state.frames.keys()];
  const away = (k) => Math.abs(Number(k.split(':')[1]) - state.slice);
  keys.sort((a, b) => away(b) - away(a));
  for (const k of keys) {
    if (total <= MAX_CACHED_BYTES || state.frames.size <= MIN_CACHED_FRAMES) break;
    total -= frameBytes(state.frames.get(k));
    state.frames.delete(k);
  }
}

// A slice that is already being decoded (fetched ahead, say) is waited for rather than decoded a
// second time. A decode that finishes after the visitor opened something else is handed back but
// not kept, because its key may by then name a slice of a different scan.
function frameAt(seriesIndex, slice) {
  const key = `${seriesIndex}:${slice}`;
  const hit = state.frames.get(key);
  if (hit) return Promise.resolve(hit);
  const loading = state.loading.get(key);
  if (loading) return loading;
  const s = state.series[seriesIndex];
  const generation = state.generation;
  const pending = decodeSlice(s.slices[slice]).then((frame) => {
    const usable = !(frame.unsupported || frame.unreadable);
    if (generation === state.generation) {
      state.loading.delete(key);
      if (usable) {
        state.frames.set(key, frame);
        evictFrames();
      }
    }
    return usable ? frame : s.firstFrame;
  });
  state.loading.set(key, pending);
  return pending;
}

// Where the browser can say when it is idle, fetching ahead waits for that; where it cannot
// (Safari), it waits for the current task to end.
const whenIdle = typeof window.requestIdleCallback === 'function'
  ? (fn) => window.requestIdleCallback(fn, { timeout: 250 })
  : (fn) => setTimeout(fn, 0);

// The next few slices in the direction the visitor is moving, and one behind, decoded while the page
// has nothing else to do, so that the next step only has to paint. It stops as soon as the visitor
// has moved: the draw for wherever they are now fetches ahead from there.
function prefetchAround(seriesIndex, slice) {
  const s = state.series[seriesIndex];
  if (!s) return;
  const generation = state.generation;
  const wanted = [];
  for (let k = 1; k <= PREFETCH_AHEAD; k++) wanted.push(slice + state.direction * k);
  wanted.push(slice - state.direction);
  whenIdle(async () => {
    for (const k of wanted) {
      if (state.generation !== generation || state.current !== seriesIndex || state.slice !== slice) return;
      if (k < 0 || k >= s.slices.length || state.frames.has(`${seriesIndex}:${k}`)) continue;
      await frameAt(seriesIndex, k);
    }
  });
}

// The window as a table with one finished pixel per stored value, built when the window changes
// rather than worked out again for every pixel of every slice. Each entry is the arithmetic the
// floating-point path below does per pixel (rescale to a 32-bit float, subtract the bottom of the
// window, scale, clamp to 0..255) stored through a Uint8ClampedArray, so it rounds exactly as
// ImageData rounds. A MONOCHROME1 picture takes 255 minus that grey. Writing the four bytes through
// a Uint8ClampedArray and reading them back as one 32-bit number keeps the byte order right on any
// machine.
let greyTable = null;

function tableFor(frame, lo, scale) {
  const t = greyTable;
  const same = t && t.slope === frame.slope && t.intercept === frame.intercept && t.invert === frame.invert
    && t.lo === lo && t.scale === scale;
  if (same && t.from <= frame.rawMin && t.to >= frame.rawMax) return t;
  // Only the stored values this frame (and, at the same window, the frames before it) use: a CT
  // slice spans a few thousand values, not all 65536.
  const from = same ? Math.min(t.from, frame.rawMin) : frame.rawMin;
  const to = same ? Math.max(t.to, frame.rawMax) : frame.rawMax;
  const pixels = new Uint32Array(to - from + 1);
  const px = new Uint8ClampedArray(4);
  const px32 = new Uint32Array(px.buffer);
  px[3] = 255;
  for (let k = 0; k < pixels.length; k++) {
    px[0] = Math.max(0, Math.min(255, (rescale(from + k, frame.slope, frame.intercept) - lo) * scale));
    const g = frame.invert ? 255 - px[0] : px[0];
    px[0] = g; px[1] = g; px[2] = g;
    pixels[k] = px32[0];
  }
  greyTable = { slope: frame.slope, intercept: frame.intercept, invert: frame.invert, lo, scale, from, to, pixels };
  return greyTable;
}

function paint(canvas, frame, window, zoom, panX, panY) {
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(frame.width, frame.height);
  const lo = window.centre - window.width / 2;
  const scale = 255 / Math.max(window.width, 1e-6);
  const n = frame.width * frame.height;
  if (frame.raw) {
    // One 32-bit store per pixel, straight from the table.
    const { pixels, from } = tableFor(frame, lo, scale);
    const out = new Uint32Array(img.data.buffer);
    const raw = frame.raw;
    for (let i = 0; i < n; i++) out[i] = pixels[raw[i] - from];
  } else if (frame.samples === 3) {
    for (let i = 0; i < n; i++) {
      img.data[i * 4] = Math.max(0, Math.min(255, (frame.values[i * 3] - lo) * scale));
      img.data[i * 4 + 1] = Math.max(0, Math.min(255, (frame.values[i * 3 + 1] - lo) * scale));
      img.data[i * 4 + 2] = Math.max(0, Math.min(255, (frame.values[i * 3 + 2] - lo) * scale));
      img.data[i * 4 + 3] = 255;
    }
  } else {
    for (let i = 0; i < n; i++) {
      const v = Math.max(0, Math.min(255, (frame.values[i] - lo) * scale));
      img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255;
    }
    if (frame.invert) {
      // After the bytes are rounded, so this is 255 minus the same grey the table path inverts.
      for (let i = 0; i < n; i++) {
        const g = 255 - img.data[i * 4];
        img.data[i * 4] = g; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = g;
      }
    }
  }
  const off = new OffscreenCanvas(frame.width, frame.height);
  off.getContext('2d').putImageData(img, 0, 0);
  ctx.save();
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const fit = Math.min(canvas.width / frame.width, canvas.height / frame.height) * zoom;
  const w = frame.width * fit;
  const h = frame.height * fit;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, (canvas.width - w) / 2 + panX, (canvas.height - h) / 2 + panY, w, h);
  ctx.restore();
  return fit;
}

let drawPending = false;
// Coalesce redraws on a timer rather than an animation frame. Drawing here is driven by what the
// visitor does, not by a running animation, and an animation frame never fires in a tab the browser
// has decided not to paint, which leaves the slice they asked for on a blank canvas.
function schedule(fn) {
  setTimeout(fn, 0);
}

function draw() {
  if (drawPending) return;
  drawPending = true;
  schedule(async () => {
    drawPending = false;
    if (state.current === null) return;
    const s = state.series[state.current];
    const [askedSeries, askedSlice] = [state.current, state.slice];
    const frame = await frameAt(askedSeries, askedSlice);
    if (state.current !== askedSeries || state.slice !== askedSlice) { draw(); return; }
    const canvas = el('canvas');
    let rect = canvas.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) {
      // The section was still being laid out. Try once more rather than drawing into nothing.
      await new Promise((r) => setTimeout(r, 50));
      rect = canvas.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2) return;
    }
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    const fit = paint(canvas, frame, { centre: state.centre, width: state.width }, state.zoom, state.panX * dpr, state.panY * dpr);
    const drawn = overlayOutlines(canvas, frame, fit, state.panX * dpr, state.panY * dpr, outlinesForSlice(), dpr);
    describeImage(el('finding'), drawn);
        describeView(s.info);
    // Which picture, and nothing else. The brightness and contrast numbers and the sort order were
    // on here too; they are a radiographer's units, and a visitor read them as things to understand.
    // The how-to is a chip under the picture, never on it, and it goes after the first change.
    el('hud').textContent = `Picture ${state.slice + 1} of ${s.slices.length}`;
    el('hint-chip').hidden = state.moved || s.slices.length < 2;
    const spacing = s.info.pixelSpacing;
    const bar = el('scalebar');
    if (spacing && spacing[0] > 0) {
      const px = (10 / spacing[0]) * fit / dpr;         // 10 mm on screen, in CSS pixels
      if (px > 12 && px < rect.width * 0.8) {
        bar.style.width = `${Math.round(px)}px`;
        bar.hidden = false;
      } else {
        bar.hidden = true;
      }
    } else {
      bar.hidden = true;
    }
    prefetchAround(state.current, state.slice);
    const waiting = drawWaiters;
    drawWaiters = [];
    for (const done of waiting) done();
  });
}

let drawWaiters = [];
/** Resolves once the next picture has been drawn, or after `ms` if none is (a hidden tab draws nothing). */
export function afterNextDraw(ms = 2000) {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    drawWaiters.push(() => { clearTimeout(timer); resolve(); });
  });
}

// Which way the picture is facing, in a visitor's words. The file says which direction each edge
// of the picture points in the body: the first three numbers are where the right-hand edge points,
// the next three where the bottom edge points, with +x the body's left, +y its back, +z its head.
// A picture taken at a slant is left unlabelled rather than rounded to the nearest side.
const SIDES = [['right side of the body', 'left side of the body'], ['front', 'back'], ['feet', 'head']];

function sideOf(v) {
  const size = v.map(Math.abs);
  const axis = size.indexOf(Math.max(...size));
  if (size[axis] < 0.85) return null;
  return { axis, toward: SIDES[axis][v[axis] > 0 ? 1 : 0], away: SIDES[axis][v[axis] > 0 ? 0 : 1] };
}

// What is true of every picture of this kind, and nothing about this one. It says which way up the
// picture is and what white and black usually are; it does not say what anything on it means.
const SHADES = {
  CT: 'On a CT picture, white is bone, grey is organs and muscle, and black is air.',
  CR: 'On an X-ray, white is bone and anything dense, and black is air.',
  DX: 'On an X-ray, white is bone and anything dense, and black is air.',
  MR: 'On an MRI, what shows up bright depends on the kind of picture, so there is no simple rule for white and black.',
};

function describeView(info) {
  const o = info.imageOrientation;
  const across = o ? sideOf(o.slice(0, 3)) : null;
  const down = o ? sideOf(o.slice(3, 6)) : null;
  const known = across && down && across.axis !== down.axis;
  const edges = known
    ? { 'edge-right': across.toward, 'edge-left': across.away, 'edge-bottom': down.toward, 'edge-top': down.away }
    : { 'edge-right': '', 'edge-left': '', 'edge-bottom': '', 'edge-top': '' };
  for (const [id, text] of Object.entries(edges)) {
    el(id).textContent = text;
    el(id).hidden = !text;
    // Phones show R and L in place of the long side names, which sat on the anatomy (open.css).
    el(id).dataset.short = text === SIDES[0][0] ? 'R' : text === SIDES[0][1] ? 'L' : '';
  }
  const parts = [];
  if (known) {
    const axes = [across.axis, down.axis].sort().join('');
    if (axes === '01') {
      parts.push(across.away === SIDES[0][0] && down.away === 'front'
        ? 'This is a slice across the body, seen as if looking up from the feet.'
        : 'This is a slice across the body.');
    } else if (axes === '02') {
      const facing = across.axis !== 0 ? '' : across.away === SIDES[0][0] ? ', seen from the front' : ', seen from behind';
      parts.push(`This is a slice through the body from side to side${facing}.`);
    } else {
      parts.push('This is a slice through the body from front to back, seen from the side.');
    }
    parts.push('The words at the edges say which side is which.');
  }
  if (SHADES[info.modality]) parts.push(SHADES[info.modality]);
  el('what').textContent = parts.join(' ');
  el('what').hidden = parts.length === 0;
}

function syncWindowSliders(frame) {
  const span = Math.max(frame.max - frame.min, 1e-6);
  const c = el('centre');
  const w = el('width');
  c.min = '0'; c.max = '1000';
  w.min = '1'; w.max = '1000';
  c.value = String(Math.round(((state.centre - frame.min) / span) * 1000));
  w.value = String(Math.round((state.width / (span * 2)) * 1000));
  c.dataset.min = String(frame.min);
  c.dataset.span = String(span);
  w.dataset.span = String(span);
}

function windowFromSliders() {
  const c = el('centre');
  const w = el('width');
  const min = Number(c.dataset.min || 0);
  const span = Number(c.dataset.span || 1);
  state.centre = min + (Number(c.value) / 1000) * span;
  state.width = Math.max((Number(w.value) / 1000) * span * 2, span / 1000);
}

function setSlice(next, byVisitor = true) {
  if (state.current === null) return;
  const s = state.series[state.current];
  const before = state.slice;
  state.slice = Math.max(0, Math.min(s.slices.length - 1, next));
  if (state.slice !== before) {
    state.direction = state.slice > before ? 1 : -1;
    forgetPointing();
  }
  if (byVisitor) state.moved = true;
  el('slice').value = String(state.slice);
  draw();
}

/** A tap that did not move is a question: what is this? */
function tapped(ev, drag) {
  if (drag && state.slice === drag.slice && Math.hypot(ev.clientX - drag.x, ev.clientY - drag.y) < 6) {
    pointAt(ev.clientX, ev.clientY, ev.pointerType !== 'mouse');
  }
}

/** A mouse over the picture names the outlined structure under it; leaving the picture forgets it. */
function wireNaming(stage, dragging) {
  stage.addEventListener('pointermove', (ev) => {
    if (ev.pointerType === 'mouse' && !dragging()) pointAt(ev.clientX, ev.clientY);
  });
  stage.addEventListener('pointerleave', (ev) => {
    if (ev.pointerType !== 'mouse' || dragging()) return;
    el('name-tip').hidden = true;
    if (state.pointedAt >= 0) { state.pointedAt = -1; state.chosen = -1; draw(); }
  });
}

function wireStage() {
  const stage = el('stage');
  let drag = null;
  stage.addEventListener('pointerdown', (ev) => {
    stage.setPointerCapture(ev.pointerId);
    drag = { x: ev.clientX, y: ev.clientY, slice: state.slice, centre: state.centre, width: state.width,
             panX: state.panX, panY: state.panY, pan: ev.button === 1 || ev.shiftKey };
    ev.preventDefault();
  });
  stage.addEventListener('pointermove', (ev) => {
    if (!drag) return;
    const dx = ev.clientX - drag.x;
    const dy = ev.clientY - drag.y;
    if (drag.pan) {
      state.panX = drag.panX + dx;
      state.panY = drag.panY + dy;
      draw();
      return;
    }
    if (Math.abs(dy) > Math.abs(dx)) {
      setSlice(drag.slice + Math.round(dy / 6));
    } else {
      const span = Number(el('centre').dataset.span || 1);
      state.centre = drag.centre + (dx / stage.clientWidth) * span;
      state.width = drag.width;
      syncOnlyValues();
      draw();
    }
  });
  stage.addEventListener('pointerup', (ev) => { tapped(ev, drag); drag = null; });
  stage.addEventListener('pointercancel', () => { drag = null; });
  wireNaming(stage, () => drag !== null);
  stage.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    state.zoom = Math.max(0.5, Math.min(12, state.zoom * (ev.deltaY < 0 ? 1.12 : 1 / 1.12)));
    draw();
  }, { passive: false });
  stage.tabIndex = 0;
  // On the page, not on the picture: the hint under the picture says the keys move through it, and
  // with the listener on the picture alone they scrolled the page until somebody clicked it first.
  // A slider that has the focus keeps its own keys.
  listen(document, 'keydown', (ev) => {
    if (state.current === null || el('viewer').hidden) return;
    if (ev.target instanceof HTMLInputElement || ev.altKey || ev.ctrlKey || ev.metaKey) return;
    const step = ev.key === 'PageUp' || ev.key === 'PageDown' ? 10 : 1;
    if (ev.key === 'ArrowDown' || ev.key === 'ArrowRight' || ev.key === 'PageDown') { setSlice(state.slice + step); ev.preventDefault(); }
    if (ev.key === 'ArrowUp' || ev.key === 'ArrowLeft' || ev.key === 'PageUp') { setSlice(state.slice - step); ev.preventDefault(); }
  });
}

function syncOnlyValues() {
  const c = el('centre');
  const min = Number(c.dataset.min || 0);
  const span = Number(c.dataset.span || 1);
  c.value = String(Math.round(((state.centre - min) / span) * 1000));
  el('width').value = String(Math.round((state.width / (span * 2)) * 1000));
}

const viewPicker = typeof window.showDirectoryPicker === 'function';

function wireIntake() {
  const drop = el('drop');
  // The step-by-step help says in advance what the browser will ask, so it has to be the question
  // this browser really asks.
  // It is said twice, beside the button that triggers the question and in the step-by-step help: a
  // visitor who met the browser's question without having opened the help had been told nothing.
  const asks = viewPicker
    ? 'Your browser will ask whether this page may read the files. Allowing it only lets the page read them, and nothing is sent.'
    : 'Your browser may ask "Upload files to this site?". That is its own wording for letting a page read them. Nothing is sent: you can switch your Wi-Fi off first and it still works.';
  for (const line of document.querySelectorAll('.browser-asks')) line.textContent = asks;
  // The page's own address, put into a message the visitor sends to themselves from their own mail
  // app. Taken from the address bar rather than written here, and nothing is sent by this page.
  const here = `${window.location.origin}${window.location.pathname}`;
  el('send-self').href = `mailto:?subject=${encodeURIComponent('Open my scan at home')}&body=${encodeURIComponent(here)}`;
  const open = (accept, directory) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    if (directory) { input.webkitdirectory = true; input.directory = true; }
    if (accept) input.accept = accept;
    input.addEventListener('change', async () => {
      if (input.files && input.files.length) handleFiles(await fromFileList(input.files));
    });
    input.click();
  };
  // Two ways to choose a folder, and the difference is what the browser says while doing it. The
  // older file input makes the browser ask "Upload files to this site?", which is a frightening
  // thing to be told to agree to on a page that promises nothing is uploaded; two visitors in three
  // said so. Where the browser has a folder picker of its own, it asks whether the page may VIEW
  // the files, which is what is actually happening, so that one is used whenever it exists.
  const help = el('drop-help');
  const pickFolder = async () => {
    if (!viewPicker) { open(null, true); return; }
    const helpText = help.textContent;
    let handle;
    // The browser's own window can open behind this one, or slowly; a visitor in round four pressed
    // the button, saw nothing change and thought it had not worked.
    help.textContent = 'A window has opened for you to choose the disc or the folder. If you cannot see it, it may be behind this one.';
    try {
      handle = await window.showDirectoryPicker({ mode: 'read' });
    } catch (err) {
      help.textContent = helpText;
      // Closing the picker is not an error. Anything else, fall back to the older way.
      if (!err || err.name !== 'AbortError') open(null, true);
      return;
    }
    help.textContent = helpText;
    handleFiles(await fromDirectoryHandle(handle));
  };
  const pickZip = () => open('.zip,application/zip', false);
  // A phone or a tablet cannot read a disc, and most cannot hand a page a folder: there the box opens
  // the ZIP chooser, which is the thing such a device can do (round four: the big button on a phone
  // was the one a phone cannot use).
  const touchOnly = window.matchMedia('(hover: none) and (pointer: coarse)').matches;
  const pickDefault = touchOnly ? pickZip : pickFolder;
  el('pick-folder').addEventListener('click', (ev) => { ev.stopPropagation(); pickFolder(); });
  el('pick-zip').addEventListener('click', (ev) => { ev.stopPropagation(); pickZip(); });
  drop.addEventListener('click', pickDefault);
  drop.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); pickDefault(); }
  });
  for (const type of ['dragenter', 'dragover']) {
    listen(window, type, (ev) => { ev.preventDefault(); drop.classList.add('over'); });
  }
  for (const type of ['dragleave', 'dragend']) {
    listen(window, type, () => drop.classList.remove('over'));
  }
  listen(window, 'drop', async (ev) => {
    ev.preventDefault();
    drop.classList.remove('over');
    if (!ev.dataTransfer) return;
    // One reading file dropped anywhere on the page is a reading, not a new scan. Making the
    // visitor aim at the right rectangle to tell the page something it can work out for itself is
    // the kind of small cruelty this page exists to avoid.
    const files = ev.dataTransfer.files;
    if (files && files.length === 1 && await isReadingFile(files[0])) {
      loadReading(files[0]);
      return;
    }
    handleFiles(await fromDataTransfer(ev.dataTransfer));
  });
  const pickReading = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.gz,.json,application/gzip,application/json';
    input.addEventListener('change', () => {
      if (input.files && input.files.length) loadReading(input.files[0]);
    });
    input.click();
  };
  el('pick-reading').addEventListener('click', (ev) => { ev.stopPropagation(); pickReading(); });
  const rdrop = el('reading-drop');
  rdrop.addEventListener('click', pickReading);
  rdrop.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); pickReading(); }
  });
  el('toggle-marks').addEventListener('click', () => {
    state.showMarks = !state.showMarks;
    el('toggle-marks').textContent = state.showMarks ? 'Hide the outlines' : 'Show the outlines';
    draw();
  });
  el('cancel').addEventListener('click', () => { state.cancelled = true; show('progress', false); });
  const startOver = () => {
    scanOpened(null);
    state.series = []; state.problems = []; state.current = null; forgetFrames();
    clearReading();
    show('series', false); show('viewer', false); show('problems', false); show('progress', false);
    show('intake', true);
    el('start-over').hidden = true;
    el('intake').scrollIntoView({ block: 'start' });
  };
  el('start-over').addEventListener('click', startOver);
  el('back').addEventListener('click', () => {
    if (state.series.length <= 1) { startOver(); return; }
    state.current = null;
    show('viewer', false);
    show('series', true);
    el('start-over').hidden = false;
  });
  el('slice').addEventListener('input', (ev) => setSlice(Number(ev.target.value)));
  // Round six's board: on a touch screen, dragging was the only way through the pictures.
  el('prev-picture').addEventListener('click', () => setSlice(state.slice - 1));
  el('next-picture').addEventListener('click', () => setSlice(state.slice + 1));
  for (const id of ['centre', 'width']) {
    el(id).addEventListener('input', () => { windowFromSliders(); draw(); });
  }
  el('reset-window').addEventListener('click', async () => {
    if (state.current === null) return;
    const frame = await frameAt(state.current, state.slice);
    const w = autoWindow(frame);
    state.centre = w.centre;
    state.width = w.width;
    state.zoom = 1; state.panX = 0; state.panY = 0;
    syncWindowSliders(frame);
    draw();
  });
  listen(window, 'resize', draw);
  // A tab that was hidden when the slice changed has no layout to draw into, so draw again when it
  // comes back rather than leaving a blank canvas.
  listen(document, 'visibilitychange', () => { if (document.visibilityState === 'visible') draw(); });
}

// The sample's way in: the same path a dropped ZIP takes, then straight to the pictures, because
// somebody trying a sample wants to see a scan and not a list with one thing in it.
// `readingFile` is a promise for the sample's outlines, fetched alongside its pictures. Resolves to
// whether the outlines are on the picture.
export async function openFiles(entries, credit, readingFile = null) {
  const { late } = (await handleFiles(entries, credit, readingFile)) || {};
  const mine = state.series;
  if (mine.length === 1) await openSeries(0);
  if (late) {
    // The outlines were slower than the pictures: the picture is up with "Outlining parts" over it.
    setStageStatus(true);
    try {
      const file = await late;
      if (file && state.series === mine) await loadReading(file);
    } finally {
      setStageStatus(false);
    }
  }
  return state.reading !== null;
}

// The sample's reading comes in the same way a dropped one does.
export const openReading = loadReading;

/** Name the biggest outlined part on the picture once, the way a tap would, so the first look at the
 * sample already shows what the outlines are for (the designers, 2026-10-11: the best part sat below
 * the fold). It looks for a point inside that part nearest its middle; any tap or slice change
 * replaces the name as usual. */
export function nameBiggestPart() {
  const v = state.view;
  if (!v || !state.reading || !state.reading.outlines) return;
  const area = new Map();
  for (const [s, text] of outlinesForSlice()) {
    const p = pointsOf(text);
    let a = 0;
    for (let i = 0, j = p.length - 2; i < p.length; j = i, i += 2) a += p[j] * p[i + 1] - p[i] * p[j + 1];
    area.set(s, (area.get(s) || 0) + Math.abs(a / 2));
  }
  let biggest = -1;
  for (const [s, a] of area) if (biggest < 0 || a > area.get(biggest)) biggest = s;
  if (biggest >= 0) namePart(biggest);
}

export function start() {
  wireIntake();
  wireStage();
  if (!window.DecompressionStream) {
    problem('This browser', 'cannot unzip a file on its own, so a ZIP will not open here. A folder still works, and any up-to-date browser can do both.');
    renderProblems();
  }
  if (new URLSearchParams(window.location.search).has('reading')) el('reading-box').hidden = false;
}

start();

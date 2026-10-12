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

// Which gesture the device has, so the viewer says tap with a finger and click or point with a mouse.
const mouseDevice = () => window.matchMedia('(hover: hover) and (pointer: fine)').matches;
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
  spotlight: -1,                    // the structure a name in the list is asking about (the others dim), or -1
  listChosen: -1,                   // the structure whose row was tapped, or -1
  ribsOpen: false,                  // whether the Ribs row of the list shows each rib
  keyOpen: true,                    // whether the list of outlined parts is open (it starts open)
  chosen: -1,                       // the part whose row the visitor tapped, or -1
  view: null,                       // where the last picture was drawn: { x0, y0, fit, dpr }
  showMarks: true,
};

function show(id, on) {
  el(id).hidden = !on;
}

// The tab's own title, kept so it can come back when the sample is closed. The sample's tab names the sample
// (wave three: the tab still said "Open your own scan" over a stranger's CT). A visitor's own set never goes in
// the title, because a tab title is kept in the browser's history.
const PAGE_TITLE = document.title;
function titleFor(credit) {
  document.title = credit && credit.title ? `rAidiology · ${credit.title}` : PAGE_TITLE;
}

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

// What a visitor calls each kind of scan; the file's own two-letter code otherwise.
const KINDS = { CT: 'CT', MR: 'MRI', CR: 'X-ray', DX: 'X-ray', MG: 'Mammogram' };

/** A set's name in words: its own description from the disc, or "CT set 3" when it has none (round eleven
 * read "CT · untitled" as a label maker). */
function setName(info) {
  const kind = KINDS[info.modality] || info.modality || 'Scan';
  const desc = (info.seriesDescription || '').trim();
  if (desc) return { kind, title: desc, full: `${kind}: ${desc}` };
  const title = `${kind} set${Number.isFinite(info.seriesNumber) ? ` ${info.seriesNumber}` : ''}`;
  return { kind, title, full: title };
}

// A date from the disc ("2023-04-01") as a person writes it, or ''.
function niceDate(iso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso || '');
  if (!m) return '';
  const d = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3]));
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
}

// Why a set's pictures would not decode, as the end of a sentence about the set.
function unsupportedWhy(what) {
  if (what === 'no pixel data') return 'didn\'t open, because its files hold no pixel data.';
  if (what === 'truncated pixel data') return 'didn\'t open, because its pixel data is cut short.';
  if (what === 'a JPEG the browser refused') return 'didn\'t open, because this browser refused its JPEG pictures.';
  return `didn't open. Its pictures are stored as ${what}, a format the viewer can't read.`;
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
  if (!parsed.ds) return { unreadable: 'it\'s damaged past its first few kilobytes' };
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
  state.credit = credit;
  titleFor(credit);
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
    read = await expand(entries, (done, total, path, zip) => {
      // Inside a ZIP the count is of the files it holds, so the words never show a fraction.
      progress(zip ? `Unpacking ${zip.unpacked} of ${plural(zip.of, 'file')} from the ZIP…`
        : `Reading ${done} of ${plural(total, 'file')}…`, done, total);
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
    progress('Adding the outlines…', 0, 0);
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
      problem(f.path, `wouldn't open: ${String(err && err.message ? err.message : err).replace(/\.$/, '')}.`);
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
  if (header.notDicom) { problem(f.path, 'isn\'t a DICOM file, so it was skipped.'); return null; }
  if (header.broken) { problem(f.path, 'looks like a DICOM file but wouldn\'t open.'); return null; }
  const { ds } = header;
  const info = describe(ds);
  if (!info.rows || !info.cols) { problem(f.path, 'holds no picture. It may be a report or an index.'); return null; }
  return { ds, info };
}

// Order each series' slices and decode its first picture. Series that cannot be opened are reported.
async function buildSeries(bySeries, read) {
  progress('Sorting the pictures into sets…', read.files.length, read.files.length);
  const series = [];
  for (const s of bySeries.values()) {
    s.orderedBy = orderSlices(s.slices);
    const frame = await decodeSlice(s.slices[0]);
    if (frame.unreadable) {
      problem(s.slices[0].path, `wouldn't open: ${frame.unreadable}.`);
      continue;
    }
    if (frame.unsupported) {
      problem(`${setName(s.info).full} (${plural(s.slices.length, 'picture')})`, unsupportedWhy(frame.unsupported));
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
  const sets = state.series.length === 1 ? '1 set' : `${state.series.length} sets`;
  el('series-summary').textContent = state.series.length
    ? `${plural(total, 'picture')} in ${sets}, read from ${(read.bytes / 1e6).toFixed(0)} MB on this device.`
    : 'None of the pictures on this disc would open. Here\'s what went wrong.';
  state.series.forEach((s, i) => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'series-card';
    const canvas = document.createElement('canvas');
    canvas.width = 220;
    canvas.height = 220;
    paint(canvas, s.firstFrame, autoWindow(s.firstFrame), 1, 0, 0, s.info.modality === 'CT' ? padBox(bodyBox(s.firstFrame), s.firstFrame) : null);
    const name = setName(s.info);
    const title = document.createElement('div');
    title.className = 'title';
    title.textContent = name.title;
    const sub = document.createElement('div');
    sub.className = 'sub';
    // A sentence a person would say, with the pixel size in words: "CT, 120 pictures of 512 by 512 pixels, taken 3 Apr 2023".
    const taken = niceDate(s.info.acquisitionDate);
    sub.textContent = `${name.title === name.kind || name.title.startsWith(`${name.kind} set`) ? '' : `${name.kind}, `}`
      + `${plural(s.slices.length, 'picture')} of ${s.info.cols} by ${s.info.rows} pixels`
      + (taken ? `, taken ${taken}` : '');
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
  state.listChosen = -1;
  state.spotlight = -1;
  el('name-tip').hidden = true;
  const out = el('reading-out');
  out.replaceChildren();
  out.hidden = true;
  el('finding').hidden = true;
  el('finding').dataset.sig = '';
  state.palette = null;
  showOutlineTools(false);
}

/** The caption strip under the picture, "Hide the outlines" and "How these outlines are made" belong to a set
 * with outlines; on any other scan they would describe outlines the viewer does not draw. */
function showOutlineTools(on) {
  el('outline-tools').hidden = !on;
  el('stage-caption').hidden = !on;
}

/** "Show the outlines" is a switch: its words stay put and its state is on or off (wave three: as a third button
 * beside Previous and Next it read as one more equal choice). The caption strip says when they are off. */
function markSwitch() {
  el('toggle-marks').setAttribute('aria-checked', String(state.showMarks));
  el('stage-caption').classList.toggle('marks-off', !state.showMarks);
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
    readingProblem('Open the scan first, then add its outline file. The file holds the outlines but none of the pictures.');
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
    readingProblem('That outline file belongs to a different scan. Open the one it was made from, or choose another file.');
    return;
  }
  state.reading = doc;
  state.readingSeries = seriesIndex;
  state.outlines = outlines;
  state.palette = buildPalette(doc, outlines);
  state.showMarks = true;
  el('finding').dataset.sig = '';
  forgetPointing();
  renderReading();
  renderSeriesMarks();
  if (state.current === state.readingSeries) {
    showOutlineTools(true);
    markSwitch();
    draw();
  }
}

/** Forget what the visitor was pointing at: a new picture, series or reading has other outlines. */
function forgetPointing() {
  state.pointedAt = -1;
  state.chosen = -1;
  state.listChosen = -1;
  state.spotlight = -1;
  el('name-tip').hidden = true;
}

function renderReading() {
  const out = el('reading-out');
  out.replaceChildren();
  const names = (state.reading.outlines && state.reading.outlines.structures) || [];
  const p = document.createElement('p');
  p.className = 'reading-note';
  p.textContent = state.outlines.size
    ? `${names.length} structure${names.length === 1 ? '' : 's'} outlined across ${state.outlines.size} slice${state.outlines.size === 1 ? '' : 's'}. ${mouseDevice() ? 'Point at one' : 'Tap one'} to see its name.`
    : 'This file has no outlines for the slices open here.';
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
    flag.textContent = `Outlines on ${outlined} slice${outlined === 1 ? '' : 's'}`;
    cards[i].append(flag);
  }
}

// The colours an outline can take: OKLCH hues 10 degrees apart at four lightnesses, all light enough to read on
// a black picture, each pulled in to the screen's gamut. How different two of them look is CIEDE2000, worked
// out once for every pair.
let paletteChoices = null;

function oklchBytes(L, C, hue) {
  const h = (hue * Math.PI) / 180;
  for (let c = C; c >= 0; c -= 0.005) {
    const a = c * Math.cos(h);
    const b = c * Math.sin(h);
    const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
    const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
    const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
    const rgb = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
      -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s];
    if (rgb.every((v) => v >= -1e-4 && v <= 1 + 1e-4)) {
      return rgb.map((v) => Math.round(255 * (v <= 0.0031308 ? 12.92 * Math.max(v, 0) : 1.055 * Math.min(v, 1) ** (1 / 2.4) - 0.055)));
    }
  }
  return null;
}

function labOf(bytes) {
  const lin = (v) => { const c = v / 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const [r, g, b] = bytes.map(lin);
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const x = f((0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047);
  const y = f(0.2126 * r + 0.7152 * g + 0.0722 * b);
  const z = f((0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883);
  return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
}

function de2000([L1, a1, b1], [L2, a2, b2]) {
  const rad = Math.PI / 180;
  const Cb = (Math.hypot(a1, b1) + Math.hypot(a2, b2)) / 2;
  const G = 0.5 * (1 - Math.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)));
  const A1 = a1 * (1 + G);
  const A2 = a2 * (1 + G);
  const C1 = Math.hypot(A1, b1);
  const C2 = Math.hypot(A2, b2);
  const h1 = ((Math.atan2(b1, A1) / rad) + 360) % 360;
  const h2 = ((Math.atan2(b2, A2) / rad) + 360) % 360;
  let dh = h2 - h1;
  if (C1 * C2 === 0) dh = 0;
  else if (dh > 180) dh -= 360;
  else if (dh < -180) dh += 360;
  const dH = 2 * Math.sqrt(C1 * C2) * Math.sin((dh / 2) * rad);
  const Lm = (L1 + L2) / 2;
  const Cm = (C1 + C2) / 2;
  let hm = h1 + h2;
  if (C1 * C2 !== 0) hm = Math.abs(h1 - h2) <= 180 ? hm / 2 : (hm < 360 ? (hm + 360) / 2 : (hm - 360) / 2);
  const T = 1 - 0.17 * Math.cos((hm - 30) * rad) + 0.24 * Math.cos(2 * hm * rad) + 0.32 * Math.cos((3 * hm + 6) * rad)
    - 0.2 * Math.cos((4 * hm - 63) * rad);
  const Rt = -Math.sin(2 * 30 * Math.exp(-(((hm - 275) / 25) ** 2)) * rad) * 2 * Math.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7));
  const Sl = 1 + (0.015 * (Lm - 50) ** 2) / Math.sqrt(20 + (Lm - 50) ** 2);
  const Sc = 1 + 0.045 * Cm;
  const Sh = 1 + 0.015 * Cm * T;
  const dl = (L2 - L1) / Sl;
  const dc = (C2 - C1) / Sc;
  const dhh = dH / Sh;
  return Math.sqrt(dl * dl + dc * dc + dhh * dhh + Rt * dc * dhh);
}

function choices() {
  if (!paletteChoices) {
    const list = [];
    for (const L of [0.62, 0.72, 0.82, 0.92]) {
      for (let hue = 0; hue < 360; hue += 10) {
        const bytes = oklchBytes(L, 0.2, hue);
        if (bytes) list.push({ css: `rgb(${bytes.join(' ')})`, lab: labOf(bytes) });
      }
    }
    const gap = new Float32Array(list.length * list.length);
    for (let i = 0; i < list.length; i++) {
      for (let j = i + 1; j < list.length; j++) gap[i * list.length + j] = gap[j * list.length + i] = de2000(list[i].lab, list[j].lab);
    }
    paletteChoices = { list, gap };
  }
  return paletteChoices;
}

/** One colour per structure, the same on every slice, chosen so that structures drawn on the same slice look
 * as different as they can (round eleven measured the spleen and left rib 7 at a distance of 4, and the old
 * golden-angle hues gave a rib and the spinal cord the same green). Structures that share a slice are
 * neighbours; the busiest are coloured first, each taking the choice furthest from its coloured neighbours,
 * then every one is chosen again four times over. On the sample CT no two neighbours end up closer than 13. */
function buildPalette(doc, outlines) {
  const count = ((doc.outlines && doc.outlines.structures) || []).length;
  const near = Array.from({ length: count }, () => new Set());
  for (const paths of outlines.values()) {
    const here = [...new Set(paths.map(([s]) => s))].filter((s) => s < count);
    for (const a of here) for (const b of here) if (a !== b) near[a].add(b);
  }
  const { list, gap } = choices();
  const k = list.length;
  const colour = new Int16Array(count).fill(-1);
  const pick = (s) => {
    let best = 0;
    let bestGap = -1;
    for (let o = 0; o < k; o++) {
      let g = Infinity;
      for (const t of near[s]) if (colour[t] >= 0) g = Math.min(g, gap[o * k + colour[t]]);
      if (g > bestGap + 1e-9) { best = o; bestGap = g; }
    }
    return best;
  };
  const order = [...Array(count).keys()].sort((a, b) => near[b].size - near[a].size || a - b);
  for (const s of order) colour[s] = pick(s);
  for (let pass = 0; pass < 4; pass++) for (const s of order) { colour[s] = -1; colour[s] = pick(s); }
  return Array.from(colour, (o) => (o >= 0 ? list[o].css : null));
}

/** A colour for structure `s`, the same on every slice. */
function colourOf(s) {
  if (state.palette && state.palette[s]) return state.palette[s];
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

/** Name the structure under a point on the picture. A mouse passing over the picture only previews a name, and
 * the chosen one comes back once it is off every outline. A tap or a click chooses what is under it, the same as
 * tapping its row, and a tap on no outline lets the choice go. A finger covers what is under and beside it, so a
 * tap also takes the nearest outline within a fingertip. */
function pointAt(clientX, clientY, above = false, choose = false) {
  const tip = el('name-tip');
  const v = state.view;
  if (!v || !state.reading || !state.reading.outlines) { tip.hidden = true; return; }
  const rect = el('canvas').getBoundingClientRect();
  const x = ((clientX - rect.left) * v.dpr - v.x0) / v.fit;
  const y = ((clientY - rect.top) * v.dpr - v.y0) / v.fit;
  let s = structureAt(x, y);
  if (s < 0 && above) s = nearestStructure(x, y, (FINGER_PX * v.dpr) / v.fit);
  if (choose) {
    if (s < 0) { forgetPointing(); draw(); return; }
    if (s !== state.chosen) { state.pointedAt = s; state.chosen = s; state.listChosen = -1; state.spotlight = -1; draw(); }
    nameTip(s);
    return;
  }
  const shown = s >= 0 ? s : state.chosen;
  if (shown !== state.pointedAt) { state.pointedAt = shown; state.spotlight = s >= 0 ? -1 : state.listChosen; draw(); }
  if (shown < 0) { tip.hidden = true; return; }
  nameTip(shown);
}

/** Write structure `s`'s name (and what it is) into the caption strip under the picture. */
function nameTip(s) {
  const tip = el('name-tip');
  // With the outlines switched off nothing is named; the choice is kept for when they come back.
  if (!state.showMarks) { tip.hidden = true; return; }
  const o = state.reading.outlines;
  const name = document.createElement('strong');
  const text = o.structures[s] || `structure ${s + 1}`;
  name.textContent = text.charAt(0).toUpperCase() + text.slice(1);      // as its row in the list writes it
  tip.replaceChildren(name);
  if (o.about && o.about[s]) tip.append(document.createTextNode(`: ${o.about[s]}`));
  tip.hidden = false;
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
function namePart(s, fromList = false) {
  if (RIB.test((state.reading?.outlines?.structures || [])[s] || '')) state.ribsOpen = true;
  const v = state.view;
  if (!v || !state.reading || !state.reading.outlines) return;
  const at = interiorPoint(s);
  if (!at) return;
  state.pointedAt = s;
  state.chosen = s;
  state.listChosen = fromList ? s : -1;
  state.spotlight = state.listChosen;
  draw();
  nameTip(s);
  pulsePart();
}

/** A tap on a row: the outline swells for a moment, so the eye finds it on the sticky picture. */
function pulsePart() {
  state.pulse = true;
  draw();
  clearTimeout(state.pulseTimer);
  state.pulseTimer = setTimeout(() => { state.pulse = false; draw(); }, 1100);
}

/** A tap on a row of the list: name that structure on the picture. Tapping the chosen row again keeps it chosen
 * and swells its outline once more, so the tap is answered (wave three: it used to let go without a word and
 * leave the caption empty). */
function choosePart(s) {
  if (state.chosen === s) {
    state.pointedAt = s;
    state.listChosen = s;
    state.spotlight = s;
    nameTip(s);
    pulsePart();
    return;
  }
  namePart(s, true);
}

/** After a step to another slice, the chosen structure stays chosen while it is outlined there, so a visitor can
 * follow one organ through the scan. Otherwise the choice is let go and the caption says how to name one. */
function followChoice() {
  const s = state.chosen;
  const here = s >= 0 && state.current === state.readingSeries && (state.outlines.get(state.slice) || []).some(([k]) => k === s);
  if (!here) { forgetPointing(); return; }
  state.pointedAt = s;
  state.spotlight = state.listChosen;
  nameTip(s);
}

const RIB = /^(left|right) rib (\d+)$/;
const pathCache = new Map();               // path text -> its points, decoded once

function outlinesForSlice() {
  if (state.current !== state.readingSeries || !state.showMarks) return [];
  return state.outlines.get(state.slice) || [];
}

/** Draw a labeller's outlines over the picture, in the picture's own pixels, and return which
 * structures were drawn. Lines rather than filled shapes: the picture underneath stays readable,
 * and an outline says "the model put this structure here" without pretending to be the organ. */
function overlayOutlines(canvas, view, paths, dpr) {
  if (!paths.length) return [];
  const ctx = canvas.getContext('2d');
  const { x0, y0, fit } = view;
  const drawn = new Set();
  ctx.save();
  ctx.lineJoin = 'round';
  // The one pointed at is drawn last, on top. While a name in the list is pointed at or tapped, the other
  // outlines step back, so it finds its own outline even where two colours are alike.
  const focus = state.pointedAt;
  const ordered = focus < 0 ? paths : [...paths.filter(([s]) => s !== focus), ...paths.filter(([s]) => s === focus)];
  for (const [s, text] of ordered) {
    const p = pointsOf(text);
    if (p.length < 4) continue;
    const pointed = s === focus;
    ctx.globalAlpha = state.spotlight >= 0 && state.spotlight === focus && !pointed ? 0.35 : 1;
    const width = Math.max(1, (pointed ? (state.pulse && s === state.chosen ? 5 : 3) : 1.25) * dpr);
    ctx.beginPath();
    ctx.moveTo(x0 + p[0] * fit, y0 + p[1] * fit);
    for (let i = 2; i < p.length; i += 2) ctx.lineTo(x0 + p[i] * fit, y0 + p[i + 1] * fit);
    ctx.closePath();
    if (pointed) {
      // A light ring under the coloured stroke so the chosen part stands out on the grey picture.
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = width + 4 * dpr;
      ctx.stroke();
    }
    ctx.strokeStyle = colourOf(s);
    ctx.lineWidth = width;
    ctx.stroke();
    if (pointed) {
      ctx.globalAlpha = state.pulse && s === state.chosen ? 0.5 : 0.25;
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
    note.textContent = state.showMarks ? 'No outlines on this slice.'
      : 'The outlines are hidden. Turn on Show the outlines to bring them back.';
    return;
  }
  const sig = drawn.join(',');
  if (note.dataset.sig !== sig) {
    note.dataset.sig = sig;
    note.replaceChildren();
    const how = document.createElement('p');
    how.textContent = `${mouseDevice() ? 'Click' : 'Tap'} a name to light it up on the slice.`;
    const all = document.createElement('details');
    all.className = 'outline-all';
    all.open = state.keyOpen;
    all.addEventListener('toggle', () => { state.keyOpen = all.open; });
    const summary = document.createElement('summary');
    summary.textContent = drawn.length === 1 ? 'Named on this slice (1)'
      : `Named on this slice (${drawn.length})`;
    const list = document.createElement('ul');
    list.className = 'parts';
    const about = state.reading.outlines.about;
    const makeRow = (s) => {
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
      // A mouse over a row previews it on the picture and in the caption, so the two always name the same
      // structure; off the row, the chosen one comes back.
      row.addEventListener('pointerenter', (ev) => {
        if (ev.pointerType === 'touch') return;
        state.pointedAt = s;
        state.spotlight = s;
        nameTip(s);
        draw();
      });
      row.addEventListener('pointerleave', (ev) => {
        if (ev.pointerType === 'touch') return;
        state.pointedAt = state.chosen;
        state.spotlight = state.listChosen;
        if (state.chosen >= 0) nameTip(state.chosen);
        else el('name-tip').hidden = true;
        draw();
      });
      item.append(row);
      return item;
    };
    // The eight or more rib rows become one, "Ribs (8)", that opens to show each rib.
    // Left ribs then right ribs, each in number order (7, 8, 9, 10), not text order.
    const ribKey = (s) => {
      const m = RIB.exec(names[s] || '');
      return m ? (m[1] === 'left' ? 0 : 1000) + Number(m[2]) : 0;
    };
    const ribs = drawn.filter((s) => RIB.test(names[s] || '')).sort((a, b) => ribKey(a) - ribKey(b));
    const grouped = ribs.length > 1;
    let ribsAdded = false;
    for (const s of drawn) {
      if (grouped && ribs.includes(s)) {
        if (ribsAdded) continue;
        ribsAdded = true;
        const item = document.createElement('li');
        const head = document.createElement('button');
        head.type = 'button';
        head.className = 'part-row rib-group';
        head.setAttribute('aria-expanded', String(state.ribsOpen));
        // Built like every other row: a swatch (a strip of the ribs' own colours), the name, and the page's
        // disclosure chevron at the end, as on "Brightness and contrast" (wave three: a lone chevron in the
        // swatch column read as a different kind of row).
        const swatch = document.createElement('span');
        swatch.className = 'outline-swatch rib-swatch';
        const stripes = ribs.slice(0, 4).map(colourOf);
        swatch.style.backgroundImage = `linear-gradient(90deg, ${stripes.map((c, i) => `${c} ${(i * 100) / stripes.length}% ${((i + 1) * 100) / stripes.length}%`).join(', ')})`;
        const text = document.createElement('span');
        text.className = 'part-name';
        text.textContent = `Ribs (${ribs.length})`;
        const chev = document.createElement('span');
        chev.className = 'rib-chevron';
        chev.setAttribute('aria-hidden', 'true');
        head.append(swatch, text, chev);
        const sub = document.createElement('ul');
        sub.className = 'parts rib-parts';
        sub.hidden = !state.ribsOpen;
        for (const r of ribs) sub.append(makeRow(r));
        head.addEventListener('click', () => {
          state.ribsOpen = !state.ribsOpen;
          head.setAttribute('aria-expanded', String(state.ribsOpen));
          sub.hidden = !state.ribsOpen;
        });
        item.append(head, sub);
        list.append(item);
      } else {
        list.append(makeRow(s));
      }
    }
    all.append(summary, list);
    note.append(how, all);
  }
  for (const row of note.querySelectorAll('.part-row:not(.rib-group)')) {
    const s = Number(row.dataset.part);
    row.classList.toggle('is-pointed', s === state.pointedAt);
    row.setAttribute('aria-pressed', String(s === state.chosen));
  }
  const group = note.querySelector('.rib-group');
  if (group) {
    group.setAttribute('aria-expanded', String(state.ribsOpen));
    group.nextElementSibling.hidden = !state.ribsOpen;
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
  // The sample is named for what it is; a visitor's set by its own description from the disc.
  el('viewer-h').textContent = (state.credit && state.credit.title) || setName(s.info).full;
  el('slice').max = String(s.slices.length - 1);
  el('slice').value = String(state.slice);
  const frame = await frameAt(index, state.slice);
  if (s.box === undefined) {
    const box = await setBox(index);
    if (box !== undefined) s.box = box;
  }
  // The frame takes the shape of what is drawn in it (the body on a CT, the whole picture otherwise), so the
  // picture fills it instead of sitting in black bars (wave three, on a phone held either way and on a computer).
  const shape = s.box ? s.box.w / s.box.h : frame.width / frame.height;
  el('viewer').style.setProperty('--pic-ratio', String(Math.min(1.8, Math.max(0.6, shape || 1)).toFixed(4)));
  const w = (Number.isFinite(s.info.windowCentre) && Number.isFinite(s.info.windowWidth) && s.info.windowWidth > 0)
    ? { centre: s.info.windowCentre, width: s.info.windowWidth }
    : autoWindow(frame);
  state.centre = w.centre;
  state.width = w.width;
  syncWindowSliders(frame);
  showOutlineTools(Boolean(state.reading && index === state.readingSeries));
  markSwitch();
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

/** Paint `frame` into `canvas`, fitted to `box` (the part of the picture to fill the canvas with: the body, on a
 * CT) or to the whole picture, then zoomed and moved. Returns where it went: image pixel (0, 0) lands at
 * (x0, y0) on the canvas, and one image pixel is `fit` canvas pixels. */
function paint(canvas, frame, window, zoom, panX, panY, box = null) {
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
  const b = box || { x: 0, y: 0, w: frame.width, h: frame.height };
  const fit = Math.min(canvas.width / b.w, canvas.height / b.h) * zoom;
  const x0 = canvas.width / 2 + panX - (b.x + b.w / 2) * fit;
  const y0 = canvas.height / 2 + panY - (b.y + b.h / 2) * fit;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, x0, y0, frame.width * fit, frame.height * fit);
  ctx.restore();
  return { fit, x0, y0 };
}

// Round eleven: on a phone the body filled 62% of the black box, so small outlines were hard to tap. A CT
// opens fitted to the body instead. The body is the largest patch denser than air (-500 HU) on a coarse grid;
// the table under the patient is a patch of its own and is left out.
function bodyBox(frame) {
  if (!frame || !frame.raw || !frame.width || !frame.height) return null;
  const { width: w, height: h, raw, slope, intercept } = frame;
  const step = Math.max(1, Math.ceil(Math.max(w, h) / 128));
  const gw = Math.ceil(w / step);
  const gh = Math.ceil(h / step);
  const dense = new Uint8Array(gw * gh);
  for (let gy = 0; gy < gh; gy++) {
    for (let gx = 0; gx < gw; gx++) dense[gy * gw + gx] = raw[gy * step * w + gx * step] * slope + intercept > -500 ? 1 : 0;
  }
  const seen = new Uint8Array(gw * gh);
  let best = null;
  const stack = [];
  for (let i = 0; i < dense.length; i++) {
    if (!dense[i] || seen[i]) continue;
    const patch = { n: 0, x0: gw, y0: gh, x1: -1, y1: -1 };
    seen[i] = 1;
    stack.push(i);
    while (stack.length) {
      const k = stack.pop();
      const x = k % gw;
      const y = (k - x) / gw;
      patch.n += 1;
      if (x < patch.x0) patch.x0 = x;
      if (x > patch.x1) patch.x1 = x;
      if (y < patch.y0) patch.y0 = y;
      if (y > patch.y1) patch.y1 = y;
      for (const j of [x > 0 ? k - 1 : -1, x < gw - 1 ? k + 1 : -1, y > 0 ? k - gw : -1, y < gh - 1 ? k + gw : -1]) {
        if (j >= 0 && dense[j] && !seen[j]) { seen[j] = 1; stack.push(j); }
      }
    }
    if (!best || patch.n > best.n) best = patch;
  }
  if (!best || best.n < dense.length * 0.02) return null;
  return { x: best.x0 * step, y: best.y0 * step, w: (best.x1 - best.x0 + 1) * step, h: (best.y1 - best.y0 + 1) * step };
}

/** Several body boxes as one, with a margin, inside the picture; null when the body already fills most of it. */
function padBox(boxes, frame) {
  const list = (Array.isArray(boxes) ? boxes : [boxes]).filter(Boolean);
  if (!list.length || !frame) return null;
  let x0 = Math.min(...list.map((b) => b.x));
  let y0 = Math.min(...list.map((b) => b.y));
  let x1 = Math.max(...list.map((b) => b.x + b.w));
  let y1 = Math.max(...list.map((b) => b.y + b.h));
  const pad = 0.05 * Math.max(x1 - x0, y1 - y0);
  x0 = Math.max(0, x0 - pad); y0 = Math.max(0, y0 - pad);
  x1 = Math.min(frame.width, x1 + pad); y1 = Math.min(frame.height, y1 + pad);
  if ((x1 - x0) * (y1 - y0) > 0.8 * frame.width * frame.height) return null;
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

/** Where the body is across a CT set: its first, middle and last pictures, so the shoulders or hips at the ends
 * stay in view as the visitor steps through. Null for anything but a CT. */
async function setBox(index) {
  const s = state.series[index];
  if (!s || s.info.modality !== 'CT') return null;
  const last = s.slices.length - 1;
  const frames = [s.firstFrame, await frameAt(index, Math.floor(last / 2)), await frameAt(index, last)];
  if (state.current !== index) return undefined;               // the visitor opened another set meanwhile
  return padBox(frames.map(bodyBox), s.firstFrame);
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
    const view = paint(canvas, frame, { centre: state.centre, width: state.width }, state.zoom, state.panX * dpr, state.panY * dpr, s.box);
    const { fit } = view;
    state.view = { ...view, dpr };
    const drawn = overlayOutlines(canvas, view, outlinesForSlice(), dpr);
    describeImage(el('finding'), drawn);
    describeView(s.info);
    // Which picture, and nothing else. The brightness and contrast numbers and the sort order were
    // on here too; they are a radiographer's units, and a visitor read them as things to understand.
    // The how-to is a chip under the picture, never on it, and it goes after the first change.
    el('hud').textContent = `${state.slice + 1} of ${s.slices.length}`;
    // The slider is what a screen reader hears on each move, so it says the picture and the outline count.
    const parts = el('finding').hidden || !drawn.length ? '' : `, ${drawn.length} structure${drawn.length === 1 ? '' : 's'} outlined`;
    el('slice').setAttribute('aria-valuetext', `Picture ${state.slice + 1} of ${s.slices.length}${parts}`);
    // How far through the set, for the filled part of the slider's track (open.css).
    el('slice').style.setProperty('--fill', `${s.slices.length > 1 ? (state.slice / (s.slices.length - 1)) * 100 : 0}%`);
    // The one line that says how to step and zoom. It stays: it is the only place that says so (wave three found
    // a second, different instruction further down, naming the arrow keys on a phone).
    el('hint-chip').hidden = el('hint-chip-side').hidden = s.slices.length < 2;
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
  CT: 'On a CT, white is bone, grey is organs and muscle, and black is air.',
  CR: 'On an X-ray, white is bone and anything dense, and black is air.',
  DX: 'On an X-ray, white is bone and anything dense, and black is air.',
  MR: 'On an MRI, what looks bright depends on how the scan was taken, so there\'s no simple rule for white and black.',
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
        ? 'This slice cuts across the body, and you\'re looking up at it from the feet.'
        : 'This slice cuts across the body.');
    } else if (axes === '02') {
      const facing = across.axis !== 0 ? '' : across.away === SIDES[0][0] ? ', seen from the front' : ', seen from behind';
      parts.push(`This slice runs from side to side through the body${facing}.`);
    } else {
      parts.push('This slice runs from front to back through the body, seen from the side.');
    }
    parts.push('The labels at the edges say which side is which.');
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
    followChoice();
  }
  if (byVisitor) state.moved = true;
  el('slice').value = String(state.slice);
  draw();
}

/** A tap that did not move is a question: what is this? */
function tapped(ev, drag) {
  if (drag && state.slice === drag.slice && Math.hypot(ev.clientX - drag.x, ev.clientY - drag.y) < 6) {
    pointAt(ev.clientX, ev.clientY, ev.pointerType !== 'mouse', true);
  }
}

/** A mouse over the picture names the outlined structure under it; leaving the picture forgets it. */
function wireNaming(stage, dragging) {
  stage.addEventListener('pointermove', (ev) => {
    if (ev.pointerType === 'mouse' && !dragging()) pointAt(ev.clientX, ev.clientY);
  });
  // Off the picture, the preview goes and the chosen structure, if there is one, is named again.
  stage.addEventListener('pointerleave', (ev) => {
    if (ev.pointerType !== 'mouse' || dragging()) return;
    const s = state.chosen;
    if (state.pointedAt !== s) { state.pointedAt = s; state.spotlight = state.listChosen; draw(); }
    if (s >= 0 && state.reading && state.reading.outlines) nameTip(s);
    else el('name-tip').hidden = true;
  });
}

const ZOOM_MIN = 0.5;
const ZOOM_MAX = 12;

/** Zoom by `factor` about a point on the screen, so what is under the pointer (or between two fingers) stays put. */
function zoomAbout(clientX, clientY, factor, from = { zoom: state.zoom, panX: state.panX, panY: state.panY }) {
  const rect = el('canvas').getBoundingClientRect();
  const px = clientX - rect.left - rect.width / 2;
  const py = clientY - rect.top - rect.height / 2;
  const zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, from.zoom * factor));
  const k = zoom / from.zoom;
  state.zoom = zoom;
  state.panX = px - (px - from.panX) * k;
  state.panY = py - (py - from.panY) * k;
}

function wireStage() {
  const stage = el('stage');
  if (mouseDevice()) {
    el('canvas').setAttribute('aria-label', 'A picture from the set you chose. Drag up or down to step through them, or sideways to change the brightness.');
  }
  let drag = null;
  // Round eleven: "Scroll or pinch to zoom in", but on a touch screen a pinch stepped through the slices. Two
  // fingers on the picture now zoom about the point between them and move it with them.
  const fingers = new Map();
  let pinch = null;
  const spread = () => {
    const [a, b] = [...fingers.values()];
    return { d: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)), x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  };
  stage.addEventListener('pointerdown', (ev) => {
    stage.setPointerCapture(ev.pointerId);
    fingers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    ev.preventDefault();
    if (fingers.size === 2) {
      // The second finger makes it a pinch: whatever the first one started is put back.
      if (drag) {
        state.centre = drag.centre;
        state.width = drag.width;
        syncOnlyValues();
        if (state.slice !== drag.slice) setSlice(drag.slice, false);
      }
      drag = null;
      pinch = { ...spread(), zoom: state.zoom, panX: state.panX, panY: state.panY };
      return;
    }
    if (fingers.size > 2 || pinch) return;
    drag = { x: ev.clientX, y: ev.clientY, slice: state.slice, centre: state.centre, width: state.width,
             panX: state.panX, panY: state.panY, pan: ev.button === 1 || ev.shiftKey };
  });
  stage.addEventListener('pointermove', (ev) => {
    if (fingers.has(ev.pointerId)) fingers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    if (pinch) {
      if (fingers.size < 2) return;
      const now = spread();
      zoomAbout(pinch.x, pinch.y, now.d / pinch.d, pinch);
      state.panX += now.x - pinch.x;
      state.panY += now.y - pinch.y;
      draw();
      return;
    }
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
  // A pinch ends when its fingers lift; one finger left behind does nothing until it lifts too.
  const lift = (ev) => {
    fingers.delete(ev.pointerId);
    if (!fingers.size) pinch = null;
  };
  stage.addEventListener('pointerup', (ev) => { if (!pinch) tapped(ev, drag); drag = null; lift(ev); });
  stage.addEventListener('pointercancel', (ev) => { drag = null; lift(ev); });
  wireNaming(stage, () => drag !== null || pinch !== null);
  stage.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    zoomAbout(ev.clientX, ev.clientY, ev.deltaY < 0 ? 1.12 : 1 / 1.12);
    draw();
  }, { passive: false });
  // A double click or a double tap puts the picture back to its first size and place.
  stage.addEventListener('dblclick', () => {
    if (state.zoom === 1 && !state.panX && !state.panY) return;
    state.zoom = 1; state.panX = 0; state.panY = 0;
    draw();
  });
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
    ? 'Your browser will ask first. Saying yes only lets this page read the files, here on this device.'
    : 'Your browser may ask "Upload files to this site?". That\'s its own wording for letting a page read the files. Nothing is sent: you can switch your Wi-Fi off first, and it still works.';
  // The numbered step is a step of its own, after the window in step 3, so it says the question in full.
  const step = viewPicker
    ? 'Your browser will ask whether this page may read the files. Say yes. They stay on this computer.'
    : 'Your browser may ask "Upload files to this site?". That\'s its wording for letting this page read them. Say yes. They stay on this computer.';
  for (const line of document.querySelectorAll('.browser-asks')) line.textContent = line.tagName === 'LI' ? step : asks;
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
    help.textContent = 'A window has opened so you can choose the disc or folder. If you can\'t see it, check behind this one.';
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
    markSwitch();
    if (state.showMarks && state.chosen >= 0) nameTip(state.chosen);
    else el('name-tip').hidden = true;
    draw();
  });
  el('cancel').addEventListener('click', () => { state.cancelled = true; show('progress', false); });
  const startOver = () => {
    scanOpened(null);
    state.series = []; state.problems = []; state.current = null; forgetFrames();
    state.credit = null;
    titleFor(null);
    clearReading();
    show('series', false); show('viewer', false); show('problems', false); show('progress', false);
    show('intake', true);
    el('start-over').hidden = true;
    // A reload after leaving the sample must show the intake, so the address drops ?sample=1.
    if (new URLSearchParams(window.location.search).has('sample')) {
      window.history.replaceState(null, '', window.location.pathname + window.location.hash);
    }
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
    problem('This browser', 'can\'t unzip files, so a ZIP won\'t open here. A folder still works, and any up-to-date browser can do both.');
    renderProblems();
  }
  if (new URLSearchParams(window.location.search).has('reading')) el('reading-box').hidden = false;
}

start();

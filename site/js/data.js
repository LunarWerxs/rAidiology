// data.js: loading + sanitising the report, and small shared helpers.
// Everything the page shows comes from data/ (see data/CONTRACT.md).

export const SEVERITY = {
  normal: { key: 'normal', label: 'Normal', color: '#7fbf9f', rank: 0 },
  'normal-variant': { key: 'normal-variant', label: 'Normal variant', color: '#7fbf9f', rank: 1 },
  incidental: { key: 'incidental', label: 'Incidental', color: '#a3a9b5', rank: 2 },
  mild: { key: 'mild', label: 'Mild', color: '#d9a441', rank: 3 },
  moderate: { key: 'moderate', label: 'Moderate', color: '#e07a3a', rank: 4 },
  severe: { key: 'severe', label: 'Severe', color: '#d64545', rank: 5 },
};

export const STATUS = {
  ok: { label: 'Looks OK', color: '#7fbf9f' },
  watch: { label: 'Worth watching', color: '#d9a441' },
  'see-doctor': { label: 'See a doctor', color: '#d64545' },
};

export const GRADE = { none: 'None', mild: 'Mild', moderate: 'Moderate', severe: 'Severe' };

export function sev(key) {
  return SEVERITY[key === 'none' ? 'normal' : key] || SEVERITY.normal;
}

export const LEVEL_ORDER = [
  ...Array.from({ length: 7 }, (_, i) => `C${i + 1}`),
  ...Array.from({ length: 12 }, (_, i) => `T${i + 1}`),
  ...Array.from({ length: 5 }, (_, i) => `L${i + 1}`),
  'S1', 'CO',
];

/** "disc_L4_L5" -> "L4-L5", "l4/5" -> "L4-L5", "Sacrum" -> "S1". */
export function normLevel(s) {
  let t = String(s ?? '').trim().toUpperCase()
    .replace(/[\s_–—\/]+/g, '-')
    .replace(/^DISC-?/, '')
    .replace(/-+/g, '-');
  t = t.replace(/\bSACRUM\b/, 'S1').replace(/\bCOCCYX\b/, 'CO');
  if (t === 'S') t = 'S1';
  const short = t.match(/^([CTL])(\d+)-(\d+)$/);
  if (short) t = `${short[1]}${short[2]}-${short[1]}${short[3]}`;
  return t;
}

/** "C5-C6, C6-C7" -> ["C5-C6", "C6-C7"]; a finding may name several levels. */
export function levelList(s) {
  return String(s ?? '').split(/[,;&]|\band\b/i).map(normLevel).filter(Boolean);
}

/** Worst severity per level, from report.levels and from findings at exactly that level. The 3D tints discs with this. */
export function levelStatusMap(report) {
  const map = new Map();
  const rank = (s) => sev(s).rank;
  for (const l of report?.levels || []) map.set(normLevel(l.level), l.status);
  for (const f of report?.findings || []) {
    for (const n of levelList(f.level)) {
      if (rank(f.severity) >= 2 && (!map.has(n) || rank(f.severity) > rank(map.get(n)))) map.set(n, f.severity);
    }
  }
  return map;
}

export function levelIndex(s) {
  const parts = (levelList(s)[0] || '').split('-');
  const idx = parts.map((p) => LEVEL_ORDER.indexOf(p));
  if (!parts.length || idx.some((i) => i < 0)) return 999;
  return idx.reduce((a, b) => a + b, 0) / idx.length;
}

export function isDiscLevel(s) {
  const parts = normLevel(s).split('-');
  if (parts.length !== 2) return false;
  const [a, b] = parts.map((p) => LEVEL_ORDER.indexOf(p));
  return a >= 0 && b === a + 1;
}

const NOTHING = new Set(['', '-', '–', '—', 'normal', 'not mentioned']);

/**
 * Discs the outside radiologist flagged: comparison rows whose level is exactly one disc level and
 * whose radiologist_short says something. Rows on the same level merge. [{ level, text }]
 */
export function reportFlaggedDiscs(report) {
  const byLevel = new Map();
  for (const r of report?.comparison?.rows || []) {
    const parts = String(r.level ?? '').split(/[,;&]|\band\b/i);
    const text = String(r.radiologist_short ?? '').trim();
    if (parts.length !== 1 || !isDiscLevel(r.level) || NOTHING.has(text.toLowerCase())) continue;
    const level = normLevel(r.level);
    const texts = byLevel.get(level) || [];
    if (!texts.includes(text)) texts.push(text);
    byLevel.set(level, texts);
  }
  return [...byLevel].map(([level, texts]) => ({ level, text: texts.join('; ') })).sort((a, b) => compareLevels(a.level, b.level));
}

export function compareLevels(a, b) {
  return levelIndex(a) - levelIndex(b);
}

export function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;
    else if (k === 'style' && typeof v === 'object') {
      for (const [sk, sv] of Object.entries(v)) {
        if (sk.startsWith('--')) n.style.setProperty(sk, sv);
        else n.style[sk] = sv;
      }
    } else if (k === 'dataset') Object.assign(n.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? '' : String(v));
  }
  append(n, kids);
  return n;
}

export function svgEl(tag, attrs = {}, ...kids) {
  const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, String(v));
  }
  append(n, kids);
  return n;
}

function append(n, kids) {
  for (const c of kids.flat(Infinity)) {
    if (c == null || c === false || c === '') continue;
    n.append(c instanceof Node ? c : String(c));
  }
}

/** Inline stroke icon from a list of SVG path strings. */
export function icon(paths, cls = '') {
  return svgEl('svg', { viewBox: '0 0 24 24', 'aria-hidden': 'true', class: cls || null },
    paths.map((d) => svgEl('path', { d })));
}

export function firstSentence(text) {
  const t = String(text || '').trim();
  const m = t.match(/^(.+?[.!?])(\s|$)/);
  return m ? m[1] : t;
}

export function fmtMonth(month) {
  const m = String(month || '').match(/^(\d{4})-(\d{2})/);
  if (!m) return String(month || '');
  const d = new Date(Date.UTC(+m[1], +m[2] - 1, 1));
  // undefined, not a fixed country: the date should read the way the person's own device
  // writes dates, which is not the same country for every reader.
  return d.toLocaleDateString(undefined, { month: 'short', year: 'numeric', timeZone: 'UTC' });
}

export function sexLabel(s) {
  return { M: 'Male', F: 'Female', X: 'Other' }[s] || '';
}

export async function loadJSON(url) {
  try {
    const res = await fetch(url, { cache: 'no-cache' });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

/** data/report.json, falling back to the clearly labelled sample. */
export async function loadReport() {
  const real = await loadJSON('data/report.json');
  if (real && typeof real === 'object') return { report: sanitizeReport(real), sample: false };
  const demo = await loadJSON('data/report.sample.json');
  if (demo && typeof demo === 'object') return { report: sanitizeReport(demo), sample: true };
  return { report: null, sample: false };
}

export async function loadStacks(sample) {
  // The sample report's marks only line up with the sample images, and no real scan should appear next to it.
  if (sample) return sampleStacks();
  const idx = await loadJSON('data/stacks/index.json');
  const list = Array.isArray(idx) ? idx.map(sanitizeStack).filter(Boolean) : [];
  return list;
}

// Privacy gate: of the patient block only label, age and sex survive sanitising.
const arr = (x) => (Array.isArray(x) ? x : []);
const str = (x) => (typeof x === 'string' ? x : x == null ? '' : String(x));
const oneOf = (x, list, dflt) => (list.includes(x) ? x : dflt);
const relUrl = (p) => str(p).trim().replace(/^[a-z]+:\/\/[^/]*\//i, '').replace(/^\/+/, '');
const LEVEL_STATUS = ['normal', 'mild', 'moderate', 'severe'];
const GRADES = ['none', 'mild', 'moderate', 'severe'];

export function sanitizeReport(raw) {
  const r = raw && typeof raw === 'object' ? raw : {};
  const p = r.patient || {};
  const s = r.study || {};
  const sum = r.summary || {};
  const m = r.method || {};
  const age = p.age == null || p.age === '' ? null : Number(p.age);
  return {
    version: r.version ?? 1,
    patient: {
      label: str(p.label).trim() || 'Anonymous',
      age: Number.isFinite(age) ? Math.round(age) : null,
      sex: oneOf(str(p.sex).toUpperCase(), ['M', 'F', 'X'], ''),
    },
    study: {
      month: str(s.month),
      field_strength: str(s.field_strength),
      contrast: !!s.contrast,
      exams: arr(s.exams).map(str).filter(Boolean),
    },
    summary: {
      status: STATUS[sum.status] ? sum.status : 'watch',
      headline: str(sum.headline),
      plain: str(sum.plain),
      technical: str(sum.technical),
    },
    regions: arr(r.regions).filter((g) => g && g.id).map((g) => ({
      id: str(g.id),
      name: str(g.name) || str(g.id),
      anchor: str(g.anchor),
      levels_present: arr(g.levels_present).map(str),
      summary: str(g.summary),
      meshes: arr(g.meshes).filter((x) => x && x.file).map((x) => ({
        name: str(x.name),
        kind: str(x.kind) || 'vertebra',
        level: str(x.level),
        file: relUrl(x.file),
        status: oneOf(x.status, LEVEL_STATUS, 'normal'),
      })),
    })),
    levels: arr(r.levels).filter((l) => l && l.level).map((l) => ({
      region: str(l.region),
      level: str(l.level),
      status: oneOf(l.status, LEVEL_STATUS, 'normal'),
      disc: str(l.disc),
      canal: oneOf(l.canal, GRADES, ''),
      foramen_left: oneOf(l.foramen_left, GRADES, ''),
      foramen_right: oneOf(l.foramen_right, GRADES, ''),
      note: str(l.note),
      measure: l.measure && typeof l.measure === 'object'
        ? Object.fromEntries(Object.entries(l.measure).filter(([, v]) => Number.isFinite(v)))
        : null,
      specialist: sanitizeSpecialist(l.specialist),
      specialist_dev: sanitizeSpecialist(l.specialist_dev),
    })),
    findings: arr(r.findings).filter((f) => f && f.id).map((f) => ({
      id: str(f.id),
      region: str(f.region),
      level: str(f.level),
      title: str(f.title) || 'Finding',
      technical_title: str(f.technical_title),
      severity: SEVERITY[f.severity] ? f.severity : 'incidental',
      confidence: oneOf(f.confidence, ['high', 'medium', 'low'], ''),
      plain: str(f.plain),
      technical: str(f.technical),
      why_it_matters: str(f.why_it_matters),
      what_to_do: str(f.what_to_do),
      anchor: f.anchor && typeof f.anchor === 'object'
        ? { mesh: str(f.anchor.mesh), point: arr(f.anchor.point).map(Number) }
        : { mesh: '', point: [] },
      views: arr(f.views).filter((v) => v && v.stack != null).map((v) => ({
        stack: str(v.stack),
        slice: Math.max(0, Math.round(Number(v.slice) || 0)),
        caption: str(v.caption),
        marks: arr(v.marks).filter((k) => k && typeof k === 'object'),
      })),
    })),
    glossary: arr(r.glossary).filter((g) => g && g.term).map((g) => ({ term: str(g.term), plain: str(g.plain) })),
    questions_for_doctor: arr(r.questions_for_doctor).map(str).filter(Boolean),
    method: {
      ai_models: arr(m.ai_models).map(str).filter(Boolean),
      readers: str(m.readers),
      limitations: arr(m.limitations).map(str).filter(Boolean),
    },
    disclaimer: str(r.disclaimer),
    comparison: sanitizeComparison(r.comparison),
    ai_report: sanitizeReportText(r.ai_report, false),
    scorecard: sanitizeScorecard(r.scorecard),
    dev_scorecard: sanitizeDevScorecard(r.dev_scorecard),
  };
}

/** The shipped specialist model's grade for one level; null unless it has a usable Pfirrmann grade. */
function sanitizeSpecialist(x) {
  const g = Math.round(Number(x?.pfirrmann));
  const c = Number(x?.pfirrmann_confidence);
  if (!(g >= 1 && g <= 5) || x?.pfirrmann_confidence == null || !(c >= 0 && c <= 1)) return null;
  const items = {};
  for (const [k, v] of Object.entries(x?.items && typeof x.items === 'object' ? x.items : {})) {
    const p = Number(v);
    if (v != null && p >= 0 && p <= 1) items[k] = p;
  }
  return { pfirrmann: g, pfirrmann_confidence: c, flags: arr(x.flags).map(str).filter(Boolean), items, not_blind: x.not_blind === true };
}

/** Only the one headline line (plus where it came from) is kept; null without text. */
function sanitizeScorecard(x) {
  const text = str(x?.headline?.text).trim();
  return text ? { headline: { text }, permission: str(x.permission), method: str(x.method) } : null;
}

/** A development model's one plain sentence (plus its method and split); null without text. */
function sanitizeDevScorecard(x) {
  const text = str(x?.text).trim();
  return text ? { text, method: str(x.method), split: str(x.split) } : null;
}


const VERDICTS = ['on par', 'behind', 'no model', 'unsettled'];
const reportLine = (x) => ({ label: str(x.label), text: str(x.text), level: str(x.level), finding: str(x.finding) });
const reportPoint = (x) => ({ text: str(x.text), level: str(x.level), finding: str(x.finding), status: SEVERITY[x.status] ? x.status : 'normal' });

/** A report in radiologist form: the AI's (`ai_report`) or the outside radiologist's (`comparison.report`). null without sections. */
function sanitizeReportText(a, outside) {
  if (!a || typeof a !== 'object') return null;
  const sections = arr(a.sections).filter((g) => g && arr(g.lines).length).map((g) => ({
    region: str(g.region),
    title: str(g.title),
    lines: arr(g.lines).filter((x) => x && x.text).map(reportLine),
  }));
  const impression = arr(a.impression).filter((x) => x && x.text).map(reportPoint);
  if (!sections.length && !impression.length) return null;
  const out = {
    title: str(a.title), author: str(a.author), byline: str(a.byline), history: str(a.history),
    exam: str(a.exam), technique: str(a.technique), who: arr(a.who).map(str).filter(Boolean),
    sections, impression,
  };
  if (outside) {
    out.impression_label = str(a.impression_label);
    out.levels_note = str(a.levels_note);
    out.levels = new Map(Object.entries(a.levels && typeof a.levels === 'object' ? a.levels : {})
      .filter(([, v]) => LEVEL_STATUS.includes(v)).map(([k, v]) => [normLevel(k), v]));
  }
  return out;
}

/** How the AI's tools compare with the field; null without items. */
function sanitizeCapability(c) {
  if (!c || typeof c !== 'object') return null;
  const items = arr(c.items).filter((x) => x && x.topic).map((x) => ({
    topic: str(x.topic), verdict: oneOf(x.verdict, VERDICTS, 'unsettled'), plain: str(x.plain),
  }));
  return items.length ? { title: str(c.title), headline: str(c.headline), items, bottom: str(c.bottom) } : null;
}

const AGREEMENT = ['agree', 'partial', 'disagree'];

/** Blind AI reading vs an outside radiologist report vs the AI's view after re-measuring. */
function sanitizeComparison(c) {
  if (!c || typeof c !== 'object') return null;
  const rows = arr(c.rows).filter((x) => x && x.topic).map((x) => ({
    region: str(x.region),
    level: str(x.level),
    topic: str(x.topic),
    blind: str(x.blind),
    radiologist: str(x.radiologist),
    after: str(x.after),
    blind_short: str(x.blind_short),
    radiologist_short: str(x.radiologist_short),
    after_short: str(x.after_short),
    model_short: str(x.model_short),
    model_dev_short: str(x.model_dev_short),
    agreement: oneOf(x.agreement, AGREEMENT, 'partial'),
    leans: oneOf(x.leans, ['ai', 'radiologist', 'both', 'unclear'], 'unclear'),
    measured: arr(x.measured).map(str).filter(Boolean),
    finding: str(x.finding),
  }));
  if (!rows.length) return null;
  const o = c.opinion || {};
  return {
    source: str(c.source),
    indication: str(c.indication),
    method: str(c.method),
    model_label: str(c.model_label),
    model_dev_label: str(c.model_dev_label),
    rows,
    report: sanitizeReportText(c.report, true),
    capability: sanitizeCapability(c.capability),
    opinion: {
      headline: str(o.headline),
      bottom_line: arr(o.bottom_line).map(str).filter(Boolean),
      paragraphs: arr(o.paragraphs).map(str).filter(Boolean),
      tell_doctor: arr(o.tell_doctor).map(str).filter(Boolean),
    },
  };
}

function sanitizeStack(s) {
  if (!s || !s.id || !s.path) return null;
  const count = Math.round(Number(s.count));
  if (!(count > 0)) return null;
  const o = s.orientation || {};
  return {
    id: str(s.id),
    label: str(s.label) || str(s.id),
    region: str(s.region),
    plane: str(s.plane),
    sequence: str(s.sequence),
    count,
    width: Number(s.width) > 0 ? Number(s.width) : 512,
    height: Number(s.height) > 0 ? Number(s.height) : 512,
    path: relUrl(s.path),
    default_slice: Math.min(count - 1, Math.max(0, Math.round(Number(s.default_slice ?? Math.floor(count / 2))))),
    orientation: { left: str(o.left), right: str(o.right), top: str(o.top), bottom: str(o.bottom) },
  };
}

/**
 * Only used with the sample report when no real image stacks exist: two procedurally drawn
 * "MRI-like" side views (clearly watermarked) so the viewer can be demonstrated.
 */
export function sampleStacks() {
  const base = { plane: 'sagittal', sequence: 'T2', count: 13, width: 512, height: 512, pixel_mm: [0.55, 0.55], path: '', default_slice: 6, orientation: { left: 'Front', right: 'Back', top: 'Head' } };
  return [
    {
      ...base, id: 'L_SAG_T2', label: 'Lower back, side view (T2) · sample image', region: 'lumbar',
      synthetic: { base: 0.46, amp: 0.04, y0: 0.05, y1: 0.9, bw: 0.075, dh: 0.036, discs: [0.14, 0.27, 0.40, 0.53, 0.66, 0.79], cord: false, dark: [4, 5], bulge: { 4: 0.014, 5: 0.026 }, spots: [{ y: 0.335, r: 0.02 }], sacrum: true },
    },
    {
      ...base, id: 'C_SAG_T2', label: 'Neck, side view (T2) · sample image', region: 'cervical',
      synthetic: { base: 0.47, amp: 0.03, y0: 0.15, y1: 0.95, bw: 0.055, dh: 0.024, discs: [0.25, 0.36, 0.47, 0.58, 0.69, 0.80], cord: true, dark: [3], bulge: { 3: 0.012 }, spots: [], skull: true },
    },
  ];
}

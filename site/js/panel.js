// panel.js: the side panel / bottom sheet. A mode switch (AI report, Radiologist, Good & bad) picks whose words the
// Report tab shows; the Scans tab is the scan viewer. Nothing from one report is drawn while the other is on.
import { el, icon, sev, STATUS, GRADE, normLevel, levelList, compareLevels, firstSentence } from './data.js';
import { createViewer, stackFigure, emptyState } from './viewer.js';

const phone = window.matchMedia('(max-width: 860px)');
const ICON = {
  why: ['M12 3a6 6 0 0 0-3.6 10.8c.6.5 1 1.2 1 2V17h5.2v-1.2c0-.8.4-1.5 1-2A6 6 0 0 0 12 3z', 'M9.8 20.5h4.4'],
  todo: ['M5 12.5l4 4 10-10'],
  cube: ['M12 3l8 4.5v9L12 21l-8-4.5v-9z', 'M12 12l8-4.5M12 12v9M12 12L4 7.5'],
  scan: ['M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3', 'M4 12h16'],
  chevron: ['M6 9l6 6 6-6'],
  close: ['M7 7l10 10M17 7L7 17'],
  copy: ['M9 9h10v10H9z', 'M5 15V5h10'],
};

// Specialist model flag keys (CONTRACT.md, levels[].specialist) in plain words.
const FLAG = {
  herniation: 'herniation', narrowing: 'narrowing', bulging: 'bulge', spondylolisthesis: 'slipped vertebra',
  up_endplate: 'upper endplate change', low_endplate: 'lower endplate change', modic: 'bone-marrow (Modic) change',
};

// Verdict colours of the "Good & bad" capability card.
const VERDICT_COLOR = { 'on par': '#7fbf9f', behind: '#e07a3a', 'no model': '#a3a9b5', unsettled: '#d9a441' };

export function createPanel(root, { report, stacks, incidentalColor = '#5cc2f2', on = {} }) {
  const $ = (s) => root.querySelector(s);
  const hasCompare = !!report?.comparison;
  const hasRadiologist = !!report?.comparison?.report;
  const switchEl = $('#mode-switch');
  const modeBtns = [...switchEl.querySelectorAll('[data-mode]')];
  if (!hasRadiologist) modeBtns.find((b) => b.dataset.mode === 'radiologist')?.remove();
  switchEl.hidden = !hasCompare;
  const panes = { report: $('#pane-report'), viewer: $('#pane-viewer') };
  const tabs = [...root.querySelectorAll('[role="tab"]')];
  const findings = [...(report?.findings || [])].sort((a, b) => sev(b.severity).rank - sev(a.severity).rank || compareLevels(a.level, b.level));
  let viewer = null;
  let mode = 'ai';
  let view = null;        // what the Report tab shows when no detail is open
  let detailOpen = false;
  let savedTop = 0;

  function showTab(name) {
    for (const t of tabs) {
      const on_ = t.id === `tab-${name}`;
      t.setAttribute('aria-selected', String(on_));
      t.tabIndex = on_ ? 0 : -1;
    }
    for (const [k, p] of Object.entries(panes)) p.hidden = k !== name;
  }
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => showTab(t.id.replace('tab-', '')));
    t.addEventListener('keydown', (e) => {
      const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      let j = null;
      if (d) j = (i + d + tabs.length) % tabs.length;
      if (e.key === 'Home') j = 0;
      if (e.key === 'End') j = tabs.length - 1;
      if (j == null) return;
      e.preventDefault();
      tabs[j].focus();
      showTab(tabs[j].id.replace('tab-', ''));
    });
  });

  // ---------- the Report tab: view, detail, back ----------
  function renderReport() {
    const pane = panes.report;
    detailOpen = false;
    pane.replaceChildren();
    if (!report) {
      pane.append(emptyState('No report yet', 'Add data/report.json (or data/report.sample.json) next to this page and reload.'));
      return;
    }
    view = mode === 'radiologist' ? radiologistView() : mode === 'compare' ? compareView() : aiView();
    pane.append(view);
    pane.scrollTop = 0;
  }

  function showDetail(node) {
    const pane = panes.report;
    if (!detailOpen) savedTop = pane.scrollTop;
    detailOpen = true;
    pane.replaceChildren(node);
    pane.scrollTop = 0;
    on.reveal?.();
  }

  function backToReport() {
    const pane = panes.report;
    detailOpen = false;
    pane.replaceChildren(view);
    pane.scrollTop = savedTop;
  }

  const backBtn = () => el('button', { class: 'btn ghost back-btn', type: 'button', onclick: backToReport }, '← Back to the report');

  function setMode(m) {
    mode = m === 'radiologist' && hasRadiologist ? 'radiologist' : m === 'compare' && hasCompare ? 'compare' : 'ai';
    for (const b of modeBtns) b.setAttribute('aria-pressed', String(b.dataset.mode === mode));
    viewer?.setMode(mode);
    showTab('report');
    renderReport();
    on.mode?.(mode);
  }
  for (const b of modeBtns) b.addEventListener('click', () => setMode(b.dataset.mode));

  // ---------- shared bits ----------
  // Long text shows its first lines on phones, with one button for the rest and for any `extra`
  // that belongs with it (CSS: .is-clamped, .clamp-extra, .more-btn; desktop shows everything).
  function clamped(cls, text, extra = null) {
    const p = el('p', { class: `${cls}${text.length > 160 ? ' is-clamped' : ''}` }, text);
    if (text.length <= 160) return [p, extra];
    const more = el('button', { class: 'more-btn', type: 'button', 'aria-expanded': 'false' }, 'Read more');
    const wrap = el('div', { class: 'clamp-wrap' }, p, extra ? el('div', { class: 'clamp-extra' }, extra) : null, more);
    more.addEventListener('click', () => {
      const open = wrap.classList.toggle('is-open');
      p.classList.toggle('is-clamped', !open);
      more.textContent = open ? 'Show less' : 'Read more';
      more.setAttribute('aria-expanded', String(open));
    });
    return wrap;
  }

  function scoreLine() {
    const main = report.scorecard?.headline?.text ? el('p', { class: 'summary-score muted small',
      title: [report.scorecard.method, report.scorecard.permission].filter(Boolean).join(' · ') }, report.scorecard.headline.text) : null;
    const dev = report.dev_scorecard?.text ? el('p', { class: 'summary-score muted small',
      title: [report.dev_scorecard.method, report.dev_scorecard.split].filter(Boolean).join(' · ') }, report.dev_scorecard.text) : null;
    return main && dev ? [main, dev] : main || dev;
  }

  function techDetails(label, title, text) {
    return el('details', { class: 'tech' }, el('summary', {}, label),
      el('div', { class: 'tech-body' }, title ? el('strong', {}, title) : null, text));
  }

  const fold = (label, ...kids) => el('details', { class: 'fold' }, el('summary', {}, label), el('div', { class: 'fold-body' }, kids));

  const dotColor = (status) => (status === 'incidental' ? incidentalColor : sev(status).color);
  const firstLevel = (text) => levelList(text)[0] || '';
  const levelEntry = (name) => report.levels.find((l) => normLevel(l.level) === normLevel(name));

  // One clickable line of a report: `label  text` (findings) or just the text (impression).
  function reportRow(item, { label = true, dot = null, onOpen = null } = {}) {
    const kids = [
      dot ? el('span', { class: 'rep-dot', style: { background: dot }, 'aria-hidden': 'true' }) : null,
      label && item.label ? el('span', { class: 'rep-label' }, item.label) : null,
      el('span', { class: 'rep-text' }, item.text),
    ];
    return onOpen
      ? el('button', { class: 'rep-row', type: 'button', onclick: onOpen }, kids)
      : el('div', { class: 'rep-row is-plain' }, kids);
  }

  // ---------- AI mode ----------
  // Every word here is the AI's. The radiologist's report never reaches this view or its details.
  function fallbackAiReport() {
    const lines = (list) => list.map((f) => ({ label: f.level || f.title, text: f.plain ? firstSentence(f.plain) : f.title, level: firstLevel(f.level), finding: f.id }));
    return {
      title: '', byline: '', exam: '', history: '', technique: '', who: [],
      impression: findings.map((f) => ({ text: f.title, level: firstLevel(f.level), finding: f.id, status: f.severity })),
      sections: report.regions.map((r) => ({ region: r.id, title: r.name, lines: lines(findings.filter((f) => f.region === r.id)) })).filter((g) => g.lines.length),
    };
  }

  function aiView() {
    const a = report.ai_report || fallbackAiReport();
    const st = STATUS[report.summary.status] || STATUS.watch;
    const open = (item) => () => showDetail(aiDetail(item));
    return el('div', { class: 'report-view' },
      el('section', { class: 'card summary-card', style: { '--tone': st.color }, 'aria-labelledby': 'summary-headline' },
        el('div', { class: 'summary-top' },
          el('span', { class: 'status-pill' }, el('span', { class: 'dot', 'aria-hidden': 'true' }), st.label),
          a.title ? el('span', { class: 'summary-exam' }, a.title) : null),
        el('h1', { class: 'headline', id: 'summary-headline' }, report.summary.headline || 'Your spine MRI, explained'),
        report.summary.plain ? clamped('summary-plain', report.summary.plain) : null,
        a.byline ? el('p', { class: 'rep-byline muted small' }, a.byline) : null,
        report.summary.technical ? techDetails('In medical terms', null, report.summary.technical) : null),
      a.impression.length ? el('div', { class: 'section-title' }, 'Impression') : null,
      a.impression.length ? el('ol', { class: 'rep-impression' }, a.impression.map((it) =>
        el('li', {}, reportRow(it, { label: false, dot: dotColor(it.status), onOpen: open(it) })))) : null,
      a.sections.length ? el('div', { class: 'section-title' }, 'Findings') : null,
      a.sections.map((g) => el('section', { class: 'rep-section' },
        el('h3', { class: 'rep-h' }, g.title),
        el('div', { class: 'rep-lines' }, g.lines.map((ln) => reportRow(ln, { onOpen: open(ln) }))))),
      !a.sections.length && !a.impression.length ? el('p', { class: 'empty-note' }, 'No findings were listed in this report.') : null,
      aiFooter(a));
  }

  function aiFooter(a) {
    const meta = [['Exam', a.exam], ['History', a.history], ['Technique', a.technique]].filter(([, v]) => v);
    const score = scoreLine();
    const who = a.who.length || score ? fold('Who made this report',
      a.who.length ? el('ul', { class: 'method-list' }, a.who.map((w) => el('li', {}, w))) : null, score) : null;
    return el('footer', { class: 'rep-foot' },
      meta.length ? el('dl', { class: 'rep-meta' }, meta.map(([k, v]) => el('div', {}, el('dt', {}, k), el('dd', {}, v)))) : null,
      who, wordsFold());
  }

  // The detail of one AI row: its text, then the finding card and/or the level's measurements.
  function aiDetail(item) {
    const f = item.finding ? findings.find((x) => x.id === item.finding) : null;
    const lvl = item.level || f?.level || '';
    const status = item.status || f?.severity || '';
    const title = item.label || f?.title || lvl || 'Detail';
    const here = !f && lvl ? findings.filter((x) => levelList(x.level).includes(firstLevel(lvl))) : [];
    const blocks = lvl ? levelBlocks(lvl) : [];
    if (f) on.finding?.(f.id);
    else if (lvl) on.level?.(firstLevel(lvl), levelEntry(firstLevel(lvl))?.region);
    return el('div', { class: 'report-detail' },
      backBtn(),
      el('div', { class: 'card detail-card', style: { '--sev': status ? dotColor(status) : 'var(--line-strong)' } },
        el('div', { class: 'summary-top' },
          el('h2', { class: 'detail-title' }, title),
          el('span', { class: 'finding-tags' },
            status ? el('span', { class: 'sev-chip', style: { '--sev': dotColor(status) } }, sev(status).label) : null,
            lvl ? el('span', { class: 'level-chip' }, lvl) : null)),
        item.text ? el('p', { class: 'lead' }, item.text) : null,
        f ? findingBody(f) : null),
      blocks,
      here.length ? el('div', { class: 'section-title' }, 'Findings here') : null,
      here.length ? el('div', { class: 'mark-list' }, here.map((x) => el('button', {
        class: 'mark-item', type: 'button', style: { '--sev': sev(x.severity).color },
        onclick: () => showDetail(aiDetail({ finding: x.id, label: x.title, level: x.level, status: x.severity })),
      }, el('span', { class: 'dot', 'aria-hidden': 'true' }), el('span', { class: 'grow' }, x.title), el('span', { class: 'sub' }, 'Open')))) : null);
  }

  function findingBody(f) {
    const s = sev(f.severity);
    const out = [];
    if (f.plain) out.push(el('p', { class: 'lead' }, f.plain));
    if (f.why_it_matters) out.push(infoBlock('Why it matters', ICON.why, f.why_it_matters));
    if (f.what_to_do) out.push(infoBlock('What to do', ICON.todo, f.what_to_do));
    if (f.confidence) {
      const n = { high: 3, medium: 2, low: 1 }[f.confidence];
      out.push(el('div', { class: 'confidence' },
        el('span', { class: 'conf-bars', 'aria-hidden': 'true' }, [1, 2, 3].map((i) => el('i', { class: i <= n ? 'on' : '' }))),
        `How sure the reading is: ${f.confidence[0].toUpperCase()}${f.confidence.slice(1)}`));
    }
    if (f.technical || f.technical_title) out.push(techDetails('In medical terms', f.technical_title, f.technical));
    if (f.views.length) out.push(el('div', { class: 'scan-grid' }, f.views.map((v) => stackFigure(stacks, v, s.color))));
    const first = f.views.find((v) => stacks.some((st) => st.id === v.stack));
    out.push(el('div', { class: 'btn-row' },
      el('button', { class: 'btn primary', type: 'button', onclick: () => on.finding?.(f.id) }, icon(ICON.cube), 'Show in 3D'),
      first ? el('button', {
        class: 'btn', type: 'button',
        onclick: () => { showTab('viewer'); viewer?.focus(first.stack, first.slice); },
      }, icon(ICON.scan), 'Open in scan viewer') : null));
    return out;
  }

  function infoBlock(title, paths, text) {
    return el('div', { class: 'info-block' }, el('h4', {}, icon(paths), title), el('p', {}, text));
  }

  // The AI's measurements for the level(s) a row names (report.levels): its own numbers and graders only.
  function levelBlocks(levelText) {
    return levelList(levelText).map(levelEntry).filter(Boolean).map((e) => {
      const rows = levelRows(e);
      if (!rows.length) return null;
      return el('div', { class: 'card level-detail' },
        el('div', { class: 'summary-top' }, el('h3', {}, e.level),
          el('span', { class: 'sev-chip', style: { '--sev': dotColor(e.status) } }, sev(e.status).label)),
        el('div', { style: { marginTop: '8px' } }, rows.map(([k, v]) => el('div', { class: 'row' }, el('span', {}, k), el('span', {}, v)))));
    }).filter(Boolean);
  }

  function levelRows(e) {
    const rows = [];
    if (e.disc) rows.push(['Disc', e.disc]);
    if (e.canal) rows.push(['Spinal canal narrowing', gradeChip(e.canal)]);
    if (e.foramen_left) rows.push(['Left nerve exit', gradeChip(e.foramen_left)]);
    if (e.foramen_right) rows.push(['Right nerve exit', gradeChip(e.foramen_right)]);
    if (e.note) rows.push(['Note', e.note]);
    const m = e.measure || {};
    const parts = [
      m.disc_height_mm != null && `disc height ${m.disc_height_mm} mm`,
      m.disc_to_csf_signal != null && `water signal ${m.disc_to_csf_signal.toFixed(2)} x fluid`,
      m.canal_ap_at_disc_mm != null && `fluid space ${m.canal_ap_at_disc_mm} mm front to back`,
      m.cord_ap_at_disc_mm != null && `cord ${m.cord_ap_at_disc_mm} mm`,
      m.cord_compression_ratio_hc != null && `cord ${Math.abs(m.cord_compression_ratio_hc).toFixed(0)}% ${m.cord_compression_ratio_hc >= 0 ? 'narrower' : 'wider'} than in healthy adults`,
    ].filter(Boolean);
    if (parts.length) rows.push(['Measured by AI', parts.join(' · ')]);
    const modelRow = (label, sp, devChip) => {
      const flags = sp.flags.map((f) => (FLAG[f] || f) + (sp.items[f] != null ? ` (${Math.round(sp.items[f] * 100)}%)` : ''));
      rows.push([label, [
        `Disc drying grade ${sp.pfirrmann} of 5 (${Math.round(sp.pfirrmann_confidence * 100)}% sure)`,
        flags.length ? ` · also flags: ${flags.join(', ')}` : '',
        devChip ? el('span', { class: 'level-chip not-blind', title: 'Not yet tested on scans it never saw; shown for comparison only' }, 'in development') : null,
        sp.not_blind ? el('span', { class: 'level-chip not-blind', title: 'This scan already feeds a mistake-log rule, so this re-run is not a blind reading.' }, 'not blind') : null,
      ]]);
    };
    if (e.specialist) modelRow('Disc grader v1', e.specialist, false);
    if (e.specialist_dev) modelRow('Disc grader v120', e.specialist_dev, true);
    return rows;
  }

  function gradeChip(g) {
    return el('span', { class: 'sev-chip', style: { '--sev': sev(g === 'none' ? 'normal' : g).color } }, GRADE[g] || g);
  }

  // The old Words tab: word list, questions for the doctor, how this was made. Collapsed under the AI report.
  function wordsFold() {
    const body = [];
    const gl = report.glossary;
    if (gl.length) {
      const dl = el('dl', { class: 'glossary' });
      const draw = (q) => {
        const t = q.trim().toLowerCase();
        const items = gl.filter((g) => !t || g.term.toLowerCase().includes(t) || g.plain.toLowerCase().includes(t));
        dl.replaceChildren(...(items.length ? items.map((g) => el('div', {}, el('dt', {}, g.term), el('dd', {}, g.plain)))
          : [el('p', { class: 'empty-note' }, 'No matching words.')]));
      };
      const search = el('input', { class: 'search', type: 'search', placeholder: 'Search words', 'aria-label': 'Search the word list' });
      search.addEventListener('input', () => draw(search.value));
      draw('');
      body.push(el('div', { class: 'stack' }, search, dl));
    }
    const qs = report.questions_for_doctor;
    if (qs.length) {
      const copy = el('button', { class: 'btn ghost', type: 'button' }, icon(ICON.copy), 'Copy');
      copy.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(qs.map((q, i) => `${i + 1}. ${q}`).join('\n'));
          copy.lastChild.textContent = 'Copied';
        } catch {
          copy.lastChild.textContent = 'Copy not available';
        }
        setTimeout(() => { copy.lastChild.textContent = 'Copy'; }, 1800);
      });
      body.push(el('div', { class: 'section-title' }, el('span', {}, 'Questions for your doctor'), copy),
        el('ol', { class: 'questions' }, qs.map((q) => el('li', {}, el('span', {}, q)))));
    }
    const m = report.method;
    if (m.ai_models.length || m.readers || m.limitations.length) {
      body.push(el('div', { class: 'section-title' }, 'How this was made'),
        el('div', { class: 'card' },
          m.readers ? el('p', { class: 'small' }, m.readers) : null,
          m.ai_models.length ? el('p', { class: 'small muted' }, `Tools: ${m.ai_models.join(', ')}`) : null,
          m.limitations.length ? el('details', { class: 'tech' }, el('summary', {}, 'Limitations'),
            el('ul', { class: 'method-list' }, m.limitations.map((l) => el('li', {}, l)))) : null));
    }
    return body.length ? fold('Words explained', body) : null;
  }

  // ---------- Radiologist mode ----------
  // Only the outside radiologist's words (report.comparison.report). No AI content in this view.
  function radiologistView() {
    const r = report.comparison.report;
    const row = (item, region, opts) => reportRow(item, { ...opts, onOpen: item.level ? () => showDetail(radiologistDetail(item, region)) : null });
    return el('div', { class: 'report-view' },
      el('section', { class: 'card summary-card', style: { '--tone': '#b58cf0' } },
        el('div', { class: 'summary-top' }, r.author ? el('span', { class: 'summary-exam' }, r.author) : null),
        el('h1', { class: 'headline' }, r.title || "Radiologist's report"),
        r.history ? el('p', { class: 'summary-plain' }, `History: ${r.history}`) : null,
        r.byline ? el('p', { class: 'rep-byline muted small' }, r.byline) : null),
      r.impression.length ? el('div', { class: 'section-title' }, 'Impression') : null,
      r.impression_label ? el('p', { class: 'rep-note muted small' }, r.impression_label) : null,
      r.impression.length ? el('ol', { class: 'rep-impression' }, r.impression.map((it) =>
        el('li', {}, row(it, '', { label: false, dot: sev(it.status).color })))) : null,
      r.sections.length ? el('div', { class: 'section-title' }, 'Findings') : null,
      r.sections.map((g) => el('section', { class: 'rep-section' },
        el('h3', { class: 'rep-h' }, g.title),
        el('div', { class: 'rep-lines' }, g.lines.map((ln) => row(ln, g.region))))));
  }

  function radiologistDetail(item, region) {
    const lvl = firstLevel(item.level);
    if (lvl) on.level?.(lvl, region);
    return el('div', { class: 'report-detail' },
      backBtn(),
      el('div', { class: 'card detail-card' },
        el('div', { class: 'summary-top' },
          el('h2', { class: 'detail-title' }, item.label || item.level),
          el('span', { class: 'level-chip' }, item.level)),
        el('p', { class: 'lead' }, item.text),
        el('div', { class: 'btn-row' },
          el('button', { class: 'btn', type: 'button', onclick: () => showTab('viewer') }, icon(ICON.scan), 'Open the scans'))));
  }

  // ---------- Good & bad mode ----------
  function compareView() {
    const cap = report.comparison.capability;
    return el('div', { class: 'report-view' }, cap ? capabilityCard(cap) : null, renderCompare());
  }

  function capabilityCard(cap) {
    return el('section', { class: 'card cap-card' },
      cap.title ? el('h3', {}, cap.title) : null,
      cap.headline ? el('p', { class: 'cap-headline' }, cap.headline) : null,
      el('ul', { class: 'cap-list' }, cap.items.map((it) => el('li', {},
        el('div', { class: 'cap-top' },
          el('span', { class: 'cap-topic' }, it.topic),
          el('span', { class: 'sev-chip', style: { '--sev': VERDICT_COLOR[it.verdict] } }, it.verdict)),
        it.plain ? el('p', { class: 'cap-plain small' }, it.plain) : null))),
      cap.bottom ? el('p', { class: 'cap-bottom muted small' }, cap.bottom) : null);
  }

  // The AI's blind reading vs the outside radiologist report vs the AI after re-checking.
  // Short phrases first; the long reasoning sits behind "Why".
  const VERDICT = {
    agree: { label: 'Agree', tone: 'agree' },
    ai: { label: 'Favors AI', tone: 'ai' },
    radiologist: { label: 'Favors radiologist', tone: 'rad' },
    both: { label: 'No clear difference', tone: 'both' },
    unclear: { label: 'Unclear', tone: 'both' },
  };
  const verdictOf = (r) => (r.leans === 'radiologist' ? VERDICT.radiologist
    : r.agreement === 'agree' ? VERDICT.agree : VERDICT[r.leans] || VERDICT.unclear);
  const isDifference = (r) => r.agreement !== 'agree';
  const COLS = [
    ['blind', 'AI, blind', 'From the images alone, before the report was shared. Never edited since.'],
    ['radiologist', 'Radiologist', 'The signed outside report.'],
    ['after', 'Re-check', 'After seeing the report: fresh AI measurers re-measured this point without knowing whose reading was whose.'],
  ];
  const short = (r, k) => r[`${k}_short`] || r[k];
  // Optional fourth column: the specialist model's opinion. Not part of the same/different check.
  const MODEL = report?.comparison?.model_label && report.comparison.rows.some((r) => r.model_short)
    ? ['model', report.comparison.model_label, 'A trained disc model, re-run on this scan after the mistake log used it, so not blind.'] : null;
  // Optional column after it: a newer model still in development. Never held-out numbers.
  const MODEL_DEV = report?.comparison?.model_dev_label && report.comparison.rows.some((r) => r.model_dev_short)
    ? ['model_dev', report.comparison.model_dev_label, 'Not yet tested on scans it never saw; shown for comparison only.'] : null;

  function compareRow(r) {
    const v = verdictOf(r);
    const same = short(r, 'blind') === short(r, 'radiologist') && short(r, 'radiologist') === short(r, 'after');
    const model = [
      MODEL && r.model_short ? [
        el('span', { class: 'cmp-k cmp-k-model', title: MODEL[2] }, MODEL[1]),
        el('span', { class: 'cmp-v cmp-v-model' }, r.model_short)] : null,
      MODEL_DEV && r.model_dev_short ? [
        el('span', { class: 'cmp-k cmp-k-model', title: MODEL_DEV[2] }, MODEL_DEV[1]),
        el('span', { class: 'cmp-v cmp-v-model' }, r.model_dev_short)] : null,
    ];
    const grid = same
      ? el('div', { class: 'cmp-grid' }, el('span', { class: 'cmp-k' }, 'All three'), el('span', { class: 'cmp-v' }, short(r, 'after')), model)
      : el('div', { class: 'cmp-grid' }, COLS.map(([k, label]) => [
        el('span', { class: `cmp-k cmp-k-${k}` }, label),
        el('span', { class: `cmp-v cmp-v-${k}` }, short(r, k)),
      ]), model);
    const details = same ? null : el('details', { class: 'cmp-why' },
      el('summary', {}, 'Why'),
      r.after ? el('p', {}, r.after) : null,
      r.measured.length ? el('ul', { class: 'cmp-measured' }, r.measured.map((m) => el('li', {}, m))) : null,
      el('div', { class: 'cmp-full' },
        r.blind ? el('p', {}, el('b', {}, 'AI, blind: '), r.blind) : null,
        r.radiologist ? el('p', {}, el('b', {}, 'Radiologist: '), r.radiologist) : null),
      r.finding ? el('button', { class: 'btn ghost', type: 'button', onclick: () => openFinding(r.finding) },
        icon(ICON.cube), 'Show on the model') : null);
    // On phones each row folds to one line (level, topic, verdict) and opens on tap.
    const chev = el('button', { class: 'cmp-chev', type: 'button', 'aria-expanded': 'false',
      'aria-label': `Details: ${r.level} ${r.topic}` }, icon(ICON.chevron));
    const head = el('header', { class: 'cmp-head' },
      el('span', { class: 'cmp-level' }, r.level),
      el('span', { class: 'cmp-topic' }, r.topic),
      el('span', { class: `cmp-pill tone-${v.tone}` }, v.label),
      chev);
    const row = el('article', { class: `cmp-row tone-${v.tone}`, 'data-diff': String(isDifference(r)) }, head, grid, details);
    head.addEventListener('click', () => {
      if (!phone.matches) return;
      const open = row.classList.toggle('is-open');
      chev.setAttribute('aria-expanded', String(open));
    });
    return row;
  }

  function renderCompare() {
    const c = report.comparison;
    const n = { agree: 0, partial: 0, disagree: 0 };
    for (const r of c.rows) n[r.agreement] += 1;
    const list = el('div', { class: 'cmp-list' }, c.rows.map(compareRow));
    const filters = [['all', `All ${c.rows.length}`], ['diff', `Differences ${n.partial + n.disagree}`], ['same', `Agreements ${n.agree}`]];
    const filterBar = el('div', { class: 'cmp-filter', role: 'group', 'aria-label': 'Show' },
      filters.map(([k, label], i) => el('button', {
        type: 'button', class: 'cmp-filter-btn', 'aria-pressed': String(i === 0),
        onclick: (e) => {
          for (const b of filterBar.children) b.setAttribute('aria-pressed', String(b === e.currentTarget));
          for (const row of list.children) row.hidden = k === 'diff' ? row.dataset.diff !== 'true' : k === 'same' ? row.dataset.diff === 'true' : false;
        },
      }, label)));
    return el('div', { class: 'cmp-view' },
      el('section', { class: 'card cmp-hero' },
        el('div', { class: 'cmp-hero-top' },
          el('h3', {}, 'The AI reading next to the radiologist\u2019s report'),
          el('button', { class: 'btn ghost', type: 'button', onclick: () => openWide() }, icon(ICON.scan), 'Table')),
        c.opinion.bottom_line.length
          ? el('ul', { class: 'cmp-bottom' }, c.opinion.bottom_line.map((b) => el('li', {}, b)))
          : c.opinion.headline ? el('p', {}, c.opinion.headline) : null,
        el('div', { class: 'cmp-score' },
          el('span', { class: 'tone-agree' }, el('b', {}, String(n.agree)), 'agree'),
          el('span', { class: 'tone-both' }, el('b', {}, String(n.partial)), 'partly'),
          el('span', { class: 'tone-ai' }, el('b', {}, String(n.disagree)), 'differ'))),
      filterBar,
      list,
      c.opinion.tell_doctor.length ? el('div', { class: 'section-title' }, 'Worth raising with your doctor') : null,
      c.opinion.tell_doctor.length ? el('ol', { class: 'questions' }, c.opinion.tell_doctor.map((q) => el('li', {}, el('span', {}, q)))) : null,
      c.opinion.paragraphs.length ? el('details', { class: 'card cmp-more' },
        el('summary', {}, 'The AI\'s full opinion'),
        c.opinion.headline ? el('p', { class: 'cmp-more-head' }, c.opinion.headline) : null,
        c.opinion.paragraphs.map((p) => el('p', {}, p))) : null,
      el('p', { class: 'cmp-foot' }, [c.method, c.source].filter(Boolean).join(' ')));
  }

  // Full-width table of the short phrases, for a side-by-side scan on a big screen.
  function openWide() {
    const c = report.comparison;
    const close = () => { overlay.remove(); document.removeEventListener('keydown', onKey); };
    const onKey = (e) => { if (e.key === 'Escape') close(); };
    const overlay = el('div', { class: 'compare-overlay', role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Side-by-side comparison' },
      el('div', { class: 'compare-sheet glass' },
        el('div', { class: 'compare-sheet-head' },
          el('h2', {}, 'Side by side'),
          el('button', { class: 'btn ghost', type: 'button', onclick: close, 'aria-label': 'Close' }, icon(ICON.close), 'Close')),
        el('div', { class: 'compare-table-wrap' },
          el('table', { class: 'compare-table' },
            el('thead', {}, el('tr', {}, el('th', {}, 'Level'), COLS.map(([, label, tip]) => el('th', { title: tip }, label)),
              MODEL ? el('th', { class: 'cmp-v-model', title: MODEL[2] }, MODEL[1]) : null,
              MODEL_DEV ? el('th', { class: 'cmp-v-model', title: MODEL_DEV[2] }, MODEL_DEV[1]) : null, el('th', {}, 'Verdict'))),
            el('tbody', {}, c.rows.map((r) => {
              const v = verdictOf(r);
              return el('tr', { class: `tone-${v.tone}` },
                el('td', { class: 'compare-point' }, el('strong', {}, r.level), el('span', {}, r.topic)),
                COLS.map(([k]) => el('td', { class: `cmp-v-${k}` }, short(r, k) || '-')),
                MODEL ? el('td', { class: 'cmp-v-model' }, r.model_short) : null,
                MODEL_DEV ? el('td', { class: 'cmp-v-model' }, r.model_dev_short) : null,
                el('td', {}, el('span', { class: `cmp-pill tone-${v.tone}` }, v.label)));
            }))))));
    document.body.append(overlay);
    document.addEventListener('keydown', onKey);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
    overlay.querySelector('button').focus();
  }

  // A tap on a 3D marker or a scan mark: open that finding's card in the AI's words (and fly there).
  // Not in Radiologist mode, where nothing of the AI's is shown.
  function openFinding(id) {
    const f = findings.find((x) => x.id === id);
    if (!f || mode === 'radiologist') return;
    showTab('report');
    showDetail(aiDetail({ finding: f.id, label: f.title, level: f.level, status: f.severity }));
  }

  viewer = createViewer(panes.viewer, { stacks, report, onOpenFinding: (id) => openFinding(id) });
  setMode('ai');

  return { openFinding, showTab, setMode };
}

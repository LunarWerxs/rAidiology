// panel.js: the side panel / bottom sheet: Overview (summary + findings), Scan viewer, Levels, Words.
import { el, svgEl, icon, sev, STATUS, GRADE, SEVERITY, LEVEL_ORDER, normLevel, levelList, compareLevels, firstSentence } from './data.js';
import { createViewer, stackFigure, emptyState } from './viewer.js';

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
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

const regionSlot = (r) => (r.anchor === 'neck' || r.anchor === 'lower_back' ? r.anchor
  : r.id === 'cervical' ? 'neck' : r.id === 'lumbar' ? 'lower_back' : null);

const cssId = (s) => String(s).replace(/[^a-zA-Z0-9_-]/g, '_');

// Specialist model flag keys (CONTRACT.md, levels[].specialist) in plain words.
const FLAG = {
  herniation: 'herniation', narrowing: 'narrowing', bulging: 'bulge', spondylolisthesis: 'slipped vertebra',
  up_endplate: 'upper endplate change', low_endplate: 'lower endplate change', modic: 'bone-marrow (Modic) change',
};

export function createPanel(root, { report, stacks, on = {} }) {
  const $ = (s) => root.querySelector(s);
  if (report?.comparison) $('#tab-compare').hidden = false;
  else { $('#tab-compare')?.remove(); $('#pane-compare')?.remove(); }
  const panes = { overview: $('#pane-overview'), viewer: $('#pane-viewer'), levels: $('#pane-levels'), words: $('#pane-words') };
  if (report?.comparison) panes.compare = $('#pane-compare');
  const tabs = [...root.querySelectorAll('[role="tab"]')];
  const findings = [...(report?.findings || [])].sort((a, b) => sev(b.severity).rank - sev(a.severity).rank || compareLevels(a.level, b.level));
  const cards = new Map();
  let levelFilter = null;
  let viewer = null;

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

  const filterSlot = el('span');
  const list = el('div', { class: 'stack' });

  function renderOverview() {
    const pane = panes.overview;
    pane.replaceChildren();
    if (!report) {
      pane.append(emptyState('No report yet', 'Add data/report.json (or data/report.sample.json) next to this page and reload.'));
      return;
    }
    pane.append(summaryCard());
    pane.append(el('div', { class: 'section-title' }, el('span', {}, 'What we found'), filterSlot), list);
    renderFindings();
    if (report.regions.length) {
      pane.append(el('div', { class: 'section-title' }, 'Scanned areas'),
        el('div', { class: 'region-grid' }, report.regions.map(regionCard)));
    }
  }

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

  function summaryCard() {
    const st = STATUS[report.summary.status] || STATUS.watch;
    const exams = [...report.study.exams];
    const counts = Object.values(SEVERITY)
      .map((s) => ({ s, n: findings.filter((f) => sev(f.severity) === s).length }))
      .filter((c) => c.n)
      .sort((a, b) => b.s.rank - a.s.rank);
    return el('section', { class: 'card summary-card', style: { '--tone': st.color }, 'aria-labelledby': 'summary-headline' },
      el('div', { class: 'summary-top' },
        el('span', { class: 'status-pill' }, el('span', { class: 'dot', 'aria-hidden': 'true' }), st.label),
        exams.length ? el('span', { class: 'summary-exam' }, exams.join(' · ')) : null),
      el('h1', { class: 'headline', id: 'summary-headline' }, report.summary.headline || 'Your spine MRI, explained'),
      report.summary.plain ? clamped('summary-plain', report.summary.plain, scoreLine()) : scoreLine(),
      counts.length ? el('div', { class: 'counts' }, counts.map(({ s, n }) =>
        el('span', { class: 'count-chip', style: { '--sev': s.color } }, el('i', { 'aria-hidden': 'true' }), `${n} ${s.label.toLowerCase()}`))) : null,
      report.summary.technical ? techDetails("The radiologist's wording", null, report.summary.technical) : null);
  }

  function scoreLine() {
    return report.scorecard?.headline?.text ? el('p', { class: 'summary-score muted small',
      title: [report.scorecard.method, report.scorecard.permission].filter(Boolean).join(' · ') }, report.scorecard.headline.text) : null;
  }

  function techDetails(label, title, text) {
    return el('details', { class: 'tech' }, el('summary', {}, label),
      el('div', { class: 'tech-body' }, title ? el('strong', {}, title) : null, text));
  }

  function regionCard(r) {
    const slot = regionSlot(r);
    return el('div', { class: 'card region-card' },
      el('h3', {}, r.name),
      el('p', {}, r.summary || `${r.levels_present.length} levels imaged.`),
      slot ? el('div', {}, el('button', { class: 'btn', type: 'button', onclick: () => on.slot?.(slot) }, icon(ICON.cube), 'Show in 3D')) : null);
  }

  function renderFindings() {
    list.replaceChildren();
    filterSlot.replaceChildren();
    if (levelFilter) {
      filterSlot.append(el('span', { class: 'filter-chip' }, `Level ${levelFilter.label}`,
        el('button', { type: 'button', 'aria-label': 'Show all findings', onclick: () => setLevelFilter(null) }, icon(ICON.close))));
    }
    const items = findings.filter((f) => !levelFilter || levelList(f.level).includes(levelFilter.key));
    if (!items.length) {
      list.append(el('p', { class: 'empty-note' }, levelFilter
        ? `Nothing specific was noted at ${levelFilter.label}.`
        : 'No findings were listed in this report.'));
    }
    for (const f of items) list.append(findingCard(f));
  }

  function findingCard(f) {
    if (cards.has(f.id)) return cards.get(f.id).article;
    const s = sev(f.severity);
    const bodyId = `fd-${cssId(f.id)}`;
    const head = el('button', { class: 'finding-head', type: 'button', 'aria-expanded': 'false', 'aria-controls': bodyId },
      el('span', { class: 'finding-tags' },
        el('span', { class: 'sev-chip' }, s.label),
        f.level ? el('span', { class: 'level-chip' }, f.level) : null),
      el('span', { class: 'finding-title' }, f.title),
      f.plain ? el('span', { class: 'finding-line' }, firstSentence(f.plain)) : null,
      icon(ICON.chevron, 'chev'));
    const body = el('div', { class: 'finding-body', id: bodyId, hidden: true });
    const article = el('article', { class: 'finding', id: `finding-${cssId(f.id)}`, style: { '--sev': s.color } }, head, body);
    let built = false;
    const setOpen = (open) => {
      if (open && !built) {
        body.append(...findingBody(f));
        built = true;
      }
      body.hidden = !open;
      head.setAttribute('aria-expanded', String(open));
      article.classList.toggle('is-open', open);
    };
    head.addEventListener('click', () => {
      const open = head.getAttribute('aria-expanded') !== 'true';
      setOpen(open);
      if (open) on.finding?.(f.id);
    });
    cards.set(f.id, { article, setOpen });
    return article;
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
    if (f.technical || f.technical_title) out.push(techDetails("The radiologist's wording", f.technical_title, f.technical));
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

  function setLevelFilter(level) {
    levelFilter = level ? { key: normLevel(level), label: level } : null;
    renderFindings();
  }

  function openFinding(id) {
    const f = findings.find((x) => x.id === id);
    if (!f) return;
    if (levelFilter && !levelList(f.level).includes(levelFilter.key)) setLevelFilter(null);
    showTab('overview');
    const card = cards.get(id);
    if (!card) return;
    card.setOpen(true);
    card.article.scrollIntoView({ block: 'start', behavior: reduceMotion.matches ? 'auto' : 'smooth' });
    card.article.classList.add('is-flash');
    setTimeout(() => card.article.classList.remove('is-flash'), 1400);
    on.reveal?.();
  }

  function renderLevels() {
    const pane = panes.levels;
    pane.replaceChildren();
    if (!report || !report.levels.length) {
      pane.append(emptyState('No level-by-level details', 'When the report lists each spinal level, a colour-coded map of your spine will appear here.'));
      return;
    }
    const regionIds = [...new Set([...report.regions.map((r) => r.id), ...report.levels.map((l) => l.region)])];
    const entryFor = (regionId, name) => report.levels.find((l) => l.region === regionId && normLevel(l.level) === name);
    const W = 300;
    const CX = 72;
    const shapes = [];
    const hits = new Map();
    let y = 4;

    for (const regionId of regionIds) {
      const region = report.regions.find((r) => r.id === regionId);
      const entries = report.levels.filter((l) => l.region === regionId);
      const names = [...entries.flatMap((l) => normLevel(l.level).split('-')), ...(region?.levels_present || []).map(normLevel)];
      const idx = names.map((n) => LEVEL_ORDER.indexOf(n)).filter((i) => i >= 0 && i < LEVEL_ORDER.length - 1);
      if (!idx.length) continue;
      const vs = LEVEL_ORDER.slice(Math.min(...idx), Math.max(...idx) + 1);
      shapes.push(svgEl('text', { x: 4, y: y + 14, fill: '#9dabc0', 'font-size': 11.5, 'font-weight': 700, 'letter-spacing': '0.08em' },
        (region?.name || regionId).toUpperCase()));
      y += 26;
      vs.forEach((v, i) => {
        const e = entryFor(regionId, v);
        const fill = e ? sev(e.status).color : '#e9e1cf';
        const shape = v === 'S1'
          ? svgEl('path', { class: 'shape', d: `M${CX - 34} ${y} H${CX + 34} L${CX + 16} ${y + 34} H${CX - 16} Z`, fill, stroke: 'transparent', 'stroke-linejoin': 'round' })
          : svgEl('rect', { class: 'shape', x: CX - 31, y, width: 62, height: 24, rx: 7, fill, stroke: 'transparent' });
        const h = v === 'S1' ? 34 : 24;
        const g = svgEl('g', { class: 'hit', role: 'button', tabindex: 0, 'aria-label': `${v}${e ? `, ${sev(e.status).label}` : ''}` },
          shape,
          svgEl('text', { x: CX, y: y + (v === 'S1' ? 15 : 16), 'text-anchor': 'middle', fill: '#141a26', 'font-size': 11.5, 'font-weight': 700 }, v === 'S1' ? 'Sacrum' : v));
        wire(g, v, regionId);
        shapes.push(g);
        y += h + 4;
        const below = vs[i + 1];
        if (!below) return;
        if (v === 'C1') { y += 8; return; }
        const name = `${v}-${below}`;
        const de = entryFor(regionId, name);
        const s = de ? sev(de.status) : null;
        const cy = y + 5;
        const dg = svgEl('g', { class: 'hit', role: 'button', tabindex: 0, 'aria-label': `${name} disc${s ? `, ${s.label}` : ''}` },
          svgEl('rect', { class: 'shape', x: CX - 27, y, width: 54, height: 10, rx: 5, fill: s ? s.color : '#6f86a8', stroke: 'transparent' }),
          svgEl('line', { x1: CX + 30, y1: cy, x2: 134, y2: cy, stroke: 'rgba(255,255,255,0.18)', 'stroke-width': 1 }),
          svgEl('text', { x: 140, y: cy + 4, fill: '#eef2f8', 'font-size': 12.5, 'font-weight': 650 }, name),
          svgEl('text', { x: 206, y: cy + 4, fill: s ? s.color : '#9dabc0', 'font-size': 12, 'font-weight': 600 }, s ? s.label : 'Not listed'));
        wire(dg, name, regionId);
        shapes.push(dg);
        y += 14;
      });
      y += 18;
    }

    function wire(g, name, regionId) {
      hits.set(`${regionId}:${name}`, g);
      const act = () => selectLevel(name, regionId);
      g.addEventListener('click', act);
      g.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); act(); }
      });
    }

    const svg = svgEl('svg', { class: 'spine-diagram', viewBox: `0 0 ${W} ${y}`, role: 'group', 'aria-label': 'Spine diagram. Choose a level to see details.' }, shapes);
    const placeholder = () => el('p', { class: 'muted small' }, 'Tap a level to see the details and focus the 3D view on it.');
    const detail = el('div', { class: 'card level-detail' }, placeholder());
    const legend = el('div', { class: 'legend' }, ['normal', 'mild', 'moderate', 'severe'].map((k) =>
      el('span', { class: 'sev-chip', style: { '--sev': sev(k).color } }, sev(k).label)));
    // On phones the diagram is taller than the screen, so the hint sits above it and a chosen level's
    // details pin to the bottom of the panel (CSS: .levels-hint, .level-detail.has-level).
    const hint = el('p', { class: 'levels-hint muted small' }, 'Tap a level for its details.');
    pane.append(el('div', { class: 'levels-wrap' }, legend, hint, el('div', { class: 'card' }, svg), detail, discHealth()));
    const clearLevel = () => {
      for (const g of hits.values()) g.classList.remove('is-selected');
      detail.classList.remove('has-level');
      detail.replaceChildren(placeholder());
      setLevelFilter(null);
    };

    // Measured water content of each disc, relative to spinal fluid on the same slices (automated).
    function discHealth() {
      const groups = regionIds.map((regionId) => {
        const rows = report.levels.filter((l) => l.region === regionId && l.measure && l.measure.disc_to_csf_signal != null);
        if (!rows.length) return null;
        const max = Math.max(...rows.map((l) => l.measure.disc_to_csf_signal));
        const region = report.regions.find((r) => r.id === regionId);
        return el('div', { class: 'disc-health-group' },
          el('div', { class: 'section-title' }, region?.name || regionId),
          rows.map((l) => {
            const v = l.measure.disc_to_csf_signal;
            const pct = Math.round((100 * v) / max);
            return el('button', { class: 'disc-bar', type: 'button', style: { '--sev': sev(l.status).color },
              title: `Disc water signal ${v.toFixed(2)} x spinal fluid${l.measure.disc_height_mm ? `, height ${l.measure.disc_height_mm} mm` : ''}`,
              onclick: () => selectLevel(normLevel(l.level), regionId) },
            el('span', { class: 'disc-bar-label' }, l.level),
            el('span', { class: 'disc-bar-track' }, el('span', { class: 'disc-bar-fill', style: { width: `${pct}%` } })),
            el('span', { class: 'disc-bar-value' }, `${pct}%`));
          }));
      }).filter(Boolean);
      if (!groups.length) return null;
      return el('div', { class: 'card disc-health' },
        el('h3', {}, 'Disc health, measured'),
        el('p', { class: 'muted small' }, 'How much water each disc holds, from the brightness of the disc on the scan compared with the spinal fluid. 100% is the best-hydrated disc in that part of your spine. Healthy discs sit close together; a short bar means a drier disc.'),
        groups);
    }

    function selectLevel(name, regionId) {
      for (const [k, g] of hits) g.classList.toggle('is-selected', k === `${regionId}:${name}`);
      const e = entryFor(regionId, name);
      const at = findings.filter((f) => levelList(f.level).includes(name));
      setLevelFilter(e?.level || name);
      on.level?.(name, regionId);
      const rows = [];
      if (e) {
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
        const sp = e.specialist;
        if (sp) {
          const flags = sp.flags.map((f) => (FLAG[f] || f) + (sp.items[f] != null ? ` (${Math.round(sp.items[f] * 100)}%)` : ''));
          rows.push(['Specialist model', [
            `Disc drying grade ${sp.pfirrmann} of 5 (${Math.round(sp.pfirrmann_confidence * 100)}% sure)`,
            flags.length ? ` · also flags: ${flags.join(', ')}` : '',
            sp.not_blind ? el('span', { class: 'level-chip not-blind', title: 'This scan already feeds a mistake-log rule, so this re-run is not a blind reading.' }, 'not blind') : null,
          ]]);
        }
      }
      detail.classList.add('has-level');
      if (phone.matches) on.expand?.();
      detail.replaceChildren(
        el('div', { class: 'summary-top' }, el('h3', {}, e?.level || name),
          e ? el('span', { class: 'sev-chip', style: { '--sev': sev(e.status).color } }, sev(e.status).label) : null,
          el('button', { class: 'icon-btn detail-close', type: 'button', 'aria-label': 'Close the level details', onclick: clearLevel }, icon(ICON.close))),
        rows.length ? el('div', { style: { marginTop: '8px' } }, rows.map(([k, v]) => el('div', { class: 'row' }, el('span', {}, k), el('span', {}, v))))
          : el('p', { class: 'muted small', style: { marginTop: '8px' } }, 'No specific notes for this level.'),
        at.length ? el('div', { class: 'section-title' }, 'Findings here') : null,
        at.length ? el('div', { class: 'mark-list' }, at.map((f) => el('button', {
          class: 'mark-item', type: 'button', style: { '--sev': sev(f.severity).color },
          onclick: () => { openFinding(f.id); on.finding?.(f.id); },
        }, el('span', { class: 'dot', 'aria-hidden': 'true' }), el('span', { class: 'grow' }, f.title), el('span', { class: 'sub' }, 'Open')))) : null);
    }
  }

  function gradeChip(g) {
    return el('span', { class: 'sev-chip', style: { '--sev': sev(g === 'none' ? 'normal' : g).color } }, GRADE[g] || g);
  }

  function renderWords() {
    const pane = panes.words;
    pane.replaceChildren();
    if (!report) {
      pane.append(emptyState('Nothing here yet', 'Word explanations and questions for your doctor appear once a report is loaded.'));
      return;
    }
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
      pane.append(el('div', { class: 'section-title' }, 'Words, explained'), el('div', { class: 'stack' }, search, dl));
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
      pane.append(el('div', { class: 'section-title' }, el('span', {}, 'Questions for your doctor'), copy),
        el('ol', { class: 'questions' }, qs.map((q) => el('li', {}, el('span', {}, q)))));
    }
    const m = report.method;
    if (m.ai_models.length || m.readers || m.limitations.length) {
      pane.append(el('div', { class: 'section-title' }, 'How this was made'),
        el('div', { class: 'card' },
          m.readers ? el('p', { class: 'small' }, m.readers) : null,
          m.ai_models.length ? el('p', { class: 'small muted' }, `Tools: ${m.ai_models.join(', ')}`) : null,
          m.limitations.length ? el('details', { class: 'tech' }, el('summary', {}, 'Limitations'),
            el('ul', { class: 'method-list' }, m.limitations.map((l) => el('li', {}, l)))) : null));
    }
    if (!pane.children.length) pane.append(emptyState('Nothing here yet', 'This report has no word list or questions.'));
  }

  // Compare tab: the AI's blind reading vs the outside radiologist report vs the AI after re-checking.
  // Short phrases first; the long reasoning sits behind "Why".
  const VERDICT = {
    agree: { label: 'Agree', tone: 'agree' },
    ai: { label: 'Favors AI', tone: 'ai' },
    radiologist: { label: 'Favors radiologist', tone: 'rad' },
    both: { label: 'Both fair', tone: 'both' },
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

  function compareRow(r) {
    const v = verdictOf(r);
    const same = short(r, 'blind') === short(r, 'radiologist') && short(r, 'radiologist') === short(r, 'after');
    const model = MODEL && r.model_short ? [
      el('span', { class: 'cmp-k cmp-k-model', title: MODEL[2] }, MODEL[1]),
      el('span', { class: 'cmp-v cmp-v-model' }, r.model_short),
    ] : null;
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
      r.finding ? el('button', { class: 'btn ghost', type: 'button', onclick: () => { openFinding(r.finding); on.finding?.(r.finding); showTab('overview'); } },
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
    const pane = panes.compare;
    pane.replaceChildren();
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
    pane.append(
      el('section', { class: 'card cmp-hero' },
        el('div', { class: 'cmp-hero-top' },
          el('h3', {}, 'AI reading vs the radiologist'),
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
              MODEL ? el('th', { class: 'cmp-v-model', title: MODEL[2] }, MODEL[1]) : null, el('th', {}, 'Verdict'))),
            el('tbody', {}, c.rows.map((r) => {
              const v = verdictOf(r);
              return el('tr', { class: `tone-${v.tone}` },
                el('td', { class: 'compare-point' }, el('strong', {}, r.level), el('span', {}, r.topic)),
                COLS.map(([k]) => el('td', { class: `cmp-v-${k}` }, short(r, k) || '-')),
                MODEL ? el('td', { class: 'cmp-v-model' }, r.model_short) : null,
                el('td', {}, el('span', { class: `cmp-pill tone-${v.tone}` }, v.label)));
            }))))));
    document.body.append(overlay);
    document.addEventListener('keydown', onKey);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
    overlay.querySelector('button').focus();
  }

  renderOverview();
  if (panes.compare) renderCompare();
  viewer = createViewer(panes.viewer, { stacks, report, onOpenFinding: (id) => { openFinding(id); on.finding?.(id); } });
  renderLevels();
  renderWords();

  return { openFinding, showTab };
}

// main.js: loads the data, then wires the 3D scene, the panel and the controls together.
import { listen, observeResize } from './lifetime.js';
import { loadReport, loadStacks, el, fmtMonth, sexLabel, reportFlaggedDiscs, levelStatusMap, sev, SEVERITY } from './data.js';
import { createPanel } from './panel.js';

const $ = (s) => document.querySelector(s);
const desktop = window.matchMedia('(min-width: 861px)');
const NOT_A_DEVICE = 'Not a medical device: not cleared by any regulator, and never for diagnosis or treatment decisions.';
const DEFAULT_DISCLAIMER = 'This page explains an MRI report in plain words; always talk to a doctor about your results.';

boot().catch(() => {
  $('#loader')?.classList.add('is-done');
  const pane = $('#pane-report');
  if (pane) pane.replaceChildren(el('div', { class: 'empty-state' }, el('h3', {}, 'Something went wrong'), el('p', {}, 'Please reload the page.')));
});

async function boot() {
  const { report, sample } = await loadReport();
  const stacks = await loadStacks(sample);

  renderHeader(report, sample);
  setupDisclaimer(`${NOT_A_DEVICE} ${report?.disclaimer || DEFAULT_DISCLAIMER}`);

  const panelEl = $('#panel');
  const getInset = () => {
    const footer = $('#disclaimer').offsetHeight || 0;
    if (desktop.matches) return { right: panelEl.offsetWidth + 16, bottom: 0, top: 0 };
    const dock = $('.dock');
    const top = dock && !dock.hidden ? dock.getBoundingClientRect().bottom + 6 : 0;
    return { right: 0, bottom: panelEl.offsetHeight + footer, top };
  };

  let scene = null;
  let incidentalColor = '#5cc2f2';
  try {
    incidentalColor = (await import('./spine.js')).INCIDENTAL_3D;
    const mod = await import('./scene.js');
    scene = mod.createScene({ canvas: $('#scene'), labelLayer: $('#labels'), tooltip: $('#tooltip'), getInset });
  } catch {
    scene = null;
  }
  if (!scene) {
    $('#scene-fallback').hidden = false;
    $('.dock').hidden = true;
  }

  // The mode switch (AI report / Radiologist / Good & bad) drives the 3D colours, rings and the colour key too.
  const setKey = wireKey(report, incidentalColor);
  const panel = createPanel(panelEl, {
    report,
    stacks,
    incidentalColor,
    on: {
      mode: (m) => { scene?.setMode(m); setKey(m); },
      finding: (id) => scene?.focusFinding(id),
      level: (level, region) => scene?.focusLevel(level, region),
      slot: (slot) => scene?.focusSlot(slot),
      reveal: () => sheet.open(),
      expand: () => sheet.full(),
    },
  });
  const sheet = createSheet(panelEl);
  wirePanelSize(panelEl, sheet);

  if (scene) {
    scene.onFindingClick = (id) => panel.openFinding(id);
    const markModel = wireDock(scene);
    const resize = () => scene.resize();
    listen(window, 'resize', resize);
    desktop.addEventListener?.('change', resize);
    if ('ResizeObserver' in window) observeResize(resize, panelEl);
    try {
      await scene.load(report);
    } catch {
      // The body and ghost spine are already on screen; a failed region load only loses the extras.
    }
    scene.setModel('skeleton').then((ok) => { if (!ok) markModel('glass'); });   // the skeleton is the default model
    scene.intro();
  }
  $('#loader').classList.add('is-done');
}

function renderHeader(report, sample) {
  const meta = $('#meta');
  meta.replaceChildren();
  if (report) {
    const p = report.patient;
    const who = [p.label, p.age != null ? `${p.age}` : '', sexLabel(p.sex)].filter(Boolean).join(' · ');
    const study = [fmtMonth(report.study.month), report.study.field_strength, report.study.contrast ? 'with contrast' : ''].filter(Boolean).join(' · ');
    if (who) meta.append(el('span', { class: 'meta-chip' }, who));
    if (study) meta.append(el('span', { class: 'meta-chip' }, study));
  }
  $('#sample-badge').hidden = !sample;
  if (sample) document.title = `Sample data · ${document.title}`;
}

// Phones: the results panel is a bottom sheet with three heights: tucked away (only the tabs show),
// half and full. Drag the grip or the tab row, or pull the content down from its top, to move it;
// a flick moves one step. A tap on the grip opens it or tucks it away; a tab tap opens it.
// The next step when the handle is tapped: peek opens to half, anything open tucks to peek.
const HANDLE_NEXT = { peek: 'half', half: 'peek', full: 'half' };

// Pointer drags on the handle, the mode row and the tabs. Each one starts, moves and ends a gesture.
function wireDragZones(zones, gesture) {
  for (const zone of zones) {
    zone.addEventListener('pointerdown', (e) => {
      if (desktop.matches || (e.pointerType === 'mouse' && e.button !== 0)) return;
      gesture.start(e.clientY, { el: zone, id: e.pointerId });
    });
    zone.addEventListener('pointermove', (e) => gesture.move(e.clientY));
    zone.addEventListener('pointerup', gesture.end);
    zone.addEventListener('pointercancel', gesture.end);
  }
}

// Pull the content down from its top to lower the sheet; push it up to raise a half sheet.
// Gestures that belong to the content itself (the slice scrubber, form fields) are left alone.
function wirePull(body, gesture) {
  let pull = null;
  body.addEventListener('touchstart', (e) => {
    if (desktop.matches || e.touches.length !== 1 || e.target.closest('.stage, input, select, textarea, .level-detail.has-level')) {
      pull = null;
      return;
    }
    const pane = [...body.children].find((p) => !p.hidden);
    pull = { y: e.touches[0].clientY, atTop: !pane || pane.scrollTop <= 0, active: false };
  }, { passive: true });
  body.addEventListener('touchmove', (e) => {
    if (!pull) return;
    const y = e.touches[0].clientY;
    const dy = y - pull.y;
    if (!pull.active) {
      if (Math.abs(dy) < 6) return;
      if (!pull.atTop || (dy < 0 && gesture.snap === 'full')) {
        pull = null;
        return;
      }
      pull.active = true;
      gesture.start(pull.y, null);
    }
    e.preventDefault();
    gesture.move(y);
  }, { passive: false });
  const release = () => {
    if (pull?.active) gesture.end();
    pull = null;
  };
  body.addEventListener('touchend', release);
  body.addEventListener('touchcancel', release);
}

function createSheet(panelEl) {
  const handle = $('#sheet-toggle');
  const tabsEl = panelEl.querySelector('.tabs');
  const modeEl = panelEl.querySelector('.mode-switch');
  const body = panelEl.querySelector('.panel-body');
  const ORDER = ['peek', 'half', 'full'];
  let snap = 'half';
  let drag = null;
  let swallowClick = false;
  const api = {
    open: () => { if (snap === 'peek') set('half'); },
    full: () => set('full'),
    half: () => set('half'),
    get snap() { return snap; },
    onSnap: null,
  };

  const sizes = () => {
    const foot = $('#disclaimer').offsetHeight || 0;
    const top = $('.topbar').getBoundingClientRect().bottom + 8;
    const vh = window.innerHeight;
    const peek = handle.offsetHeight + modeEl.offsetHeight + tabsEl.offsetHeight + 1;
    return { peek, half: Math.max(peek, Math.round(vh * 0.5) - foot), full: Math.max(peek, vh - foot - top) };
  };
  function set(name) {
    snap = name;
    api.onSnap?.();
    if (desktop.matches) {
      panelEl.style.height = '';
      delete panelEl.dataset.snap;
      return;
    }
    panelEl.dataset.snap = name;
    panelEl.style.height = `${sizes()[name]}px`;
    handle.setAttribute('aria-expanded', String(name !== 'peek'));
    handle.querySelector('.sr-only').textContent = name === 'peek' ? 'Open the results panel' : 'Tuck the results panel away';
  }

  function start(y, pointer) {
    drag = { y0: y, h0: panelEl.getBoundingClientRect().height, y, t: performance.now(), v: 0, moved: false, pointer };
  }
  function move(y) {
    if (!drag) return;
    const dy = y - drag.y0;
    if (!drag.moved) {
      if (Math.abs(dy) < 6) return;
      drag.moved = true;
      panelEl.classList.add('is-dragging');
      if (drag.pointer) drag.pointer.el.setPointerCapture?.(drag.pointer.id);
    }
    const now = performance.now();
    drag.v = 0.7 * drag.v + 0.3 * ((y - drag.y) / Math.max(1, now - drag.t));   // px per ms, positive = down
    drag.y = y;
    drag.t = now;
    const s = sizes();
    panelEl.style.height = `${Math.min(s.full, Math.max(s.peek, drag.h0 - dy))}px`;
  }
  function end() {
    const d = drag;
    drag = null;
    if (!d?.moved) return false;
    panelEl.classList.remove('is-dragging');
    const s = sizes();
    const h = panelEl.getBoundingClientRect().height;
    let i;
    if (Math.abs(d.v) > 0.4) {
      // a flick: one step from where the drag started, in its direction
      i = Math.max(0, Math.min(2, ORDER.indexOf(snap) + (d.v < 0 ? 1 : -1)));
    } else {
      i = ORDER.reduce((best, k, j) => (Math.abs(s[k] - h) < Math.abs(s[ORDER[best]] - h) ? j : best), 0);
    }
    set(ORDER[i]);
    swallowClick = true;   // the click that ends a drag is not a tap
    setTimeout(() => { swallowClick = false; }, 60);
    return true;
  }

  const gesture = { start, move, end, get snap() { return snap; } };
  wireDragZones([handle, modeEl, tabsEl], gesture);
  panelEl.addEventListener('click', (e) => {
    if (swallowClick) {
      swallowClick = false;
      e.stopPropagation();
      e.preventDefault();
    }
  }, true);
  handle.addEventListener('click', () => set(HANDLE_NEXT[snap]));
  for (const zone of [modeEl, tabsEl]) zone.addEventListener('click', () => { if (snap === 'peek') set('half'); });
  wirePull(body, gesture);

  listen(window, 'resize', () => set(snap));
  desktop.addEventListener?.('change', () => set(snap));
  set('half');
  return api;
}

// The panel's width button. Desktop: normal -> wide -> full -> normal (remembered, Escape leaves full).
// Phone: the same button flips the sheet between full and half.
function wirePanelSize(panelEl, sheet) {
  const btn = $('#panel-size');
  const KEY = 'raidiology.panelSize';
  const SIZES = ['normal', 'wide', 'full'];
  const NEXT = {
    normal: 'Make the results panel wider',
    wide: 'Make the results panel full screen',
    full: 'Return the results panel to normal size',
  };
  let size = 'normal';
  try { size = localStorage.getItem(KEY); } catch { /* private mode: no memory */ }
  if (!SIZES.includes(size)) size = 'normal';

  function show() {
    let label;
    if (desktop.matches) {
      document.documentElement.dataset.panel = size;
      btn.dataset.size = size;
      label = NEXT[size];
    } else {
      delete document.documentElement.dataset.panel;
      const full = sheet.snap === 'full';
      btn.dataset.size = full ? 'full' : 'normal';
      label = full ? 'Shrink the results panel to half height' : 'Expand the results panel to full height';
    }
    btn.setAttribute('aria-label', label);
    btn.title = label;
  }
  function setSize(next) {
    size = next;
    try { localStorage.setItem(KEY, size); } catch { /* ignore */ }
    show();
  }
  btn.addEventListener('click', () => {
    if (desktop.matches) setSize(SIZES[(SIZES.indexOf(size) + 1) % SIZES.length]);
    else if (sheet.snap === 'full') sheet.half();
    else sheet.full();
  });
  listen(document, 'keydown', (e) => {
    if (e.key === 'Escape' && desktop.matches && size === 'full' && $('#model-menu').hidden) setSize('normal');
  });
  sheet.onSnap = show;
  desktop.addEventListener?.('change', show);
  show();
}

// The 3D colour key (#rings-legend): one swatch per status colour the 3D shows in the current mode, taken from the
// same status map the 3D tints with. AI: the AI's colours. Radiologist: its colours plus its short note. Good & bad:
// the AI's colours plus the violet ring entry. Returns the function that rebuilds the key for a mode.
function wireKey(report, incidentalColor) {
  const key = $('#rings-legend');
  const flagged = reportFlaggedDiscs(report).length > 0;
  const swatch = (color, label) => el('span', { class: 'key-item' }, el('span', { class: 'key-dot', style: `background:${color}` }), label);
  const aiEntries = () => {
    const present = new Set([...levelStatusMap(report).values()].map((s) => sev(s).key));
    return [
      ['incidental', 'Noted by the AI, normal for age', incidentalColor],
      ['mild', 'Mild', SEVERITY.mild.color],
      ['moderate', 'Moderate', SEVERITY.moderate.color],
      ['severe', 'Severe', SEVERITY.severe.color],
    ].filter(([k]) => present.has(k)).map(([, label, color]) => swatch(color, label));
  };
  return (mode) => {
    const items = [];
    if (mode === 'radiologist') {
      const r = report?.comparison?.report;
      const present = new Set(r?.levels?.values() || []);
      for (const k of ['mild', 'moderate', 'severe']) if (present.has(k)) items.push(swatch(SEVERITY[k].color, SEVERITY[k].label));
      if (r?.levels_note) items.push(el('span', { class: 'key-note' }, r.levels_note));
    } else {
      items.push(...aiEntries());
      if (mode === 'compare' && flagged) items.push(el('span', { class: 'key-item' }, el('span', { class: 'rings-key' }), "Ring: the radiologist's report flagged it"));
    }
    key.replaceChildren(...items);
    key.hidden = !items.length;
  };
}

function setupDisclaimer(text) {
  const box = $('#disclaimer');
  const btn = $('#disclaimer-btn');
  $('#disclaimer-text').textContent = text;
  btn.title = text;
  btn.addEventListener('click', () => {
    const open = !box.classList.contains('is-open');
    box.classList.toggle('is-open', open);
    btn.setAttribute('aria-expanded', String(open));
  });
}

// Floating menu (outside the dock, which scrolls sideways on phones and would clip it).
function wireModelPicker(scene) {
  const btn = $('#model-btn');
  const menu = $('#model-menu');
  const label = $('#model-label');
  const items = [...menu.querySelectorAll('[data-model]')];
  const navItems = [...menu.querySelectorAll('button[role^="menuitem"]')];   // models and the two toggle rows
  const place = () => {
    const r = btn.getBoundingClientRect();
    menu.style.left = `${Math.max(8, Math.min(r.left, window.innerWidth - menu.offsetWidth - 8))}px`;
    const below = r.top < window.innerHeight / 2;
    menu.style.top = below ? `${r.bottom + 8}px` : '';
    menu.style.bottom = below ? '' : `${window.innerHeight - r.top + 8}px`;
  };
  const setOpen = (open) => {
    menu.hidden = !open;
    btn.setAttribute('aria-expanded', String(open));
    if (open) {
      place();
      (items.find((i) => i.getAttribute('aria-checked') === 'true') || items[0]).focus();
    }
  };
  btn.addEventListener('click', (e) => { e.stopPropagation(); setOpen(menu.hidden); });
  listen(document, 'click', (e) => { if (!menu.hidden && !menu.contains(e.target)) setOpen(false); });
  listen(document, 'keydown', (e) => { if (e.key === 'Escape' && !menu.hidden) { setOpen(false); btn.focus(); } });
  listen(window, 'resize', () => { if (!menu.hidden) place(); });
  menu.addEventListener('keydown', (e) => {
    const i = navItems.indexOf(document.activeElement);
    const d = { ArrowDown: 1, ArrowUp: -1 }[e.key];
    if (d && i >= 0) { e.preventDefault(); navItems[(i + d + navItems.length) % navItems.length].focus(); }
  });
  // Shows `name` as the chosen model (menu check and button label) without loading anything.
  const mark = (name) => {
    for (const o of items) o.setAttribute('aria-checked', String(o.dataset.model === name));
    label.textContent = items.find((o) => o.dataset.model === name)?.querySelector('strong').textContent || label.textContent;
  };
  for (const it of items) {
    it.addEventListener('click', async () => {
      mark(it.dataset.model);
      setOpen(false);
      btn.classList.add('is-loading');
      const ok = await scene.setModel(it.dataset.model);
      btn.classList.remove('is-loading');
      if (!ok) label.textContent = `${label.textContent} (unavailable)`;
    });
  }
  return mark;
}

// On phones the zoom buttons ride the top edge of the bottom sheet (and hide when the sheet is
// nearly full height, where they would sit under the dock). On desktop CSS puts them left of the panel.
function wireZoomPlacement() {
  const box = $('#zoom-ctl');
  const panelEl = $('#panel');
  const place = () => {
    if (desktop.matches) { box.style.bottom = ''; box.style.visibility = ''; return; }
    const top = panelEl.getBoundingClientRect().top;
    const dock = $('.dock').getBoundingClientRect().bottom;
    box.style.bottom = `${Math.round(window.innerHeight - top + 12)}px`;
    box.style.visibility = top - 12 - box.offsetHeight < dock + 6 ? 'hidden' : '';
  };
  if ('ResizeObserver' in window) observeResize(place, panelEl);
  listen(window, 'resize', place);
  desktop.addEventListener?.('change', place);
  panelEl.addEventListener('transitionend', place);
  place();
}

function wireDock(scene) {
  for (const b of document.querySelectorAll('.dock [data-action]')) {
    b.addEventListener('click', () => {
      const a = b.dataset.action;
      if (a === 'reset') scene.reset();
      else scene.focusSlot(a);
    });
  }
  const markModel = wireModelPicker(scene);
  for (const b of document.querySelectorAll('[data-zoom]')) {
    b.addEventListener('click', () => scene.zoomBy(b.dataset.zoom === 'in' ? 1.4 : 1 / 1.4));
  }
  wireZoomPlacement();
  // Ghost spine and Labels are toggle rows at the bottom of the model menu.
  const setters = { ghost: scene.setGhost, labels: scene.setLabels };
  for (const b of document.querySelectorAll('[data-toggle]')) {
    b.addEventListener('click', () => {
      const on = b.getAttribute('aria-checked') !== 'true';
      b.setAttribute('aria-checked', String(on));
      setters[b.dataset.toggle]?.(on);
    });
  }
  return markModel;
}

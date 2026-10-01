// main.js: loads the data, then wires the 3D scene, the panel and the controls together.
import { loadReport, loadStacks, el, fmtMonth, sexLabel } from './data.js';
import { createPanel } from './panel.js';

const $ = (s) => document.querySelector(s);
const desktop = window.matchMedia('(min-width: 861px)');
const NOT_A_DEVICE = 'Not a medical device: not cleared by any regulator, and never for diagnosis or treatment decisions.';
const DEFAULT_DISCLAIMER = 'This page explains an MRI report in plain words; always talk to a doctor about your results.';

boot().catch(() => {
  $('#loader')?.classList.add('is-done');
  const pane = $('#pane-overview');
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
  try {
    const mod = await import('./scene.js');
    scene = mod.createScene({ canvas: $('#scene'), labelLayer: $('#labels'), tooltip: $('#tooltip'), getInset });
  } catch {
    scene = null;
  }
  if (!scene) {
    $('#scene-fallback').hidden = false;
    $('.dock').hidden = true;
  }

  const panel = createPanel(panelEl, {
    report,
    stacks,
    on: {
      finding: (id) => scene?.focusFinding(id),
      level: (level, region) => scene?.focusLevel(level, region),
      slot: (slot) => scene?.focusSlot(slot),
      reveal: () => sheet.open(),
      expand: () => sheet.full(),
    },
  });
  const sheet = createSheet(panelEl);

  if (scene) {
    scene.onFindingClick = (id) => panel.openFinding(id);
    wireDock(scene);
    const resize = () => scene.resize();
    window.addEventListener('resize', resize);
    desktop.addEventListener?.('change', resize);
    if ('ResizeObserver' in window) new ResizeObserver(resize).observe(panelEl);
    try {
      await scene.load(report);
    } catch {
      // The body and ghost spine are already on screen; a failed region load only loses the extras.
    }
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
function createSheet(panelEl) {
  const handle = $('#sheet-toggle');
  const tabsEl = panelEl.querySelector('.tabs');
  const body = panelEl.querySelector('.panel-body');
  const ORDER = ['peek', 'half', 'full'];
  let snap = 'half';
  let drag = null;
  let swallowClick = false;

  const sizes = () => {
    const foot = $('#disclaimer').offsetHeight || 0;
    const top = $('.topbar').getBoundingClientRect().bottom + 8;
    const vh = window.innerHeight;
    const peek = handle.offsetHeight + tabsEl.offsetHeight + 1;
    return { peek, half: Math.max(peek, Math.round(vh * 0.5) - foot), full: Math.max(peek, vh - foot - top) };
  };
  function set(name) {
    snap = name;
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

  for (const zone of [handle, tabsEl]) {
    zone.addEventListener('pointerdown', (e) => {
      if (desktop.matches || (e.pointerType === 'mouse' && e.button !== 0)) return;
      start(e.clientY, { el: zone, id: e.pointerId });
    });
    zone.addEventListener('pointermove', (e) => move(e.clientY));
    zone.addEventListener('pointerup', end);
    zone.addEventListener('pointercancel', end);
  }
  panelEl.addEventListener('click', (e) => {
    if (swallowClick) {
      swallowClick = false;
      e.stopPropagation();
      e.preventDefault();
    }
  }, true);
  handle.addEventListener('click', () => set(snap === 'peek' ? 'half' : snap === 'half' ? 'peek' : 'half'));
  tabsEl.addEventListener('click', () => { if (snap === 'peek') set('half'); });

  // Pull the content down from its top to lower the sheet; push it up to raise a half sheet.
  // Gestures that belong to the content itself (the slice scrubber, form fields) are left alone.
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
      if (!pull.atTop || (dy < 0 && snap === 'full')) {
        pull = null;
        return;
      }
      pull.active = true;
      start(pull.y, null);
    }
    e.preventDefault();
    move(y);
  }, { passive: false });
  const release = () => {
    if (pull?.active) end();
    pull = null;
  };
  body.addEventListener('touchend', release);
  body.addEventListener('touchcancel', release);

  window.addEventListener('resize', () => set(snap));
  desktop.addEventListener?.('change', () => set(snap));
  set('half');
  return { open: () => { if (snap === 'peek') set('half'); }, full: () => set('full') };
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
  document.addEventListener('click', (e) => { if (!menu.hidden && !menu.contains(e.target)) setOpen(false); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !menu.hidden) { setOpen(false); btn.focus(); } });
  window.addEventListener('resize', () => { if (!menu.hidden) place(); });
  menu.addEventListener('keydown', (e) => {
    const i = items.indexOf(document.activeElement);
    const d = { ArrowDown: 1, ArrowUp: -1 }[e.key];
    if (d && i >= 0) { e.preventDefault(); items[(i + d + items.length) % items.length].focus(); }
  });
  for (const it of items) {
    it.addEventListener('click', async () => {
      for (const o of items) o.setAttribute('aria-checked', String(o === it));
      label.textContent = it.querySelector('strong').textContent;
      setOpen(false);
      btn.classList.add('is-loading');
      const ok = await scene.setModel(it.dataset.model);
      btn.classList.remove('is-loading');
      if (!ok) label.textContent = `${label.textContent} (unavailable)`;
    });
  }
}

function wireDock(scene) {
  for (const b of document.querySelectorAll('.dock [data-action]')) {
    b.addEventListener('click', () => {
      const a = b.dataset.action;
      if (a === 'reset') scene.reset();
      else scene.focusSlot(a);
    });
  }
  wireModelPicker(scene);
  const setters = { body: scene.setBody, ghost: scene.setGhost, labels: scene.setLabels };
  for (const b of document.querySelectorAll('.dock [data-toggle]')) {
    b.addEventListener('click', () => {
      const on = b.getAttribute('aria-pressed') !== 'true';
      b.setAttribute('aria-pressed', String(on));
      setters[b.dataset.toggle]?.(on);
    });
  }
}

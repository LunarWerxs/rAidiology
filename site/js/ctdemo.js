// The home page's live look at the sample CT: 46 real slices with the open research model's
// outlines over them. Names appear only here, on the sample. The same sample opens full size in
// the open viewer (open.html?sample=1). Every slice change is announced on window as
// `raidiology:slice` so the 3D skeleton can move its scan plane.
import { prefersReducedMotion } from './motion.js';

const NS = 'http://www.w3.org/2000/svg';
const PLAY_MS = 6000;

function make(tag, cls, attrs) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
  return e;
}

function svgEl(tag, attrs) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
  return e;
}

export function mountCtDemo(box, opts = {}) {
  const ac = new AbortController();
  const on = (t, type, fn, o) => t.addEventListener(type, fn, { signal: ac.signal, ...o });
  const src = new URL(box.dataset.src || opts.src || 'sample/demo/demo.json', document.baseURI);
  let demo = null;
  let index = 0;
  let activeId = null;       // the outline named by the chip
  let pinned = false;        // set by a tap or click, so leaving the picture keeps it
  let raf = 0;
  let playing = false;
  let started = false;
  let io = null;
  let dead = false;
  const loaded = new Set();

  box.classList.add('ctdemo', 'is-loading');
  box.setAttribute('aria-busy', 'true');
  const phSrc = box.querySelector('.demo-ph img');
  box.replaceChildren();

  const lightbox = make('div', 'ctdemo__lightbox');
  const img = make('img', 'ctdemo__img', { alt: 'A slice of the sample CT', draggable: 'false' });
  const svg = svgEl('svg', { class: 'ctdemo__svg', 'aria-hidden': 'true' });
  const chip = make('div', 'ctdemo__chip');
  chip.hidden = true;
  // The placeholder frame from the HTML stays under the live picture until it paints, so nothing shifts and no text shows.
  const ph = phSrc ? phSrc.cloneNode() : null;
  if (ph) { ph.classList.add('ctdemo__ph'); ph.alt = ''; ph.setAttribute('aria-hidden', 'true'); }
  lightbox.append(...(ph ? [ph] : []), img, svg, chip);

  const bar = make('div', 'ctdemo__bar');
  const prev = make('button', 'ctdemo__step', { type: 'button', 'aria-label': 'Previous slice' });
  prev.textContent = '‹';
  const next = make('button', 'ctdemo__step', { type: 'button', 'aria-label': 'Next slice' });
  next.textContent = '›';
  const range = make('input', 'ctdemo__range', { type: 'range', min: '0', max: '0', step: '1', value: '0', 'aria-label': 'Picture' });
  bar.append(prev, range, next);
  const count = make('p', 'ctdemo__count');
  count.setAttribute('aria-live', 'off');
  const credit = make('p', 'ctdemo__credit');
  const link = make('a', 'ctdemo__link', { href: 'open.html?sample=1' });
  link.textContent = 'Open the full viewer';
  const foot = make('div', 'ctdemo__foot');
  foot.append(count);
  box.append(lightbox, bar, foot, credit);
  const line = make('span', 'ctdemo__line');
  line.textContent = 'Sample CT from TotalSegmentator (CC BY 4.0). Outlines by an open research model.';
  credit.append(line, link);
  box.tabIndex = 0;
  box.setAttribute('role', 'group');
  box.setAttribute('aria-label', 'Sample CT viewer. Arrow keys change the slice.');

  function failed() {
    box.classList.remove('is-loading');
    box.classList.add('is-failed');
    box.removeAttribute('aria-busy');
    credit.replaceChildren(link); // the full viewer is still one tap away
  }
  on(img, 'load', () => {
    box.classList.remove('is-loading');
    box.removeAttribute('aria-busy');
    if (ph) ph.remove();
  }, { once: true });
  on(img, 'error', failed, { once: true });

  const frameUrl = (i) => new URL(demo.frames[i], src).href;
  const mid = () => Math.floor((demo.count - 1) / 2);

  function drawOutlines() {
    svg.replaceChildren();
    const sl = demo.slices[index];
    const byId = new Map();
    for (const p of sl.paths) byId.set(p.s, (byId.get(p.s) || '') + p.d);
    const items = [];
    for (const [id, d] of byId) {
      const meta = demo.byId.get(id);
      const path = svgEl('path', { d, 'fill-rule': 'evenodd', stroke: meta.color, 'data-id': id });
      path.setAttribute('class', 'ctdemo__outline');
      items.push(path);
      svg.append(path);
    }
    // Biggest first underneath, so the smallest outline wins where two overlap.
    const area = new Map();
    for (const p of items) { const b = p.getBBox(); area.set(p, b.width * b.height); }
    items.sort((a, b) => area.get(b) - area.get(a)).forEach((p) => svg.append(p));
    for (const p of items) {
      on(p, 'pointerenter', (e) => { if (e.pointerType === 'mouse' && !pinned) setActive(Number(p.dataset.id)); });
      on(p, 'click', (e) => { e.stopPropagation(); if (moved) return; pinned = true; setActive(Number(p.dataset.id)); });
    }
    if (activeId !== null && !byId.has(activeId)) { activeId = null; pinned = false; }
    paintActive();
  }

  function paintActive() {
    for (const p of svg.children) p.classList.toggle('is-active', Number(p.dataset.id) === activeId);
    const el = [...svg.children].find((p) => Number(p.dataset.id) === activeId);
    if (!el) { chip.hidden = true; return; }
    const b = el.getBBox();
    const [w, h] = demo.size;
    const nm = demo.byId.get(activeId).name;
    chip.textContent = nm.charAt(0).toUpperCase() + nm.slice(1);
    chip.hidden = false;
    const cx = ((b.x + b.width / 2) / w) * 100;
    const top = (b.y / h) * 100;
    chip.style.left = `${Math.min(86, Math.max(14, cx))}%`;
    chip.style.top = `${top < 14 ? Math.min(((b.y + b.height) / h) * 100 + 2, 88) : top - 2}%`;
    chip.classList.toggle('is-below', top < 14);
  }

  function setActive(id) {
    activeId = id;
    paintActive();
  }

  function clear() {
    pinned = false;
    activeId = null;
    paintActive();
  }

  function show(i, fromUser) {
    i = Math.max(0, Math.min(demo.count - 1, i));
    if (fromUser) stopPlay();
    if (i === index && img.getAttribute('src')) return;
    index = i;
    img.src = frameUrl(i);
    range.value = String(i);
    count.textContent = `Picture ${i + 1} of ${demo.count}`;
    range.setAttribute('aria-valuetext', count.textContent);
    drawOutlines();
    window.dispatchEvent(new CustomEvent('raidiology:slice', {
      detail: { index: i, count: demo.count, level: demo.slices[i].level || null },
    }));
  }

  function restingOutline() {
    // One outline named on the middle slice: the liver if it is there, else the largest.
    const ps = [...svg.children];
    const liver = ps.find((p) => demo.byId.get(Number(p.dataset.id)).name === 'liver');
    let pick = liver;
    if (!pick) {
      let best = 0;
      for (const p of ps) { const b = p.getBBox(); if (b.width * b.height > best) { best = b.width * b.height; pick = p; } }
    }
    if (pick) setActive(Number(pick.dataset.id));
  }

  function stopPlay() {
    if (!playing) return;
    playing = false;
    cancelAnimationFrame(raf);
  }

  function play() {
    // Middle, up to the first slice, down to the last, back to the middle: about 6 s once.
    const m = mid();
    const n = demo.count - 1;
    const legs = [[m, 0], [0, n], [n, m]];
    const total = legs.reduce((s, [a, b]) => s + Math.abs(b - a), 0);
    playing = true;
    const t0 = performance.now();
    const tick = (now) => {
      if (!playing) return;
      const t = Math.min(1, (now - t0) / PLAY_MS);
      let at = t * total;
      let pos = m;
      for (const [a, b] of legs) {
        const len = Math.abs(b - a);
        if (at <= len) { pos = a + Math.sign(b - a) * at; break; }
        at -= len;
        pos = b;
      }
      const i = Math.round(pos);
      if (loaded.has(i) || i === index) show(i, false);
      if (t < 1) { raf = requestAnimationFrame(tick); return; }
      stopPlay();
      show(m, false);
      restingOutline();
    };
    raf = requestAnimationFrame(tick);
  }

  function loadAll() {
    const order = [...Array(demo.count).keys()].sort((a, b) => Math.abs(a - mid()) - Math.abs(b - mid()));
    let pending = order.length;
    let go = prefersReducedMotion() ? () => {} : play;
    const done = () => { if (--pending === 0 && !dead) { const f = go; go = () => {}; f(); } };
    for (const i of order) {
      const im = new Image();
      im.onload = im.onerror = () => { loaded.add(i); done(); };
      im.src = frameUrl(i);
    }
  }

  // Scrubbing by dragging sideways on the picture; vertical swipes belong to the page.
  let drag = null;
  let moved = false;
  on(lightbox, 'pointerdown', (e) => {
    if (e.button > 0) return;
    stopPlay();
    drag = { x: e.clientX, i: index, id: e.pointerId };
    moved = false;
  });
  on(lightbox, 'pointermove', (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    const dx = e.clientX - drag.x;
    if (!moved && Math.abs(dx) < 6) return;
    if (!moved) { moved = true; try { lightbox.setPointerCapture(e.pointerId); } catch { /* gone */ } }
    const w = lightbox.clientWidth || 1;
    show(drag.i + Math.round((dx / w) * demo.count), true);
  });
  const endDrag = () => { drag = null; setTimeout(() => { moved = false; }, 0); };
  on(lightbox, 'pointerup', endDrag);
  on(lightbox, 'pointercancel', endDrag);
  on(lightbox, 'click', () => { if (!moved) clear(); });
  on(lightbox, 'pointerleave', (e) => { if (e.pointerType === 'mouse' && !pinned && !drag) clear(); });
  on(box, 'touchstart', stopPlay, { passive: true });

  on(range, 'input', () => show(Number(range.value), true));
  on(prev, 'click', () => show(index - 1, true));
  on(next, 'click', () => show(index + 1, true));
  on(box, 'keydown', (e) => {
    if (e.target === range) return;
    const k = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1 }[e.key];
    if (k) { e.preventDefault(); show(index + k, true); }
    else if (e.key === 'Home') { e.preventDefault(); show(0, true); }
    else if (e.key === 'End') { e.preventDefault(); show(demo.count - 1, true); }
  });

  function begin() {
    if (started || dead) return;
    started = true;
    loadAll();
  }

  fetch(src).then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status))))).then((d) => {
    if (dead) return;
    demo = d;
    demo.byId = new Map(d.structures.map((s) => [s.id, s]));
    lightbox.style.aspectRatio = '1 / 1';
    img.width = d.size[0];
    img.height = d.size[1];
    svg.setAttribute('viewBox', `0 0 ${d.size[0]} ${d.size[1]}`);
    range.max = String(d.count - 1);
    index = -1;
    show(mid(), false);
    if (prefersReducedMotion()) restingOutline();
    if ('IntersectionObserver' in window) {
      io = new IntersectionObserver((entries) => {
        if (entries.some((en) => en.isIntersecting)) { io.disconnect(); begin(); }
      }, { threshold: 0.4 });
      io.observe(box);
    } else {
      begin();
    }
  }).catch(failed);

  return {
    dispose() {
      dead = true;
      stopPlay();
      if (io) io.disconnect();
      ac.abort();
      box.replaceChildren();
    },
  };
}

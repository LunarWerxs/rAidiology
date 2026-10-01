// viewer.js: the 2D scan viewer (slice scrolling + finding marks) and the small scan figures
// used inside finding cards.
import { el, svgEl, sev, icon } from './data.js';

const OPPOSITE = { head: 'Feet', feet: 'Head', front: 'Back', back: 'Front', left: 'Right', right: 'Left', up: 'Down', down: 'Up' };

export function sliceUrl(stack, i) {
  return stack.path.replace('{i}', String(i).padStart(2, '0'));
}

/** Marks from findings[].views, drawn in image pixels so circles stay round at any size. */
function drawMarks(svg, sets, W, H) {
  svg.replaceChildren();
  const S = Math.min(W, H);
  for (const { marks, color } of sets) {
    for (const k of marks) {
      const x = Number(k.x) * W;
      const y = Number(k.y) * H;
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const stroke = (node) => [
        svgEl(node.tag, { ...node.attrs, fill: 'none', stroke: 'rgba(0,0,0,0.65)', 'stroke-width': 5, 'vector-effect': 'non-scaling-stroke', 'stroke-linecap': 'round' }),
        svgEl(node.tag, { ...node.attrs, fill: 'none', stroke: color, 'stroke-width': 2.4, 'vector-effect': 'non-scaling-stroke', 'stroke-linecap': 'round' }),
      ];
      if (k.type === 'circle') {
        svg.append(...stroke({ tag: 'circle', attrs: { cx: x, cy: y, r: Math.max(2, (Number(k.r) || 0.03) * W) } }));
      } else if (k.type === 'arrow') {
        // The arrow points AT (x, y); its tail sits at (x + dx, y + dy).
        const tx = x + (Number(k.dx) || 0) * W;
        const ty = y + (Number(k.dy) || 0) * H;
        const ang = Math.atan2(y - ty, x - tx);
        const len = S * 0.03;
        const bx = x - Math.cos(ang) * len * 0.8;
        const by = y - Math.sin(ang) * len * 0.8;
        svg.append(...stroke({ tag: 'line', attrs: { x1: tx, y1: ty, x2: bx, y2: by } }));
        const pts = [
          [x, y],
          [x - Math.cos(ang - 0.45) * len, y - Math.sin(ang - 0.45) * len],
          [x - Math.cos(ang + 0.45) * len, y - Math.sin(ang + 0.45) * len],
        ].map((p) => p.join(',')).join(' ');
        svg.append(svgEl('polygon', { points: pts, fill: color, stroke: 'rgba(0,0,0,0.65)', 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke' }));
      } else if (k.type === 'rect' || k.type === 'box') {
        svg.append(...stroke({ tag: 'rect', attrs: { x, y, width: (Number(k.w) || 0.05) * W, height: (Number(k.h) || 0.05) * H, rx: S * 0.01 } }));
      } else if (k.type === 'line') {
        svg.append(...stroke({ tag: 'line', attrs: { x1: x, y1: y, x2: (Number(k.x2) || 0) * W, y2: (Number(k.y2) || 0) * H } }));
      } else if ((k.type === 'text' || k.type === 'label') && k.text) {
        svg.append(svgEl('text', { x, y, fill: color, 'font-size': S * 0.045, 'font-weight': 700, stroke: 'rgba(0,0,0,0.75)', 'stroke-width': S * 0.008, 'paint-order': 'stroke' }, String(k.text)));
      }
    }
  }
}

/** Seeded MRI-like side view for the sample report only; watermarked so it is never mistaken for a scan. */
function drawSynthetic(canvas, stack, i) {
  const S = stack.synthetic;
  const W = canvas.width;
  const H = canvas.height;
  const c = canvas.getContext('2d');
  const mid = (stack.count - 1) / 2;
  const off = mid ? (i - mid) / mid : 0;
  const central = Math.exp(-off * off * 4);
  const X = (x) => x * W;
  const Y = (y) => y * H;
  const xc = (y) => S.base - S.amp * Math.sin((Math.PI * (y - S.y0)) / (S.y1 - S.y0));
  const round = (x, y, w, h, r) => {
    c.beginPath();
    if (c.roundRect) c.roundRect(x, y, w, h, r);
    else c.rect(x, y, w, h);
  };

  c.fillStyle = '#030405';
  c.fillRect(0, 0, W, H);
  const tissue = c.createLinearGradient(X(0.05), 0, X(0.97), 0);
  tissue.addColorStop(0, '#16171b');
  tissue.addColorStop(0.35, '#26272c');
  tissue.addColorStop(0.75, '#303136');
  tissue.addColorStop(1, '#1d1e22');
  c.fillStyle = tissue;
  c.beginPath();
  c.ellipse(X(0.52), Y(0.5), X(0.46), Y(0.66), 0, 0, Math.PI * 2);
  c.fill();
  c.strokeStyle = 'rgba(205,205,210,0.32)';
  c.lineWidth = X(0.035);
  c.beginPath();
  c.ellipse(X(0.52), Y(0.5), X(0.43), Y(0.63), 0, -0.9, 0.9);
  c.stroke();

  if (S.skull) {
    c.fillStyle = '#3a3b40';
    c.beginPath();
    c.ellipse(X(0.5), Y(0.02), X(0.3), Y(0.14), 0, 0, Math.PI * 2);
    c.fill();
  }

  const d = S.discs;
  const step = d[1] - d[0];
  const bh = step - S.dh * 1.5;
  const bodies = [d[0] - step / 2, ...d.slice(0, -1).map((y, k) => (y + d[k + 1]) / 2)];
  if (!S.sacrum) bodies.push(d[d.length - 1] + step / 2);
  for (const y of bodies) {
    const g = c.createLinearGradient(X(xc(y) - S.bw), 0, X(xc(y) + S.bw), 0);
    g.addColorStop(0, '#77787e');
    g.addColorStop(0.5, '#9a9ba1');
    g.addColorStop(1, '#6e6f75');
    c.fillStyle = g;
    round(X(xc(y) - S.bw), Y(y - bh / 2), X(2 * S.bw), Y(bh), X(0.012));
    c.fill();
    c.strokeStyle = '#1b1c20';
    c.lineWidth = 2;
    c.stroke();
    c.fillStyle = `rgba(120,121,127,${0.25 + 0.5 * central})`;
    c.beginPath();
    c.ellipse(X(xc(y) + S.bw + 0.13), Y(y + 0.01), X(0.028), Y(bh * 0.32), 0.25, 0, Math.PI * 2);
    c.fill();
  }
  if (S.sacrum) {
    const y = d[d.length - 1] + S.dh / 2 + 0.01;
    c.fillStyle = '#86878d';
    c.beginPath();
    c.moveTo(X(xc(y) - S.bw), Y(y));
    c.lineTo(X(xc(y) + S.bw), Y(y));
    c.lineTo(X(xc(y) + S.bw + 0.1), Y(y + 0.2));
    c.lineTo(X(xc(y) + 0.03), Y(y + 0.22));
    c.closePath();
    c.fill();
  }

  d.forEach((y, k) => {
    const x = xc(y);
    c.fillStyle = '#3a3b40';
    c.beginPath();
    c.ellipse(X(x), Y(y), X(S.bw * 1.02), Y(S.dh / 2), 0, 0, Math.PI * 2);
    c.fill();
    c.fillStyle = S.dark.includes(k) ? '#55565c' : '#d9dade';
    c.beginPath();
    c.ellipse(X(x - 0.005), Y(y), X(S.bw * 0.55), Y(S.dh * 0.3), 0, 0, Math.PI * 2);
    c.fill();
  });

  c.globalAlpha = 0.25 + 0.75 * central;
  c.lineCap = 'round';
  c.strokeStyle = '#eceff3';
  c.lineWidth = X(0.045);
  c.beginPath();
  for (let y = 0.02; y <= 0.98; y += 0.01) {
    const x = X(xc(y) + S.bw + 0.035);
    if (y === 0.02) c.moveTo(x, Y(y));
    else c.lineTo(x, Y(y));
  }
  c.stroke();
  c.strokeStyle = '#6c6e75';
  c.lineWidth = X(S.cord ? 0.02 : 0.004);
  const cordEnd = S.cord ? 0.98 : 0.2;
  c.beginPath();
  for (let y = 0.02; y <= cordEnd; y += 0.01) {
    const x = X(xc(y) + S.bw + 0.035);
    if (y === 0.02) c.moveTo(x, Y(y));
    else c.lineTo(x, Y(y));
  }
  c.stroke();
  c.globalAlpha = 1;

  for (const [k, amount] of Object.entries(S.bulge || {})) {
    const y = d[Number(k)];
    c.fillStyle = '#404147';
    c.beginPath();
    c.ellipse(X(xc(y) + S.bw + amount * 0.3), Y(y), X(amount + 0.008), Y(S.dh * 0.55), 0, 0, Math.PI * 2);
    c.fill();
  }
  for (const s of S.spots || []) {
    c.fillStyle = 'rgba(235,236,240,0.85)';
    c.beginPath();
    c.arc(X(xc(s.y)), Y(s.y), X(s.r), 0, Math.PI * 2);
    c.fill();
  }

  const img = c.getImageData(0, 0, W, H);
  const px = img.data;
  let seed = (i + 1) * 9973;
  for (let p = 0; p < px.length; p += 4) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    const n = (((seed >> 16) & 255) - 128) * 0.09;
    px[p] += n;
    px[p + 1] += n;
    px[p + 2] += n;
  }
  c.putImageData(img, 0, 0);

  c.font = `600 ${Math.round(W * 0.028)}px system-ui, sans-serif`;
  c.fillStyle = 'rgba(255,255,255,0.4)';
  c.textAlign = 'right';
  c.fillText('SAMPLE IMAGE · not a real scan', W - 12, H - 14);
}

/** Image (or sample canvas) with an SVG mark layer on top, sized by the stack's aspect ratio. */
export function buildStage(stack, { labels = false } = {}) {
  const W = stack.width;
  const H = stack.height;
  const root = el('div', { class: 'stage', style: { aspectRatio: `${W} / ${H}` } });
  const media = stack.synthetic
    ? el('canvas', { class: 'stage-media', width: W, height: H, 'aria-hidden': 'true' })
    : el('img', { class: 'stage-media', alt: '', decoding: 'async' });
  const missing = el('div', { class: 'stage-missing', hidden: true }, 'This image is not available yet');
  const marks = svgEl('svg', { class: 'stage-marks', viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'xMidYMid meet', 'aria-hidden': 'true' });
  root.append(media, marks, missing);
  if (!stack.synthetic) {
    media.addEventListener('error', () => { missing.hidden = false; media.style.visibility = 'hidden'; });
    media.addEventListener('load', () => { missing.hidden = true; media.style.visibility = ''; });
  }
  if (labels) {
    const o = stack.orientation || {};
    const bottom = o.bottom || OPPOSITE[String(o.top).toLowerCase()] || '';
    for (const [pos, text] of [['top', o.top], ['bottom', bottom], ['left', o.left], ['right', o.right]]) {
      if (text) root.append(el('span', { class: `orient ${pos}` }, text));
    }
  }
  return {
    el: root,
    show(i, sets) {
      if (stack.synthetic) drawSynthetic(media, stack, i);
      else {
        const url = sliceUrl(stack, i);
        if (media.getAttribute('src') !== url) media.src = url;
      }
      drawMarks(marks, sets, W, H);
    },
  };
}

export function stackFigure(stacks, view, color) {
  const stack = stacks.find((s) => s.id === view.stack);
  const caption = el('figcaption', {}, view.caption || stack?.label || 'Scan view', stack ? ` · slice ${view.slice + 1}` : '');
  if (!stack) {
    return el('figure', { class: 'scan-fig' }, el('div', { class: 'stage', style: { aspectRatio: '1 / 1' } },
      el('div', { class: 'stage-missing' }, 'Scan images are not available yet')), caption);
  }
  const stage = buildStage(stack);
  stage.show(Math.min(view.slice, stack.count - 1), [{ marks: view.marks, color }]);
  return el('figure', { class: 'scan-fig' }, stage.el, caption);
}

export function emptyState(title, text) {
  return el('div', { class: 'empty-state' },
    icon(['M4 7V5a1 1 0 0 1 1-1h2M17 4h2a1 1 0 0 1 1 1v2M20 17v2a1 1 0 0 1-1 1h-2M7 20H5a1 1 0 0 1-1-1v-2', 'M9 12h6M12 9v6']),
    el('h3', {}, title), el('p', {}, text));
}

export function createViewer(root, { stacks, report, onOpenFinding }) {
  root.replaceChildren();
  if (!stacks.length) {
    root.append(emptyState('No scan images yet', 'When the image series are added, you can scroll through every slice here and see where each finding is.'));
    return { focus() {} };
  }
  const findings = report?.findings || [];
  const select = el('select', { class: 'select', id: 'stack-select' },
    stacks.map((s) => el('option', { value: s.id }, s.label)));
  const stageHost = el('div');
  const range = el('input', { type: 'range', min: 0, max: 0, value: 0, step: 1, 'aria-label': 'Slice' });
  const count = el('span', { class: 'slice-count', 'aria-live': 'polite' });
  const prev = el('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Previous slice' }, icon(['M15 6l-6 6 6 6']));
  const next = el('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Next slice' }, icon(['M9 6l6 6-6 6']));
  const here = el('div', { class: 'mark-list' });
  const elsewhere = el('div', { class: 'mark-list' });
  const hereTitle = el('div', { class: 'section-title' }, 'Marked on this slice');
  const elseTitle = el('div', { class: 'section-title' }, 'Findings in this series');

  root.append(el('div', { class: 'viewer' },
    el('div', { class: 'field' }, el('label', { for: 'stack-select' }, 'Series'), select),
    stageHost,
    el('div', { class: 'slice-row' }, prev, range, next, count),
    el('p', { class: 'hint' }, 'Scroll over the image, drag the slider or use the arrow keys to move through the slices.'),
    hereTitle, here, elseTitle, elsewhere));

  let stack = null;
  let stage = null;
  let slice = 0;
  let wheelAcc = 0;

  const viewsFor = (s) => findings.flatMap((f) => f.views.filter((v) => v.stack === s.id).map((v) => ({ f, v })));

  function render() {
    range.value = String(slice);
    count.textContent = `${slice + 1} / ${stack.count}`;
    const all = viewsFor(stack);
    const now = all.filter(({ v }) => v.slice === slice);
    stage.show(slice, now.map(({ f, v }) => ({ marks: v.marks, color: sev(f.severity).color })));
    here.replaceChildren(...(now.length ? now.map(({ f }) => item(f, 'Open', () => onOpenFinding?.(f.id)))
      : [el('p', { class: 'hint' }, 'Nothing is marked on this slice.')]));
    hereTitle.hidden = false;
    elsewhere.replaceChildren(...all.map(({ f, v }) => item(f, `Slice ${v.slice + 1}`, () => go(v.slice))));
    elseTitle.hidden = !all.length;
    const preload = [slice - 1, slice + 1].filter((k) => k >= 0 && k < stack.count);
    if (!stack.synthetic) for (const k of preload) new Image().src = sliceUrl(stack, k);
  }

  function item(f, sub, onClick) {
    const s = sev(f.severity);
    return el('button', { class: 'mark-item', type: 'button', style: { '--sev': s.color }, onclick: onClick },
      el('span', { class: 'dot', 'aria-hidden': 'true' }),
      el('span', { class: 'grow' }, el('strong', {}, f.level ? `${f.level} · ` : ''), f.title),
      el('span', { class: 'sub' }, sub));
  }

  function go(i) {
    const k = Math.max(0, Math.min(stack.count - 1, i));
    if (k === slice) return;
    slice = k;
    render();
  }

  function setStack(id, sliceIndex) {
    stack = stacks.find((s) => s.id === id) || stacks[0];
    select.value = stack.id;
    stage = buildStage(stack, { labels: true });
    stage.el.tabIndex = 0;
    stage.el.setAttribute('role', 'img');
    stage.el.setAttribute('aria-label', `${stack.label}. Use arrow keys to change slice.`);
    stage.el.addEventListener('wheel', (e) => {
      e.preventDefault();
      wheelAcc += e.deltaY;
      if (Math.abs(wheelAcc) >= 40) {
        go(slice + Math.sign(wheelAcc));
        wheelAcc = 0;
      }
    }, { passive: false });
    stage.el.addEventListener('keydown', (e) => {
      const d = { ArrowUp: -1, ArrowLeft: -1, ArrowDown: 1, ArrowRight: 1, PageUp: -3, PageDown: 3 }[e.key];
      if (d) { e.preventDefault(); go(slice + d); }
      if (e.key === 'Home') { e.preventDefault(); go(0); }
      if (e.key === 'End') { e.preventDefault(); go(stack.count - 1); }
    });
    stageHost.replaceChildren(stage.el);
    range.max = String(stack.count - 1);
    slice = Math.max(0, Math.min(stack.count - 1, sliceIndex ?? stack.default_slice));
    render();
  }

  select.addEventListener('change', () => setStack(select.value));
  range.addEventListener('input', () => go(Number(range.value)));
  prev.addEventListener('click', () => go(slice - 1));
  next.addEventListener('click', () => go(slice + 1));
  setStack(stacks[0].id);

  return {
    /** Jump to a stack/slice, e.g. from a finding's scan view. */
    focus(stackId, sliceIndex) {
      if (stacks.some((s) => s.id === stackId)) setStack(stackId, sliceIndex);
    },
  };
}

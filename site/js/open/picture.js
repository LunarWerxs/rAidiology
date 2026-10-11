// Full-size pictures for the two donation pages: a small greyscale painter and a widget that scrolls
// through every picture of a list. It reads the bytes it is given (the de-identified files that will
// be, or were, sent) and draws them at their own pixel size. No network, nothing stored.

import { readDataset, decodeFrame, autoWindow, rescale } from './dicom.js';

async function frameOf(bytes) {
  const ds = readDataset(bytes);
  if (!ds) throw new Error('it isn\'t a DICOM file');
  const frame = await decodeFrame(ds, 0);
  if (frame.unsupported) throw new Error(frame.unsupported);
  return frame;
}

/** Paint one decoded greyscale frame at its own size. `full` shows the whole brightness range. */
export function paintFull(canvas, frame, full) {
  const { width, height } = frame;
  canvas.width = width;
  canvas.height = height;
  const w = full ? { centre: (frame.min + frame.max) / 2, width: Math.max(frame.max - frame.min, 1e-6) } : autoWindow(frame);
  const lo = w.centre - w.width / 2;
  const scale = 255 / Math.max(w.width, 1e-6);
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(width, height);
  const out = img.data;
  const n = width * height;
  const grey = (v) => {
    const g = Math.max(0, Math.min(255, (v - lo) * scale));
    return frame.invert ? 255 - g : g;
  };
  for (let i = 0; i < n; i++) {
    let g;
    if (frame.samples === 3) {
      out[i * 4] = Math.max(0, Math.min(255, (frame.values[i * 3] - lo) * scale));
      out[i * 4 + 1] = Math.max(0, Math.min(255, (frame.values[i * 3 + 1] - lo) * scale));
      out[i * 4 + 2] = Math.max(0, Math.min(255, (frame.values[i * 3 + 2] - lo) * scale));
      out[i * 4 + 3] = 255;
      continue;
    }
    g = grey(frame.raw ? rescale(frame.raw[i], frame.slope, frame.intercept) : frame.values[i]);
    out[i * 4] = g; out[i * 4 + 1] = g; out[i * 4 + 2] = g; out[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

const make = (tag, props = {}, text) => {
  const e = document.createElement(tag);
  Object.assign(e, props);
  if (text !== undefined) e.textContent = text;
  return e;
};

/**
 * Put a viewer for `images` ([{name, bytes}], in the order to show) into `container`.
 * Every picture is reachable: slider, Previous, Next, and the arrow keys. Pictures are decoded
 * only when shown. Returns { destroy }.
 */
export function mountPreview(container, images) {
  container.replaceChildren();
  container.classList.add('preview');
  if (!images.length) { container.append(make('p', { className: 'fine' }, 'There are no pictures to show.')); return { destroy() {} }; }
  const label = make('p', { className: 'fine', role: 'status' });
  const wrap = make('div', { className: 'preview-stage', tabIndex: 0 });
  const canvas = make('canvas', { className: 'preview-canvas' });
  canvas.setAttribute('aria-label', 'One picture, shown at full size.');
  wrap.append(canvas);
  const prev = make('button', { type: 'button', className: 'ghost small' }, 'Previous');
  const next = make('button', { type: 'button', className: 'ghost small' }, 'Next');
  const slider = make('input', { type: 'range', min: 0, max: images.length - 1, value: 0 });
  const sliderLabel = make('label', { className: 'slider' }, 'Picture ');
  sliderLabel.append(slider);
  const fullBox = make('input', { type: 'checkbox' });
  const fullLabel = make('label', { className: 'check' });
  fullLabel.append(fullBox, ' Show the full brightness range');
  const controls = make('div', { className: 'controls' });
  controls.append(prev, sliderLabel, next, fullLabel);
  const message = make('p', { className: 'fine' });
  container.append(label, wrap, controls, message);

  let at = 0;
  let seq = 0;
  const frames = new Map();
  const show = async (i) => {
    at = Math.max(0, Math.min(images.length - 1, i));
    slider.value = String(at);
    const mine = ++seq;
    label.textContent = `${at + 1} of ${images.length}`;
    try {
      if (!frames.has(at)) {
        frames.set(at, await frameOf(images[at].bytes));
        if (frames.size > 6) frames.delete(frames.keys().next().value);
      }
      if (mine !== seq) return;
      message.textContent = '';
      paintFull(canvas, frames.get(at), fullBox.checked);
    } catch (e) {
      if (mine === seq) message.textContent = `This picture can't be shown (${e.message}).`;
    }
  };
  prev.addEventListener('click', () => show(at - 1));
  next.addEventListener('click', () => show(at + 1));
  slider.addEventListener('input', () => show(Number(slider.value)));
  fullBox.addEventListener('change', () => show(at));
  wrap.addEventListener('keydown', (ev) => {
    if (ev.key === 'ArrowRight' || ev.key === 'ArrowDown') { ev.preventDefault(); show(at + 1); }
    if (ev.key === 'ArrowLeft' || ev.key === 'ArrowUp') { ev.preventDefault(); show(at - 1); }
  });
  show(0);
  return { destroy() { seq++; frames.clear(); container.replaceChildren(); } };
}

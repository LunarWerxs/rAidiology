// Small DOM helpers shared by the two donation pages.

/** Make an element; `props` are assigned, `kids` are appended (strings become text). */
export function h(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  Object.assign(e, props);
  for (const k of kids) e.append(k);
  return e;
}

/** Hand a file to the browser's own save; nothing is sent anywhere. */
export function saveFile(name, data, type) {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const a = h('a', { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}

/** A progress block (text + bar) whose handle has set(text, fraction). */
export function progressBlock() {
  const fill = h('div', { className: 'fill' });
  const text = h('p', {}, '');
  const box = h('div', { className: 'progress', role: 'status' }, text, h('div', { className: 'track' }, fill));
  box.setAttribute('aria-live', 'polite');
  return { box, set(t, f) { text.textContent = t; if (f !== undefined) fill.style.width = `${Math.round(Math.max(0, Math.min(1, f)) * 100)}%`; } };
}

// The listeners and observers a page sets up that live as long as the page. Each one is recorded
// here, so the page has one teardown that takes them all down when the document is really going
// away. A page kept in the back-forward cache (persisted) keeps them, so it still works on return.
const undo = [];

export function listen(target, type, fn, options) {
  target?.addEventListener?.(type, fn, options);
  undo.push(() => target?.removeEventListener?.(type, fn, options));
  return fn;
}

export function observeResize(callback, target) {
  const observer = new ResizeObserver(callback);
  observer.observe(target);
  undo.push(() => observer.disconnect());
  return observer;
}

export function teardown() {
  while (undo.length) undo.pop()();
}

if (typeof window !== 'undefined') {
  window.addEventListener('pagehide', (e) => { if (!e.persisted) teardown(); });
}

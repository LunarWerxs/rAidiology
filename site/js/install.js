// Installable web app: registers the service worker, shows the Install button when the browser offers it.
const root = document.documentElement;
const buttons = () => document.querySelectorAll('#install-app, [data-install-app]');
const standalone = () => matchMedia('(display-mode: standalone)').matches || navigator.standalone === true;

function hint() {
  const ua = navigator.userAgent;
  const touchMac = /Macintosh/.test(ua) && navigator.maxTouchPoints > 1;
  if (/iPhone|iPad|iPod/.test(ua) || touchMac) return 'ios';
  if (/Firefox\//.test(ua)) return 'firefox';
  if (/Chrome\/|Edg\/|CriOS\//.test(ua)) return 'chromium';
  if (/Macintosh/.test(ua) && /Safari\//.test(ua)) return 'safari-mac';
  return 'other';
}
root.classList.add('install-hint-' + hint());

function setHidden(hidden) { buttons().forEach((b) => { b.hidden = hidden; }); }

let deferred = null;
if (standalone()) root.classList.add('is-standalone');

addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferred = e;
  if (!standalone()) setHidden(false);
});
addEventListener('appinstalled', () => { deferred = null; setHidden(true); });

document.addEventListener('click', (e) => {
  const b = e.target.closest && e.target.closest('#install-app, [data-install-app]');
  if (!b || !deferred) return;
  const ev = deferred;
  deferred = null;
  ev.prompt();
  ev.userChoice.finally(() => setHidden(true));
});

function notice(worker) {
  if (document.getElementById('update-ready')) return;
  const bar = document.createElement('div');
  bar.id = 'update-ready';
  bar.setAttribute('role', 'status');
  bar.style.cssText = 'position:fixed;left:50%;bottom:16px;transform:translateX(-50%);z-index:9999;background:#0b1a2b;color:#fff;padding:10px 14px;border-radius:8px;font:14px/1.3 system-ui,sans-serif;display:flex;gap:12px;align-items:center';
  const t = document.createElement('span');
  t.textContent = 'A new version is ready';
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.textContent = 'Reload';
  btn.style.cssText = 'background:#fff;color:#0b1a2b;border:0;border-radius:6px;padding:6px 10px;font:inherit;cursor:pointer';
  btn.addEventListener('click', () => {
    navigator.serviceWorker.addEventListener('controllerchange', () => location.reload(), { once: true });
    worker.postMessage('skip-waiting');
  });
  bar.append(t, btn);
  document.body.appendChild(bar);
}

if ('serviceWorker' in navigator && window.isSecureContext) {
  navigator.serviceWorker.register('sw.js').then((reg) => {
    const watch = (w) => w.addEventListener('statechange', () => {
      if (w.state === 'installed' && navigator.serviceWorker.controller) notice(w);
    });
    if (reg.waiting && navigator.serviceWorker.controller) notice(reg.waiting);
    reg.addEventListener('updatefound', () => { if (reg.installing) watch(reg.installing); });
  }).catch(() => {});
}

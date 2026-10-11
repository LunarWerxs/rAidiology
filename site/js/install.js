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

// A new version takes over by itself (sw.js skips waiting). A page opened a moment ago reloads once so
// it and its styles come from the same version; a page in use (an open scan) is never reloaded.
if ('serviceWorker' in navigator && window.isSecureContext) {
  const updating = Boolean(navigator.serviceWorker.controller);
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (!updating || performance.now() > 15000 || /open(\.html)?$/.test(location.pathname)) return;
    if (sessionStorage.getItem('raidiology-reloaded')) return;
    sessionStorage.setItem('raidiology-reloaded', '1');
    location.reload();
  });
  navigator.serviceWorker.register('sw.js').catch(() => {});
}

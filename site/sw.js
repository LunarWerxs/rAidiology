// rAidiology service worker. The precache list and version are written by release/site/build_sw.py.
// precache:start
const VERSION = 'e9622150807b';
const PRECACHE = [
  './css/brand.css',
  './css/home.css',
  './css/money.css',
  './css/open.css',
  './css/pages.css',
  './css/results.css',
  './css/style.css',
  './data/models/skeleton.glb',
  './data/models/skeleton.json',
  './data/scan-pictures.json',
  './data/stats.json',
  './donation.html',
  './fonts/Inter-Variable.woff2',
  './fonts/OFL-Inter.txt',
  './fonts/SOURCES.json',
  './health-data-privacy.html',
  './img/apple-touch-icon.png',
  './img/favicon.svg',
  './img/icon-192.png',
  './img/icon-512.png',
  './img/icon-maskable-512.png',
  './img/logo-512.png',
  './img/logo.svg',
  './img/sample-slice.webp',
  './img/scans/ct-lungs.webp',
  './img/scans/ct-organs.webp',
  './img/scans/mammogram.webp',
  './img/scans/mri-spine.webp',
  './img/scans/xray-chest.webp',
  './img/scans/xray-spine.webp',
  './img/scans/xray-wrist.webp',
  './index.html',
  './js/body.js',
  './js/data.js',
  './js/donation.js',
  './js/install.js',
  './js/lifetime.js',
  './js/main.js',
  './js/motion.js',
  './js/nav.js',
  './js/open/collect.js',
  './js/open/deid.js',
  './js/open/dicom.js',
  './js/open/dom_util.js',
  './js/open/donate.js',
  './js/open/donate_core.js',
  './js/open/donate_crypto.js',
  './js/open/donation_host.js',
  './js/open/donation_public_key.json',
  './js/open/findings.js',
  './js/open/inflate-worker.js',
  './js/open/permission.js',
  './js/open/picture.js',
  './js/open/sample.js',
  './js/open/terms_v1.js',
  './js/open/viewer.js',
  './js/open/zip.js',
  './js/open/zipwrite.js',
  './js/panel.js',
  './js/scene.js',
  './js/skeleton.js',
  './js/spine.js',
  './js/viewer.js',
  './manifest.webmanifest',
  './models.html',
  './open.html',
  './pricing.html',
  './run.html',
  './support.html',
  './thanks.html',
];
// precache:end

const CACHE = 'raidiology-' + VERSION;
const BYPASS = ['/api/', '/report', '/data/report', '/data/meshes/', '/data/stacks/', '/launcher/'];

const FRESH = /\.(?:html|css|js|mjs|json|webmanifest)$/;

function networkFirst(request, fallback) {
  return fetch(request).then((res) => {
    if (res.ok && res.type === 'basic' && !res.redirected) {
      const copy = res.clone();
      caches.open(CACHE).then((c) => c.put(request, copy));
    }
    return res;
  }).catch(() => caches.match(request, { ignoreSearch: true })
    .then((hit) => hit || (fallback ? caches.match(fallback) : Response.error())));
}

// A new version takes over at once. Waiting for a click left an open page with the new HTML and the old
// cached stylesheets and scripts, which is a broken page.
self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(PRECACHE)));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('raidiology-') && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  const scope = new URL(self.registration.scope).pathname;
  const rel = '/' + url.pathname.slice(scope.length);
  if (BYPASS.some((p) => rel.startsWith(p))) return;

  // Pages and the code and data that draw them come from the network first, so a deploy never meets an
  // older stylesheet or script; the cache answers only when the network does not (offline).
  if (request.mode === 'navigate') {
    event.respondWith(networkFirst(request, './index.html'));
    return;
  }
  if (FRESH.test(url.pathname)) {
    event.respondWith(networkFirst(request, null));
    return;
  }
  if (rel.startsWith('/sample/')) {
    event.respondWith(
      caches.open(CACHE).then((c) => c.match(request).then((hit) => hit || fetch(request).then((res) => {
        if (res.ok) c.put(request, res.clone());
        return res;
      })))
    );
    return;
  }
  event.respondWith(caches.match(request).then((hit) => hit || fetch(request)));
});

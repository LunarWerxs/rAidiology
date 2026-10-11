// rAidiology service worker. The precache list and version are written by release/site/build_sw.py.
// precache:start
const VERSION = 'cb108d6bd771';
const PRECACHE = [
  './css/brand.css',
  './css/home.css',
  './css/money.css',
  './css/open.css',
  './css/style.css',
  './data/models/skeleton.glb',
  './data/models/skeleton.json',
  './data/scan-pictures.json',
  './data/stats.json',
  './donation.html',
  './fonts/Inter-Variable.woff2',
  './fonts/OFL-Inter.txt',
  './fonts/OFL-SourceSerif4.txt',
  './fonts/SOURCES.json',
  './fonts/SourceSerif4-Variable.woff2',
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

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(PRECACHE)));
});

self.addEventListener('message', (event) => {
  if (event.data === 'skip-waiting') self.skipWaiting();
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

  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request).catch(() => caches.match(request, { ignoreSearch: true }).then((r) => r || caches.match('./index.html')))
    );
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

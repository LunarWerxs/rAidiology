// Home page: wakes the live 3D and the sample CT after first paint, and runs the small support chooser. The
// "Why back this" run counts carry data-log attributes that tools/build_momentum.py writes from the run ledger.
// "Try the sample scan" is a plain link to the viewer (open.html?sample=1). The stage keeps its poster until the 3D
// is running, and keeps it if an import fails. Where the sample CT card sits over the stage (wide screens), the 3D
// moves the body clear of it.

const stage = document.querySelector('.hero3d');
const demo = document.getElementById('demo');

// The CT demo announces every slice on window; the 3D may mount after the first one, so keep the latest.
let lastSlice = null;
addEventListener('raidiology:slice', (e) => { lastSlice = e.detail; });

let heroJob = null;
let ctJob = null;

function mountHero() {
  if (!stage) return null;
  heroJob ||= import('./hero3d.js')
    .then((m) => {
      const hero = m.mountHero3D(stage, { avoid: demo });
      if (lastSlice && hero && typeof hero.setSlice === 'function') hero.setSlice(lastSlice);
      return hero;
    })
    .catch(() => null);
  return heroJob;
}

// The script never loaded: stop the "loading" look and say so (ctdemo.js does this itself for a failed fetch).
function demoFailed() {
  demo.classList.remove('is-loading');
  demo.classList.add('is-failed');
  demo.removeAttribute('aria-busy');
  if (!demo.querySelector('.ctdemo__link')) {
    const a = document.createElement('a');
    a.className = 'ctdemo__link';
    a.href = 'open.html?sample=1';
    a.textContent = 'Open in the viewer';
    demo.append(a);
  }
}

function mountCt() {
  if (!demo) return null;
  ctJob ||= import('./ctdemo.js').then((m) => m.mountCtDemo(demo)).catch(() => { demoFailed(); return null; });
  return ctJob;
}

function afterLoadAndIdle(fn) {
  const idle = () => ('requestIdleCallback' in window ? requestIdleCallback(fn, { timeout: 1500 }) : setTimeout(fn, 300));
  if (document.readyState === 'complete') idle();
  else addEventListener('load', idle, { once: true });
}

afterLoadAndIdle(() => {
  mountHero();
  mountCt();
});

// Support chooser: Monthly or Once, three amounts, the GPU hours they buy, and a link that carries the choice.
const RATE = 1.59;
const chooser = document.getElementById('chooser');
if (chooser) {
  const segs = [...chooser.querySelectorAll('.seg-b')];
  const amts = [...chooser.querySelectorAll('.amt')];
  const hours = document.getElementById('chooser-hours');
  const go = document.getElementById('chooser-go');
  const state = { amount: 15, freq: 'month' };

  const render = () => {
    const monthly = state.freq === 'month';
    segs.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.freq === state.freq)));
    amts.forEach((b) => b.setAttribute('aria-pressed', String(Number(b.dataset.amt) === state.amount)));
    hours.textContent = `That pays for about ${Math.floor(state.amount / RATE)} hours of model training${monthly ? ' a month' : ''}.`;
    go.textContent = `Pay $${state.amount} ${monthly ? 'a month' : 'once'}`;
    go.href = `support.html?amount=${state.amount}&freq=${state.freq}`;
  };

  segs.forEach((b) => b.addEventListener('click', () => { state.freq = b.dataset.freq; render(); }));
  amts.forEach((b) => b.addEventListener('click', () => { state.amount = Number(b.dataset.amt); render(); }));
  render();
}

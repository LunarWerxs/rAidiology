// Support page: one chosen-amount card (frequency, chips, box, slider, live GPU hours, checkout button).
const RATE = 1.59;
const MIN = 3;
const MAX = { month: 1000, once: 5000 };
const FIXED = {
  5: 'https://buy.stripe.com/dRm14nbkHfzG8KOftJ9Zm00',
  15: 'https://buy.stripe.com/bJeeVdewTgDK6CG95l9Zm02',
  50: 'https://buy.stripe.com/fZu9AT60n9bi6CGftJ9Zm03',
};
const ONCE = 'https://buy.stripe.com/7sY9ATagDbjq5yC95l9Zm01';
const LEVELS = [
  [50, 'Sponsor', 'the Supporter perks, plus your name or logo in the README sponsors section.'],
  [15, 'Supporter', 'a supporter role on Discord, your name in the supporters list, a vote on what to train next.'],
  [5, 'Backer', 'a supporter role on Discord.'],
];

const $ = (id) => document.getElementById(id);
const box = $('give-amt');

if (box) {
  const range = $('give-range');
  const go = $('give-go');
  const err = $('give-err');
  const note = $('give-note');
  const hours = $('give-hours');
  const level = $('give-level');
  const segs = [...document.querySelectorAll('.seg-b')];
  const chips = [...document.querySelectorAll('.chip')];
  const state = { amount: 15, freq: 'month', busy: false };
  const money = (n) => '$' + n.toLocaleString('en-US');

  function parse(text) {
    const n = parseInt(String(text).replace(/[^0-9]/g, ''), 10);
    return Number.isFinite(n) ? n : 0;
  }

  function problem() {
    if (state.amount < MIN) return 'The smallest amount is $3.';
    if (state.amount > MAX[state.freq]) {
      return state.freq === 'month'
        ? 'A month tops out at $1,000. For more, pick Once (up to $5,000) or email us.'
        : 'One payment tops out at $5,000. For more, email us.';
    }
    return '';
  }

  function target() {
    if (problem()) return '#give';
    if (state.freq === 'once') return `${ONCE}?prefilled_amount=${state.amount * 100}`;
    return FIXED[state.amount] || '#give';
  }

  function render(from) {
    const bad = problem();
    const n = state.amount;
    const monthly = state.freq === 'month';
    segs.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.freq === state.freq)));
    chips.forEach((c) => c.setAttribute('aria-pressed', String(Number(c.dataset.amt) === n)));
    if (from !== 'box') box.value = n ? String(n) : '';
    if (from !== 'range') range.value = String(Math.min(200, Math.max(MIN, n || MIN)));
    box.setAttribute('aria-invalid', String(Boolean(bad)));
    note.hidden = !bad;
    note.textContent = bad;
    if (bad) {
      hours.textContent = 'Pick an amount to see the GPU time it buys';
      level.textContent = '';
      go.textContent = 'Choose an amount';
    } else {
      hours.textContent = `About ${Math.floor(n / RATE)} hours of A100 GPU training${monthly ? ' a month' : ''}`;
      const lv = LEVELS.find((l) => n >= l[0]);
      level.textContent = !monthly ? '' : lv ? `${lv[1]}: ${lv[2]}` : 'Levels: Backer at $5+, Supporter at $15+, Sponsor at $50+.';
      go.textContent = `Give ${money(n)} ${monthly ? 'a month' : 'once'}`;
    }
    go.setAttribute('aria-disabled', String(Boolean(bad) || state.busy));
    go.href = target();
    err.hidden = true;
  }

  function set(n, from) {
    state.amount = n;
    render(from);
  }

  segs.forEach((b) => b.addEventListener('click', () => { state.freq = b.dataset.freq; render(); }));
  chips.forEach((c) => c.addEventListener('click', () => set(Number(c.dataset.amt))));
  range.addEventListener('input', () => set(parse(range.value), 'range'));
  box.addEventListener('input', () => set(parse(box.value), 'box'));
  box.addEventListener('blur', () => render());

  go.addEventListener('click', async (e) => {
    if (problem()) { e.preventDefault(); box.focus(); return; }
    if (state.freq === 'once' || FIXED[state.amount]) return; // a plain link
    e.preventDefault();
    if (state.busy) return;
    state.busy = true;
    err.hidden = true;
    go.setAttribute('aria-disabled', 'true');
    go.textContent = 'Opening checkout...';
    try {
      const res = await fetch('/api/support/checkout', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ amount: state.amount }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.url) { location.href = data.url; return; }
      throw new Error(data.error || 'Checkout could not start. Please try again.');
    } catch (x) {
      state.busy = false;
      render();
      err.textContent = x.message || 'Checkout could not start. Please try again.';
      err.hidden = false;
    }
  });

  const q = new URLSearchParams(location.search);
  if (q.get('freq') === 'once' || q.get('freq') === 'month') state.freq = q.get('freq');
  const a = parseInt(q.get('amount'), 10);
  if (a > 0) state.amount = a;
  render();
}

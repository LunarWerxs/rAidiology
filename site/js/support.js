// Support page: one chosen-amount card (frequency, three tier chips, Other amount box, live GPU hours, checkout button),
// plus a small refresh of the training-log counts at the bottom of this file.
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

// Shown under Once, so the perks line never vanishes. Pricing says perks are for monthly members.
const ONCE_NOTE = 'A one-time payment buys GPU time. The app stays free. Perks such as the Discord supporter role are for monthly members.';

const $ = (id) => document.getElementById(id);
const box = $('give-amt');

if (box) {
  const go = $('give-go');
  const err = $('give-err');
  const note = $('give-note');
  const minHelp = $('give-min');
  const hours = $('give-hours');
  const level = $('give-level');
  const segs = [...document.querySelectorAll('.seg-b')];
  const chips = [...document.querySelectorAll('.chip')];
  const next = $('give-next');
  const state = { amount: 15, freq: 'month', busy: false, custom: false, touched: false };
  const ICON = '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" focusable="false"><path d="M12 3 2 21h20L12 3z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 10v5M12 18v.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';

  // A message with a small warning icon. role="alert" only on the render where it first appears.
  function show(el, text) {
    const first = el.hidden;
    el.hidden = false;
    el.innerHTML = `<span class="msg-ic">${ICON}</span><span></span>`;
    el.lastChild.textContent = text;
    if (first) el.setAttribute('role', 'alert'); else el.removeAttribute('role');
  }
  function hide(el) {
    el.hidden = true;
    el.textContent = '';
    el.removeAttribute('role');
  }
  const money = (n) => '$' + n.toLocaleString(undefined);

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
    chips.forEach((c) => c.setAttribute('aria-pressed', String(from !== 'box' && Number(c.dataset.amt) === n)));
    if (from !== 'box') box.value = state.custom && n ? String(n) : '';
    const showBad = Boolean(bad) && (state.touched || !state.custom);
    box.setAttribute('aria-invalid', String(showBad));
    if (showBad) show(note, bad); else hide(note);
    next.textContent = monthly
      ? 'Next: a secure Stripe checkout page, then a receipt by email. Cancel any time.'
      : 'Next: a secure Stripe checkout page, then a receipt by email.';
    const low = n > 0 && n < MIN || (state.custom && box.value.trim() !== '' && n < MIN);
    if (minHelp) minHelp.hidden = showBad && n < MIN;
    if (bad) {
      hours.textContent = low ? 'Enter $3 or more to see the GPU time it buys' : 'Pick an amount to see the GPU time it buys';
      level.textContent = '';
      go.textContent = low ? 'Enter $3 or more' : 'Choose an amount';
    } else {
      hours.textContent = `About ${Math.floor(n / RATE)} hours of A100 GPU training${monthly ? ' a month' : ''}`;
      const lv = LEVELS.find((l) => n >= l[0]);
      level.textContent = !monthly ? ONCE_NOTE : lv ? `${lv[1]}: ${lv[2]}` : 'Levels: Backer at $5+, Supporter at $15+, Sponsor at $50+.';
      go.textContent = `Give ${money(n)} ${monthly ? 'a month' : 'once'}`;
    }
    go.setAttribute('aria-disabled', String(Boolean(bad) || state.busy));
    go.href = target();
    hide(err);
  }

  function set(n, from) {
    state.amount = n;
    render(from);
  }

  segs.forEach((b) => b.addEventListener('click', () => { state.freq = b.dataset.freq; render(); }));
  chips.forEach((c) => c.addEventListener('click', () => { state.custom = false; state.touched = false; set(Number(c.dataset.amt)); }));
  box.addEventListener('input', () => { state.custom = true; set(parse(box.value), 'box'); });
  box.addEventListener('blur', () => { if (state.custom) { state.touched = true; render(); } });

  go.addEventListener('click', async (e) => {
    if (problem()) { e.preventDefault(); box.focus(); return; }
    if (state.freq === 'once' || FIXED[state.amount]) return; // a plain link
    e.preventDefault();
    if (state.busy) return;
    state.busy = true;
    hide(err);
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
      show(err, x.message || 'Checkout could not start. Please try again.');
    }
  });

  const q = new URLSearchParams(location.search);
  if (q.get('freq') === 'once' || q.get('freq') === 'month') state.freq = q.get('freq');
  const a = parseInt(q.get('amount'), 10);
  if (a > 0) {
    state.amount = a;
    state.custom = !chips.some((c) => Number(c.dataset.amt) === a);
    state.touched = state.custom;
  }
  render();
}

// Support page: one chosen-amount card (how often, three tier chips, an Other amount box, the training
// time it buys, the checkout button). The choice is kept on this device so thanks.html can name it in the perks email.
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
  [50, 'Sponsor', 'everything Supporters get, plus a name or logo in the sponsors section of the GitHub page'],
  [15, 'Supporter', 'a role on Discord, your name in the supporters list and a vote on what we train next'],
  [5, 'Backer', 'a role on Discord'],
];
const PERK_LINES = [
  [5, 'A role on Discord (from $5)'],
  [15, 'Your name in the supporters list and a vote on what we train next (from $15)'],
  [50, "A name or logo on the project's GitHub page ($50)"],
];
// A typed amount is judged once the typing pauses, so "1" on the way to "15" never flashes an error.
const SETTLE_MS = 700;
const KEY = 'raidiology-support-choice';
const KEEP_MS = 2 * 864e5;
const MAIL = 'mailto:contact@lunarwerx.com?subject=' + encodeURIComponent('My rAidiology supporter perks') + '&body=';

const $ = (id) => document.getElementById(id);
const money = (n) => '$' + n.toLocaleString('en-US');
const levelFor = (n) => LEVELS.find((l) => n >= l[0]);

function planName(c) {
  if (c.freq !== 'month') return `${money(c.amount)} once`;
  const lv = levelFor(c.amount);
  return lv ? `${lv[1]}, ${money(c.amount)} a month` : `${money(c.amount)} a month`;
}

function perksMail(c) {
  const lines = ['Discord username: '];
  if (c) lines.push(`My payment: ${planName(c)}`);
  lines.push('Email I paid with: ', 'Name for the supporters list (optional): ', '');
  return MAIL + encodeURIComponent(lines.join('\n'));
}

function remember(c) {
  try { localStorage.setItem(KEY, JSON.stringify({ amount: c.amount, freq: c.freq, at: Date.now() })); } catch (e) { /* storage off: thanks.html reads the same without it */ }
}

function recall() {
  try {
    const c = JSON.parse(localStorage.getItem(KEY) || 'null');
    if (c && c.amount >= MIN && (c.freq === 'month' || c.freq === 'once') && Date.now() - c.at < KEEP_MS) return c;
    localStorage.removeItem(KEY);
  } catch (e) { /* storage off */ }
  return null;
}

// "label: rest", with the label in bold, built from text nodes so no amount ever passes through innerHTML.
function lead(el, label, rest) {
  const b = document.createElement('strong');
  b.textContent = label;
  el.replaceChildren(b, rest ? ' ' + rest : '');
}

const box = $('give-amt');

if (box) {
  const go = $('give-go');
  const err = $('give-err');
  const note = $('give-note');
  const minHelp = $('give-min');
  const readout = $('give-readout');
  const hours = $('give-hours');
  const level = $('give-level');
  const claim = $('give-claim');
  const renew = $('give-renew');
  const perksLink = $('perks-mail');
  const bar = $('pay-bar');
  const barBtn = bar && bar.querySelector('button');
  const barHours = $('bar-hours');
  const narrow = window.matchMedia('(max-width: 719px)');
  const segs = [...document.querySelectorAll('.seg-b')];
  const chips = [...document.querySelectorAll('.chip')];
  const state = { amount: 15, freq: 'once', busy: false, custom: false, settled: true };
  const ICON = '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" focusable="false"><path d="M12 3 2 21h20L12 3z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 10v5M12 18v.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
  let timer = 0;
  const perkSwitch = level.querySelector('.perk-switch') || Object.assign(document.createElement('button'), { type: 'button', className: 'perk-switch', textContent: 'Switch to Monthly' });
  perkSwitch.addEventListener('click', () => { state.freq = 'month'; render(); });

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

  function parse(text) {
    const n = parseInt(String(text).replace(/[^0-9]/g, ''), 10);
    return Number.isFinite(n) ? n : 0;
  }

  // What stops checkout, if anything: the message under the box and the words on the disabled button.
  function problem() {
    const n = state.amount;
    if (!n) return { msg: '', btn: 'Choose an amount' };
    if (n < MIN) return { msg: 'Enter $3 or more.', btn: 'Pay $3 or more' };
    if (n > MAX[state.freq]) {
      return state.freq === 'month'
        ? { msg: 'Monthly support tops out at $1,000. For more, pick Once (up to $5,000) or email us.', btn: 'Pay up to $1,000 a month' }
        : { msg: 'One payment tops out at $5,000. For more, email us.', btn: 'Pay up to $5,000 once' };
    }
    return null;
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

    // While someone is still typing, a low amount only greys the button; the message waits for the pause.
    const showBad = Boolean(bad && bad.msg) && (state.settled || !state.custom);
    box.setAttribute('aria-invalid', String(showBad));
    if (showBad) show(note, bad.msg); else hide(note);
    if (minHelp) minHelp.hidden = showBad && n < MIN;

    if (!bad) {
      readout.hidden = false;
      const h = `${Math.floor(n / RATE)} hours`;
      hours.replaceChildren('That pays for about ', Object.assign(document.createElement('strong'), { textContent: h }),
        ` ${monthly ? 'a month ' : ''}of rented GPU time, training our research models.`);
      if (barHours) {
        barHours.replaceChildren('That pays for about ', Object.assign(document.createElement('strong'), { textContent: h }),
          ` of training${monthly ? ' a month' : ''}.`);
      }
      const lv = levelFor(n);
      if (monthly) {
        const ul = document.createElement('ul');
        ul.className = 'perk-list';
        for (const [min, text] of PERK_LINES) {
          const li = document.createElement('li');
          li.textContent = text;
          if (lv && lv[0] === min) li.className = 'is-yours';
          ul.append(li);
        }
        const b = document.createElement('strong');
        b.textContent = 'Perks with monthly support:';
        level.replaceChildren(b, ul);
      } else if (!level.contains(perkSwitch)) {
        // One persistent button: a blur-triggered render must not replace it between mousedown and click.
        const b = document.createElement('strong');
        b.textContent = 'Perks (a role on Discord and more) go to monthly backers.';
        level.replaceChildren(b, ' ', perkSwitch);
      }
      claim.hidden = !(monthly && lv);
      go.textContent = `Pay ${money(n)} ${monthly ? 'a month' : 'once'}`;
    } else {
      if (state.settled || !state.custom || !n) readout.hidden = true;
      if (barHours) barHours.replaceChildren();
      go.textContent = bad.btn;
    }
    renew.hidden = !monthly;
    go.setAttribute('aria-disabled', String(Boolean(bad) || state.busy));
    go.href = target();
    if (barBtn) {
      barBtn.textContent = go.textContent;
      barBtn.setAttribute('aria-disabled', go.getAttribute('aria-disabled'));
    }
    if (perksLink) perksLink.href = perksMail(bad ? null : { amount: n, freq: state.freq });
    hide(err);
  }

  function settleSoon() {
    clearTimeout(timer);
    state.settled = false;
    timer = setTimeout(() => { state.settled = true; render('box'); }, SETTLE_MS);
  }

  // The keyboard covers the bottom half of a phone, Pay included, so while the box has focus a copy of the
  // button rides just above the keyboard. visualViewport says how much of the window the keyboard covers.
  function placeBar() {
    const vv = window.visualViewport;
    const covered = vv ? Math.max(0, window.innerHeight - vv.height - vv.offsetTop) : 0;
    bar.style.setProperty('--kb', `${Math.round(covered)}px`);
  }
  function showBar() {
    if (!bar || !narrow.matches) return;
    placeBar();
    bar.hidden = false;
  }
  function hideBar() { if (bar) bar.hidden = true; }

  segs.forEach((b) => b.addEventListener('click', () => { state.freq = b.dataset.freq; render(); }));
  chips.forEach((c) => c.addEventListener('click', () => {
    clearTimeout(timer);
    Object.assign(state, { custom: false, settled: true, amount: Number(c.dataset.amt) });
    render();
  }));
  box.addEventListener('input', () => {
    state.custom = true;
    state.amount = parse(box.value);
    settleSoon();
    render('box');
  });
  box.addEventListener('blur', () => {
    hideBar();
    if (!state.custom) return;
    clearTimeout(timer);
    state.settled = true;
    render();
  });
  box.addEventListener('focus', showBar);
  box.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); go.click(); } });
  if (bar && barBtn) {
    // pointerdown would move focus to the button and close the keyboard before the tap lands.
    barBtn.addEventListener('pointerdown', (e) => e.preventDefault());
    barBtn.addEventListener('click', () => go.click());
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', () => { if (!bar.hidden) placeBar(); });
      window.visualViewport.addEventListener('scroll', () => { if (!bar.hidden) placeBar(); });
    }
  }

  go.addEventListener('click', async (e) => {
    if (problem()) {
      e.preventDefault();
      clearTimeout(timer);
      state.settled = true;
      render();
      box.focus();
      return;
    }
    remember(state);
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
      throw new Error(data.error || 'Checkout didn\'t open. Try again in a moment.');
    } catch (x) {
      state.busy = false;
      render();
      show(err, x.message || 'Checkout didn\'t open. Try again in a moment.');
    }
  });

  const q = new URLSearchParams(location.search);
  if (q.get('freq') === 'once' || q.get('freq') === 'month') state.freq = q.get('freq');
  const a = parseInt(q.get('amount'), 10);
  if (a > 0) {
    state.amount = a;
    state.custom = !chips.some((c) => Number(c.dataset.amt) === a);
  }
  render();
}

// thanks.html: name the choice made on the support page, and put it in the perks email.
const chosen = $('thx-choice');
if (chosen) {
  const c = recall();
  const mail = $('thx-perks-mail');
  if (c) {
    chosen.textContent = c.freq === 'month' ? `You chose ${planName(c)}.` : `You chose to pay ${planName(c)}.`;
    chosen.hidden = false;
  }
  if (mail) mail.href = perksMail(c);
}

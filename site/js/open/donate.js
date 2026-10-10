// The "Donate this scan" flow on the open page. Nothing here runs, and nothing about donating shows,
// until a scan of the visitor's own is open and GET <donation host>/api/donation/status answers
// {"open": true}. A failed call, a CORS error or an unset host leaves the panel hidden, silently.
// Nothing at all is sent until the visitor ticks the three boxes and presses Send. The only network
// calls are to the donation host (donation_host.js; no cookies, no auth) and one same-origin read of
// the public key.
//
// viewer.js calls scanOpened(provider) when a scan opens (provider: async () => [{ name, bytes }])
// and scanOpened(null) when it closes or is the sample.

import { fillPanel } from './terms_v1.js';
import { dateText, POLICY_URL } from './permission.js';
import { fetchStatus, judgeScan, buildDonation, sendDonation, explainError, sentZip } from './donate_core.js';
import { DONATION_BASE, donationFetch, donationConfigured } from './donation_host.js';
import { h, saveFile, progressBlock } from './dom_util.js';
import { mountPreview } from './picture.js';

const el = (id) => document.getElementById(id);
const PRIVACY = 'health-data-privacy.html';
let provider = null;
let token = 0;                      // bumped whenever the open scan changes, so old work is dropped

function reset() {
  token++;
  el('donate-offer').hidden = true;
  el('donate').hidden = true;
  el('donate').replaceChildren();
}

export async function scanOpened(next) {
  reset();
  provider = next;
  if (!next) return;
  const mine = token;
  const open = donationConfigured() && await fetchStatus({ baseUrl: DONATION_BASE, fetch: donationFetch });
  if (mine !== token || !open) return;
  const offer = el('donate-offer');
  offer.hidden = false;
  const button = el('donate-start');
  button.onclick = () => { offer.hidden = true; begin(mine); };
}

function step(title) {
  const box = h('div', { className: 'step' }, h('h3', {}, title));
  return box;
}

function cancelButton(mine) {
  const b = h('button', { type: 'button', className: 'ghost' }, 'No thanks, close this');
  b.onclick = () => { if (mine === token) { el('donate').hidden = true; el('donate').replaceChildren(); el('donate-offer').hidden = false; } };
  return b;
}

async function begin(mine) {
  const root = el('donate');
  root.replaceChildren();
  root.hidden = false;
  root.append(h('h2', { id: 'donate-h' }, 'Donate this scan'));
  root.setAttribute('aria-labelledby', 'donate-h');
  const prog = progressBlock();
  root.append(prog.box);
  prog.set('Reading the files on this device…', 0);
  let judged;
  try {
    const files = await provider();
    judged = await judgeScan(files, { onProgress: (done, total) => prog.set(`Checking pictures ${done} of ${total}…`, done / Math.max(total, 1)) });
  } catch (e) {
    prog.box.remove();
    root.append(h('p', { className: 'warn-note' }, `This scan could not be checked: ${e.message}`), cancelButton(mine));
    return;
  }
  if (mine !== token) return;
  prog.box.remove();
  chooseSeries(mine, root, judged);
}

function seriesName(s) {
  return `${s.modality || 'Scan'}${s.description ? ' · ' + s.description : ''} · ${s.count} picture${s.count === 1 ? '' : 's'}`;
}

function chooseSeries(mine, root, judged) {
  const box = step('1. Which sets of pictures can be sent?');
  box.append(h('p', { className: 'fine' }, 'Nothing has been sent. Your browser checked each set of pictures on this device. A set that is refused stays here.'));
  const list = h('ul', { className: 'series-list' });
  const boxes = [];
  for (const s of judged.series) {
    const li = h('li');
    if (s.accepted) {
      const cb = h('input', { type: 'checkbox', checked: true });
      boxes.push({ cb, s });
      li.append(h('label', {}, cb, h('span', {}, h('strong', {}, 'Can be sent: '), seriesName(s))));
    } else {
      li.className = 'refused';
      li.append(h('strong', {}, 'Refused: '), seriesName(s), h('br'), s.reason || 'This set cannot be sent.');
    }
    list.append(li);
  }
  if (!judged.series.length) list.append(h('li', {}, 'No scan pictures were found.'));
  box.append(list);
  const go = h('button', { type: 'button', className: 'cta' }, 'Continue');
  const sync = () => { go.disabled = !boxes.some((b) => b.cb.checked); };
  boxes.forEach((b) => b.cb.addEventListener('change', sync));
  sync();
  if (!boxes.length) go.hidden = true;
  go.onclick = () => review(mine, root, boxes.filter((b) => b.cb.checked).map((b) => b.s));
  box.append(h('div', { className: 'actions' }, go, cancelButton(mine)));
  root.append(box);
  box.scrollIntoView({ block: 'start' });
}

async function review(mine, root, chosen) {
  root.replaceChildren(h('h2', {}, 'Donate this scan'));
  const prog = progressBlock();
  root.append(prog.box);
  prog.set('Removing everything that could identify you…', 0);
  let built;
  try {
    built = await buildDonation(chosen, {
      onProgress: (i, n, done, total) => prog.set(`Set ${i + 1} of ${n}: picture ${done} of ${total}…`, (i + done / Math.max(total, 1)) / n),
    });
  } catch (e) {
    prog.box.remove();
    root.append(h('p', { className: 'warn-note' }, e.message), cancelButton(mine));
    return;
  }
  if (mine !== token) return;
  prog.box.remove();

  const total = built.images.length;
  const pics = step(`2. Look at every picture that will be sent (${total})`);
  pics.append(h('p', { className: 'fine' }, 'This is exactly what would leave your browser, at full size. Look through all of them. If you see a name, a number or anything else about you, do not send it.'));
  const host = h('div');
  pics.append(host);
  mountPreview(host, built.images);

  const tags = step('3. The details that will be sent with them');
  tags.append(h('p', { className: 'fine' }, 'For each set, these are the technical details kept, with the values from its first picture. Nothing else is sent.'));
  built.list.forEach((s, i) => {
    tags.append(h('p', {}, h('strong', {}, `Set ${i + 1}: ${s.modality}, ${s.bodyPart}`)));
    const body = h('tbody');
    for (const t of s.preview) body.append(h('tr', {}, h('td', {}, t.name), h('td', {}, t.value === '' || t.value == null ? '(empty)' : String(t.value))));
    const table = h('table', { className: 'tag-table' }, h('thead', {}, h('tr', {}, h('th', {}, 'Detail'), h('th', {}, 'Value'))), body);
    tags.append(h('div', { className: 'tag-scroll' }, table));
  });

  const consent = step('4. Your permission');
  const panelHost = h('div', { className: 'donate-panel-host' });
  panelHost.innerHTML = fillPanel(dateText(new Date()), PRIVACY);   // the contract's own words, unchanged
  const save = h('button', { type: 'button', className: 'ghost' }, 'Save a copy of what will be sent');
  save.onclick = () => saveFile('rAidiology-what-will-be-sent.zip', sentZip(built), 'application/zip');
  const note = h('p', { className: 'fine' }, 'You can keep a copy of exactly what would be sent. It is made on this device.');
  const status = h('div', { role: 'status' });
  consent.append(note, h('div', { className: 'actions' }, save), panelHost, status);
  root.append(pics, tags, consent);

  const ids = ['donate-eligible', 'donate-agree', 'donate-storage'].map((i) => panelHost.querySelector('#' + i));
  const send = panelHost.querySelector('#donate-send');
  const no = panelHost.querySelector('#donate-no');
  const sync = () => { send.disabled = !ids.every((c) => c.checked); };
  ids.forEach((c) => { c.checked = false; c.addEventListener('change', sync); });
  sync();
  no.onclick = () => { if (mine === token) { root.hidden = true; root.replaceChildren(); el('donate-offer').hidden = false; } };
  send.onclick = () => doSend(mine, root, built, { send, no, save, ids, status });
  pics.scrollIntoView({ block: 'start' });
}

async function publicKey() {
  const res = await fetch(new URL('./donation_public_key.json', import.meta.url));
  if (!res.ok) throw new Error('The public key could not be loaded.');
  return res.json();
}

async function doSend(mine, root, built, ui) {
  if (!ui.ids.every((c) => c.checked)) return;
  for (const b of [ui.send, ui.no, ui.save, ...ui.ids]) b.disabled = true;
  const prog = progressBlock();
  ui.status.replaceChildren(prog.box);
  prog.set('Locking and sending…', 0);
  try {
    const publicJwk = await publicKey();
    const done = await sendDonation({
      built, publicJwk, baseUrl: DONATION_BASE, fetch: donationFetch, now: new Date(),
      onProgress: (n, total) => prog.set(`Sending part ${n} of ${total}…`, n / Math.max(total, 1)),
    });
    if (mine !== token) return;
    success(root, done);
  } catch (e) {
    prog.box.remove();
    ui.status.replaceChildren(h('p', { className: 'warn-note' }, explainError(e)));
    for (const b of [ui.send, ui.no, ui.save, ...ui.ids]) b.disabled = false;
  }
}

function success(root, done) {
  const box = step('Thank you. Your scan was sent.');
  box.append(
    h('p', {}, 'This is your donation number. You need it to see, save or delete your scan.'),
    h('div', { className: 'number-big', id: 'donate-number' }, done.display),
    h('p', { className: 'warn-note' }, 'We cannot show this number again. We do not keep it, so if you lose it we cannot recover it, and we cannot find your scan for you. Save it now.'),
  );
  const copy = h('button', { type: 'button', className: 'ghost' }, 'Copy the number');
  copy.onclick = async () => { try { await navigator.clipboard.writeText(done.display); copy.textContent = 'Copied'; } catch { copy.textContent = 'Select the number and copy it'; } };
  const saveNum = h('button', { type: 'button', className: 'ghost' }, 'Save the number as a text file');
  saveNum.onclick = () => saveFile('rAidiology-donation-number.txt', `rAidiology donation number: ${done.display}\nSee, save or delete it on this site's donation page (donation.html)\n`, 'text/plain');
  const perm = h('button', { type: 'button', className: 'cta' }, 'Save your permission copy');
  perm.onclick = () => saveFile('rAidiology-my-permission.html', done.permissionHtml, 'text/html');
  box.append(
    h('div', { className: 'actions' }, perm, copy, saveNum),
    h('p', {}, 'The permission copy is one file with the words you agreed to, today’s date and your number.'),
    h('p', {}, 'To see, save or delete your scan later, go to ', h('a', { href: '/donation.html' }, 'the donation page'), ' and type the number.'),
  );
  root.replaceChildren(box);
  box.scrollIntoView({ block: 'start' });
}

// The see-or-delete page. With the donation number: look it up, open a copy here, save a copy, save
// the permission copy, or delete. Without it: open the same scan files locally and show each
// series' code to email. The only network calls are to the donation host (donation_host.js).

import { openDonation, deleteDonation, parseDonationNumber, displayNumber } from './open/donate_crypto.js';
import { fromDirectoryHandle, fromFileList, expand } from './open/collect.js';
import { judgeScan, codesOf, openedZip, recordedPermissionHtml, explainError } from './open/donate_core.js';
import { monthText, hasTerms } from './open/permission.js';
import { mountPreview } from './open/picture.js';
import { DONATION_BASE, donationFetch, donationConfigured } from './open/donation_host.js';
import { h, saveFile } from './open/dom_util.js';

const el = (id) => document.getElementById(id);
let busy = false;

function wireNumber() {
  el('number-form').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    if (busy) return;
    const msg = el('number-msg');
    const found = el('found');
    found.hidden = true;
    found.replaceChildren();
    const hex = parseDonationNumber(el('number').value);
    if (!hex) { msg.textContent = 'That does not look like a donation number. Check it and try again.'; return; }
    if (!donationConfigured()) { msg.textContent = 'Donations are not open yet.'; return; }
    busy = true;
    msg.textContent = 'Looking it up and unlocking it on this device…';
    try {
      const opened = await openDonation({ baseUrl: DONATION_BASE, fetch: donationFetch, numberText: hex });
      msg.textContent = '';
      showRecord(found, opened, hex);
    } catch (e) {
      msg.textContent = e.status === 404
        ? 'No donation is held under that number. It may have been deleted, or the number may be wrong.'
        : explainError(e);
    } finally {
      busy = false;
    }
  });
}

function showRecord(root, opened, hex) {
  const { consent, files, id } = opened;
  const display = displayNumber(hex);
  root.hidden = false;
  root.replaceChildren(h('h2', {}, 'Your donation'));
  const rec = h('ul', { className: 'series-list' });
  rec.append(h('li', {}, h('strong', {}, 'Number: '), display));
  rec.append(h('li', {}, h('strong', {}, 'Donated in: '), monthText(consent.month)));
  rec.append(h('li', {}, h('strong', {}, 'Terms version: '), String(consent.terms)));
  consent.series.forEach((s, i) => {
    rec.append(h('li', {}, h('strong', {}, `Set ${i + 1}: `), `${s.modality}, ${s.bodyPart}, code ${s.code}`));
  });
  rec.append(h('li', {}, h('strong', {}, 'Pictures: '), String(files.length)));
  root.append(rec);

  const view = h('div');
  const open = h('button', { type: 'button', className: 'ghost' }, 'Open a copy');
  open.onclick = () => {
    open.disabled = true;
    const images = files.filter((f) => /\.dcm$/.test(f.name));
    view.replaceChildren(h('p', { className: 'fine' }, `${images.length} pictures, as they were sent.`));
    const host = h('div');
    view.append(host);
    mountPreview(host, images);
  };
  const save = h('button', { type: 'button', className: 'ghost' }, 'Save a copy');
  save.onclick = () => saveFile('rAidiology-my-donation.zip', openedZip(files.filter((f) => /\.dcm$/.test(f.name))), 'application/zip');
  const perm = h('button', { type: 'button', className: 'ghost' }, 'Save the permission copy');
  if (!hasTerms(consent.terms)) {
    perm.disabled = true;
    perm.title = `This site does not hold the words of terms version ${consent.terms}.`;
  }
  perm.onclick = () => saveFile('rAidiology-my-permission.html', recordedPermissionHtml({ consent, numberDisplay: display }), 'text/html');
  const del = h('button', { type: 'button', className: 'ghost danger' }, 'Delete my donation');
  const confirmBox = h('div', { className: 'warn-note', hidden: true });
  const yes = h('button', { type: 'button', className: 'cta' }, 'Yes, delete it');
  const keep = h('button', { type: 'button', className: 'ghost' }, 'No, keep it');
  confirmBox.append(h('p', {}, 'Delete this donation? This also withdraws your permission. It cannot be undone, and the number will stop working.'), h('div', { className: 'actions' }, yes, keep));
  del.onclick = () => { confirmBox.hidden = false; };
  keep.onclick = () => { confirmBox.hidden = true; };
  yes.onclick = async () => {
    yes.disabled = true;
    try {
      await deleteDonation({ baseUrl: DONATION_BASE, fetch: donationFetch, id });
      root.replaceChildren(
        h('h2', {}, 'Your donation is deleted'),
        h('p', {}, 'It is gone from our storage, and your permission is withdrawn. The daily pull removes our local copies within a day. A model that has already learnt from it cannot unlearn it.'),
      );
    } catch (e) {
      yes.disabled = false;
      confirmBox.append(h('p', {}, `It could not be deleted: ${explainError(e)}`));
    }
  };
  root.append(h('div', { className: 'actions' }, open, save, perm, del), confirmBox, view);
}

// ---- Without the number ----------------------------------------------------------------------

async function codesFrom(entries) {
  const prog = el('code-progress');
  const text = el('code-progress-text');
  const fill = el('code-progress-fill');
  const list = el('codes');
  list.replaceChildren();
  prog.hidden = false;
  const set = (t, f) => { text.textContent = t; fill.style.width = `${Math.round(f * 100)}%`; };
  try {
    set('Reading the files on this device…', 0);
    const read = await expand(entries, (done, total) => set(`Reading ${done} of ${total}…`, done / Math.max(total, 1)));
    const files = [];
    for (const f of read.files) files.push({ name: f.path, bytes: await f.load() });
    const judged = await judgeScan(files, { onProgress: (d, t) => set(`Checking pictures ${d} of ${t}…`, d / Math.max(t, 1)) });
    const out = await codesOf(judged);
    if (!out.length) list.append(h('li', {}, 'No scan pictures were found.'));
    for (const { series: s, code } of out) {
      const name = `${s.modality || 'Scan'}${s.description ? ' · ' + s.description : ''} · ${s.count} picture${s.count === 1 ? '' : 's'}`;
      if (code) {
        list.append(h('li', {}, h('strong', {}, name), h('br'), 'Code: ', h('code', {}, code), h('br'),
          'To have a donation removed, email this code to ', h('a', { href: `mailto:contact@lunarwerx.com?subject=${encodeURIComponent('Delete my donation')}&body=${encodeURIComponent(`Please find and delete the donation with this code: ${code}`)}` }, 'contact@lunarwerx.com'), '.'));
      } else {
        list.append(h('li', { className: 'refused' }, h('strong', {}, name), h('br'), `No code: ${s.reason || 'this set would not have been sent.'}`));
      }
    }
  } catch (e) {
    list.append(h('li', {}, `This could not be read: ${e.message}`));
  } finally {
    prog.hidden = true;
  }
}

function wireScan() {
  const pick = (accept, directory) => {
    const input = h('input', { type: 'file', multiple: true });
    if (directory) { input.webkitdirectory = true; input.directory = true; }
    if (accept) input.accept = accept;
    input.addEventListener('change', async () => { if (input.files && input.files.length) codesFrom(await fromFileList(input.files)); });
    input.click();
  };
  el('pick-folder').addEventListener('click', async () => {
    if (!window.showDirectoryPicker) { pick(null, true); return; }
    try {
      codesFrom(await fromDirectoryHandle(await window.showDirectoryPicker({ mode: 'read' })));
    } catch (err) {
      if (!err || err.name !== 'AbortError') pick(null, true);
    }
  });
  el('pick-zip').addEventListener('click', () => pick('.zip,application/zip', false));
}

function start() { wireNumber(); wireScan(); }
start();

// The donor's saved copy of the permission: one standalone HTML file holding the panel's words as
// shown (from terms_v1.js, never rewritten), the date, the terms version and the donation number.
// No DOM, so Node builds it too. The words live in the terms module of the version on record; a
// later version adds its own module here and an entry in TERMS.

import { TERMS_VERSION, PANEL_HTML, fillPanel } from './terms_v1.js';

export const POLICY_URL = new URL('health-data-privacy.html', location.href).href;

const TERMS = { [TERMS_VERSION]: { PANEL_HTML, fillPanel } };

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
  'October', 'November', 'December'];

/** "October 3, 2026", from the browser's own clock, the same in every locale. */
export const dateText = (d = new Date()) => `${MONTHS[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;

/** "2026-10", the month on the consent record. */
export const monthOf = (d = new Date()) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;

/** "2026-10" back to "October 2026". */
export function monthText(month) {
  const m = /^(\d{4})-(\d{2})$/.exec(month || '');
  return m && MONTHS[+m[2] - 1] ? `${MONTHS[+m[2] - 1]} ${m[1]}` : String(month);
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export const hasTerms = (version) => Object.prototype.hasOwnProperty.call(TERMS, version);

// The panel's words with the form controls turned into a record: the three boxes shown ticked and
// fixed, the buttons removed. Nothing else about the words changes.
function asRecord(html) {
  return html
    .replace(/<p><button[\s\S]*?<\/p>/, '')
    .replace(/<input type="checkbox"[^>]*>/g, '<input type="checkbox" checked disabled>');
}

/**
 * The permission copy as an HTML file (a string).
 *   version       the terms version on record
 *   date          the text to put where the panel says Date (the full date at sending; the month later)
 *   numberDisplay the donation number as shown (ABCD-...)
 *   note          optional one line under the heading (the donation page says what the date is)
 */
export function permissionCopyHtml({ version = TERMS_VERSION, date, numberDisplay, note = '', policyUrl = POLICY_URL }) {
  if (!hasTerms(version)) throw new Error(`terms version ${version} is not on this site`);
  const body = asRecord(TERMS[version].fillPanel(esc(date), policyUrl));
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>My permission to donate a scan to rAidiology</title>
<style>body{font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif;max-width:46rem;margin:24px auto;padding:0 18px;color:#1c2b3a}.record{border:1px solid #c3cfdc;border-radius:10px;padding:12px 18px;margin-bottom:20px;font-size:19px;line-height:1.5}</style>
</head>
<body>
<h1>My permission to donate a scan to rAidiology</h1>
<div class="record">
<p><strong>Donation number:</strong> ${esc(numberDisplay)}</p>
<p><strong>Terms version:</strong> ${esc(version)}</p>
<p><strong>Date:</strong> ${esc(date)}</p>${note ? `\n<p>${esc(note)}</p>` : ''}
</div>
${body}
</body>
</html>
`;
}

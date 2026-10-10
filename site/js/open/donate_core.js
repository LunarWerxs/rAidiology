// The donation flow without a DOM: everything the donate page and the see-or-delete page do that is
// not drawing. Node runs it too (tools/donate_flow_check.mjs). The only network calls are the ones
// in donate_crypto.js, and only to the donation host.

import { judgeFiles, newUidMap, deidentifySeries, buildManifest, readAgeProof, seriesCode } from './deid.js';
import { TERMS_VERSION, panelSha256 } from './terms_v1.js';
import { makeDonationNumber, lockDonation, uploadDonation, donationId } from './donate_crypto.js';
import { zipStore } from './zipwrite.js';
import { monthOf, monthText, dateText, permissionCopyHtml } from './permission.js';

export const OUTSIDE_TEXT = 'Donations are only open in the US, outside Maryland and New York.';

const url = (base, path) => `${String(base || '').replace(/\/+$/, '')}/api/donation/${path}`;

/** True only when GET /api/donation/status answers {"open": true}. Any failure is "closed". */
export async function fetchStatus({ baseUrl = '', fetch: f = globalThis.fetch } = {}) {
  if (baseUrl.includes('__DONATION_HOST__')) return false;
  try {
    const res = await f(url(baseUrl, 'status'), { cache: 'no-store' });
    if (!res.ok) return false;
    const body = await res.json();
    return !!body && body.open === true;
  } catch {
    return false;
  }
}

/** Group the files into series and decide each one (deid.judgeFiles). */
export const judgeScan = (files, { now = new Date(), onProgress } = {}) => judgeFiles(files, { now, onProgress });

/**
 * Rebuild the chosen series as de-identified files. Refuses unless every picture of every chosen
 * series proves an age of 18 or older (fails closed). Returns { list, manifest, images }, where
 * `images` is every picture of every series in order.
 */
export async function buildDonation(selected, { now = new Date(), onProgress } = {}) {
  if (!selected.length) throw new Error('No series is chosen.');
  for (const s of selected) {
    if (!s.accepted) throw new Error('A series that was refused cannot be sent.');
    for (const f of s.files) {
      const proof = readAgeProof(f.bytes, now);
      const years = proof.ok ? (Number.isFinite(proof.years) ? proof.years : null) : null;
      if (years === null || years < 18) {
        throw new Error('This scan cannot be donated: it does not show that the person was 18 or older.');
      }
    }
  }
  const uidMap = newUidMap();
  const list = [];
  for (let i = 0; i < selected.length; i++) {
    list.push(await deidentifySeries(selected[i], {
      uidMap, seriesIndex: i,
      onProgress: onProgress ? (done, total) => onProgress(i, selected.length, done, total) : undefined,
    }));
  }
  const manifest = buildManifest(list);
  const images = list.flatMap((s) => s.images.map((im) => ({ name: im.name, bytes: im.bytes })));
  return { list, manifest, images };
}

/** Lock and upload a built donation. Returns the number and the donor's permission copy. */
export async function sendDonation({ built, publicJwk, baseUrl = '', fetch: f = globalThis.fetch, now = new Date(), onProgress }) {
  const { hex } = makeDonationNumber();
  const consent = {
    terms: TERMS_VERSION,
    panelSha256: await panelSha256(),
    month: monthOf(now),
    series: built.manifest.series.map((s) => ({ code: s.code, modality: s.modality, bodyPart: s.bodyPart, tags: s.tags })),
  };
  const locked = await lockDonation({ manifest: built.manifest, entries: built.images, consent, publicJwk, numberHex: hex });
  await uploadDonation({ baseUrl, fetch: f, locked, onProgress });
  const date = dateText(now);
  return {
    numberHex: hex,
    display: locked.display,
    id: locked.id,
    date,
    permissionHtml: permissionCopyHtml({ version: TERMS_VERSION, date, numberDisplay: locked.display }),
  };
}

/** Plain words for a failed call. */
export function explainError(e) {
  const status = e && e.status;
  if (status === 403) return OUTSIDE_TEXT;
  if (status === 409) return 'That number is already in use. Please try again.';
  if (status === 413) return 'This scan is too large to send.';
  if (status === 503 || status === 404) return 'Donations are not open right now.';
  if (e && /Failed to fetch|NetworkError|Load failed/i.test(String(e.message))) {
    return 'The connection failed. Nothing is kept from a partly sent donation. Please try again.';
  }
  return (e && e.message) ? String(e.message) : 'Something went wrong.';
}

/** The de-identified .dcm files as sent, one store-only ZIP. */
export const sentZip = (built) => zipStore(built.images.map((im) => ({ name: im.name, bytes: im.bytes })));

/** The files of an opened donation (openDonation().files), as one store-only ZIP. */
export const openedZip = (files) => zipStore(files.map((f) => ({ name: f.name, bytes: f.bytes })));

/** The series codes of a judged-and-accepted set, the same way the manifest makes them. */
export async function codesOf(judged, { now = new Date(), onProgress } = {}) {
  const uidMap = newUidMap();
  const out = [];
  let index = 0;
  for (const s of judged.series) {
    if (!s.accepted) { out.push({ series: s, code: null }); continue; }
    const d = await deidentifySeries(s, { uidMap, seriesIndex: index++, onProgress });
    out.push({ series: s, code: d.code, built: d });
  }
  return out;
}

/** The permission copy for a donation on record: the month stands where the panel says Date. */
export function recordedPermissionHtml({ consent, numberDisplay }) {
  return permissionCopyHtml({
    version: consent.terms,
    date: monthText(consent.month),
    numberDisplay,
    note: 'Only the month is kept on record, so the month stands in for the date.',
  });
}

export { donationId, seriesCode };

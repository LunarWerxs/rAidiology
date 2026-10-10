// Locking and unlocking a donation (docs/DONATED_SCANS.md, "The build: files, formats and endpoints").
// WebCrypto only, so the same file runs in the browser and in Node (tools/donation_roundtrip.mjs).
//
//   number   16 random bytes, shown as 32 upper-case hex characters in eight groups of four
//   id       hex SHA-256 of "raidiology-donation-id:" + 32 lower-case hex characters (all the server sees)
//   donor    HKDF-SHA-256(number bytes, salt "raidiology-donation", info "donor-key") -> AES-KW 256
//   data     random AES-GCM 256, wrapped by the owner's RSA-OAEP key and by the donor key
//   bundle   entries of [4-byte big-endian header length][JSON {name, bytes}][bytes]; first is manifest.json
//   pieces   the bundle cut in 8 MiB pieces; piece n = 12-byte IV + AES-GCM, additional data "<id>:<n>",
//            the last one "<id>:<n>:last". Pieces are numbered from 0.

export const PIECE_SIZE = 4 * 1024 * 1024; // 4 MiB: the upload service runs where a request holds at most 6 MB
export const IV_BYTES = 12;
export const TAG_BYTES = 16;

const subtle = globalThis.crypto.subtle;
const enc = new TextEncoder();
const dec = new TextDecoder();

export const toHex = (u8) => Array.from(u8, (b) => b.toString(16).padStart(2, '0')).join('');
export function fromHex(s) {
  const out = new Uint8Array(s.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(s.slice(2 * i, 2 * i + 2), 16);
  return out;
}
export function toBase64(u8) {
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode(...u8.subarray(i, i + 0x8000));
  return btoa(s);
}
export const fromBase64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

// ---- the number, the id, the donor key -----------------------------------------------------------

/** A new donation number: {hex (32 lower-case hex), display ("ABCD-...", eight groups of four)}. */
export function makeDonationNumber() {
  const hex = toHex(globalThis.crypto.getRandomValues(new Uint8Array(16)));
  return { hex, display: displayNumber(hex) };
}

export const displayNumber = (hex) => hex.toUpperCase().match(/.{4}/g).join('-');

/** Read a typed number back: case-blind, every non-hex character ignored. Returns 32 lower-case hex or null. */
export function parseDonationNumber(text) {
  const hex = String(text).toLowerCase().replace(/[^0-9a-f]/g, '');
  return hex.length === 32 ? hex : null;
}

function needHex(hex) {
  const h = parseDonationNumber(hex);
  if (!h) throw new Error('not a donation number');
  return h;
}

export async function donationId(numberHex) {
  const h = needHex(numberHex);
  const d = await subtle.digest('SHA-256', enc.encode(`raidiology-donation-id:${h}`));
  return toHex(new Uint8Array(d));
}

export async function donorKey(numberHex) {
  const base = await subtle.importKey('raw', fromHex(needHex(numberHex)), 'HKDF', false, ['deriveKey']);
  return subtle.deriveKey(
    { name: 'HKDF', hash: 'SHA-256', salt: enc.encode('raidiology-donation'), info: enc.encode('donor-key') },
    base, { name: 'AES-KW', length: 256 }, false, ['wrapKey', 'unwrapKey'],
  );
}

// ---- the data key and its two wraps --------------------------------------------------------------

export const makeDataKey = () => subtle.generateKey({ name: 'AES-GCM', length: 256 }, true, ['encrypt', 'decrypt']);

/** The public key's fingerprint, as tools/donation_keys.mjs prints it (and keys.json carries it). */
export async function keyFingerprint(jwk) {
  const d = await subtle.digest('SHA-256', enc.encode(`${jwk.e}.${jwk.n}`));
  return toHex(new Uint8Array(d)).slice(0, 16);
}

/** keys.json: {owner, donor, fingerprint}; the wraps are base64. publicJwk is donation_public_key.json. */
export async function wrapDataKey(dataKey, numberHex, publicJwk) {
  const pub = await subtle.importKey('jwk', publicJwk, { name: 'RSA-OAEP', hash: 'SHA-256' }, false, ['wrapKey']);
  const owner = await subtle.wrapKey('raw', dataKey, pub, { name: 'RSA-OAEP' });
  const donor = await subtle.wrapKey('raw', dataKey, await donorKey(numberHex), 'AES-KW');
  return { owner: toBase64(new Uint8Array(owner)), donor: toBase64(new Uint8Array(donor)), fingerprint: await keyFingerprint(publicJwk) };
}

const GCM_USAGES = ['encrypt', 'decrypt'];

/** The donor's side: the number opens the donor wrap. */
export async function unwrapWithNumber(donorWrapB64, numberHex) {
  return subtle.unwrapKey('raw', fromBase64(donorWrapB64), await donorKey(numberHex), 'AES-KW',
    { name: 'AES-GCM', length: 256 }, false, GCM_USAGES);
}

/** The owner's side (the pull does this in Python; here for tests): the private JWK opens the owner wrap. */
export async function unwrapWithOwnerKey(ownerWrapB64, privateJwk) {
  const priv = await subtle.importKey('jwk', privateJwk, { name: 'RSA-OAEP', hash: 'SHA-256' }, false, ['unwrapKey']);
  return subtle.unwrapKey('raw', fromBase64(ownerWrapB64), priv, { name: 'RSA-OAEP' },
    { name: 'AES-GCM', length: 256 }, false, GCM_USAGES);
}

// ---- the bundle ----------------------------------------------------------------------------------

function entryHeader(name, length) {
  const json = enc.encode(JSON.stringify({ name, bytes: length }));
  const out = new Uint8Array(4 + json.length);
  new DataView(out.buffer).setUint32(0, json.length);
  out.set(json, 4);
  return out;
}

// An entry's bytes may be a Uint8Array, or a function returning one (or a promise of one), so a big
// CT is read only when its turn comes. {name, bytes, size} lets the caller state the size up front
// when bytes is a function.
const entrySize = (e) => (typeof e.bytes === 'function' ? e.size : e.bytes.length);

/** Sizes before anything is built: {plainBytes, pieces, storedBytes} (storedBytes = what is uploaded). */
export function planBundle(manifest, entries) {
  const m = enc.encode(JSON.stringify(manifest));
  let plain = entryHeader('manifest.json', m.length).length + m.length;
  for (const e of entries) {
    const size = entrySize(e);
    if (!Number.isInteger(size)) throw new Error(`entry ${e.name}: size unknown`);
    plain += entryHeader(e.name, size).length + size;
  }
  const pieces = Math.max(1, Math.ceil(plain / PIECE_SIZE));
  return { plainBytes: plain, pieces, storedBytes: plain + pieces * (IV_BYTES + TAG_BYTES) };
}

/** The plain bundle as an async generator of pieces (each PIECE_SIZE, the last one shorter). */
export async function* bundlePieces(manifest, entries) {
  let buf = new Uint8Array(PIECE_SIZE);
  let fill = 0;
  async function* feed(chunk) {
    let off = 0;
    while (off < chunk.length) {
      const take = Math.min(PIECE_SIZE - fill, chunk.length - off);
      buf.set(chunk.subarray(off, off + take), fill);
      fill += take; off += take;
      if (fill === PIECE_SIZE) { yield buf; buf = new Uint8Array(PIECE_SIZE); fill = 0; }
    }
  }
  const m = enc.encode(JSON.stringify(manifest));
  yield* feed(entryHeader('manifest.json', m.length));
  yield* feed(m);
  for (const e of entries) {
    const bytes = typeof e.bytes === 'function' ? await e.bytes() : e.bytes;
    yield* feed(entryHeader(e.name, bytes.length));
    yield* feed(bytes);
  }
  if (fill > 0) yield buf.slice(0, fill);
}

/** Read a plain bundle (async iterable of chunks) back: async generator of {name, bytes}. */
export async function* readBundle(chunks) {
  let queue = [];
  let have = 0;
  const it = chunks[Symbol.asyncIterator] ? chunks[Symbol.asyncIterator]() : chunks[Symbol.iterator]();
  async function take(n) {
    while (have < n) {
      const { value, done } = await it.next();
      if (done) return null;
      if (value.length) { queue.push(value); have += value.length; }
    }
    const out = new Uint8Array(n);
    let got = 0;
    while (got < n) {
      const head = queue[0];
      const k = Math.min(head.length, n - got);
      out.set(head.subarray(0, k), got);
      got += k;
      if (k === head.length) queue.shift(); else queue[0] = head.subarray(k);
    }
    have -= n;
    return out;
  }
  for (;;) {
    const lenBytes = await take(4);
    if (lenBytes === null) return;
    const hl = new DataView(lenBytes.buffer).getUint32(0);
    const hdr = await take(hl);
    if (hdr === null) throw new Error('bundle cut short (header)');
    const { name, bytes } = JSON.parse(dec.decode(hdr));
    const body = await take(bytes);
    if (body === null) throw new Error(`bundle cut short (${name})`);
    yield { name, bytes: body };
  }
}

/** Parse a whole plain bundle: {manifest, files: [{name, bytes}]}. */
export async function parseBundle(chunks) {
  let manifest = null;
  const files = [];
  for await (const e of readBundle(chunks)) {
    if (manifest === null) {
      if (e.name !== 'manifest.json') throw new Error('bundle does not start with manifest.json');
      manifest = JSON.parse(dec.decode(e.bytes));
    } else files.push(e);
  }
  if (manifest === null) throw new Error('empty bundle');
  return { manifest, files };
}

// ---- pieces --------------------------------------------------------------------------------------

const aad = (id, n, last) => enc.encode(`${id}:${n}${last ? ':last' : ''}`);

export async function encryptPiece(dataKey, id, n, last, plain) {
  const iv = globalThis.crypto.getRandomValues(new Uint8Array(IV_BYTES));
  const ct = new Uint8Array(await subtle.encrypt({ name: 'AES-GCM', iv, additionalData: aad(id, n, last) }, dataKey, plain));
  const out = new Uint8Array(IV_BYTES + ct.length);
  out.set(iv, 0);
  out.set(ct, IV_BYTES);
  return out;
}

export async function decryptPiece(dataKey, id, n, last, stored) {
  if (stored.length < IV_BYTES + TAG_BYTES) throw new Error(`piece ${n} too short`);
  const iv = stored.subarray(0, IV_BYTES);
  return new Uint8Array(await subtle.decrypt({ name: 'AES-GCM', iv, additionalData: aad(id, n, last) },
    dataKey, stored.subarray(IV_BYTES)));
}

/** Plain pieces in, async generator of {n, data} (stored form) out. Looks one piece ahead to mark the last. */
export async function* encryptPieces(dataKey, id, plainPieces) {
  let n = 0;
  let prev = null;
  for await (const p of plainPieces) {
    if (prev) yield { n: n - 1, data: await encryptPiece(dataKey, id, n - 1, false, prev) };
    prev = p; n++;
  }
  if (!prev) throw new Error('nothing to encrypt');
  yield { n: n - 1, data: await encryptPiece(dataKey, id, n - 1, true, prev) };
}

/** Stored pieces (async iterable of Uint8Array, in order) in, plain pieces out. Fails if reordered, altered or cut short. */
export async function* decryptPieces(dataKey, id, storedPieces) {
  let n = 0;
  let prev = null;
  for await (const p of storedPieces) {
    if (prev) yield await decryptPiece(dataKey, id, n - 1, false, prev);
    prev = p; n++;
  }
  if (!prev) throw new Error('no pieces');
  yield await decryptPiece(dataKey, id, n - 1, true, prev);
}

/** Everything the page does in one go: lock a manifest and files; returns what to send. */
export async function lockDonation({ manifest, entries, consent, publicJwk, numberHex }) {
  const number = numberHex || makeDonationNumber().hex;
  const id = await donationId(number);
  const dataKey = await makeDataKey();
  const keys = await wrapDataKey(dataKey, number, publicJwk);
  const plan = planBundle(manifest, entries);
  const pieces = encryptPieces(dataKey, id, bundlePieces(manifest, entries));
  return { number, display: displayNumber(number), id, keys, plan, pieces, consent: { ...consent, pieces: plan.pieces, bytes: plan.storedBytes } };
}

// ---- the public routes ---------------------------------------------------------------------------

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const url = (base, path) => `${base.replace(/\/+$/, '')}/api/donation/${path}`;

async function withRetry(fn, tries, delayMs) {
  let last;
  for (let i = 0; i < tries; i++) {
    try { return await fn(); } catch (e) {
      last = e;
      if (e.noRetry) break;
      await wait(delayMs * (i + 1));
    }
  }
  throw last;
}

async function ok(res, what) {
  if (res.ok) return res;
  const e = new Error(`${what}: HTTP ${res.status}`);
  e.status = res.status;
  e.noRetry = res.status >= 400 && res.status < 500 && res.status !== 408 && res.status !== 429;
  throw e;
}

/**
 * POST <id> {consent, keys}, PUT each piece (retried), POST <id>/finish {pieces, bytes}.
 * `locked` is what lockDonation returned. onProgress(done, total) is optional.
 */
export async function uploadDonation({ baseUrl, fetch: f = globalThis.fetch, locked, onProgress, tries = 4, retryDelayMs = 800 }) {
  const { id, keys, consent, plan } = locked;
  await ok(await f(url(baseUrl, id), {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ consent, keys }),
  }), 'start');
  let done = 0;
  for await (const { n, data } of locked.pieces) {
    await withRetry(async () => ok(await f(url(baseUrl, `${id}/p/${n}`), {
      method: 'PUT', headers: { 'content-type': 'application/octet-stream' }, body: data,
    }), `piece ${n}`), tries, retryDelayMs);
    onProgress?.(++done, plan.pieces);
  }
  await withRetry(async () => ok(await f(url(baseUrl, `${id}/finish`), {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ pieces: plan.pieces, bytes: plan.storedBytes }),
  }), 'finish'), tries, retryDelayMs);
  return { id, pieces: plan.pieces };
}

/** GET <id>: {consent, donor} for the donation page (assumed shape of the Function's reply). */
export async function fetchDonation({ baseUrl, fetch: f = globalThis.fetch, id }) {
  const res = await ok(await f(url(baseUrl, id)), 'lookup');
  return res.json();
}

/** GET <id>/p/<n> for each piece, in order, as an async generator of stored pieces. */
export async function* downloadPieces({ baseUrl, fetch: f = globalThis.fetch, id, pieces, tries = 4, retryDelayMs = 800 }) {
  for (let n = 0; n < pieces; n++) {
    yield await withRetry(async () => new Uint8Array(await (await ok(await f(url(baseUrl, `${id}/p/${n}`)), `piece ${n}`)).arrayBuffer()), tries, retryDelayMs);
  }
}

/** The donation page's whole read: number in, {consent, manifest, files} out. */
export async function openDonation({ baseUrl, fetch: f = globalThis.fetch, numberText }) {
  const number = parseDonationNumber(numberText);
  if (!number) throw new Error('not a donation number');
  const id = await donationId(number);
  const { consent, donor } = await fetchDonation({ baseUrl, fetch: f, id });
  const dataKey = await unwrapWithNumber(donor, number);
  const plain = decryptPieces(dataKey, id, downloadPieces({ baseUrl, fetch: f, id, pieces: consent.pieces }));
  return { id, consent, ...(await parseBundle(plain)) };
}

/** DELETE <id>. */
export async function deleteDonation({ baseUrl, fetch: f = globalThis.fetch, id }) {
  await ok(await f(url(baseUrl, id), { method: 'DELETE' }), 'delete');
}

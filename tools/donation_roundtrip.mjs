// Round trip of a donation with a THROWAWAY key pair (never the real key in %USERPROFILE%\.raidiology):
// build a bundle, lock it with site/js/open/donate_crypto.js, lay it out like the bucket (d/<id>/...) in
// a temp folder, unlock it as the donor in JS, then have tools/donations_pull.py --selftest unlock it
// with the owner key. Also checks that a reordered piece and a dropped last piece fail.
// Prints "roundtrip: ok" and exits 0, else prints what failed and exits 1.
import { webcrypto as wc, randomBytes } from 'node:crypto';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import * as dc from '../site/js/open/donate_crypto.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const same = (a, b) => a.length === b.length && Buffer.compare(Buffer.from(a), Buffer.from(b)) === 0;
const fail = (msg) => { throw new Error(msg); };
const rejects = async (fn) => { try { await fn(); return false; } catch { return true; } };
const collect = async (gen) => { const out = []; for await (const x of gen) out.push(x); return out; };

const tmp = mkdtempSync(join(tmpdir(), 'donation-roundtrip-'));
try {
  const pair = await wc.subtle.generateKey(
    { name: 'RSA-OAEP', modulusLength: 3072, publicExponent: new Uint8Array([1, 0, 1]), hash: 'SHA-256' },
    true, ['wrapKey', 'unwrapKey']);
  const publicJwk = await wc.subtle.exportKey('jwk', pair.publicKey);
  const privateJwk = await wc.subtle.exportKey('jwk', pair.privateKey);

  const sizes = [1234, 300000, 17 * 1024 * 1024 + 7];
  const entries = sizes.map((n, i) => ({ name: `s0/i${i}.dcm`, bytes: new Uint8Array(randomBytes(n)) }));
  const manifest = {
    terms: 1,
    series: [{ code: 'abcdef0123456789', modality: 'CT', bodyPart: 'CHEST',
      images: entries.map((e) => ({ name: e.name, sha256: dc.toHex(new Uint8Array(randomBytes(32))) })), tags: ['00080060'] }],
  };

  const num = dc.makeDonationNumber();
  if (!/^([0-9A-F]{4}-){7}[0-9A-F]{4}$/.test(num.display)) fail('display form');
  if (dc.parseDonationNumber(num.display.toLowerCase().replace(/-/g, ' . ')) !== num.hex) fail('typed number not read back');
  if (dc.parseDonationNumber('1234') !== null) fail('short number accepted');

  const locked = await dc.lockDonation({
    manifest, entries, publicJwk, numberHex: num.hex,
    consent: { terms: 1, panelSha256: '0'.repeat(64), month: '2026-10', series: [{ code: manifest.series[0].code, modality: 'CT', bodyPart: 'CHEST', tags: ['00080060'] }] },
  });
  const stored = (await collect(locked.pieces)).map((p) => p.data);
  if (stored.length < 3) fail(`only ${stored.length} pieces`);
  if (stored.length !== locked.plan.pieces) fail('plan disagrees with the pieces');
  if (stored.reduce((s, p) => s + p.length, 0) !== locked.plan.storedBytes) fail('plan disagrees with the bytes');
  if (stored.slice(0, -1).some((p) => p.length !== dc.PIECE_SIZE + 28)) fail('piece size');

  // the bucket layout
  const dir = join(tmp, 'd', locked.id);
  mkdirSync(join(dir, 'p'), { recursive: true });
  stored.forEach((p, n) => writeFileSync(join(dir, 'p', String(n).padStart(6, '0')), p));
  writeFileSync(join(dir, 'keys.json'), JSON.stringify(locked.keys));
  writeFileSync(join(dir, 'consent.json'), JSON.stringify({ ...locked.consent, complete: true }));
  writeFileSync(join(tmp, 'private_key.jwk'), JSON.stringify(privateJwk));

  const checkOpened = ({ manifest: m, files }, who) => {
    if (JSON.stringify(m) !== JSON.stringify(manifest)) fail(`${who}: manifest differs`);
    if (files.length !== entries.length) fail(`${who}: file count`);
    files.forEach((f, i) => { if (f.name !== entries[i].name || !same(f.bytes, entries[i].bytes)) fail(`${who}: ${f.name} differs`); });
  };

  // the donor, with the number only
  const key = await dc.unwrapWithNumber(locked.keys.donor, dc.parseDonationNumber(num.display));
  checkOpened(await dc.parseBundle(dc.decryptPieces(key, locked.id, stored)), 'donor');
  // the owner, in JS
  const okey = await dc.unwrapWithOwnerKey(locked.keys.owner, privateJwk);
  checkOpened(await dc.parseBundle(dc.decryptPieces(okey, locked.id, stored)), 'owner (js)');
  // a wrong number cannot open the donor wrap
  const other = dc.makeDonationNumber().hex;
  if (!await rejects(() => dc.unwrapWithNumber(locked.keys.donor, other))) fail('wrong number unwrapped');

  // reordered, dropped last, altered
  const swapped = [stored[1], stored[0], ...stored.slice(2)];
  if (!await rejects(() => collect(dc.decryptPieces(key, locked.id, swapped)))) fail('reordered pieces decrypted');
  if (!await rejects(() => collect(dc.decryptPieces(key, locked.id, stored.slice(0, -1))))) fail('dropped last piece decrypted');
  const flipped = stored.map((p, i) => (i === 1 ? Uint8Array.from(p, (b, j) => (j === 500 ? b ^ 1 : b)) : p));
  if (!await rejects(() => collect(dc.decryptPieces(key, locked.id, flipped)))) fail('altered piece decrypted');

  // the upload and download helpers against a fake of the Function's routes
  const fake = new Map();
  const f = async (u, init = {}) => {
    const path = new URL(u).pathname.replace('/api/donation/', '');
    const m = (init.method || 'GET');
    const res = (status, body) => new Response(body, { status });
    if (m === 'POST' && !path.includes('/')) { if (fake.has(path)) return res(409, ''); fake.set(path, { ...JSON.parse(init.body), p: [] }); return res(200, '{}'); }
    const [id, kind, n] = path.split('/');
    const d = fake.get(id);
    if (!d) return res(404, '');
    if (m === 'PUT') { if (kind !== 'p') return res(400, ''); d.p[Number(n)] = new Uint8Array(init.body); return res(200, ''); }
    if (m === 'POST' && kind === 'finish') { const b = JSON.parse(init.body); d.done = d.p.length === b.pieces; return res(d.done ? 200 : 400, ''); }
    if (m === 'GET' && !kind) return res(200, JSON.stringify({ consent: d.consent, donor: d.keys.donor }));
    if (m === 'GET' && kind === 'p') return res(200, d.p[Number(n)]);
    return res(404, '');
  };
  const locked2 = await dc.lockDonation({ manifest, entries: entries.slice(0, 2), publicJwk, consent: { terms: 1 } });
  await dc.uploadDonation({ baseUrl: 'https://example.invalid', fetch: f, locked: locked2 });
  if (!fake.get(locked2.id)?.done) fail('upload did not finish');
  const got = await dc.openDonation({ baseUrl: 'https://example.invalid', fetch: f, numberText: locked2.display });
  if (!same(got.files[1].bytes, entries[1].bytes)) fail('download helper bytes differ');

  // the pull, with the owner key in Python
  const py = spawnSync('python', [join(ROOT, 'tools', 'donations_pull.py'), '--selftest', tmp], { encoding: 'utf8' });
  if (py.status !== 0) fail(`donations_pull.py --selftest exit ${py.status}: ${(py.stdout + py.stderr).trim().slice(-400)}`);
  const out = join(tmp, 'unlocked', locked.id);
  if (readdirSync(join(out, 's0')).length !== entries.length) fail('python: file count');
  const pyManifest = JSON.parse(readFileSync(join(out, 'manifest.json'), 'utf8'));
  if (JSON.stringify(pyManifest) !== JSON.stringify(manifest)) fail('python: manifest differs');
  entries.forEach((e) => { if (!same(readFileSync(join(out, e.name)), e.bytes)) fail(`python: ${e.name} differs`); });

  console.log('roundtrip: ok');
} catch (e) {
  console.log(`roundtrip: FAILED: ${e.message}`);
  process.exitCode = 1;
} finally {
  rmSync(tmp, { recursive: true, force: true });
}

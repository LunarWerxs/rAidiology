// The key pair that locks donated scans (docs/DONATED_SCANS.md, "What leaves the browser", step 5).
//
// The browser makes a fresh AES-GCM key for each donation, encrypts the files with it, and wraps that
// key with this pair's PUBLIC key (RSA-OAEP, 3072 bits, SHA-256), so Cloudflare stores only bytes it
// cannot open. The PRIVATE key never enters this repository or the cloud: it is written to
// %USERPROFILE%\.raidiology\donation_private_key.jwk (outside every synced folder), and the owner keeps
// one copy in their password manager. Node's WebCrypto is the browser's own API, so the formats match
// exactly what site/js/open/ uses to encrypt.
//
// Prints only the public key's file and its SHA-256 fingerprint, never the private key.
// Refuses to overwrite an existing private key: a new pair would leave every earlier donation locked
// for good.
//
// Usage: node tools/donation_keys.mjs            (make the pair once)
//        node tools/donation_keys.mjs --show     (print the fingerprint of the key in use)
import { webcrypto as wc, createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const PUBLIC = join(ROOT, 'site', 'js', 'open', 'donation_public_key.json');
const PRIVATE = join(homedir(), '.raidiology', 'donation_private_key.jwk');

const fingerprint = (jwk) => createHash('sha256').update(`${jwk.e}.${jwk.n}`).digest('hex').slice(0, 16);

if (process.argv.includes('--show')) {
  if (!existsSync(PUBLIC)) { console.log(`no public key at ${PUBLIC}`); process.exit(1); }
  const jwk = JSON.parse(readFileSync(PUBLIC, 'utf8'));
  console.log(`public key ${PUBLIC}, fingerprint ${fingerprint(jwk)}; private key ${existsSync(PRIVATE) ? 'present' : 'MISSING'} at ${PRIVATE}`);
  process.exit(0);
}
if (existsSync(PRIVATE)) {
  console.log(`a private key already exists at ${PRIVATE}; not replaced (run with --show)`);
  process.exit(1);
}
const pair = await wc.subtle.generateKey(
  { name: 'RSA-OAEP', modulusLength: 3072, publicExponent: new Uint8Array([1, 0, 1]), hash: 'SHA-256' },
  true, ['wrapKey', 'unwrapKey'],
);
const pub = await wc.subtle.exportKey('jwk', pair.publicKey);
const priv = await wc.subtle.exportKey('jwk', pair.privateKey);
mkdirSync(dirname(PRIVATE), { recursive: true });
writeFileSync(PRIVATE, JSON.stringify(priv), { mode: 0o600, flag: 'wx' });
writeFileSync(PUBLIC, JSON.stringify({ kty: pub.kty, n: pub.n, e: pub.e, alg: pub.alg, ext: true, key_ops: ['wrapKey'] }, null, 1) + '\n');
console.log(`public key written to ${PUBLIC}, fingerprint ${fingerprint(pub)}; private key written to ${PRIVATE} (not printed)`);

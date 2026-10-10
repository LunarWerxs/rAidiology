// A store-only ZIP writer (no compression), for "Save a copy of what will be sent" and for the donation
// page's "Save a copy". No DOM, so Node runs it too (tools/donate_flow_check.mjs reads the result back
// with zip.js). Every entry carries the same fixed 1980-01-01 time, so a saved ZIP holds no date.

const TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

const DOS_TIME = 0;
const DOS_DATE = (0 << 9) | (1 << 5) | 1;      // 1980-01-01
const UTF8 = 0x0800;

/** files: [{ name, bytes: Uint8Array }] in, one ZIP (Uint8Array) out. Plain ZIP, no ZIP64: under 4 GiB and 65535 files. */
export function zipStore(files) {
  const enc = new TextEncoder();
  const prepared = files.map((f) => ({ name: enc.encode(f.name), bytes: f.bytes, crc: crc32(f.bytes) }));
  if (prepared.length > 0xffff) throw new Error('too many files for one ZIP');
  let total = 22;
  for (const p of prepared) total += 30 + p.name.length + p.bytes.length + 46 + p.name.length;
  if (total >= 0xffffffff) throw new Error('too big for one ZIP');
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  let o = 0;
  const offsets = [];
  for (const p of prepared) {
    offsets.push(o);
    dv.setUint32(o, 0x04034b50, true);
    dv.setUint16(o + 4, 20, true);
    dv.setUint16(o + 6, UTF8, true);
    dv.setUint16(o + 8, 0, true);
    dv.setUint16(o + 10, DOS_TIME, true);
    dv.setUint16(o + 12, DOS_DATE, true);
    dv.setUint32(o + 14, p.crc, true);
    dv.setUint32(o + 18, p.bytes.length, true);
    dv.setUint32(o + 22, p.bytes.length, true);
    dv.setUint16(o + 26, p.name.length, true);
    dv.setUint16(o + 28, 0, true);
    out.set(p.name, o + 30);
    out.set(p.bytes, o + 30 + p.name.length);
    o += 30 + p.name.length + p.bytes.length;
  }
  const cdStart = o;
  prepared.forEach((p, i) => {
    dv.setUint32(o, 0x02014b50, true);
    dv.setUint16(o + 4, 20, true);
    dv.setUint16(o + 6, 20, true);
    dv.setUint16(o + 8, UTF8, true);
    dv.setUint16(o + 10, 0, true);
    dv.setUint16(o + 12, DOS_TIME, true);
    dv.setUint16(o + 14, DOS_DATE, true);
    dv.setUint32(o + 16, p.crc, true);
    dv.setUint32(o + 20, p.bytes.length, true);
    dv.setUint32(o + 24, p.bytes.length, true);
    dv.setUint16(o + 28, p.name.length, true);
    dv.setUint32(o + 42, offsets[i], true);
    out.set(p.name, o + 46);
    o += 46 + p.name.length;
  });
  dv.setUint32(o, 0x06054b50, true);
  dv.setUint16(o + 8, prepared.length, true);
  dv.setUint16(o + 10, prepared.length, true);
  dv.setUint32(o + 12, o - cdStart, true);
  dv.setUint32(o + 16, cdStart, true);
  return out;
}

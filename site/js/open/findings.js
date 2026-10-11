// Reading a rAidiology reading file: the few kilobytes the CT outliner on somebody's own machine leaves
// behind, so this page can show them their own pictures with the anatomy outlined and named. Only the
// outlines section is used; nothing else in the file is read or shown.
//
// The file never carries an identifier. It refers to the series and to each image by a one-way
// fingerprint, and this module computes the same fingerprint from the scan the browser is already
// holding. So the matching happens here, on the visitor's device, out of two things they brought
// themselves, and the file on its own tells a stranger nothing about whose scan it is.
//
// Nothing here fetches. The file is read with FileReader the same way the DICOM files are.

export const VERSION = 1;

// Gzip because that is what the writer produces and what every browser can already undo; a plain
// .json is accepted too, because somebody who unzips it to look inside should not find that the
// page then refuses the thing they were trying to understand.
async function unpack(bytes) {
  if (!(bytes[0] === 0x1f && bytes[1] === 0x8b)) return bytes;
  if (typeof DecompressionStream !== 'function') {
    throw new Error('This browser cannot un-gzip the file. Open it in a current Chrome, Edge, Firefox or Safari.');
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const MASK = 0xffffffffffffffffn;

function fallbackHash(text) {
  // Only reached when crypto.subtle is missing, which means the page was opened straight off the
  // disk rather than over https. It still has to agree with itself across one session, so the two
  // halves of the matching line up; it never has to agree with the writer, because a file written
  // by the writer always arrives at a page served over https.
  let h = FNV_OFFSET;
  for (let i = 0; i < text.length; i++) {
    h = ((h ^ BigInt(text.charCodeAt(i) & 0xff)) * FNV_PRIME) & MASK;
  }
  return h.toString(16).padStart(16, '0');
}

export async function fingerprint(uid) {
  const text = String(uid || '').trim();
  if (!text) return '';
  if (!(globalThis.crypto && globalThis.crypto.subtle)) return fallbackHash(text);
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('').slice(0, 16);
}

export function looksLikeFindings(name, bytes) {
  if (/\.raidiology\.json(\.gz)?$/i.test(name)) return true;
  if (bytes[0] === 0x7b) {                                  // a bare '{'
    return new TextDecoder().decode(bytes.subarray(0, 200)).includes('raidiology_findings');
  }
  return false;
}

export async function parse(bytes) {
  const text = new TextDecoder().decode(await unpack(bytes));
  let doc;
  try {
    doc = JSON.parse(text);
  } catch {
    throw new Error('That file is not a rAidiology reading: it is not readable as JSON.');
  }
  if (doc.raidiology_findings !== VERSION) {
    throw new Error(`That file says it is rAidiology file version ${doc.raidiology_findings}, and this page reads version ${VERSION}.`);
  }
  if (!doc.scan || !doc.outlines || !Array.isArray(doc.outlines.slices)) {
    throw new Error('That file holds no anatomy outlines, so there is nothing to show.');
  }
  return doc;
}

/** Work out which of the loaded series this reading is about, and which image each outline is on.
 *
 * Two ways round, in order. The fingerprint is the real one: it matches the exact image even when
 * the person loaded the series in a different order or from a different folder. The position in the
 * series is the fallback, for a disc whose files carry no image identifier at all. A reading that
 * matches no loaded series is not an error and not silently dropped: the caller is told, because
 * "you opened a reading of a different scan" is the single most likely mistake here. */
export async function attach(doc, series) {
  const want = String(doc.scan.series_fingerprint || '');
  let hit = -1;
  for (let i = 0; i < series.length && hit < 0; i++) {
    const fp = await fingerprint(series[i].info.seriesUid);
    if (fp && fp === want) hit = i;
  }
  if (hit < 0) {
    return { seriesIndex: -1, outlines: new Map() };
  }
  const slices = series[hit].slices;
  const prints = await Promise.all(slices.map((s) => fingerprint(s.info.sopUid)));
  const place = (item) => {
    if (item.image_fingerprint) return prints.indexOf(item.image_fingerprint);
    return Number.isInteger(item.image_index) && item.image_index < slices.length ? item.image_index : -1;
  };
  // The reading's outlines, matched to images the same two ways.
  const outlines = new Map();
  for (const row of doc.outlines.slices) {
    const at = place(row);
    if (at >= 0 && Array.isArray(row.paths)) outlines.set(at, row.paths);
  }
  return { seriesIndex: hit, outlines };
}

/** One outline as core/findings.py's encode_path wrote it: base64 of zigzag varints, each the step
 * from the last point in half pixels. Returns the points as a flat array, x0, y0, x1, y1, ... in the
 * image's own pixels. Arithmetic rather than bit operators, which would wrap at 32 bits. */
export function decodePath(text) {
  const raw = atob(text);
  const out = [];
  let z = 0;
  let scale = 1;
  let x = 0;
  let y = 0;
  let odd = false;
  for (let i = 0; i < raw.length; i++) {
    const b = raw.charCodeAt(i);
    z += (b & 0x7f) * scale;
    scale *= 128;
    if (b >= 0x80) continue;
    const d = z % 2 === 1 ? -(z + 1) / 2 : z / 2;
    if (odd) { y += d; out.push(x / 2, y / 2); } else { x += d; }
    odd = !odd;
    z = 0;
    scale = 1;
  }
  return out;
}

// A DICOM reader for the browser, with no library behind it: parse a file's data set, describe the
// series it belongs to, and decode one frame's pixels.
//
// It handles what a real hospital disc actually contains. Measured on a real Agfa IMPAX CD: 203 of
// 211 files were Explicit VR Little Endian, 16-bit MONOCHROME2, and 8 were JPEG Baseline. Anything
// else (JPEG 2000, JPEG-LS, JPEG Lossless) is reported by name rather than failing quietly, so the
// page can say plainly that this disc stores its images in a format it cannot open yet.
//
// A patient's name, ID, date of birth, address, referring physician and institution are never read
// into the element map at all. They cannot be shown or kept because they are never there.

const SKIP_TAGS = new Set([
  '00100010', '00100020', '00100030', '00100032', '00100040', '00101000', '00101001', '00101040',
  '00102154', '00080080', '00080081', '00080090', '00081040', '00081048', '00081050', '00081060',
  '00080092', '00080094', '00324000', '00401001', '00402016', '00402017',
]);

// Value representations whose length field is 4 bytes long, after two reserved bytes.
const LONG_VR = new Set(['OB', 'OW', 'OF', 'OL', 'OD', 'OV', 'SQ', 'UT', 'UN', 'UC', 'UR']);
const TEXT_VR = new Set(['AE', 'AS', 'CS', 'DA', 'DS', 'DT', 'IS', 'LO', 'LT', 'PN', 'SH', 'ST', 'TM', 'UC', 'UI', 'UR', 'UT']);

export const transferSyntaxNames = Object.freeze({
  '1.2.840.10008.1.2': 'Implicit VR Little Endian',
  '1.2.840.10008.1.2.1': 'Explicit VR Little Endian',
  '1.2.840.10008.1.2.1.99': 'Deflated Explicit VR Little Endian',
  '1.2.840.10008.1.2.2': 'Explicit VR Big Endian',
  '1.2.840.10008.1.2.5': 'RLE Lossless',
  '1.2.840.10008.1.2.4.50': 'JPEG Baseline',
  '1.2.840.10008.1.2.4.51': 'JPEG Extended',
  '1.2.840.10008.1.2.4.57': 'JPEG Lossless',
  '1.2.840.10008.1.2.4.70': 'JPEG Lossless, first-order prediction',
  '1.2.840.10008.1.2.4.80': 'JPEG-LS Lossless',
  '1.2.840.10008.1.2.4.81': 'JPEG-LS Near-lossless',
  '1.2.840.10008.1.2.4.90': 'JPEG 2000 Lossless',
  '1.2.840.10008.1.2.4.91': 'JPEG 2000',
  '1.2.840.10008.1.2.4.92': 'JPEG 2000 Part 2 Lossless',
  '1.2.840.10008.1.2.4.93': 'JPEG 2000 Part 2',
});

const UNCOMPRESSED = new Set(['1.2.840.10008.1.2', '1.2.840.10008.1.2.1', '1.2.840.10008.1.2.2']);
const JPEG_BY_BROWSER = new Set(['1.2.840.10008.1.2.4.50', '1.2.840.10008.1.2.4.51']);

// The standard VR (DICOM PS3.6) of every tag this app reads, for data sets in Implicit VR Little
// Endian, where the file carries no VR. Covers what describe() and decodeFrame() read, the donation
// allow-list (docs/DONATED_SCANS.md) and every tag deid.js reads to refuse a series. A tag not listed
// is 'UN', which nums() reads as text only. Keys are group+element, 8 upper-case hex digits.
// Two tags allow two VRs: PixelPaddingValue (0028,0120) is US or SS, and is SS exactly when
// PixelRepresentation (0028,0103) is 1 (see vrOf); (7FE0,0010) PixelData is OB or OW, and in
// implicit VR it is always OW.
export const IMPLICIT_VR = Object.freeze({
  '00080008': 'CS', '00080016': 'UI', '00080018': 'UI', '00080020': 'DA', '00080021': 'DA', '00080022': 'DA',
  '00080060': 'CS', '00080070': 'LO', '00081030': 'LO', '0008103E': 'LO', '00081090': 'LO',
  '00100040': 'CS', '00101010': 'AS',
  '00180015': 'CS', '00180020': 'CS', '00180021': 'CS', '00180023': 'CS', '00180050': 'DS', '00180060': 'DS',
  '00180080': 'DS', '00180081': 'DS', '00180087': 'DS', '00180088': 'DS', '00180091': 'IS', '00181030': 'LO',
  '00181164': 'DS', '00181210': 'SH', '00181314': 'DS', '00185101': 'CS',
  '0020000D': 'UI', '0020000E': 'UI', '00200011': 'IS', '00200013': 'IS', '00200020': 'CS', '00200032': 'DS',
  '00200037': 'DS', '00200052': 'UI',
  '00280002': 'US', '00280004': 'CS', '00280006': 'US', '00280008': 'IS', '00280010': 'US', '00280011': 'US',
  '00280030': 'DS', '00280100': 'US', '00280101': 'US', '00280102': 'US', '00280103': 'US',
  '00280120': 'US', '00280301': 'CS', '00281040': 'CS', '00281041': 'SS', '00281050': 'DS', '00281051': 'DS',
  '00281052': 'DS', '00281053': 'DS', '00281054': 'LO', '00281201': 'OW', '00281202': 'OW', '00281203': 'OW',
  '00282110': 'CS', '7FE00010': 'OW',
});

// The VR of an implicit-VR element: the dictionary's, with PixelPaddingValue made signed when the
// picture's pixels are. Tags come in ascending order, so PixelRepresentation is already in the map.
function vrOf(key, map) {
  const vr = IMPLICIT_VR[key] || 'UN';
  if (key === '00280120') {
    const rep = map.get('00280103');
    return rep && rep.bytes.length >= 2 && (rep.bytes[0] | (rep.bytes[1] << 8)) === 1 ? 'SS' : vr;
  }
  return vr;
}

function tagKey(group, element) {
  return (group.toString(16).padStart(4, '0') + element.toString(16).padStart(4, '0')).toUpperCase();
}

function ascii(bytes, from, length) {
  let s = '';
  for (let i = from; i < from + length; i++) s += String.fromCharCode(bytes[i]);
  return s;
}

// A file with the 128-byte preamble starts "DICM" at 128. Some discs write the data set with no
// preamble at all, so a plausible group-2 or group-8 tag at offset 0 counts too.
export function isDicom(bytes) {
  if (!bytes || bytes.length < 136) return false;
  if (ascii(bytes, 128, 4) === 'DICM') return true;
  const group = bytes[0] | (bytes[1] << 8);
  if (group !== 0x0002 && group !== 0x0008) return false;
  const vr = ascii(bytes, 4, 2);
  return /^[A-Z]{2}$/.test(vr) || (bytes[6] | (bytes[7] << 8)) < 0x1000;
}

// Walk the items of an undefined-length sequence (or of encapsulated pixel data) and return the
// offset just past its sequence delimiter. Items may themselves have undefined length, which is
// what a plain forward scan for the delimiter gets wrong: a nested item's own delimiter ends the
// walk early and every tag after it is read from the middle of a value.
function skipItems(view, bytes, p, end, explicit, little) {
  while (p + 8 <= end) {
    const group = view.getUint16(p, little);
    const element = view.getUint16(p + 2, little);
    const len = view.getUint32(p + 4, little);
    if (group === 0xfffe && element === 0xe0dd) return p + 8;
    if (group !== 0xfffe || element !== 0xe000) return p;     // not an item: stop where we are
    p += 8;
    p = len === 0xffffffff ? skipItemContents(view, bytes, p, end, explicit, little) : p + len;
  }
  return p;
}

function skipItemContents(view, bytes, p, end, explicit, little) {
  while (p + 8 <= end) {
    const group = view.getUint16(p, little);
    const element = view.getUint16(p + 2, little);
    if (group === 0xfffe && element === 0xe00d) return p + 8;
    let vr = '';
    let len = 0;
    if (explicit) {
      vr = ascii(bytes, p + 4, 2);
      if (LONG_VR.has(vr)) { len = view.getUint32(p + 8, little); p += 12; }
      else { len = view.getUint16(p + 6, little); p += 8; }
    } else {
      len = view.getUint32(p + 4, little);
      p += 8;
    }
    p = len === 0xffffffff ? skipItems(view, bytes, p, end, explicit, little) : p + len;
  }
  return p;
}

// Returns whether the pixel-data tag itself was reached. Tags are stored in ascending order, so once
// it has been, every tag describe() reads is already in the map: that is how a file's first few
// kilobytes are known to be enough without reading the rest of it.
function parseInto(map, view, bytes, start, end, explicit, little, depth) {
  let p = start;
  let sawPixelData = false;
  while (p + 8 <= end) {
    const group = view.getUint16(p, little);
    const element = view.getUint16(p + 2, little);
    if (group === 0xfffe) {                       // item, item delimiter or sequence delimiter
      const len = view.getUint32(p + 4, little);
      p += 8;
      if (element === 0xe000 && len !== 0xffffffff) p += len;
      continue;
    }
    let vr = '';
    let len = 0;
    if (explicit) {
      vr = ascii(bytes, p + 4, 2);
      if (LONG_VR.has(vr)) {
        len = view.getUint32(p + 8, little);
        p += 12;
      } else {
        len = view.getUint16(p + 6, little);
        p += 8;
      }
    } else {
      len = view.getUint32(p + 4, little);
      p += 8;
    }
    const key = tagKey(group, element);
    if (!explicit) vr = vrOf(key, map);
    if (key === '7FE00010') sawPixelData = true;
    if (len === 0xffffffff) {
      // Undefined length: encapsulated pixel data, whose fragments are kept whole, or a sequence,
      // which is walked past because nothing we read lives inside one.
      const from = p;
      const after = skipItems(view, bytes, p, end, explicit, little);
      if (key === '7FE00010') {
        const to = Math.max(from, after - 8);     // everything up to the sequence delimiter
        map.set(key, { vr: vr || 'OB', offset: from, length: to - from, encapsulated: true,
                       bytes: bytes.subarray(from, to) });
      }
      p = after;
      continue;
    }
    if (vr === 'SQ') {
      p += len;                                   // sequences hold no top-level value we read
      continue;
    }
    if (!SKIP_TAGS.has(key) && p + len <= end) {
      map.set(key, { vr, offset: p, length: len, encapsulated: false, bytes: bytes.subarray(p, p + len) });
    }
    p += len;
  }
  return sawPixelData;
}

export function readDataset(bytes) {
  if (!isDicom(bytes)) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const hasPreamble = ascii(bytes, 128, 4) === 'DICM';
  let transferSyntax = '1.2.840.10008.1.2.1';
  const elements = new Map();
  let p = 0;
  if (hasPreamble) {
    const meta = new Map();
    // The file-meta group is always Explicit VR Little Endian, whatever the data set uses.
    let q = 132;
    let metaEnd = bytes.length;
    const first = tagKey(view.getUint16(q, true), view.getUint16(q + 2, true));
    if (first === '00020000') {
      const len = view.getUint16(q + 6, true);
      metaEnd = q + 8 + len + view.getUint32(q + 8, true);
    }
    parseInto(meta, view, bytes, q, Math.min(metaEnd, bytes.length), true, true, 0);
    const ts = meta.get('00020010');
    if (ts) transferSyntax = ascii(bytes, ts.offset, ts.length).replace(/\0+$/, '').trim();
    // Walk the meta group again to find where it ends, since the length tag may be absent.
    q = 132;
    while (q + 8 <= bytes.length && view.getUint16(q, true) === 0x0002) {
      const vr = ascii(bytes, q + 4, 2);
      if (LONG_VR.has(vr)) q += 12 + view.getUint32(q + 8, true);
      else q += 8 + view.getUint16(q + 6, true);
    }
    p = q;
  }
  const little = transferSyntax !== '1.2.840.10008.1.2.2';
  const explicit = transferSyntax !== '1.2.840.10008.1.2';
  const pixelDataSeen = parseInto(elements, view, bytes, p, bytes.length, explicit, little, 0);
  return { transferSyntax, transferSyntaxName: transferSyntaxNames[transferSyntax] || transferSyntax,
           elements, little, explicit, bytes, pixelDataSeen };
}

export function text(elements, key) {
  const e = elements.get(key);
  if (!e) return '';
  let s = '';
  for (let i = 0; i < e.bytes.length; i++) s += String.fromCharCode(e.bytes[i]);
  return s.replace(/\0+$/, '').trim();
}

export function nums(elements, key, little = true) {
  const e = elements.get(key);
  if (!e) return [];
  if (TEXT_VR.has(e.vr) || e.vr === 'UN') {
    const s = text(elements, key);
    if (/^[\s\d+\-.eE\\]*$/.test(s) && s !== '') {
      return s.split('\\').map((v) => Number(v)).filter((v) => Number.isFinite(v));
    }
  }
  const dv = new DataView(e.bytes.buffer, e.bytes.byteOffset, e.bytes.byteLength);
  const out = [];
  if (e.vr === 'US' || e.vr === 'OW') for (let i = 0; i + 2 <= e.bytes.length; i += 2) out.push(dv.getUint16(i, little));
  else if (e.vr === 'SS') for (let i = 0; i + 2 <= e.bytes.length; i += 2) out.push(dv.getInt16(i, little));
  else if (e.vr === 'UL') for (let i = 0; i + 4 <= e.bytes.length; i += 4) out.push(dv.getUint32(i, little));
  else if (e.vr === 'SL') for (let i = 0; i + 4 <= e.bytes.length; i += 4) out.push(dv.getInt32(i, little));
  else if (e.vr === 'FL') for (let i = 0; i + 4 <= e.bytes.length; i += 4) out.push(dv.getFloat32(i, little));
  else if (e.vr === 'FD') for (let i = 0; i + 8 <= e.bytes.length; i += 8) out.push(dv.getFloat64(i, little));
  return out;
}

export function num(elements, key, fallback = null, little = true) {
  const v = nums(elements, key, little);
  return v.length ? v[0] : fallback;
}

function dicomDate(s) {
  return /^\d{8}$/.test(s) ? `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}` : '';
}

export function describe(ds) {
  const e = ds.elements;
  const L = ds.little;
  const g = (k, f = null) => num(e, k, f, L);
  return {
    sopClassUid: text(e, '00080016'),
    // The image's own identifier. Kept so a reading written elsewhere can be matched to the exact
    // picture it is about; it is hashed before it is ever compared, and it is not a patient tag.
    sopUid: text(e, '00080018'),
    modality: text(e, '00080060'),
    studyUid: text(e, '0020000D'),
    seriesUid: text(e, '0020000E') || text(e, '0020000D') || 'unknown-series',
    seriesNumber: g('00200011', null),
    seriesDescription: text(e, '0008103E') || text(e, '00081030') || '',
    instanceNumber: g('00200013', null),
    acquisitionDate: dicomDate(text(e, '00080022')) || dicomDate(text(e, '00080021')) || dicomDate(text(e, '00080020')),
    rows: g('00280010', 0),
    cols: g('00280011', 0),
    bitsAllocated: g('00280100', 8),
    bitsStored: g('00280101', g('00280100', 8)),
    highBit: g('00280102', null),
    pixelRepresentation: g('00280103', 0),
    samplesPerPixel: g('00280002', 1),
    photometric: text(e, '00280004').toUpperCase(),
    planarConfiguration: g('00280006', 0),
    numberOfFrames: Math.max(1, Number(text(e, '00280008')) || g('00280008', 1) || 1),
    rescaleSlope: g('00281053', 1) ?? 1,
    rescaleIntercept: g('00281052', 0) ?? 0,
    windowCentre: g('00281050', null),
    windowWidth: g('00281051', null),
    pixelSpacing: nums(e, '00280030', L).length >= 2 ? nums(e, '00280030', L).slice(0, 2)
      : (nums(e, '00181164', L).length >= 2 ? nums(e, '00181164', L).slice(0, 2) : null),
    sliceThickness: g('00180050', null),
    spacingBetweenSlices: g('00180088', null),
    imagePosition: nums(e, '00200032', L).length === 3 ? nums(e, '00200032', L) : null,
    imageOrientation: nums(e, '00200037', L).length === 6 ? nums(e, '00200037', L) : null,
  };
}

// The fragments of an encapsulated pixel-data element, skipping the basic offset table.
function fragments(el, little) {
  const b = el.bytes;
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const out = [];
  let p = 0;
  let first = true;
  while (p + 8 <= b.length) {
    const group = dv.getUint16(p, little);
    const element = dv.getUint16(p + 2, little);
    const len = dv.getUint32(p + 4, little);
    p += 8;
    if (group !== 0xfffe || element !== 0xe000) break;
    if (!(first && len === 0)) {
      if (first) first = false;                   // a non-empty first item is the offset table
      else out.push(b.subarray(p, p + len));
    } else {
      first = false;
    }
    p += len;
  }
  return out.length ? out : [b];
}

function rleDecode(src, rows, cols, samples, bytesPerSample) {
  const dv = new DataView(src.buffer, src.byteOffset, src.byteLength);
  const count = dv.getUint32(0, true);
  const out = new Uint8Array(rows * cols * samples * bytesPerSample);
  const pixels = rows * cols;
  for (let seg = 0; seg < count; seg++) {
    const from = dv.getUint32(4 + seg * 4, true);
    const to = seg + 1 < count ? dv.getUint32(4 + (seg + 1) * 4, true) : src.length;
    // Segments run most significant byte first within a sample, samples outermost.
    const sample = Math.floor(seg / bytesPerSample);
    const byteInSample = bytesPerSample - 1 - (seg % bytesPerSample);
    let o = sample * pixels * bytesPerSample + byteInSample;
    const stride = samples > 1 ? samples * bytesPerSample : bytesPerSample;
    let p = from;
    while (p < to && o < out.length) {
      const n = dv.getInt8(p); p += 1;
      if (n >= 0) {
        for (let i = 0; i <= n && p < to; i++) { out[o] = src[p]; p += 1; o += stride; }
      } else if (n !== -128) {
        const v = src[p]; p += 1;
        for (let i = 0; i < 1 - n; i++) { out[o] = v; o += stride; }
      }
    }
  }
  return out;
}

async function jpegToGrey(fragment, rows, cols) {
  const bitmap = await createImageBitmap(new Blob([fragment], { type: 'image/jpeg' }));
  const w = bitmap.width || cols;
  const h = bitmap.height || rows;
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return { data: ctx.getImageData(0, 0, w, h).data, width: w, height: h };
}

function paletteTables(ds) {
  const e = ds.elements;
  const L = ds.little;
  const r = nums(e, '00281201', L);
  const g = nums(e, '00281202', L);
  const b = nums(e, '00281203', L);
  return r.length && g.length && b.length ? { r, g, b } : null;
}

// Typed arrays use the computer's own byte order. Every browser in use is little-endian, but this is
// checked rather than assumed, so a big-endian machine reads the bytes one at a time and stays right.
const HOST_LITTLE = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

// The stored numbers of a frame, as a typed array of their own type. Little-endian 16-bit data is
// viewed in place, and copied first only when it starts on an odd byte, which a view cannot do.
// Only big-endian data needs a byte-by-byte DataView read. 8-bit data is unsigned; 16-bit data is
// signed exactly when the file says so.
function storedSamples(buf, from, count, bytesPer, signed, little) {
  if (bytesPer === 1) return new Uint8Array(buf, from, count);
  const Type = signed ? Int16Array : Uint16Array;
  if (little && HOST_LITTLE) {
    return from % 2 === 0 ? new Type(buf, from, count) : new Type(buf.slice(from, from + count * 2));
  }
  const dv = new DataView(buf, from, count * 2);
  const out = new Type(count);
  for (let i = 0; i < count; i++) out[i] = signed ? dv.getInt16(i * 2, little) : dv.getUint16(i * 2, little);
  return out;
}

export async function decodeFrame(ds, frame = 0) {
  const d = describe(ds);
  const el = ds.elements.get('7FE00010');
  if (!el || !d.rows || !d.cols) {
    return { unsupported: 'no pixel data', transferSyntaxName: ds.transferSyntaxName };
  }
  const n = d.rows * d.cols;
  const samples = Math.max(1, d.samplesPerPixel);
  const palette = samples === 1 && d.photometric === 'PALETTE COLOR' ? paletteTables(ds) : null;
  if (JPEG_BY_BROWSER.has(ds.transferSyntax)) return jpegFrame(ds, el, d, samples, frame);
  const got = storedFrame(ds, el, d, samples, frame);
  if (!got) return { unsupported: ds.transferSyntaxName, transferSyntaxName: ds.transferSyntaxName };
  if (got.unsupported) return got;
  const { stored } = got;

  // A plain greyscale frame keeps its stored numbers as they are; the viewer turns them into grey
  // levels through a table, which is where rescaling now happens too.
  if (samples === 1 && !palette) return greyFrame(stored, d.cols, d.rows, d);
  // Colour and palette frames are few, and stay floating point. set() is a native int-to-float
  // copy, and every 8- or 16-bit integer is exact in a 32-bit float.
  const values = new Float32Array(n * samples);
  values.set(stored);
  return finish(values, d.cols, d.rows, samples, d, palette, false);
}

// The stored numbers of one frame of uncompressed or RLE pixel data: { stored }, { unsupported } when
// the pixel data is cut short, or null for any other transfer syntax.
function storedFrame(ds, el, d, samples, frame) {
  const n = d.rows * d.cols;
  const signed = d.pixelRepresentation === 1;
  if (UNCOMPRESSED.has(ds.transferSyntax) && !el.encapsulated) {
    const bytesPer = d.bitsAllocated > 8 ? 2 : 1;
    const per = n * samples * bytesPer;
    const from = el.bytes.byteOffset + frame * per;
    const buf = el.bytes.buffer;
    if (from + per > buf.byteLength) return { unsupported: 'truncated pixel data', transferSyntaxName: ds.transferSyntaxName };
    return { stored: storedSamples(buf, from, n * samples, bytesPer, signed, ds.little) };
  }
  if (ds.transferSyntax === '1.2.840.10008.1.2.5') {
    const frags = fragments(el, ds.little);
    const src = frags[Math.min(frame, frags.length - 1)];
    const bytesPer = d.bitsAllocated > 8 ? 2 : 1;
    const raw = rleDecode(src, d.rows, d.cols, samples, bytesPer);
    // RLE puts the most significant byte first, so its 16-bit samples are big-endian.
    return { stored: storedSamples(raw.buffer, raw.byteOffset, n * samples, bytesPer, signed, false) };
  }
  return null;
}

// A JPEG frame, decoded by the browser into 8-bit pixels.
async function jpegFrame(ds, el, d, samples, frame) {
  const frags = fragments(el, ds.little);
  const src = frags[Math.min(frame, frags.length - 1)];
  let rgba;
  try {
    rgba = await jpegToGrey(src, d.rows, d.cols);
  } catch {
    return { unsupported: 'a JPEG the browser refused', transferSyntaxName: ds.transferSyntaxName };
  }
  return finish(jpegValues(rgba, samples), rgba.width, rgba.height, samples === 1 ? 1 : 3, d, null, true);
}

// The browser's RGBA pixels as floats: one value per pixel for greyscale, three for colour.
function jpegValues(rgba, samples) {
  const total = rgba.width * rgba.height;
  if (samples === 1) {
    const values = new Float32Array(total);
    // The browser has already turned YBR into RGB; for a greyscale frame the channels match.
    for (let i = 0; i < total; i++) values[i] = rgba.data[i * 4];
    return values;
  }
  const values = new Float32Array(total * 3);
  for (let i = 0; i < total; i++) {
    values[i * 3] = rgba.data[i * 4];
    values[i * 3 + 1] = rgba.data[i * 4 + 1];
    values[i * 3 + 2] = rgba.data[i * 4 + 2];
  }
  return values;
}

// The modality rescale of one stored value, rounded to a 32-bit float the way storing it in a
// Float32Array rounds it, so a raw frame and a floating-point frame of the same pixels agree to the
// last bit. Every grey level of a raw frame comes through here.
export function rescale(stored, slope, intercept) {
  return Math.fround(stored * slope + intercept);
}

function rescaleOf(d) {
  return {
    slope: Number.isFinite(d.rescaleSlope) ? d.rescaleSlope : 1,
    intercept: Number.isFinite(d.rescaleIntercept) ? d.rescaleIntercept : 0,
  };
}

// MONOCHROME1 means low values are white. The numbers are left as the file stored them, so that the
// file's own window centre and width (in those numbers) mean what the file says, and the viewer
// inverts the grey level at the very end. Flipping the numbers about a frame's own minimum and
// maximum would move the file's window, and move it differently on every slice.
function invertsOf(d, samples) {
  return samples === 1 && d.photometric === 'MONOCHROME1';
}

function greyFrame(raw, width, height, d) {
  const { slope, intercept } = rescaleOf(d);
  let rawMin = raw[0];
  let rawMax = raw[0];
  for (let i = 1; i < raw.length; i++) {
    const v = raw[i];
    if (v < rawMin) rawMin = v;
    else if (v > rawMax) rawMax = v;
  }
  // Rescaling is monotonic, so the lowest and highest rescaled values come from the lowest and
  // highest stored ones: the same min and max a pass over every rescaled pixel would find.
  const a = rescale(rawMin, slope, intercept);
  const b = rescale(rawMax, slope, intercept);
  let min = Math.min(a, b);
  let max = Math.max(a, b);
  if (!Number.isFinite(min)) { min = 0; max = 1; }
  return { width, height, samples: 1, raw, rawMin, rawMax, slope, intercept, invert: invertsOf(d, 1), min, max };
}

function finish(values, width, height, samples, d, palette, alreadyDisplayReady) {
  if (!alreadyDisplayReady) {
    if (palette) {
      const out = new Float32Array(width * height * 3);
      const top = palette.r.length - 1;
      for (let i = 0; i < width * height; i++) {
        const k = Math.min(top, Math.max(0, Math.round(values[i])));
        out[i * 3] = palette.r[k]; out[i * 3 + 1] = palette.g[k]; out[i * 3 + 2] = palette.b[k];
      }
      values = out;
      samples = 3;
    } else if (samples === 3 && d.planarConfiguration === 1) {
      const out = new Float32Array(values.length);
      const n = width * height;
      for (let i = 0; i < n; i++) {
        out[i * 3] = values[i];
        out[i * 3 + 1] = values[n + i];
        out[i * 3 + 2] = values[2 * n + i];
      }
      values = out;
    }
    const { slope, intercept } = rescaleOf(d);
    if (slope !== 1 || intercept !== 0) {
      for (let i = 0; i < values.length; i++) values[i] = values[i] * slope + intercept;
    }
  }
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] < min) min = values[i];
    if (values[i] > max) max = values[i];
  }
  if (!Number.isFinite(min)) { min = 0; max = 1; }
  return { width, height, samples, values, invert: invertsOf(d, samples), min, max };
}

// How many pixels fall in each of 1024 bins between min and max. A raw frame counts each stored
// value once and then drops each count into the bin its rescaled value falls in, which builds the
// very same histogram as a pass over every rescaled pixel, without rescaling every pixel.
function rawHistogram(frame, bins, scale) {
  const { raw, rawMin, rawMax, slope, intercept, min } = frame;
  const counts = new Uint32Array(rawMax - rawMin + 1);
  for (let i = 0; i < raw.length; i++) counts[raw[i] - rawMin] += 1;
  const hist = new Uint32Array(bins);
  for (let k = 0; k < counts.length; k++) {
    if (counts[k]) hist[Math.round((rescale(k + rawMin, slope, intercept) - min) * scale)] += counts[k];
  }
  return hist;
}

function valueHistogram(values, bins, scale, min) {
  const hist = new Uint32Array(bins);
  for (let i = 0; i < values.length; i++) hist[Math.round((values[i] - min) * scale)] += 1;
  return hist;
}

// A 1024-bin histogram, so this stays fast on a 1024x1024 frame.
export function autoWindow(frame) {
  const { min, max } = frame;
  if (!(max > min)) return { centre: min || 0, width: 1 };
  const bins = 1024;
  const scale = (bins - 1) / (max - min);
  const hist = frame.raw ? rawHistogram(frame, bins, scale) : valueHistogram(frame.values, bins, scale, min);
  const total = frame.raw ? frame.raw.length : frame.values.length;
  const want = [0.01 * total, 0.99 * total];
  let seen = 0;
  let lo = min;
  let hi = max;
  let gotLow = false;
  for (let b = 0; b < bins; b++) {
    seen += hist[b];
    if (!gotLow && seen >= want[0]) { lo = min + b / scale; gotLow = true; }
    if (seen >= want[1]) { hi = min + b / scale; break; }
  }
  if (hi <= lo) { lo = min; hi = max; }
  return { centre: (lo + hi) / 2, width: Math.max(hi - lo, 1e-6) };
}

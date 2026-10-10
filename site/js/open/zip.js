// zip.js: reads a ZIP in the visitor's own browser with DecompressionStream, so no library and no
// upload is needed. Large archives are read by slicing the Blob, never by loading it whole.

const SIG_EOCD = 0x06054b50;
const SIG_EOCD64_LOCATOR = 0x07064b50;
const SIG_EOCD64 = 0x06064b50;
const SIG_CENTRAL = 0x02014b50;
const SIG_LOCAL = 0x04034b50;
const EOCD_MIN = 22;
// The comment can be up to 65535 bytes, so the record sits somewhere in the last 66 KB.
const TAIL_SCAN = 66 * 1024;
const YIELD_EVERY = 32;

const METHOD_NAMES = {
  1: 'Shrink', 6: 'Implode', 9: 'Deflate64', 12: 'BZIP2', 14: 'LZMA', 93: 'Zstandard', 95: 'XZ', 98: 'PPMd', 99: 'AES encryption',
};

export function zipSupported() {
  if (typeof DecompressionStream !== 'function') return false;
  try {
    new DecompressionStream('deflate-raw');
    return true;
  } catch {
    return false;
  }
}

function makeSource(input) {
  if (typeof Blob !== 'undefined' && input instanceof Blob) {
    return {
      size: input.size,
      async read(start, end) {
        return new Uint8Array(await input.slice(start, end).arrayBuffer());
      },
      stream(start, end) {
        return input.slice(start, end).stream();
      },
    };
  }
  let bytes;
  if (input instanceof Uint8Array) bytes = input;
  else if (input instanceof ArrayBuffer) bytes = new Uint8Array(input);
  else if (ArrayBuffer.isView(input)) bytes = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  else throw new Error('This is not a file the ZIP reader can open.');
  return {
    size: bytes.length,
    async read(start, end) {
      return bytes.subarray(start, Math.min(end, bytes.length));
    },
    stream(start, end) {
      return new Blob([bytes.subarray(start, end)]).stream();
    },
  };
}

function view(u8) {
  return new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
}

// JavaScript numbers are exact up to 2^53, far beyond any real archive, so BigInt is not needed.
function u64(dv, at) {
  return dv.getUint32(at, true) + dv.getUint32(at + 4, true) * 0x100000000;
}

const utf8 = new TextDecoder('utf-8');
function latin1(u8) {
  let s = '';
  for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]);
  return s;
}

async function findEnd(src) {
  const tailStart = Math.max(0, src.size - TAIL_SCAN);
  const tail = await src.read(tailStart, src.size);
  const dv = view(tail);
  let at = -1;
  for (let i = tail.length - EOCD_MIN; i >= 0; i--) {
    if (dv.getUint32(i, true) === SIG_EOCD) { at = i; break; }
  }
  if (at < 0) throw new Error('This does not look like a ZIP file: its table of contents is missing, so it may be damaged or only partly downloaded.');

  let count = dv.getUint16(at + 10, true);
  let cdSize = dv.getUint32(at + 12, true);
  let cdOffset = dv.getUint32(at + 16, true);

  // Zip64 places a locator just before the classic record; trust it whenever it is present.
  const locAbs = tailStart + at - 20;
  if (locAbs >= 0) {
    const loc = locAbs >= tailStart ? tail.subarray(at - 20, at) : await src.read(locAbs, locAbs + 20);
    const ldv = view(loc);
    if (loc.length === 20 && ldv.getUint32(0, true) === SIG_EOCD64_LOCATOR) {
      const recAt = u64(ldv, 8);
      const rec = await src.read(recAt, recAt + 56);
      const rdv = view(rec);
      if (rec.length < 56 || rdv.getUint32(0, true) !== SIG_EOCD64) {
        throw new Error('This large ZIP file has a damaged table of contents, so it cannot be opened.');
      }
      count = u64(rdv, 32);
      cdSize = u64(rdv, 40);
      cdOffset = u64(rdv, 48);
    }
  }
  if (cdOffset + cdSize > src.size) {
    throw new Error('This ZIP file is incomplete: its table of contents points past the end of the file.');
  }
  return { count, cdSize, cdOffset };
}

function readCentral(cd, count) {
  const dv = view(cd);
  const entries = [];
  let p = 0;
  for (let n = 0; n < count; n++) {
    if (p + 46 > cd.length || dv.getUint32(p, true) !== SIG_CENTRAL) {
      throw new Error('This ZIP file has a damaged table of contents, so it cannot be opened.');
    }
    const flags = dv.getUint16(p + 8, true);
    const method = dv.getUint16(p + 10, true);
    let csize = dv.getUint32(p + 20, true);
    let usize = dv.getUint32(p + 24, true);
    const nameLen = dv.getUint16(p + 28, true);
    const extraLen = dv.getUint16(p + 30, true);
    const commentLen = dv.getUint16(p + 32, true);
    let offset = dv.getUint32(p + 42, true);
    const rawName = cd.subarray(p + 46, p + 46 + nameLen);
    const name = (flags & 0x0800) ? utf8.decode(rawName) : latin1(rawName);

    // The Zip64 extra field holds only the values whose classic slot is saturated, in this order.
    let e = p + 46 + nameLen;
    const extraEnd = e + extraLen;
    while (e + 4 <= extraEnd) {
      const id = dv.getUint16(e, true);
      const len = dv.getUint16(e + 2, true);
      if (id === 0x0001) {
        let q = e + 4;
        const end = q + len;
        if (usize === 0xffffffff && q + 8 <= end) { usize = u64(dv, q); q += 8; }
        if (csize === 0xffffffff && q + 8 <= end) { csize = u64(dv, q); q += 8; }
        if (offset === 0xffffffff && q + 8 <= end) { offset = u64(dv, q); q += 8; }
      }
      e += 4 + len;
    }
    entries.push({ name, flags, method, csize, usize, offset });
    p = extraEnd + commentLen;
  }
  return entries;
}

function unsafeReason(name) {
  if (name.startsWith('/') || name.startsWith('\\')) return 'its path starts at the top of the disk, so it was not unpacked';
  if (/^[A-Za-z]:/.test(name)) return 'its path names a drive letter, so it was not unpacked';
  if (name.split(/[\\/]/).includes('..')) return 'its path tries to step outside the archive, so it was not unpacked';
  return '';
}

function joinChunks(chunks, size) {
  const out = new Uint8Array(size);
  let at = 0;
  for (const c of chunks) {
    if (at >= size) break;
    const part = c.length <= size - at ? c : c.subarray(0, size - at);
    out.set(part, at);
    at += part.length;
  }
  return out;
}

// `limit` is how many bytes are wanted from the front of the entry. Inflating stops once it has
// them, so the head of a large entry costs about as much as the head and no more.
async function collectStream(stream, expected, limit = Infinity) {
  const reader = stream.getReader();
  const chunks = [];
  let total = 0;
  while (total < limit) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
  }
  if (total >= limit) {
    // The rest is not wanted now; it is inflated again, whole, when the picture is shown.
    reader.cancel().catch(() => {});
    return joinChunks(chunks, limit);
  }
  if (expected && total !== expected) {
    throw new Error(`it unpacked to ${total} bytes instead of the ${expected} the archive promised`);
  }
  if (chunks.length === 1) return chunks[0];
  return joinChunks(chunks, total);
}

async function extract(src, entry, limit = Infinity) {
  const head = await src.read(entry.offset, entry.offset + 30);
  const hdv = view(head);
  if (head.length < 30 || hdv.getUint32(0, true) !== SIG_LOCAL) {
    throw new Error('its entry in the archive is damaged');
  }
  const start = entry.offset + 30 + hdv.getUint16(26, true) + hdv.getUint16(28, true);
  const end = start + entry.csize;
  if (end > src.size) throw new Error('the archive ends before this file does');

  if (entry.method === 0) {
    const bytes = await src.read(start, Math.min(end, start + limit));
    // A subarray of caller bytes would keep the whole archive alive; copy so it can be freed.
    return bytes.slice();
  }
  const stream = src.stream(start, end).pipeThrough(new DecompressionStream('deflate-raw'));
  return collectStream(stream, entry.usize, limit);
}

// What inflate-worker.js calls: the same extract() the page's own thread runs, so an entry comes
// out byte for byte the same whichever thread unpacked it.
export function extractEntry(blob, entry, limit) {
  return extract(makeSource(blob), entry, limit);
}

function errorText(err) {
  const msg = err && err.message ? err.message : String(err);
  return msg.replace(/\.$/, '');
}

// Inflating is CPU work on whichever thread does it, so a big archive is unpacked by a few workers
// beside the page rather than on the thread that draws it and answers the visitor. The worker is
// this site's own file next to this one; it is handed the visitor's File and hands back bytes.
// A small archive (the sample) stays on the page's thread, where it is done before a worker would
// even have started.
const WORKER_URL = new URL('./inflate-worker.js', import.meta.url);
const WORKERS_FROM_ENTRIES = 64;
const WORKERS_FROM_BYTES = 32 * 1024 * 1024;
const WORKER_START_MS = 5000;
const WORKER_IDLE_MS = 30 * 1000;

let poolStart = null;

function workerCount() {
  const cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 2;
  return Math.max(1, Math.min(6, cores - 1));
}

function makePool(size) {
  const slots = [];
  const queue = [];
  let nextId = 1;
  let idleTimer = 0;

  function spawn() {
    let isModule = false;
    let worker;
    try {
      // A browser without module workers ignores `type` without ever reading it, and would then
      // fail on the worker's import line. Reading the option is how this knows in advance.
      worker = new Worker(WORKER_URL, { get type() { isModule = true; return 'module'; } });
    } catch {
      return null;                                  // a page opened from file://, or workers turned off
    }
    if (!isModule) { worker.terminate(); return null; }
    const slot = { worker, job: null, onReady: null, onFail: null };
    worker.onmessage = (ev) => answered(slot, ev.data);
    worker.onerror = (ev) => { ev.preventDefault(); failed(slot); };
    slots.push(slot);
    return slot;
  }

  function answered(slot, data) {
    if (data && data.ready) { if (slot.onReady) slot.onReady(); return; }
    const job = slot.job;
    if (!job || !data || data.id !== job.id) return;
    slot.job = null;
    if ('error' in data) job.reject(new Error(data.error));
    else job.resolve(data.bytes);
    pump();
  }

  function failed(slot) {
    slot.worker.terminate();
    const at = slots.indexOf(slot);
    if (at >= 0) slots.splice(at, 1);
    if (slot.onFail) slot.onFail();
    if (slot.job) slot.job.reject(new Error('the unpacking helper stopped working'));
    slot.job = null;
    pump();
  }

  function send(slot, job) {
    slot.job = job;
    const { offset, csize, usize, method } = job.entry;
    slot.worker.postMessage({ id: job.id, blob: job.blob, entry: { offset, csize, usize, method }, limit: job.limit });
  }

  function pump() {
    clearTimeout(idleTimer);
    for (const slot of slots) if (!slot.job && queue.length) send(slot, queue.shift());
    while (queue.length && slots.length < size) {
      const slot = spawn();
      if (!slot) break;
      send(slot, queue.shift());
    }
    if (!slots.length) {
      while (queue.length) queue.shift().reject(new Error('the unpacking helper could not be started'));
    }
    // Idle workers are let go after a while and started again on the next job.
    if (!queue.length && slots.every((s) => !s.job)) {
      idleTimer = setTimeout(() => { for (const s of slots.splice(0)) s.worker.terminate(); }, WORKER_IDLE_MS);
    }
  }

  function run(blob, entry, limit) {
    return new Promise((resolve, reject) => {
      queue.push({ id: nextId++, blob, entry, limit, resolve, reject });
      pump();
    });
  }

  // Resolves true once one worker has loaded and said so, false if none can.
  function start() {
    return new Promise((resolve) => {
      const first = spawn();
      if (!first) { resolve(false); return; }
      first.onReady = () => resolve(true);
      first.onFail = () => resolve(false);
      setTimeout(() => resolve(false), WORKER_START_MS);
    });
  }

  function close() {
    clearTimeout(idleTimer);
    for (const s of slots.splice(0)) s.worker.terminate();
  }

  return { run, start, close };
}

// One pool for the whole page, made the first time a big archive needs it. It resolves to null
// where this browser cannot run the worker, and such a browser unpacks on the page's own thread.
function sharedPool() {
  if (!poolStart) {
    if (typeof Worker !== 'function') {
      poolStart = Promise.resolve(null);
    } else {
      const pool = makePool(workerCount());
      poolStart = pool.start().then((ok) => {
        if (ok) return pool;
        pool.close();
        return null;
      });
    }
  }
  return poolStart;
}

async function poolFor(input, wanted) {
  if (!(typeof Blob !== 'undefined' && input instanceof Blob)) return null;
  const packed = wanted.reduce((n, p) => n + p.entry.csize, 0);
  if (wanted.length < WORKERS_FROM_ENTRIES && packed < WORKERS_FROM_BYTES) return null;
  return sharedPool();
}

function planEntry(entry) {
  const path = entry.name;
  // Folders carry no data; their files arrive as their own entries.
  if (path.endsWith('/') || path.endsWith('\\')) return { entry, path };
  const unsafe = unsafeReason(path);
  if (unsafe) return { entry, path, why: unsafe };
  if (entry.flags & 0x0001) return { entry, path, why: 'it is password protected, and this page cannot unlock it' };
  if (entry.method !== 0 && entry.method !== 8) {
    const label = METHOD_NAMES[entry.method] ? ` (${METHOD_NAMES[entry.method]})` : '';
    return { entry, path, why: `it is packed with compression method ${entry.method}${label}, which this browser cannot unpack` };
  }
  return { entry, path, wanted: true };
}

function settled(promise) {
  return promise.then((bytes) => ({ bytes }), (err) => ({ err }));
}

function keep(bytes) {
  return async () => bytes;
}

// One unpacked entry. `head` is its first bytes; `whole` says whether that is all of it. `load`
// gives the whole entry, inflating it again only when the head was not all of it. Built in its own
// function so the loader holds the entry and not the head.
function entryFile(path, entry, head, whole, unpack) {
  return {
    path: path.replace(/\\/g, '/'),
    size: whole ? head.byteLength : entry.usize,
    head,
    whole,
    load: whole ? keep(head) : () => unpack(entry, Infinity),
  };
}

// `headBytes` caps what is inflated now from each entry; the rest waits for `load`. Without it every
// entry is inflated whole, as the archive's own sizes promise, and checked against them.
export async function unzip(blobOrBytes, onProgress, { headBytes = Infinity } = {}) {
  const src = makeSource(blobOrBytes);
  const { count, cdSize, cdOffset } = await findEnd(src);
  const cd = await src.read(cdOffset, cdOffset + cdSize);
  const entries = readCentral(cd, count);
  const plan = entries.map(planEntry);

  const pool = await poolFor(blobOrBytes, plan.filter((p) => p.wanted));
  const inflateOnThisThread = (entry, limit) => extract(src, entry, limit);
  const unpack = pool ? (entry, limit) => pool.run(blobOrBytes, entry, limit) : inflateOnThisThread;
  const limitOf = (entry) => (entry.usize > headBytes ? headBytes : Infinity);
  // With workers every entry is handed out at once and the loop below takes them back in order, so
  // the files and the reasons come out in the same order either way.
  const started = pool
    ? plan.map((p) => (p.wanted ? settled(unpack(p.entry, limitOf(p.entry))) : null))
    : null;

  const files = [];
  const skipped = [];
  const total = entries.length;
  for (let i = 0; i < total; i++) {
    const { entry, path, why, wanted } = plan[i];
    if (why) {
      skipped.push({ path, why });
    } else if (wanted) {
      const limit = limitOf(entry);
      const got = started ? await started[i] : await settled(unpack(entry, limit));
      if (got.err) skipped.push({ path, why: `it could not be unpacked: ${errorText(got.err)}` });
      else files.push(entryFile(path, entry, got.bytes, limit === Infinity, unpack));
    }
    if (typeof onProgress === 'function') onProgress(i + 1, total, path);
    if ((i + 1) % YIELD_EVERY === 0) await new Promise((resolve) => setTimeout(resolve, 0));
  }
  return { files, skipped };
}

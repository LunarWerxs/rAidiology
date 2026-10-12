// collect.js: turns a drop or a file picker into one flat list of files, each with its first bytes
// in memory and a way to read the rest, unzipping any ZIP on the way. Everything stays in the
// visitor's browser.
import { unzip, zipSupported } from './zip.js';

const MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024;

// Hospital discs bundle a viewer, its help pages and an index; none of these are scans.
const NOT_SCANS = new Set(['exe', 'dll', 'ini', 'htm', 'html', 'pdf', 'txt', 'xml', 'css', 'js', 'jar', 'dat', 'cab', 'msi']);

function baseName(path) {
  const parts = path.split('/');
  return parts[parts.length - 1];
}

function extensionOf(path) {
  const name = baseName(path);
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : '';
}

function isZip(path) {
  return extensionOf(path) === 'zip';
}

// Each reason finishes a sentence that starts with the file's path, as the viewer lists them.
function notScanReason(path) {
  if (baseName(path).toUpperCase() === 'DICOMDIR') return 'is the disc\'s index (DICOMDIR), so it was skipped.';
  const ext = extensionOf(path);
  if (NOT_SCANS.has(ext)) return `is a .${ext} file from the disc's own software, so it was skipped.`;
  return '';
}

function limitError() {
  const gb = MAX_TOTAL_BYTES / (1024 * 1024 * 1024);
  return new Error(`These files add up to more than ${gb} GB, the most this page can hold in your browser's memory. Open one folder at a time.`);
}

function readBatch(reader) {
  return new Promise((resolve, reject) => reader.readEntries(resolve, reject));
}

function entryFile(entry) {
  return new Promise((resolve, reject) => entry.file(resolve, reject));
}

async function walk(entry, prefix, out, skipped) {
  const path = prefix ? `${prefix}/${entry.name}` : entry.name;
  if (entry.isFile) {
    try {
      out.push({ path, file: await entryFile(entry) });
    } catch {
      skipped.push({ path, why: 'was kept from this page by the browser.' });
    }
    return;
  }
  if (!entry.isDirectory) return;
  const reader = entry.createReader();
  // readEntries hands back at most 100 entries per call, so keep asking until it returns none.
  for (;;) {
    let batch;
    try {
      batch = await readBatch(reader);
    } catch {
      skipped.push({ path, why: 'is a folder the browser kept this page from listing.' });
      return;
    }
    if (!batch.length) break;
    for (const child of batch) await walk(child, path, out, skipped);
  }
}

export async function fromDataTransfer(dataTransfer) {
  const items = dataTransfer && dataTransfer.items ? Array.from(dataTransfer.items) : [];
  // Entries must be taken synchronously: the browser empties the DataTransfer after the first await.
  const roots = [];
  for (const item of items) {
    if (item.kind !== 'file') continue;
    const entry = typeof item.webkitGetAsEntry === 'function' ? item.webkitGetAsEntry() : null;
    if (entry) roots.push(entry);
  }
  const fallback = dataTransfer && dataTransfer.files ? Array.from(dataTransfer.files) : [];
  if (!roots.length) return fallback.map((file) => ({ path: file.name, file }));

  const out = [];
  const skipped = [];
  for (const root of roots) await walk(root, '', out, skipped);
  // A file the browser refused to hand over cannot be retried here, so it simply drops out;
  // a folder that could not be listed leaves nothing to show, which the page reports as empty.
  return out;
}

// A folder chosen through the browser's own "let this page view these files" picker, where the
// browser has one. It hands over read-only handles; nothing here can write, and nothing leaves.
export async function fromDirectoryHandle(handle) {
  const out = [];
  const walkHandle = async (dir, prefix) => {
    for await (const [name, child] of dir.entries()) {
      const path = `${prefix}/${name}`;
      if (child.kind === 'directory') {
        await walkHandle(child, path);
      } else {
        try {
          out.push({ path, file: await child.getFile() });
        } catch {
          // A file the browser will not hand over simply drops out, as in a drag and drop.
        }
      }
    }
  };
  await walkHandle(handle, handle.name || 'scan');
  return out;
}

export async function fromFileList(fileList) {
  return Array.from(fileList || []).map((file) => ({
    path: file.webkitRelativePath || file.name,
    file,
  }));
}

function errorText(err) {
  const msg = err && err.message ? err.message : String(err);
  return msg.replace(/\.$/, '');
}

// Only the front of each file is read when it is opened: a scan file keeps everything that says
// what it is in its first few kilobytes, and the pixels after that are read when the picture is
// shown. A file whose header runs past this is read whole by the viewer.
export const HEAD_BYTES = 64 * 1024;

// Reading one file at a time leaves the disk idle between requests; a few in flight at once keep it
// busy. Measured on a 500-file folder: 1226 ms one at a time, 281 ms eight at a time.
const READ_POOL = 8;

// Starts every job now, at most `width` at a time, and returns one promise per job in the jobs'
// own order. A promise never rejects: it settles to { value } or { err }, so the caller decides.
function startPool(jobs, width) {
  const out = jobs.map(() => Promise.withResolvers());
  let next = 0;
  const lane = async () => {
    for (;;) {
      while (next < jobs.length && !jobs[next]) next++;
      if (next >= jobs.length) return;
      const i = next++;
      try {
        out[i].resolve({ value: await jobs[i]() });
      } catch (err) {
        out[i].resolve({ err });
      }
    }
  };
  for (let k = 0; k < width; k++) lane();
  return out.map((o) => o.promise);
}

async function readHead(file) {
  return new Uint8Array(await file.slice(0, HEAD_BYTES).arrayBuffer());
}

// Each loader is made in a function of its own so that it holds the File, not the head bytes.
function readWhole(file) {
  return async () => new Uint8Array(await file.arrayBuffer());
}

function keep(bytes) {
  return async () => bytes;
}

// What every file found becomes: its path, its declared size, its first bytes, whether those are
// all of it, and a way to read all of it when its picture is wanted.
function plainFile(path, file, head) {
  const whole = head.byteLength >= file.size;
  return { path, size: file.size, head, whole, load: whole ? keep(head) : readWhole(file) };
}

// Checking the sizes first means a too-large drop fails before any memory is spent on it.
function checkDeclaredSize(list) {
  let declared = 0;
  for (const { file } of list) declared += file && typeof file.size === 'number' ? file.size : 0;
  if (declared > MAX_TOTAL_BYTES) throw limitError();
}

// Counts a kept file's bytes against the total, and stops the drop when it is past the limit.
function addBytes(out, size) {
  out.bytes += size;
  if (out.bytes > MAX_TOTAL_BYTES) throw limitError();
}

// The files inside one ZIP: each is kept under the archive's path, or skipped with the reason.
function keepZipFiles(path, result, out) {
  for (const s of result.skipped) out.skipped.push({ path: `${path}/${s.path}`, why: s.why });
  for (const f of result.files) {
    const inner = `${path}/${f.path}`;
    const innerNotScan = notScanReason(inner);
    if (isZip(inner)) {
      out.skipped.push({ path: inner, why: 'is a ZIP inside the ZIP. Unzip the outer one, then drop this one here.' });
    } else if (innerNotScan) {
      out.skipped.push({ path: inner, why: innerNotScan });
    } else {
      addBytes(out, f.size);
      out.files.push({ ...f, path: inner });
    }
  }
}

// Unpacks one ZIP. The File itself goes to unzip so a large archive is read in slices, not all at once.
async function expandZip(path, file, i, total, onProgress, out) {
  if (!zipSupported()) {
    out.skipped.push({ path, why: 'wouldn\'t open, because this browser can\'t unpack ZIP files. Unzip it first, then drop the folder here.' });
    return;
  }
  let result = null;
  try {
    // The fourth argument counts the files inside the ZIP, so a caller can say "12 of 40" rather than a fraction.
    result = await unzip(file, (done, n, inner) => {
      if (typeof onProgress === 'function') onProgress(i + done / Math.max(n, 1), total, `${path}/${inner}`, { unpacked: done, of: n });
    }, { headBytes: HEAD_BYTES });
  } catch (err) {
    out.skipped.push({ path, why: `wouldn't open as a ZIP file (${errorText(err)}).` });
  }
  if (result) keepZipFiles(path, result, out);
}

export async function expand(entries, onProgress) {
  const list = Array.from(entries || []);
  const out = { files: [], skipped: [], bytes: 0 };
  checkDeclaredSize(list);

  const report = (done, path) => {
    if (typeof onProgress === 'function') onProgress(done, list.length, path);
  };

  // Plain files are read ahead, READ_POOL at a time, while the loop below takes them in order, so
  // the files and the reasons come out in the same order as one at a time.
  const heads = startPool(list.map(({ path, file }) => (
    notScanReason(path) || isZip(path) ? null : () => readHead(file)
  )), READ_POOL);

  for (let i = 0; i < list.length; i++) {
    const { path, file } = list[i];
    const notScan = notScanReason(path);
    if (notScan) {
      out.skipped.push({ path, why: notScan });
    } else if (isZip(path)) {
      await expandZip(path, file, i, list.length, onProgress, out);
    } else {
      const got = await heads[i];
      if (got.err) {
        out.skipped.push({ path, why: `wouldn't open (${errorText(got.err)}).` });
      } else {
        addBytes(out, file.size);
        out.files.push(plainFile(path, file, got.value));
      }
    }
    report(i + 1, path);
  }
  return out;
}

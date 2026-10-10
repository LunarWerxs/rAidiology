// inflate-worker.js: unpacks one ZIP entry beside the page, so a big archive does not hold up the
// thread that draws the pictures. It is this site's own file, loaded from the same place as the
// page. It is handed the visitor's File and an entry's position in it, and hands back the bytes; it
// has no other way in or out, and nothing in it can reach a network.

import { extractEntry } from './zip.js';

self.onmessage = async (ev) => {
  const { id, blob, entry, limit } = ev.data;
  try {
    let bytes = await extractEntry(blob, entry, limit);
    // Only a whole buffer can be handed over without copying it on the way.
    if (bytes.byteOffset !== 0 || bytes.byteLength !== bytes.buffer.byteLength) bytes = bytes.slice();
    self.postMessage({ id, bytes }, [bytes.buffer]);
  } catch (err) {
    self.postMessage({ id, error: err && err.message ? err.message : String(err) });
  }
};

// Says it loaded, which is how the page knows this browser can run it at all.
self.postMessage({ ready: true });

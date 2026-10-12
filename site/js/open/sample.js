// The sample scan: the one thing on this page that comes over the network. It is a GET of a file
// this site serves itself, it sends nothing, and it only runs when somebody asks for it. It is kept
// out of viewer.js so that file can go on saying it never fetches: a visitor's own scan is read off
// their disk by code that has no way to reach a network at all.
//
// It exists because every visitor in the first test arrived without their disc to hand and left
// without having seen what the page does.

import { restoreFromAddress, openFiles, afterNextDraw, nameBiggestPart } from './viewer.js';

const SAMPLE = 'sample/sample-ct.zip';
// The sample's own reading, written by the MCP server's read_ct (tools/build_sample_reading.py): what
// a person's AI leaves them, every structure outlined. Round four's visitors said seeing the pictures
// does not explain them; this is the part of rAidiology that does, shown on a scan that is nobody's.
const SAMPLE_READING = 'sample/sample-ct.raidiology.json.gz';
// `source` is the attribution the licence asks for, changes included. Three visitors read it as
// something they were supposed to understand, so the viewer draws it smaller, on its own line.
const CREDIT = {
  // The viewer's heading for it (round eleven read "CT · Sample scan (not yours)" as a label maker).
  title: 'Sample CT of the chest and belly',
  text:'This is a sample, not your scan. It\'s a real CT of an adult\'s chest and belly, shared for '
    + 'research with nothing in it that identifies anyone. A CT is a stack of thin slices through the '
    + 'body, and you can step through them one by one. Your own CT, MRI or X-ray opens the same way.',
  short: 'A sample CT from TotalSegmentator (CC BY 4.0), not your scan.',
  source: 'Where the sample comes from, for the record: the TotalSegmentator dataset by Wasserthal and '
    + 'others, 2023, under licence CC BY 4.0. We left out every second slice.',
};

// The reading is fetched at the same moment as the pictures, not after them. It is 48 KB beside a
// 5 MB zip, so it is ready long before the pictures are unpacked, and the first picture on screen
// already has its outlines. Fetched afterwards (as it was) it started only once the zip had been
// downloaded, unzipped and opened, then had to be unpacked and matched slice by slice, and the
// outlines drew seconds after a plain grey picture. Null when it cannot be had: the pictures still open.
function fetchReading() {
  return fetch(SAMPLE_READING)
    .then((res) => (res.ok ? res.blob() : null))
    .then((blob) => (blob ? new File([blob], 'sample-ct.raidiology.json.gz') : null))
    .catch(() => null);
}

// The bar under "Opening the sample CT": the share of the download that has arrived, when the server says how
// big the file is. Without that it keeps its sliding stripe, which claims no amount.
function showDownloaded(fraction) {
  const track = document.getElementById('sample-track');
  const pct = Math.round(Math.min(1, Math.max(0, fraction)) * 100);
  track.classList.add('known');
  track.setAttribute('aria-valuenow', String(pct));
  document.getElementById('sample-fill').style.inlineSize = `${pct}%`;
}

async function download(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`the sample answered ${res.status}`);
  const size = Number(res.headers.get('content-length')) || 0;
  if (!size || !res.body || typeof res.body.getReader !== 'function') return res.blob();
  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.byteLength;
    showDownloaded(got / size);
  }
  return new Blob(parts);
}

// The tab's title before the sample was asked for, to put back if it cannot load.
const PAGE_TITLE = document.title;

async function loadSample() {
  const root = document.documentElement;
  root.classList.add('sample-started');
  // The tab names the sample from the first moment, as the heading does (wave three: it said "Open your own scan").
  document.title = `rAidiology · ${CREDIT.title}`;
  const button = document.getElementById('pick-sample');
  const label = button.textContent;
  button.disabled = true;
  button.textContent = 'Loading the sample CT…';
  // The sample button on this page takes the same viewer-shaped wait as the link from the home page (round
  // eleven: pressed here, it left the visitor on "Open your own scan" with a greyed button).
  if (!root.classList.contains('is-sample') || root.classList.contains('sample-done')) {
    root.classList.remove('sample-done');
    root.classList.add('is-sample');
    window.scrollTo(0, 0);
  }
  const track = document.getElementById('sample-track');
  track.classList.remove('known');
  track.removeAttribute('aria-valuenow');
  document.getElementById('sample-fill').style.inlineSize = '';
  try {
    const reading = fetchReading();
    const blob = await download(SAMPLE);
    const outlined = await openFiles([{ path: 'sample-ct.zip', file: new File([blob], 'sample-ct.zip') }], CREDIT, reading);
    document.getElementById('sample-next').hidden = false;
    if (outlined) showSampleReading();
    else restoreFromAddress();
    button.textContent = label;
  } catch {
    button.textContent = 'The sample didn\'t load. Try again';
    // Back to the ordinary page, where that message and the button are.
    document.documentElement.classList.remove('is-sample');
    document.title = PAGE_TITLE;
  }
  document.documentElement.classList.add('sample-done');
  button.disabled = false;
}

function showSampleReading() {
  // Round six, three visitors of five: "it's called rAidiology... where is the AI?"
  document.getElementById('sample-reading-about').hidden = false;
  document.getElementById('outline-caption').hidden = false;
  // One part named at once, so the outlines explain themselves before anyone taps. Once the picture
  // with the outlines on it has been drawn, because the name needs to know where they are.
  // Back from another page puts the slice and the chosen structure from the address back instead.
  afterNextDraw().then(async () => { if (!(await restoreFromAddress())) nameBiggestPart(); });
}

function start() {
  document.getElementById('pick-sample').addEventListener('click', loadSample);
  if (new URLSearchParams(window.location.search).has('sample')) loadSample();
}

start();

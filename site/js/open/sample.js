// The sample scan: the one thing on this page that comes over the network. It is a GET of a file
// this site serves itself, it sends nothing, and it only runs when somebody asks for it. It is kept
// out of viewer.js so that file can go on saying it never fetches: a visitor's own scan is read off
// their disk by code that has no way to reach a network at all.
//
// It exists because every visitor in the first test arrived without their disc to hand and left
// without having seen what the page does.

import { openFiles, openReading } from './viewer.js';

const SAMPLE = 'sample/sample-ct.zip';
// The sample's own reading, written by the MCP server's read_ct (tools/build_sample_reading.py): what
// a person's AI leaves them, every structure outlined. Round four's visitors said seeing the pictures
// does not explain them; this is the part of rAidiology that does, shown on a scan that is nobody's.
const SAMPLE_READING = 'sample/sample-ct.raidiology.json.gz';
// `source` is the attribution the licence asks for, changes included. Three visitors read it as
// something they were supposed to understand, so the viewer draws it smaller, on its own line.
const CREDIT = {
  text: 'This is a sample, not your scan. It is a real CT of an adult\'s chest and abdomen, shared for '
    + 'research, with nothing in it that identifies anyone. An MRI or an X-ray from your own disc '
    + 'opens the same way.',
  short: 'Sample, not your scan. TotalSegmentator sample, CC BY 4.0.',
  source: 'Where the sample comes from, for the record: TotalSegmentator dataset, Wasserthal and '
    + 'others, 2023, licence CC BY 4.0; every second picture left out.',
};

async function loadSample() {
  const button = document.getElementById('pick-sample');
  const label = button.textContent;
  button.disabled = true;
  button.textContent = 'Fetching the sample…';
  try {
    const res = await fetch(SAMPLE);
    if (!res.ok) throw new Error(`the sample answered ${res.status}`);
    const blob = await res.blob();
    await openFiles([{ path: 'sample-ct.zip', file: new File([blob], 'sample-ct.zip') }], CREDIT);
    // Straight to the outlines (round six's board, and a visitor who found them "hidden behind a
    // separate link at the bottom"): grey slices with nothing on them only frightened people.
    document.getElementById('sample-next').hidden = false;
    await loadSampleReading();
    button.textContent = label;
  } catch {
    button.textContent = 'The sample could not be fetched. Try again';
  }
  button.disabled = false;
}

async function loadSampleReading() {
  const res = await fetch(SAMPLE_READING);
  if (!res.ok) return;                     // the pictures still open without their outlines
  await openReading(new File([await res.blob()], 'sample-ct.raidiology.json.gz'));
  // Round six, three visitors of five: "it's called rAidiology... where is the AI?"
  document.getElementById('sample-reading-about').hidden = false;
  document.getElementById('outline-caption').hidden = false;
}

function start() {
  document.getElementById('pick-sample').addEventListener('click', loadSample);
  if (new URLSearchParams(window.location.search).has('sample')) loadSample();
}

start();

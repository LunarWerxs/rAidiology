# rAidiology as an MCP server

The visitor's own AI opens their scan, on their own computer. No upload, no account, no API key of
ours, no graphics card to rent. They point Claude Desktop (or ChatGPT, or anything that speaks MCP)
at this server, say "have a look at the CD from the hospital", and it does.

```jsonc
// Claude Desktop: claude_desktop_config.json
{
  "mcpServers": {
    "raidiology": {
      "command": "<path to python>",
      "args": ["<path to this repository>/mcp/server.py"]
    }
  }
}
```

The Python it runs needs `pip install onnxruntime numpy pandas pillow pydicom`; reading an X-ray
needs no torch and no graphics card. Reading a CT also needs `pip install torch nibabel psutil
SimpleITK` (a processor-only torch is enough) and the CT weights in `Models/ct-segmenter-v1/` beside
the repository. Check it without a client:
`python mcp/server.py --selftest <a folder or ZIP of DICOM files>`.

## The rule this server is built around

**The AI explains. These tools grade.**

That is not a preference, it is what the scorecards say. On the held-out X-ray test the trained
detector scored 0.976 and Claude Opus 5.5 looking at the same pictures scored 0.690
(`modalities/xray/eval/SCORECARD.md`); on spine MRI a Claude reader did not beat even the
measurement rules. An assistant that eyeballs a radiograph and gives a verdict is a worse product
wearing better numbers.

So `read_xray` is where a yes or no comes from, and `render_slice` hands over a picture while saying
in its own answer that what the AI sees in it is description only. Every answer carries how often the
model was right, what it misses, who it is not for, and that rAidiology is not a medical device.

## What it can do today

| tool | what it does |
|---|---|
| `about` | what rAidiology is, what has been measured, how to use the rest |
| `find_scans` | the scan series in a folder or ZIP from a hospital CD: modality, description, how many images, matrix size. It never reads the name, patient number or date of birth |
| `read_xray` | the measured fracture detector on a radiograph, with its held-out accuracy and its limits |
| `render_slice` | writes one image as a PNG for the AI to look at and describe |
| `read_ct` | the measured CT segmenter: which organs, bones and large vessels it found, roughly how large each is, and each one's own Dice on the held-out test scans. Anatomy, not disease. About a minute a scan on a processor (45 to 53 s for a 91-slice scan here, in bf16 where the processor has it). With `findings`, it also saves a findings file the viewer draws as outlines over the person's own slices |
| `write_findings` | saves the reading as a small file the person drops into rAidiology's own viewer, which then shows them their images with what was found marked on them |

Spine MRI is not here yet: its segmentation step wants a graphics card, so it stays in the
server-side pipeline for now.

**CT runs in a child process** (`modalities/ct/eval/segmenter.py scan`, the code the scorecard
measured): the CT code has modules named like the X-ray ones, and torch stays out of the server.
A DICOM series becomes one volume through SimpleITK, slices ordered by where they are, not by
their image numbers. `python tools/check_ct_series.py --read` writes a development scan as a
DICOM series (shuffled, image numbers reversed), and requires it to come back as the same voxels in
the same place and to get the same answer as the NIfTI file the scorecard used; with the patient
geometry or the slice order broken on purpose, it fails.

**On a processor it gives the scored answer.** The scorecard was read on a graphics card in half
precision; version 1's fold 0 model reading 60 of its held-out scans on the processor in full
precision scored 0.7301 mean Dice against the card's 0.7301 on the same scans (difference +0.0000,
95% CI -0.0000 to +0.0000) and invented structures exactly as often (2026-10-02).

## What the person is left with

A chat message is a bad place to leave somebody's scan. `write_findings` saves a file of about a
kilobyte beside their scan; they open rAidiology's "Open your own scan" page (`site/open.html`,
wherever it is published), drop in the scan folder and then that file, and they are looking at their
own pictures with the images the model flagged marked on them and the shading showing where the
score came from. Set `RAIDIOLOGY_VIEWER` to have the server name a particular address when it tells
them where to go. It works offline, it uploads nothing, and
the file holds no name, patient number or date of birth: the series and each image are referred to
by a one-way fingerprint that the browser recomputes from the scan it already has. The format is
written down in [docs/FINDINGS_FILE.md](../docs/FINDINGS_FILE.md) so anybody can write one.

## What it costs the person

A computer, Python, ONNX Runtime (about 15 MB; no torch) and the five folds as one 45 MB graph.
Measured on 2026-10-02: **0.07 seconds per image on four CPU threads, no GPU**, for the full
five-model ensemble, and 1.0 second from a cold start to the first answer. That is the whole reason
this route works: the thing that was measured is small enough to run on a laptop. The graph is made
from the fold weights by `tools/export_detector_onnx.py`, carries their sha256, and gave the same
decision as the scored torch reading on all 400 development pictures it was checked on.

## How it talks

MCP over stdin and stdout, JSON-RPC 2.0, written out by hand in `server.py`. No library: a medical
tool is a bad place to take a dependency for 200 lines of protocol, and it keeps the install to this
repository. `--selftest` drives it the way a client would, so a change that breaks the protocol
fails loudly.

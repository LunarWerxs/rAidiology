# CT: labelling the structures in a scan

Started 2026-10-01, once X-ray shipped. Steps follow [../README.md](../README.md) ("How a new scan
type gets a trained model"); all six are done, and the segmenter shipped on 2026-10-02.

## The scored question, fixed before any model is trained

> **Where is each organ and bone in this scan?** One label per voxel, over the structures the
> TotalSegmentator dataset marks.

That is the job TotalSpineSeg does for spine MRI, and it is what the 3D view and every measurement
downstream need. It is scored per structure, by Dice overlap against the dataset's own expert masks,
never as one average over everything: an average hides a structure the model never finds.

| step | state |
|---|---|
| 1. The question above | fixed |
| 2. Licence-checked data, row in `LICENSING.md` before the download | done: TotalSegmentator dataset v2.0.1, CC BY 4.0, 1,228 CT scans, 23.6 GB |
| 3. `python fetch_data.py all` | done: 23.58 GB in 27 minutes, md5 verified, 1,228 scans unpacked |
| 4. Split locked, metric contract written, baselines scored | done: `eval/SCORECARD.md`, checked cell by cell against an independent recomputation |
| 5. Train on development folds only, from scratch | done: a 3D U-Net, five folds, no pretrained weights; out of fold on development 0.7255 mean Dice (0.7212 to 0.7295) |
| 6. Open the test once, scorecard with an independent recomputation | done 2026-10-02, **ships**: 0.7667 mean Dice on the 89 test scans against bone-window's 0.1776, and both recomputations agree on all 216 cells |

### What steps 5 and 6 settled

**It ships.** `eval/final_eval.py` opened the held-out test once (it wrote its latch to the run
ledger before reading a scan, and refuses a second run). On the 89 test scans, read by all five
fold models with their probabilities averaged, the segmenter scores **0.7667 mean Dice (0.7517 to
0.7811)** against bone-window's 0.1776: a difference of **+0.5891, 95% CI +0.5728 to +0.6055**. The
rule, written before anything was trained, was that interval above zero. Every stratum (abdomen,
thorax, head or neck, scanner maker, institute, marked pathology) lands between 0.757 and 0.793.

**It seldom outlines what is not there:** 4.2% of the absent (scan, structure) pairs on test (3.2%
to 5.3%), where bone-window does it 86% of the time.

**Where it is weakest:** the two kidney cysts score 0 (they are on 4 and 3 test scans: small and
rare), then the neck vertebrae C5 and C6, the right carotid artery and the portal vein, between
0.51 and 0.58. The full list, weakest first, is in `eval/SCORECARD.md`.

**What it took:** five folds of 37 to 53 minutes on one RTX 4070 Ti at about half the card, because
the owner was using the machine (`core/polite.py`), and 20 minutes for the test evaluation.

### What step 4 settled

**The test set is the authors' own 89 scans**, kept rather than re-cut. Nobody here chose it after
seeing a result, and it is the split the published numbers for this dataset were produced on, so a
comparison means something. Their other 1,139 scans are development, in five folds of 228,
stratified by study type, institute and marked pathology.

**The unit is the scan, not the patient**, because this dataset publishes no patient identifier.
Two scans of one person could fall on both sides. That is the same limitation FracAtlas has on the
X-ray side and it goes on the scorecard rather than being hoped away.

**A method hands in three voxel counts** per scan and structure - how many the truth marks, how many
it marks, how many both mark - and nothing else. Everything scored is computable from those, so the
scorer never opens a scan and the independent recomputation never has to agree with it about how a
file was read. [eval/METRICS_SPEC.md](eval/METRICS_SPEC.md) is the contract.

**Two questions, kept apart.** Dice on structures the scan actually contains, and how often a method
outlines one that is not there. A scan holds about 69 of the 117 structures at the median, so the
absent ones are most of the data, and a metric that averages them together is reporting mostly on
silence.

### The number to beat

Four baselines, declared before anything was trained, scored on the 1,139 development scans
([eval/SCORECARD.md](eval/SCORECARD.md)):

| baseline | mean Dice (95% CI) | invents a structure that is not there |
|---|---:|---:|
| always-empty | 0.0000 | 0% |
| where-it-usually-is | 0.0859 (0.0821 to 0.0900) | 100% |
| nearest-scan | 0.1550 (0.1506 to 0.1594) | 11% |
| **bone-window** | **0.1847 (0.1798 to 0.1894)** | 87% |

**0.1847 is the number a trained segmenter has to beat**, with the confidence interval of the
difference above zero, before it goes anywhere near the app.

Two things in that table are worth carrying forward. The strongest baseline is a Hounsfield
threshold inside a learned region, which is classical engineering and not a straw man: on CT,
brightness is in fixed units that mean the same thing on every scanner, which is exactly what it
could not be on an X-ray. And it wins while outlining a structure that is not in the scan 87% of
the time, where single-atlas copying does that only 11% of the time. Dice never sees that, because
Dice is only defined where a structure is present. It is on the scorecard as its own column for
that reason, and a segmenter that beats 0.1847 while inventing less is better than the number alone
will say.

### How the segmenter is trained (step 5)

`eval/segmenter.py` is a 3D U-Net, trained from scratch on the development folds and nothing else.
Three decisions in it are worth knowing before touching it, and each came from a measurement.

**It reads the scan at 3 mm and is scored at 1.5 mm.** At 3 mm a 128-voxel cube spans 38 cm of body,
enough to tell the eleventh rib from the twelfth; the probabilities are interpolated back onto the
scan's own grid before a label is chosen, and the three counts are taken there, against the expert
masks as published. `eval/prepare.py` builds the cache that makes this fast: each scan's 118 files
are read once, and a training sample is a memory-mapped crop. Every one of its 133,263 truth counts
matches the baselines' rows. The best that 3 mm labels could ever score at 1.5 mm is 0.8837 mean
Dice (`prepare.py ceiling`), and the scoring path has its own positive control: handed the right
answer it scores above that ceiling, and handed the right answer one coarse voxel off it scores
0.72.

**Coarse first, then fine.** A plain 3 mm schedule had reached 0.07 mean Dice on held-out scans a
quarter of the way through. A 6 mm phase shows the same stretch of body in an eighth of the
arithmetic, so the network learns where everything is many times faster, and a 3 mm phase then
sharpens the edges. The schedule is a fixed number of steps: nothing is stopped early or chosen by
looking at held-out scans.

**A 1.5 mm phase exists, and lost (2026-10-07, idea 98 in docs/efficiency).** `prepare.py hi15` writes
every scan at the dataset's own 1.5 mm grid beside the 3 mm cache (`cache/hi15/`, 50 GB for the 1,139 development scans); a phase
`{"mm": 1.5, ...}` trains on 128-voxel cubes drawn from it; `--set init_from=V` starts a version from
version V's net of the same fold; a version whose last phase is 1.5 mm is read at 1.5 mm, each window
owning the voxels nearest its centre. Fine-tuned 8,000 steps from the 3 mm net on fold 0, it read
0.7910 against 0.8107 (-0.0196, -0.0242 to -0.0151), losing on organs, ribs and vertebrae alike, and
it drew an absent structure 40% of the time against 9%. A 128-voxel cube at 1.5 mm spans only 19 cm,
too little to tell which rib or vertebra it holds. The next try at fine detail is a cascade: the 3 mm
read given to the 1.5 mm net as an input.

**Two seeds read together.** `predict dev --version N --fold K --with-version M --tag T` reads a fold
with both versions' nets, their probabilities averaged, with version N's own reading. v114 and v115
(one recipe, seeds 0 and 1) together scored +0.0078 (+0.0069 to +0.0088) over v114 alone on fold 0 and
drew absent structures less often (0.086 to 0.055) (idea 97). Idea 102 trains v115 on the other four
folds and reads it with v116 on all five.

**The expert masks overlap**, in 0.15% of marked voxels, so one label per voxel cannot score a
perfect Dice on every structure at once. The truth counts are taken from each mask separately, as
the contract says; a flattened label map would have undercounted them.

    python eval/prepare.py dev                 the cache, once (about ten minutes)
    python eval/prepare.py check               every truth count against the baselines' rows
    python eval/segmenter.py train             five folds; a fold already trained is skipped
    python eval/segmenter.py predict dev       out-of-fold rows for the scorer
    python eval/score.py dev                   the development scorecard
    python eval/final_eval.py                  the held-out test, once
    python eval/model_card.py                  models/segmenter.md, from the files above

Training gives way to whoever is at the machine (`core/polite.py`: about half the graphics card
while there is somebody at the keyboard) and caps its own memory on the card
(`segmenter.limit_card_memory`), because a full card on Windows stalls rather than fails.

**Measured for speed on 2026-10-02** (`python core/ledger.py tried` has every number). Training is
bound by the card, not the data: crops waited 0 to 2% of the time. A 3 mm step spent 147 of its
321 ms converting memory layouts around the per-scan normalisation, so models from version 2 on are
laid out plainly, and the loss lets the softmax cast to fp32 itself instead of copying the logits
first (the same loss to the last bit). A 3 mm step went from 3.07 to 4.79 a second and a 6 mm step
from 15.5 to 18.1, so a fold's steps at the full card take about 18 minutes instead of 25. Version
1 is still read channels-last, as it was scored. Batching the 6 mm phase harder gains nothing: 31
to 34 cubes a second at two to eight a step. Reading used to move a long scan's running total into
ordinary memory past 1 GB; it now stays on the card whenever it fits beside the reading's measured
room, which every prepared scan does, and is summed in place: 2.7 times faster on the largest scan
with one model, a gigabyte less at the peak, and label for label the same.

### Version 2: more accurate, and it invents less (development, 2026-10-02)

Trained in `../_tools/venv-next`, all five folds: crops that pick the structure first across every
training scan, a sixth level in the network, compiled, about 16 minutes a fold on the whole card.
It reads with the one-piece cleanup and a size floor (a structure predicted smaller than 1% of the
real ones is dropped). On all 1,139 development scans, each read by the fold model that never saw it:
**0.7363 mean Dice against version 1's 0.7255, +0.0108 (95% CI +0.0099 to +0.0117), and it outlines
a structure the scan does not contain 8.0% of the time against 14.1%.** Without the floor it was
+0.0116 but invented more (15.1%), so the floor is what makes it better on both of the scorecard's
questions. The detail is in `docs/efficiency/README.md`.

It does not ship yet. Version 1's numbers come from the test set it opened once; how a second model
meets a test set is the owner's decision (docs/HANDOFF.md). Until then the MCP server reads with
version 1.

### Checked by a second implementation

`eval/recompute.py` was written from the metric contract by a worker told not to look at the
scorer. Its first run disagreed, and it was right: the stratum filter lower-cased the column but not
the value, so the institute stratum declared as `I` matched nothing, vanished from the scorecard,
and its complement quietly became every scan while printing a plausible number. Nothing crashed.
After fixing that and three smaller things, **the two implementations agree on all 144 cells.**

## How CT fits the rest of rAidiology

A CT is a 3D stack like an MRI, so most of the spine MRI path carries over: `core/dicom/` turns the
disc into NIfTI volumes, a segmentation model finds the structures (the job TotalSpineSeg does for
spine MRI), measurements and 3D meshes follow, and the viewer shows them. Two differences: CT
brightness is in fixed Hounsfield units (so bone and soft-tissue windows are standard), and the
segmenter must be one we may use commercially.

## Other questions worth training for later

1. **A segmenter for the 3D view**, trained on the TotalSegmentator dataset (below), so every CT gets
   its bones and organs labelled the way spine MRI gets its vertebrae and discs.
2. **Spine CT**: vertebra labelling and shape, trained on VerSe, reusing the spine reading and 3D
   view.
3. **Lung nodules**, scored against LIDC-IDRI, where four thoracic radiologists read every scan, so
   the scorecard can compare with more than one expert.

## Candidate data and models (licences read at the source on 2026-10-01)

Commercially usable:

| Dataset or model | What it is | Expert labels | Licence |
|---|---|---|---|
| [TotalSegmentator dataset v2](https://zenodo.org/records/10047292) | 1,228 clinical CT scans | masks of 117 anatomical structures | CC BY 4.0 |
| [VerSe](https://github.com/anjany/verse) ('19 + '20) | 374 spine CT scans, 355 patients | vertebra masks and centre points | CC BY-SA 4.0 |
| [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) | chest CT, 1,010 patients | lung nodule marks and ratings by 4 thoracic radiologists | CC BY 3.0 (plus TCIA's usage policy) |
| [AMOS](https://zenodo.org/records/7155725) | 500 CT + 100 MRI abdominal scans | masks of 15 organs | CC BY 4.0 |
| [MedSAM](https://github.com/bowang-lab/MedSAM) weights | general medical segmentation model | none (a model) | Apache-2.0 |
| [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) weights | CT segmentation models | none (models) | Apache-2.0 for the open tasks; some tasks need a separate licence. Check per task |

Not usable in a paid product: CTSpine1K and KiTS23 (CC BY-NC-SA), NVIDIA's NV-Segment-CTMR weights
(non-commercial licence); RSNA's cervical spine fracture and intracranial haemorrhage sets, CQ500,
DeepLesion, AbdomenCT-1K and the MONAI bundles (per-bundle licences) have not been confirmed.

Before any download: add the row to `LICENSING.md`.

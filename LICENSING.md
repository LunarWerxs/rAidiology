# Licensing

rAidiology's code is licensed under the **PolyForm Noncommercial License 1.0.0**, verbatim in
[LICENSE](LICENSE).

Required Notice: Copyright (c) 2026 LunarWerx (https://github.com/LunarWerxs)

It is source-available, not open source: it is free for noncommercial use, and commercial use
needs a separate paid licence from LunarWerx. **The LICENSE text is what binds.** Everything on
this page marked "LunarWerx reads this as" is LunarWerx's interpretation, not licence text.

## Free to use

The licence allows two kinds of free use.

- **Personal use.** In the licence's words: "Personal use for research, experiment, and testing for
  the benefit of public knowledge, personal study, private entertainment, hobby projects, amateur
  pursuits, or religious observance, without any anticipated commercial application".
  LunarWerx reads this as covering a person looking at their own scan to understand it.
- **Noncommercial organisations.** In the licence's words: "Use by any charitable organization,
  educational institution, public research organization, public safety or health organization,
  environmental protection organization, or government institution is use for a permitted purpose
  regardless of the source of funding or obligations resulting from the funding."
  LunarWerx reads this as letting such organisations use rAidiology's own code free. It does not
  cover third-party weights or data (the tables below), and rAidiology must not be used for
  diagnosis or treatment decisions in any setting.

## Needs a commercial licence

Any other use with an anticipated commercial application. LunarWerx reads this as including:

- clinicians, private clinics, private hospitals and imaging centres using it in paid work;
- insurers, claims adjusters, law firms and expert witnesses;
- running it to support an insurance or legal claim for money (ask first if that is you);
- companies building a product or service on it, including a hosted app, an API or an MCP server.

Commercial licences: contact LunarWerx (https://github.com/LunarWerxs). LunarWerx can license only
code it owns: outside contributions need a copyright assignment first.

## A hosted rAidiology

If rAidiology is offered as a hosted service (the web app, an API or an
MCP server), that service's own terms of service govern using it, including anything about what
users may do with its outputs. The LICENSE covers the code, not its outputs.

## Not a medical device

rAidiology is not cleared or approved by the FDA or any other regulator. It is for education and a
second look, not for diagnosis or treatment decisions, and a commercial licence does not change
that.

- **In the US:** software that analyses medical images for a medical purpose is generally
  regulated as a device: the exemption for clinical decision support software excludes functions that "acquire, process, or analyze a
  medical image" (FD&C Act section 520(o)(1)(E)).
- **Even for patients:** what counts as a device depends on the intended use and the claims made,
  so a patient-facing "second look" can still be regulated if it is marketed as diagnosis.
- **Elsewhere:** other markets have their own rules, for example the EU Medical Device Regulation.

Selling it for diagnosis needs regulatory clearance first.

## No patient data

This repository holds no patient data. Development used private data that is not part of
this repository and is not covered by the LICENSE. What you make from your own scan is yours.

## Third-party parts

"If distributed" means shipping copies of rAidiology to someone. For the code licences below (LGPL,
Apache, MIT, BSD) a hosted service is not distribution, except where a row says otherwise. CC BY 4.0
material (BodyParts3D, SPIDER) needs its credit whenever it is shown or shared, hosted or not.

| Part | Used for | Licence | What anyone shipping or hosting it must do |
|---|---|---|---|
| [three.js](https://threejs.org/) | 3D viewer | MIT | Keep its copyright notice. |
| [TotalSpineSeg](https://github.com/neuropoly/totalspineseg) code | Labels vertebrae, discs, cord and canal | LGPL-3.0 | If distributed: ship the LGPL-3.0 and GPL-3.0 texts, give a prominent notice, and keep it replaceable (LGPL section 4). |
| TotalSpineSeg trained weights | Same | Released with the code | **Unsettled.** Mainly trained on whole-spine (OpenNeuro ds005616), SPIDER (CC BY 4.0) and Spine Generic. Its sacrum labels came from [SynthRAD2023](https://zenodo.org/records/7260705) (CC BY-NC 4.0), MRSpineSeg and GoldAtlas (terms not confirmed), and TotalSegmentator. Get NeuroPoly's written OK before selling, or retrain on commercial-OK data. |
| [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) | Runs TotalSpineSeg | Apache-2.0 | If distributed: include the Apache-2.0 text and any NOTICE, and mark modified files. |
| [SpineReport](https://github.com/ivadomed/SpineReport) | Optional independent measurements | LGPL-3.0 | Same as TotalSpineSeg code. |
| [BodyParts3D](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/) (DBCLS) | Reference skeleton (`site/data/models/`) | CC BY 4.0 (licence updated 2025-02-25) | Show the credit "BodyParts3D, © The Database Center for Life Science, licensed under CC Attribution 4.0 International (https://creativecommons.org/licenses/by/4.0/); modified (parts merged and simplified) by LunarWerx" (the app shows it in the model menu, with the licence link). |
| pydicom, nibabel, trimesh (MIT); SimpleITK (Apache-2.0); scikit-image, scipy (BSD-3); Pillow (MIT-CMU) | Pipeline | Permissive | If distributed: keep their licence texts. |
| [anthropic](https://github.com/anthropics/anthropic-sdk-python) Python SDK | AI-panel baseline (`modalities/spine_mri/eval/panel_runner.py`) | MIT | If distributed: keep its licence text. |
| [Spinal Cord Toolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox) 7.3 (installed outside the repository by `modalities/spine_mri/sct/install_sct.ps1`, not shipped) | Cervical cord vs healthy adults (`modalities/spine_mri/sct/`) | LGPL-3.0; data as in the specialist table below | Same as TotalSpineSeg code if ever distributed. |

## Data and models for the specialist upgrade

The specialist upgrade (`modalities/spine_mri/eval/`) intends to use only commercial-OK sources. Open items are marked.

| Source | Licence | Checked | Used in | Status |
|---|---|---|---|---|
| [SPIDER](https://zenodo.org/records/10159290) lumbar MRI with per-disc gradings (`images.zip`, `masks.zip`, `overview.csv`, `radiological_gradings.csv`, 3.76 GB) | CC BY 4.0 (the Zenodo record's licence field and the paper's Data Records section) | 2026-09-30 | Phase 1 scorecard, Phase 2 training, lumbar percentiles | Use. Cite it: van der Graaf, van Hooff, Buckens et al., "Lumbar spine segmentation in MR images: a dataset and a public benchmark", Scientific Data 11, 264 (2024), doi:10.1038/s41597-024-03090-w; data doi:10.5281/zenodo.10159290, CC BY 4.0. One musculoskeletal radiologist graded every disc. Kept outside the repository, never committed. |
| TotalSpineSeg released weights, release r20260730 (Dataset101 step 1, Dataset102 step 2) | See the third-party table | 2026-09-30 | Every phase | **Permission pending:** research results only; nothing that depends on these weights goes into a paid product until NeuroPoly agrees in writing. Not yet requested. |
| [Spinal Cord Toolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox) code | LGPL-3.0 (repository LICENSE) | 2026-09-30 | Phase 3 cervical cord | Allowed, with the LGPL duties in the third-party table. |
| SCT healthy-cord database, [PAM50-normalized-metrics](https://github.com/spinalcordtoolbox/PAM50-normalized-metrics) r20250321 (what `-normalize-hc` compares against) | MIT (repository licence); the 203 healthy adults come from [spine-generic data-multi-subject](https://github.com/spine-generic/data-multi-subject), CC BY 4.0 | 2026-09-30 | Phase 3 cervical cord | Commercial-OK. Cite Valošek, Bédard et al., Imaging Neuroscience 2024. |
| [PAM50 template](https://github.com/spinalcordtoolbox/PAM50) r20250730 (SCT reads `PAM50/template/PAM50_levels.nii.gz` for `-normalize-hc`) | **None found:** no licence in the repository, its README or the release zip | 2026-09-30 | Phase 3 cervical cord (needed by `-normalize-hc`) | **Permission pending:** to be asked in the same NeuroPoly request. Without a licence nothing is granted, so the healthy-control comparison is not commercial-clean even though the database itself is MIT. |
| SCT cord model for `sct_deepseg` ([contrast-agnostic-softseg-spinalcord](https://github.com/sct-modalities/spine_mri/contrast-agnostic-softseg-spinalcord)) | MIT (repository licence) | 2026-09-30 | Phase 3 cervical cord | Commercial-OK. |
| [whole-spine](https://openneuro.org/datasets/ds005616) (OpenNeuro ds005616) | CC0 (its `dataset_description.json`) | 2026-09-30 | Plan B only (retraining TotalSpineSeg), not downloaded | Commercial-OK. Its first labels came from PAM50 registration (TotalSpineSeg README), so the PAM50 answer applies. |
| [Spine Generic](https://github.com/spine-generic) single- and multi-subject | CC BY 4.0 (repository licences) | 2026-09-30 | Plan B only, not downloaded | Commercial-OK with credit. Same PAM50 note. |
| [SynthRAD2023](https://zenodo.org/records/7260705) | CC BY-NC 4.0 (Zenodo record) | 2026-09-30 | Nowhere directly; it is why the TotalSpineSeg weights are pending | **Not used.** |
| [GoldAtlas](https://zenodo.org/records/583096) | None on the record; access is restricted (by request) | 2026-09-30 | Nowhere directly (TotalSpineSeg sacrum labels) | **Not used.** |
| [MRSpineSeg challenge](https://paperswithcode.com/dataset/mrspineseg-challenge) | Terms not published (registration-gated site) | 2026-09-30 | Nowhere directly (TotalSpineSeg sacrum labels) | **Not used.** |
| [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | Apache-2.0 (repository licence) | 2026-09-30 | Nowhere directly (NeuroPoly used its outputs for some sacrum labels) | Not used by rAidiology. |
| Claude Opus 5.5 (`claude-opus-5-5`) | Anthropic's terms | 2026-09-30 | Phase 1 AI-panel baseline, Phase 4 mistake-log re-runs | The panel runs on the Anthropic API (`--backend api`) or, with no key, as Claude Opus 5.5 sub-agents in Claude Code (`--backend agents`); the scored runs used sub-agents. Reads SPIDER images (CC BY 4.0) and, for the mistake log, private development data. |
| [MedGemma 1.5](https://developers.google.com/health-ai-developer-foundations/terms) | Health AI Developer Foundations terms | 2026-09-30 | Phase 3 extra reader | Allowed with conditions, listed after this table. **Anyone who runs MedGemma must accept these terms themselves** and get the weights from [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it). LunarWerx accepted them on 2026-09-30; its scored runs used a mirror whose safetensors files have the same SHA-256 as Google's release (`5e4c75b0ef1fb009caee567ab244f9e354e915fda748d1b76179cc453a39b4b5`, `958e39df78c35ddb812fbcb8b5e7f46e07f1b153c1589f2d3f0b41c3b0748d30`). |
| The disc grader rAidiology trains (Phase 2) | rAidiology's own (trained from scratch on SPIDER, no pretrained backbone) | 2026-09-30 | Phase 2 onward | Commercial-OK once trained on commercial-OK inputs. Its crops are centred with TotalSpineSeg disc labels, so it inherits **permission pending** until NeuroPoly answers (see its model card). |
| PyTorch (BSD-3-Clause), scikit-learn (BSD-3-Clause), pandas (BSD-3-Clause) | Permissive | 2026-09-30 | Phase 1 scoring, Phase 2 training | If distributed: keep their licence texts. |
| [RSNA LumbarDISC](https://arxiv.org/abs/2506.09162) (RSNA 2024 challenge data) | Non-commercial | 2026-09-30 | Nowhere | **Not used.** |
| [SpineNet](https://github.com/rwindsor1/SpineNet/blob/main/LICENCE.md) | Non-commercial, including its outputs | 2026-09-30 | Nowhere | **Not used.** |
| ImageNet-pretrained weights (any backbone) | ImageNet's terms are non-commercial | 2026-09-30 | Nowhere | **Not used:** the grader trains from scratch. |

Plan B (retrain TotalSpineSeg on commercial-OK data) starts only if NeuroPoly refuses. Two facts
for that day: its recipe asks for about 3.5 TB of disk, and its first labels for whole-spine and
Spine Generic came from PAM50 registration, so the PAM50 answer matters for it too.

MedGemma's conditions:

- A hosted service counts as distribution, and the section 3.2 use restrictions must go into
  rAidiology's own terms.
- rAidiology must ship Google's notice and mark any modified files.
- It may not be used in a way that makes Google a medical-device manufacturer.
- rAidiology must indemnify Google.
- Its [Prohibited Use Policy](https://developers.google.com/health-ai-developer-foundations/prohibited-use-policy)
  bans "making automated decisions in domains that affect material or individual rights or
  well-being (e.g., finance, legal, employment, healthcare, housing, insurance, and social welfare)".
  So MedGemma may never make or set a decision on its own in any rAidiology product. It can only be
  a labelled second opinion that a person reviews, and it must never feed an insurer's or legal
  user's decision.

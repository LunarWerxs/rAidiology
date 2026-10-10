# How a release is made

This page is for the maintainers of rAidiology. It lists the steps of a release, in order. Nothing is published until a maintainer has read the list of files in step c and said yes.

Models and code are released under the [PolyForm Noncommercial License 1.0.0](../LICENSE). The models are not a medical device.

A release has three parts:

- the code: a clean copy of the repository on GitHub, with no history,
- the models: one gated repository per model on Hugging Face, so a person accepts the licence before downloading,
- the paperwork: [CONTRIBUTING.md](../CONTRIBUTING.md) and [CLA.md](../CLA.md).

Two of the tools below (the provenance scripts in step b and the export script in step c) live in the maintainers' working copy. They are not part of this public repository.

## What is in this folder

| Path | What it is |
|---|---|
| `cards/` | One model card per model, as a Markdown file, for example `cards/disc-grader.md`. |
| `huggingface/README.template.md` | The front matter and gate form that every Hugging Face model repository gets. Its body is the placeholder `{{CARD}}`. |
| `huggingface/publish.py` | Builds a model repository from a card, the licence and the weight folders you list. It uploads only with `--push`. |

A card says, in plain words: what the model does, its scores (each with the test set it was measured on), the datasets it learned from and the credit each one asks for, and that it is not a medical device. A card has no front matter of its own, because the template supplies it. Links in a card must be full web addresses (or a `#` link to a place in the same card), because the card is shown on Hugging Face, outside this repository. `publish.py` refuses a card with any other link and names the line.

## Steps

### a. Pick the standing weights for each card

For each card, choose the weight files it describes. These must be the files whose scores the card quotes. For a model made of several fold networks, choose the version folder that holds them (for example `../Models/disc-grader-v133`). Write down the paths. Weights are never kept in this repository.

### b. Run the provenance records and commit them

Run this before every release, so that there is a dated proof of the exact bytes you are about to publish. These records are private. The export in step c leaves them out.

```
python records/provenance/manifest.py work
python records/provenance/manifest.py shipped
python records/provenance/stamp.py <each new manifest file>
```

- `manifest.py work` lists the experiment weights with their size, date and sha256. `manifest.py shipped` does the same for the models kept beside the repository. Each writes a new file under `records/models/`.
- `stamp.py` asks two outside timestamp services to sign each manifest's sha256, so the date does not rest on our own clock. Only the hash is sent, never the file. It writes the signed replies under `records/provenance/stamps/`.
- Commit the new manifests and stamps.
- Then check each weight file you picked in step a: its sha256 (the dry run in step f prints it) must appear in the `shipped` manifest if the file sits in the models folder beside the repository, or in the `work` manifest (which lists `.pt` files only) if it sits under `work/eval/`. If it does not, you picked the wrong file, or the manifest is out of date. Config files (`meta.json`, `grader.json`) are in neither manifest: compare each with the sha256 the card lists, or with the one the dry run prints.

### c. Build the public copy and read the file list

Run the export script, `tools/public_export.py`, in the maintainers' working copy, and give it a new folder to write to. (`python tools/public_export.py --help` lists what else it needs.)

- The new folder must not exist yet.
- The script copies the files that may go public, swaps a few passages for neutral text, and refuses to finish if it finds private material or an image or volume file that is not on its approved list.
- If it stops with any error, delete the new folder yourself before you try again. Only a run that ends with "public copy: N files ... none found" has been checked. Fix the cause. Do not weaken the check.

Then read the file list. List every file in the new folder (for example with `find <folder> -type f`). Look for:

- anything under `records/`, `ledger/`, `cases/`, `functions/`, `.github/`, `.claude/` or `data/efficiency/`, and anything under `docs/` other than `docs/FINDINGS_FILE.md` and `docs/COST_MODEL.md`: these must not be there,
- any weight file (`.pt`, `.pth`, `.onnx`, `.safetensors`, `.bin`), scan or image that is not an approved sample,
- anything that names a person, a private scan, or a private web address.

Send the file list to the maintainer who decides, and wait for a yes.

### d. Check the public repository's visibility

```
gh repo view LunarWerxs/rAidiology --json visibility,url
```

The public repository is public. Everything pushed to it is visible to everyone at once, and cannot be fully taken back, because people copy it. Never push the working copy, or its history, to the public repository. Only the folder from step c goes there.

### e. Push the public copy and create the GitHub release

Only after the yes from step c:

1. Clone the public repository into a new folder.
2. Delete everything in it except the `.git` folder, then copy in the folder from step c.
3. Run `git status` and `git diff --stat`. Read them. They should match the file list you approved.
4. Commit and push.
5. Create the release:

```
gh release create <tag> --repo LunarWerxs/rAidiology --title "<title>" --notes-file <notes.md>
```

The notes say what changed and which models changed, link to each model's page on Hugging Face, and say that the models are not a medical device. Do not attach weights to the GitHub release. The licence form only works on Hugging Face.

### f. Publish each model on Hugging Face

The models go to the LunarWerx organisation on Hugging Face. A maintainer creates the organisation once and logs in on their own computer with `hf auth login`. Never paste a token into a command, a file or a chat. `publish.py` never asks for one, reads one or prints one.

For each model, first run a dry run:

```
python release/huggingface/publish.py <card> --weights-dir <version folder> [<version folder> ...] --repo-id <org>/<name>
```

It builds the folder and prints each file by its path in the repository (for example `disc-grader-v133/fold0.pt`), with its size and sha256. It uploads nothing. Check the list: the card text, `LICENSE`, and only the weight files you meant. From each version folder it takes only the fold networks (`fold0.pt`, `fold1.pt` and so on, with an optional prefix such as `LA-`), `.onnx` and `.safetensors` files, and the config files `meta.json`, `grader.json` and `config.json`, which it takes only beside a weight file. It says how many other files it left behind, such as training logs, scores and scratch networks. A model whose files have other names needs `--include <pattern>`. It refuses a scan, an image or an archive, and a JSON file under any other name.

Each folder keeps its own name in the repository. If two folders have the same name (for example two folders called `model`), give one a name: `--weights-dir segmenter-v121=<path>`. The script stops if two files would get the same path.

When the list is right, run the same command with `--push`. This creates the repository as private if it is new, uploads the folder, and turns on gated access with automatic approval. Then open the model page in a browser where you are logged out. You should see the licence form, and you should not be able to download a file.

When that looks right, make the repository public by running the same command with `--push --public`, or by changing its visibility in the repository settings on Hugging Face.

### g. The contributor licence agreement check (set up once)

The check is a GitHub Action, CLA Assistant Lite. Its workflow lives here as `release/github/cla.yml`, and the export
in step c copies it into the public copy as `.github/workflows/cla.yml`. No outside service and no sign-in is needed.

- A pull request's author signs by replying with the sentence the bot gives. The bot keeps the signatures in
  `signatures/cla-v1.json` on the public repository's `cla-signatures` branch. That branch was created once, on
  2026-10-10, holding only a README. Never delete it, and never push the export to it: the export replaces `main`,
  and the signatures live apart so a release cannot wipe them.
- The maintainer account (`LunarWerxs`) and Dependabot are on the workflow's allowlist and are never asked to sign.
- If `CLA.md` changes in a way that needs new signatures, change `path-to-signatures` in `cla.yml` to a new file
  (`signatures/cla-v2.json`), so everyone signs the new text. Earlier signatures stay in the old file.
- Merge a pull request only when its CLA check is green.

## Before saying yes

- Every card says what the model does, shows its scores with their test sets, names its datasets and their credits, and says it is not a medical device.
- The manifests and stamps from step b are committed.
- The file list from step c has been read by a person.
- The public repository's visibility has been checked.
- Each Hugging Face repository shows the licence form to a logged-out visitor.

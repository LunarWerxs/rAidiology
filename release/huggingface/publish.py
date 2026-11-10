"""Build one model repository for Hugging Face, and (only with --push) upload it as a gated repository.

A card is a file in release/cards/ (for example release/cards/disc-grader.md). This script puts together a
folder with three kinds of file:

  README.md   release/huggingface/README.template.md with {{CARD}} replaced by the card text
  LICENSE     the PolyForm Noncommercial 1.0.0 text at the root of this repository
  weights     the version folders you list after --weights-dir. Each folder keeps its name, so
              ../Models/disc-grader-v133/fold0.pt goes in as disc-grader-v133/fold0.pt, the path the card's table uses

From a folder it takes only the fold networks (fold0.pt to fold9.pt, with an optional prefix such as LA-), .onnx and
.safetensors files, and a config file by its known name (meta.json, grader.json, config.json) that sits beside a weight
file. Training logs, scores, fingerprints and scratch networks stay behind, and the count is printed. A model with
other file names needs --include PATTERN. A name like segmenter-v121=PATH gives a folder another name in the
repository, for when two folders have the same name (for example two folders called model).

Run it once without --push. That is a dry run: it builds the folder in a temporary place, prints every file
with its size and sha256, and uploads nothing. Read the list. Then run it again with --push.

  python release/huggingface/publish.py disc-grader --weights-dir ../Models/disc-grader-v133 \\
      ../Models/disc-grader-v134 ../Models/disc-grader-v140 --repo-id YourOrg/rAidiology-disc-grader
  python release/huggingface/publish.py disc-grader --weights-dir ../Models/disc-grader-v133 \\
      ../Models/disc-grader-v134 ../Models/disc-grader-v140 --repo-id YourOrg/rAidiology-disc-grader --push

Links in a card must be full web addresses (http:// or https://) or a #anchor in the same card, because the card is
shown on Hugging Face, outside this repository. A card with any other link is refused, and the line is named.

Paid and free models get different gates, and there is no flag to change which. The paid models are the cards
Schedule A of COMMERCIAL-LICENSE.md lists as covered. They go up on request only: a manual gate, so nobody downloads
until review_requests.py answers (universities, schools, public research and health bodies and governments approved
by their email address; everyone else declined with the way to the commercial licence and its free evaluation). A
paid model's repository must be named rAidiology-<card>, the name the reviewer looks for. After pushing a paid model
this script answers the requests once and installs the reviewer's timer, so the answering needs no one. The free
models get the automatic gate: accept the licence form and you are in. The licence is the one list: move a model in
or out of Schedule A and this script follows. If the licence is missing or its Schedule A cannot be read, nothing is
built.

--push creates the repository as PRIVATE if it does not exist, uploads the folder, then turns on gated access
(automatic for a free model, on request for a paid one, as above). The repository stays private until you
add --public, which makes it public after the gate is on. A repository that already exists, is public and is not
gated is refused, so a file is never exposed without the form.

Login: this script never asks for, reads, prints or logs a token. huggingface_hub uses the login you made
yourself (run `hf auth login`, or set HF_TOKEN in your own shell). Standard library plus huggingface_hub; the
latter is imported only when --push runs."""
import argparse
import fnmatch
import hashlib
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(HERE)
ROOT = os.path.dirname(RELEASE)
CARDS = os.path.join(RELEASE, "cards")
TEMPLATE = os.path.join(HERE, "README.template.md")
LICENSE = os.path.join(ROOT, "LICENSE")
COMMERCIAL = os.path.join(ROOT, "COMMERCIAL-LICENSE.md")

PLACEHOLDER = "{{CARD}}"
GATE_PLACEHOLDER = "{{GATE_NOTE}}"
# A covered model's row in Schedule A names its card as `release/cards/<name>.md`.
PAID_ROW = re.compile(r"`release/cards/([A-Za-z0-9._-]+)\.md`")
# What the gate form says above its fields, by gate. No double quote: it sits inside a quoted YAML string.
GATE_NOTES = {
    "auto": "Access is granted automatically once you accept.",
    "manual": ("Access is on request. A request from a university, school, public research or health body, or "
               "government email address is approved automatically. Any other request is declined with the way in: "
               "a company gets the commercial licence, which starts with a free 60-day evaluation copy, and a charity "
               "or a person writes to contact@lunarwerx.com."),
}
# The file kinds models are saved as here (records/provenance/manifest.py lists the same).
WEIGHT_SUFFIXES = (".pt", ".pth", ".onnx", ".safetensors", ".bin", ".xml")
# What a weights folder gives up by default. Everything else in it (training logs, scores, fingerprints, scratch
# networks) stays behind, so a scan, an image, an archive or a person's data can never be picked up by accident.
FOLDER_PATTERNS = ("*fold[0-9].pt", "*.onnx", "*.safetensors")
# A JSON file goes out only under one of these names, and only beside a weight file in the same folder.
CONFIG_NAMES = ("meta.json", "grader.json", "config.json", "finder.json")
RESERVED = ("readme.md", "license")  # names the script writes itself; a weights folder cannot take them
# Markdown link targets that are not a full web address or an anchor in the same card are refused (see read_card).
LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*<?([^)\s>]*)")        # [text](target) and ![alt](target)
LINK_DEF = re.compile(r"^ {0,3}\[[^\]\n]+\]:[ \t]*<?(\S*)", re.M)  # [label]: target
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,}).*?^ {0,3}\1[`~]*[ \t]*$", re.M | re.S)
CODE_SPAN = re.compile(r"(`+)(?:(?!\1).)+?\1")                    # one line only: an odd backtick must not hide later links
HTML_LINK = re.compile(r"""(?:href|src)\s*=\s*["']([^"']*)""", re.I)  # <a href="..."> and <img src="...">
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
MACHINE_PATH = re.compile(r"[A-Za-z]:[\\/]|/Users/|/home/")
TOKENISH = re.compile(r"(hf_[A-Za-z0-9]{6,}|Bearer\s+\S+)")


def die(message, code=2):
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


def scrub(text):
    """Anything that looks like a Hugging Face token is hidden before an error is shown."""
    return TOKENISH.sub("***", str(text))


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def split_front_matter(text):
    """Return (front matter block including both --- lines, the rest), or (None, text) if there is none."""
    lines = text.lstrip("\ufeff").splitlines(keepends=True)
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return "".join(lines[: i + 1]), "".join(lines[i + 1:])
    return None, text


def blank_code(text):
    """The text with fenced code and inline code replaced by spaces, newlines kept: a link-shaped bit of code is not
    a link, and line numbers still match the card."""
    def blank(m):
        return re.sub(r"[^\n]", " ", m.group(0))
    return CODE_SPAN.sub(blank, FENCE.sub(blank, text))


def check_links(name, text):
    """A card is shown on Hugging Face, so a link into this repository (../../LICENSE) leads nowhere there. Refuse every
    link target that is not a full web address or an anchor in the same card, and name each line."""
    clean = blank_code(text)
    bad = []
    for pattern in (LINK, LINK_DEF, HTML_LINK):
        for m in pattern.finditer(clean):
            target = m.group(1)
            if not target.lower().startswith(("http://", "https://", "#")):
                bad.append((clean.count("\n", 0, m.start(1)) + 1, target))
    if bad:
        shown = "; ".join(f"line {n}: '{t}'" for n, t in sorted(bad)[:12])
        more = f" (and {len(bad) - 12} more)" if len(bad) > 12 else ""
        die(f"release/cards/{name}.md has {len(bad)} link(s) that are not a full web address (http:// or https://) "
            f"or a #anchor: {shown}{more}. The card is shown on Hugging Face, where a path in this repository leads nowhere")


def paid_cards():
    """The cards of the paid models: Schedule A's covered table in COMMERCIAL-LICENSE.md, lower-cased (Windows finds
    release/cards/Chest-Xray.md as chest-xray.md, so the name is compared without case). Fails closed: a licence that
    is missing, or whose Schedule A cannot be read or lists nothing, stops the build."""
    if not os.path.isfile(COMMERCIAL):
        die("no COMMERCIAL-LICENSE.md at the root of the repository; without it a paid model cannot be told from a "
            "free one, so nothing is built")
    with open(COMMERCIAL, encoding="utf-8") as f:
        text = f.read()
    start = text.find("## Schedule A")
    covered = text.find("Covered (code and weights):", start) if start >= 0 else -1
    end = text.find("Not covered:", covered) if covered >= 0 else -1
    if end < 0:
        die("COMMERCIAL-LICENSE.md has no Schedule A with a 'Covered (code and weights):' list followed by 'Not "
            "covered:'; without it a paid model cannot be told from a free one, so nothing is built")
    cards = {c.lower() for c in PAID_ROW.findall(text[covered:end])}
    if not cards:
        die("Schedule A of COMMERCIAL-LICENSE.md lists no covered card (rows naming `release/cards/<name>.md`); "
            "nothing is built")
    return cards


def gate_for(card):
    """'manual' (on request) for a paid model, 'auto' for a free one. Decided by Schedule A alone."""
    return "manual" if card.lower() in paid_cards() else "auto"


def check_repo_name(card, gate, repo_id):
    """A paid model's repository is named rAidiology-<card>, so review_requests.py finds it and answers its requests."""
    if gate == "manual" and repo_id and repo_id.split("/", 1)[1].lower() != f"raidiology-{card.lower()}":
        die(f"{card} is a paid model (Schedule A of COMMERCIAL-LICENSE.md), so its repository must be named "
            f"rAidiology-{card}; review_requests.py answers requests only there, and anywhere else they would wait")


def read_card(name):
    name = name[:-3] if name.endswith(".md") else name
    if not NAME.fullmatch(name):
        die(f"'{name}' is not a card name; give a file name from release/cards/ without a folder")
    path = os.path.join(CARDS, name + ".md")
    if not os.path.isfile(path):
        have = sorted(f[:-3] for f in os.listdir(CARDS) if f.endswith(".md")) if os.path.isdir(CARDS) else []
        die(f"no card release/cards/{name}.md; cards found: {', '.join(have) if have else 'none (the folder is empty or missing)'}")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    front, _ = split_front_matter(text)
    if front is not None:
        die(f"release/cards/{name}.md starts with its own front matter; remove it, the template supplies it")
    if "{{" in text:
        die(f"release/cards/{name}.md still holds a {{{{ placeholder")
    if "not a medical device" not in text.lower():
        die(f"release/cards/{name}.md does not say 'not a medical device'; every card must")
    check_links(name, text)
    return name, text.strip()


def build_readme(card_text, gate):
    with open(TEMPLATE, encoding="utf-8") as f:
        front, body = split_front_matter(f.read())
    if front is None:
        die("release/huggingface/README.template.md has no front matter")
    if body.count(PLACEHOLDER) != 1:
        die(f"the template body must hold {PLACEHOLDER} exactly once")
    if front.count(GATE_PLACEHOLDER) != 1:
        die(f"the template front matter must hold {GATE_PLACEHOLDER} exactly once")
    front = front.replace(GATE_PLACEHOLDER, GATE_NOTES[gate])
    return (front + body.replace(PLACEHOLDER, card_text)).replace("\r\n", "\n").rstrip("\n") + "\n"


def check_license():
    if not os.path.isfile(LICENSE):
        die("no LICENSE at the root of the repository")
    with open(LICENSE, encoding="utf-8") as f:
        head = f.read(4000)
    if "PolyForm Noncommercial License 1.0.0" not in head or "Required Notice:" not in head:
        die("LICENSE is not the PolyForm Noncommercial License 1.0.0 with its Required Notice line")


def collect_weights(specs, include, tensors_only=False):
    """Walk each weights folder (a --weights-dir value: DIR, or NAME=DIR) and pick its files. Each file goes into the
    repository as <folder name>/<path inside the folder>, so two versions that share file names (fold0.pt) never clash.
    Returns (picked, left_behind): picked is a sorted list of (path in the repository, source file) and left_behind is
    how many other files were passed over."""
    for pattern in include:
        if not pattern.lower().endswith(WEIGHT_SUFFIXES):
            die(f"--include {pattern}: only patterns ending in {', '.join(WEIGHT_SUFFIXES)} can be added this way")
    base = tuple(p for p in FOLDER_PATTERNS if not (tensors_only and p.endswith(".pt")))
    patterns = tuple(p.lower() for p in base + tuple(include))
    chosen, seen, left_behind = [], {}, 0
    for spec in specs:
        name, sep, folder = spec.partition("=")
        if not sep:
            folder = spec
            name = os.path.basename(os.path.abspath(folder))
        folder = os.path.abspath(folder)
        if not NAME.fullmatch(name) or name.lower() in RESERVED:
            die(f"'{name}' cannot be the name of a folder in the repository; write NAME=PATH to choose a name made of "
                "letters, digits, '.', '_' and '-'")
        if not os.path.isdir(folder):
            die(f"weights folder not found: {folder}")
        before = len(chosen)
        for top, dirs, files in os.walk(folder):
            dirs.sort()
            for f in sorted(files):
                low = f.lower()
                if not (low in CONFIG_NAMES or any(fnmatch.fnmatchcase(low, p) for p in patterns)):
                    left_behind += 1
                    continue
                src = os.path.join(top, f)
                if "partial" in low:
                    die(f"{src}: a partial checkpoint is not a release")
                rel = "/".join([name] + os.path.relpath(src, folder).split(os.sep))
                if rel.lower() in seen:
                    die(f"two files would both be {rel} ({seen[rel.lower()]} and {src}); give one folder another name "
                        "with NAME=PATH")
                if low.endswith(".json") and MACHINE_PATH.search(open(src, encoding="utf-8", errors="replace").read()):
                    die(f"{src}: holds a path on this machine; clean it before it goes into a release")
                seen[rel.lower()] = src
                chosen.append((rel, src))
        if len(chosen) == before:
            die(f"no weight files found under {folder}")
    # A config file goes out only beside a weight file in the same folder.
    weight_dirs = {rel.rsplit("/", 1)[0] for rel, _ in chosen if not rel.lower().endswith(".json")}
    for rel, _ in chosen:
        if rel.lower().endswith(".json") and rel.rsplit("/", 1)[0] not in weight_dirs:
            die(f"{rel}: a config file goes out only beside a weight file in the same folder")
    return sorted(chosen), left_behind


def stage(folder, readme, weights):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(readme)
    shutil.copyfile(LICENSE, os.path.join(folder, "LICENSE"))
    for rel, src in weights:
        dst = os.path.join(folder, *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            os.link(src, dst)  # no second copy of a big file; removing the staging folder leaves the original alone
        except OSError:
            shutil.copy2(src, dst)


def show(folder):
    rels = []
    for top, _, files in os.walk(folder):
        rels += [os.path.relpath(os.path.join(top, f), folder).replace(os.sep, "/") for f in files]
    print(f"{'file':<52} {'size':>10}  sha256")
    for rel in sorted(rels):
        path = os.path.join(folder, *rel.split("/"))
        print(f"{rel:<52} {human(os.path.getsize(path)):>10}  {sha256_of(path)}")


def check_metadata(folder):
    """Send the built README.md to Hugging Face's own metadata check (the one an upload runs) and stop on any error it
    names. It once refused a gate field whose name held a '.', after the repository had been created. Uses the login
    huggingface_hub finds; the token is sent only to huggingface.co and never shown."""
    import json
    import urllib.error
    import urllib.request
    from huggingface_hub import get_token
    with open(os.path.join(folder, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    headers = {"Content-Type": "application/json"}
    token = get_token()
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request("https://huggingface.co/api/validate-yaml", method="POST", headers=headers,
                                 data=json.dumps({"content": readme, "repoType": "model"}).encode("utf-8"))
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            answer = json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        answer = json.loads(exc.read() or b"{}")
    errors = [e.get("message", str(e)) for e in answer.get("errors", [])]
    if errors:
        die("Hugging Face refuses this README.md's metadata, so nothing was sent: " + "; ".join(errors), 1)
    print("Hugging Face's metadata check: passed")


def push(folder, repo_id, card, public, gate):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        die("--push needs huggingface_hub (pip install huggingface_hub)")
    api = HfApi()  # finds the login by itself; no token is read or shown here
    try:
        print(f"logged in as: {api.whoami().get('name', '(unknown)')}")
        check_metadata(folder)  # Hugging Face's own check, before anything is created or sent
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        if not info.private and not info.gated:
            die(f"{repo_id} exists, is public and is not gated; make it private (or gate it) on the Hub, then run again", 1)
        api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type="model", commit_message=f"Release {card}")
        print("uploaded the folder")
        api.update_repo_settings(repo_id=repo_id, gated=gate, repo_type="model")
        print("gated access is on (" + ("on request" if gate == "manual" else "automatic approval") + ")")
        if public:
            api.update_repo_settings(repo_id=repo_id, private=False, repo_type="model")
            print("the repository is now public")
        if gate == "manual":
            sys.path.insert(0, HERE)
            import review_requests  # the one place requests are answered
            review_requests.review(repo_id.split("/", 1)[0], act=True)
            review_requests.install_timer(30)
    except SystemExit:
        raise
    except Exception as exc:  # a network or permission error; show its kind and text, never anything that looks like a token
        die(f"{type(exc).__name__}: {scrub(exc)}", 1)
    state = "public" if public else "private (add --public when you are ready)"
    print(f"done: https://huggingface.co/{repo_id} is {state}, gated " + ("on request" if gate == "manual" else "automatically"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("card", help="a card name: a file in release/cards/, with or without .md")
    ap.add_argument("--weights-dir", nargs="*", default=[], metavar="[NAME=]DIR",
                    help="version folders to take weights from; each keeps its name in the repository (NAME=DIR picks another)")
    ap.add_argument("--include", action="append", default=[], metavar="PATTERN",
                    help="also take weight files matching this name pattern from the folders (can repeat), for example 'chest-*.pt'")
    ap.add_argument("--tensors-only", action="store_true",
                    help="leave the pickled fold networks (fold0.pt ...) behind; send only .onnx and .safetensors files and the config file")
    ap.add_argument("--repo-id", help="the Hugging Face repository, as ORG/NAME (needed for --push)")
    ap.add_argument("--push", action="store_true", help="upload; without it this is a dry run")
    ap.add_argument("--public", action="store_true", help="with --push: make the gated repository public afterwards")
    ap.add_argument("--out", help="build the folder here (must be new or empty) and keep it; default is a temporary folder")
    a = ap.parse_args()

    if a.public and not a.push:
        die("--public only makes sense with --push")
    if a.repo_id and not REPO_ID.fullmatch(a.repo_id):
        die("--repo-id must look like ORG/NAME")
    if a.push and not a.repo_id:
        die("--push needs --repo-id ORG/NAME")
    if a.push and not a.weights_dir:
        die("--push needs at least one folder after --weights-dir")
    if a.include and not a.weights_dir:
        die("--include only makes sense with --weights-dir")

    check_license()
    card, card_text = read_card(a.card)
    gate = gate_for(card)
    check_repo_name(card, gate, a.repo_id)
    readme = build_readme(card_text, gate)
    weights, left_behind = collect_weights(a.weights_dir, a.include, a.tensors_only)

    if a.out:
        out = os.path.abspath(a.out)
        if os.path.exists(out) and os.listdir(out):
            die(f"{out} is not empty")
        temp = None
    else:
        temp = tempfile.mkdtemp(prefix="raidiology-hf-")
        out = os.path.join(temp, card)
    try:
        stage(out, readme, weights)
        print(f"card: {card}    target: {a.repo_id or '(no --repo-id given)'}    folder: {out}")
        if not weights:
            print("note: no weights folder given; the folder holds only README.md and LICENSE")
        else:
            print(f"left behind: {left_behind} other file(s) in the weights folders (logs, scores, scratch networks, anything not named above)")
        show(out)
        if a.push:
            push(out, a.repo_id, card, a.public, gate)
        else:
            print("\nDry run: nothing was uploaded. Read the list above, then repeat the command with --push.")
    finally:
        if temp:
            shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    main()

"""How a model leaves: a model licensed to companies goes to Hugging Face only on request, answered by
review_requests.py; a noncommercial-only one behind the automatic gate; and the public export lets no weight file out."""
import importlib.util
import re
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF = ROOT / "release" / "huggingface"
PUBLISH = HF / "publish.py"
EXPORT = ROOT / "tools" / "public_export.py"
PAID = {"chest-xray", "fracture-detector", "ct-segmenter", "lung-nodules", "mammography", "spine-xray-corners",
        "disc-grader"}
# The owner's rule: every model trains and runs only on what he can sell, so every card is in Schedule A. A card may
# sit outside it only while it is being made sellable, named here with what blocks it and what clears it. Empty since
# 2026-11-10: the disc grader's own disc finder (SPIDER masks, CC BY 4.0) replaced TotalSpineSeg.
NOT_SELLABLE_YET = {}
FREE = set(NOT_SELLABLE_YET)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_publish(*args):
    return subprocess.run([sys.executable, str(PUBLISH), *args], capture_output=True, text=True, cwd=ROOT)


def test_paid_list_is_schedule_a():
    assert load(PUBLISH, "publish").paid_cards() == PAID


def test_every_card_is_sellable_or_named_with_its_fix():
    """A model the owner cannot sell is a defect, not a category (2026-11-09: two were found trained or run on parts
    he could not sell). A new card fails here until it is in Schedule A or named in NOT_SELLABLE_YET with its fix."""
    cards = {p.stem for p in (ROOT / "release" / "cards").glob("*.md")}
    assert cards == PAID | FREE, "a new card goes in Schedule A, or in NOT_SELLABLE_YET with what blocks it"
    assert not PAID & FREE
    for card, why in NOT_SELLABLE_YET.items():
        assert len(why) > 40 and "cleared by" in why, card
        text = (ROOT / "release" / "cards" / f"{card}.md").read_text(encoding="utf-8")
        assert "not covered by the commercial licence" in text, card
    for card in PAID:
        text = (ROOT / "release" / "cards" / f"{card}.md").read_text(encoding="utf-8")
        assert "not covered by the commercial licence" not in text and "noncommercial use only" not in text, card
        assert "COMMERCIAL-LICENSE.md" in text, card


def test_gate_by_schedule_a():
    mod = load(PUBLISH, "publish")
    for card in PAID:
        assert mod.gate_for(card) == mod.gate_for(card.upper()) == "manual", card
    for card in FREE:
        assert mod.gate_for(card) == "auto", card


def test_each_card_builds_with_its_own_gate_note(tmp_path):
    for card in sorted(PAID | FREE):
        out = tmp_path / card
        repo = f"LunarWerx/rAidiology-{card}"
        r = run_publish(card, "--repo-id", repo, "--out", str(out))
        assert r.returncode == 0, (card, r.stderr)
        readme = (out / "README.md").read_text(encoding="utf-8")
        front = readme.split("\n---\n", 1)[0]
        assert "{{" not in readme, card
        if card in PAID:
            assert "Access is on request." in front and "granted automatically" not in front, card
        else:
            assert "granted automatically" in front and "on request" not in front, card


def test_paid_repo_must_be_named_for_the_reviewer():
    r = run_publish("chest-xray", "--repo-id", "LunarWerx/chest")
    assert r.returncode == 2 and "must be named rAidiology-chest-xray" in r.stderr, r.stderr
    r = run_publish("disc-grader", "--repo-id", "LunarWerx/rAidiology-disc-grader")
    assert r.returncode == 0, r.stderr


def test_push_gets_the_cards_gate(tmp_path, monkeypatch):
    """The upload step itself receives the gate Schedule A gives the card (a missed argument once stopped a release)."""
    mod = load(PUBLISH, "publish")
    folder = tmp_path / "v9"
    folder.mkdir()
    (folder / "fold0.pt").write_bytes(b"not a real network")
    calls = []
    monkeypatch.setattr(mod, "push", lambda *args: calls.append(args))
    for card, gate in (("disc-grader", "manual"), ("chest-xray", "manual")):
        monkeypatch.setattr(sys, "argv", ["publish.py", card, "--weights-dir", str(folder),
                                          "--repo-id", f"LunarWerx/rAidiology-{card}", "--push"])
        mod.main()
        assert calls[-1][1:] == (f"LunarWerx/rAidiology-{card}", card, False, gate), calls[-1]


def test_gate_field_names_pass_hugging_faces_key_rule():
    """Hugging Face refuses a metadata key holding '.' or '$' (it refused 'License 1.0.0' in a field name)."""
    lines = (HF / "README.template.md").read_text(encoding="utf-8").splitlines()
    start = lines.index("extra_gated_fields:")
    names = []
    for line in lines[start + 1:]:
        if not line.startswith("  "):
            break
        m = re.match(r"^  ([^ -][^:]*):", line)
        if m:
            names.append(m.group(1))
    assert len(names) >= 8, names
    for name in names:
        assert "." not in name and "$" not in name, name


def test_visitor_text_never_splits_models_into_paid_and_free():
    """All seven models are under one licence (free for noncommercial use, companies pay); what differs is whether a
    model is also licensed to companies. 'Paid model' / 'free model' told the owner his models were given away
    (2026-11-09), so no text a visitor reads may say it."""
    rr = reviewer()
    pub = load(PUBLISH, "publish")
    texts = {"decline reason": rr.DECLINE_REASON, **{f"gate note {k}": v for k, v in pub.GATE_NOTES.items()}}
    for rel in ["README.md", "reading/local/README.md", "reading/local/ui/index.html", "reading/local/models.json",
                "site/data/stats.json", "site/pricing.html", *[f"release/cards/{c}.md" for c in PAID | FREE]]:
        texts[rel] = (ROOT / rel).read_text(encoding="utf-8")
    bad = re.compile(r"\b(paid|free) (model|ones)\b|\bfive paid\b|\btwo free\b", re.I)
    found = {k: bad.findall(v) for k, v in texts.items() if bad.search(v)}
    assert not found, found


def test_paid_list_fails_closed(tmp_path, monkeypatch):
    mod = load(PUBLISH, "publish")
    broken = tmp_path / "COMMERCIAL-LICENSE.md"
    broken.write_text("# a licence with no Schedule A\n", encoding="utf-8")
    monkeypatch.setattr(mod, "COMMERCIAL", str(broken))
    try:
        mod.paid_cards()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("a licence without Schedule A must stop the build")


def test_export_holds_every_weight_file():
    src = EXPORT.read_text(encoding="utf-8")
    m = re.search(r"^MEDIA\s*=\s*(re\.compile\(.*?\))\s*$", src, re.M | re.S)
    media = eval(m.group(1), {"re": re})
    for name in ("fold0.pt", "m.pth", "m.onnx", "m.safetensors", "openvino.bin", "M.ONNX"):
        assert media.search("models/" + name), name
    assert not media.search("tools/public_export.py")


def reviewer():
    return load(HF / "review_requests.py", "review_requests")


ACCEPT = ["a@mit.edu", "x@cs.ox.ac.uk", "x@tum.de", "x@ethz.ch", "x@u-tokyo.ac.jp", "x@nih.gov", "x@nhs.net",
          "x@trust.nhs.uk", "x@ons.gov.uk", "x@hc-sc.gc.ca", "x@ec.europa.eu", "x@who.int", "x@unam.mx",
          "x@usp.br", "x@sydney.edu.au", "X@MIT.EDU"]
REJECT = ["x@gmail.com", "x@acme-ai.com", "x@fakeedu.com", "x@edu.example.com", "x@evilgov.uk", "x@mit.edu.evil.com",
          "x@clinic.org", "", None, "not-an-address", "a@b@mit.edu", "x@-bad-.edu"]


def test_reviewer_approves_only_by_address():
    rr = reviewer()
    unis = rr.read_domains(rr.UNIVERSITIES)
    assert len(unis) > 10000
    for email in ACCEPT:
        assert rr.decide(email, unis, set())[0] == "accept", email
    for email in REJECT:
        verdict, why = rr.decide(email, unis, set())
        assert verdict == "reject" and why == rr.DECLINE_REASON, email


def test_hand_approved_domain_and_its_subdomains():
    rr = reviewer()
    assert rr.decide("x@clinic.org", set(), {"clinic.org"})[0] == "accept"
    assert rr.decide("x@radiology.clinic.org", set(), {"clinic.org"})[0] == "accept"
    assert rr.decide("x@notclinic.org", set(), {"clinic.org"})[0] == "reject"


class Req:
    def __init__(self, username, email):
        self.username, self.email = username, email


class FakeApi:
    def __init__(self, gated):
        self.gated, self.answers = gated, []

    def list_models(self, author):
        return [types.SimpleNamespace(id=f"{author}/{n}") for n in self.gated]

    def model_info(self, repo_id):
        return types.SimpleNamespace(gated=self.gated[repo_id.split("/", 1)[1]])

    def list_pending_access_requests(self, repo):
        return [Req("student", "s@mit.edu"), Req("company", "c@acme-ai.com")]

    def list_rejected_access_requests(self, repo):
        return [Req("charity", "c@clinic.org"), Req("still-no", "n@gmail.com")]

    def accept_access_request(self, repo, user):
        self.answers.append((repo, user, "accept", None))

    def reject_access_request(self, repo, user, rejection_reason=None):
        self.answers.append((repo, user, "reject", rejection_reason))


def test_review_answers_paid_manual_repos_only(tmp_path, monkeypatch):
    rr = reviewer()
    api = FakeApi({"rAidiology-chest-xray": "manual", "rAidiology-disc-grader": "auto",
                   "rAidiology-mammography": "auto", "other-model": "manual"})
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=lambda: api))
    monkeypatch.setattr(rr, "LOG", str(tmp_path / "log.txt"))
    approved = tmp_path / "approved.txt"
    approved.write_text("# by hand\nclinic.org\n", encoding="utf-8")
    monkeypatch.setattr(rr, "APPROVED", str(approved))

    assert rr.review("LunarWerx", act=False) == 0 and api.answers == []  # a dry run answers nobody
    rr.review("LunarWerx", act=True)
    repo = "LunarWerx/rAidiology-chest-xray"
    assert api.answers == [(repo, "student", "accept", None),
                           (repo, "company", "reject", rr.DECLINE_REASON),
                           (repo, "charity", "accept", None)]  # declined before, approved once its domain was added
    assert len((tmp_path / "log.txt").read_text(encoding="utf-8").splitlines()) == 3

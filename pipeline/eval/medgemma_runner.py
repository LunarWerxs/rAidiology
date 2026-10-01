"""MedGemma 1.5 (google/medgemma-1.5-4b-it) as an extra reader, and its combination with the grader.

Anyone who runs it must first accept Google's Health AI Developer Foundations terms
(https://developers.google.com/health-ai-developer-foundations/terms) and get the weights from
google/medgemma-1.5-4b-it (about 8.6 GB) into ../Models/medgemma-1.5-4b-it. (For the scored runs,
LunarWerx accepted those terms on 2026-09-30 and used a mirror whose two safetensors files have the
same SHA-256 as Google's release; LICENSING.md lists the hashes.) Its Prohibited Use Policy bans automated healthcare decisions, so in
the app MedGemma may only ever be a labelled second opinion a person reviews: it never sets a
level's status, the overall status or a grade on its own.

Per disc it sees the same crop the grader uses (crops.py), rendered as images: the middle three T2
planes and the middle T1 plane. Scores are token probabilities, not free text: for each binary
item P("Yes") against P("No") after a yes/no question, and for Pfirrmann the distribution over the
tokens "1".."5". Temperatures (one per output) are cross-fitted on development folds exactly like
the grader's (fold k gets values fitted on the other four).
The combination rule, fixed before any MedGemma number exists: the mean of the grader's and
MedGemma's calibrated probabilities (per binary item; for Pfirrmann the mean class distribution,
then the median grade). Thresholds: Youden's J, cross-fitted on development folds.
MedGemma and the combination enter the final evaluation only if the combination's development
Pfirrmann kappa beats the grader's (`qualify`).
Usage: python medgemma_runner.py <dev|test> | qualify"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from prepare import WORK  # noqa: E402
from spider import ITEMS  # noqa: E402

MODEL_ID = os.path.normpath(os.path.join(HERE, "..", "..", "..", "Models", "medgemma-1.5-4b-it"))
LIBS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "_tools", "medgemma_libs"))  # transformers, kept out of the shared venv
EVAL = os.path.dirname(WORK)
QUESTIONS = {
    "herniation": "Is there a disc herniation (focal displacement of disc material beyond the disc space) at this disc?",
    "narrowing": "Is this disc space narrowed compared with the discs above it?",
    "bulging": "Is there a broad, symmetric disc bulge beyond the vertebral body margins at this disc?",
    "spondylolisthesis": "Has the vertebra above this disc slipped forward or backward on the one below?",
    "up_endplate": "Is there an endplate change or Schmorl node in the endplate above this disc?",
    "low_endplate": "Is there an endplate change or Schmorl node in the endplate below this disc?",
    "modic": "Is there a Modic change of any type in the bone marrow next to this disc?",
}
CONTEXT = ("These are sagittal lumbar spine MRI crops centred on one intervertebral disc, front of the patient on "
           "the left, head at the top. Images 1-3: T2-weighted, three neighbouring sagittal planes (middle is image 2). "
           "Image 4: T1-weighted, middle plane (blank if there is no T1).")


def to_images(x):
    from PIL import Image
    out = []
    for seq, plane in ((0, 1), (0, 2), (0, 3), (1, 2)):
        a = np.clip(x[seq, plane].astype(np.float32), 0, 1.2) / 1.2
        im = Image.fromarray((a * 255).astype(np.uint8)).resize((336, 240), Image.BICUBIC).convert("RGB")
        out.append(im)
    return out


class Reader:
    def __init__(self):
        import torch
        sys.path.insert(0, LIBS)
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
        vocab = self.proc.tokenizer
        self.yes, self.no = vocab.convert_tokens_to_ids("Yes"), vocab.convert_tokens_to_ids("No")
        self.digits = [vocab.convert_tokens_to_ids(str(k)) for k in range(1, 6)]

    def next_logits(self, images, question):
        msgs = [{"role": "user", "content": [*({"type": "image", "image": im} for im in images),
                                             {"type": "text", "text": CONTEXT + " " + question}]}]
        inp = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                            return_tensors="pt").to(self.model.device, dtype=self.torch.bfloat16)
        with self.torch.no_grad():
            return self.model(**inp).logits[0, -1].float().cpu().numpy()

    def read_full(self, x):
        """One full pass per question (the reference `read` is checked against)."""
        ims = to_images(x)
        return self.answers([self.next_logits(ims, q) for q in ASKED])

    def answers(self, z):
        out = {"pfirrmann_logits": z[0][self.digits].tolist()}
        for b, zb in zip(QUESTIONS, z[1:]):
            out[b] = float(zb[self.yes] - zb[self.no])
        return out

    def read(self, x):
        """The images and context are the same for every question, so they go through the model once;
        each question then continues from a copy of that cache. Same logits as `read_full` (`selfcheck`)."""
        torch = self.torch
        ims = to_images(x)
        msgs = [[{"role": "user", "content": [*({"type": "image", "image": im} for im in ims),
                                              {"type": "text", "text": CONTEXT + " " + q}]}] for q in ASKED]
        # the images are processed once (first question); every question's text tail is tokenized alone
        enc = self.proc.apply_chat_template(msgs[0], add_generation_prompt=True, tokenize=True, return_dict=True,
                                            return_tensors="pt").to(self.model.device, dtype=torch.bfloat16)
        texts = [self.proc.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]
        raw = [self.proc.tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]
        c = 0
        while c < min(len(r) for r in raw) - 1 and all(r[c] == raw[0][c] for r in raw):
            c += 1
        tails = [r[c:] for r in raw]
        n = enc["input_ids"].shape[1] - len(tails[0])
        assert enc["input_ids"][0, n:].tolist() == tails[0], "tokenized tail differs from the processor's"
        pre = {k: (v[:, :n] if k in ("input_ids", "attention_mask", "token_type_ids") else v) for k, v in enc.items()}
        # all questions in one batch on the shared cache; right padding, so a question's last real
        # token never sees a pad
        k, m, dev = len(tails), max(len(t) for t in tails), self.model.device
        ids = torch.full((k, m), self.proc.tokenizer.pad_token_id, dtype=torch.long, device=dev)
        att = torch.zeros((k, n + m), dtype=torch.long, device=dev)
        att[:, :n] = 1
        for i, t in enumerate(tails):
            ids[i, :len(t)] = torch.tensor(t, device=dev)
            att[i, n:n + len(t)] = 1
        with torch.no_grad():
            cache = self.model(**pre, use_cache=True).past_key_values
            cache.batch_repeat_interleave(k)
            o = self.model(input_ids=ids, past_key_values=cache, use_cache=True, attention_mask=att,
                           cache_position=torch.arange(n, n + m, device=dev))
        z = [o.logits[i, len(t) - 1].float().cpu().numpy() for i, t in enumerate(tails)]
        return self.answers(z)


ASKED = ["What is the Pfirrmann grade of the middle disc? Answer with one digit from 1 to 5."] + \
        [q + " Answer Yes or No." for q in QUESTIONS.values()]


def selfcheck(n=3):
    """The cached `read` must give the logits of one full pass per question."""
    import time
    from grader import load_split_crops
    reader = Reader()
    x, _ = load_split_crops("dev", "tss")
    for i in range(n):
        t0 = time.time()
        a = reader.read(x[i])
        t1 = time.time()
        b = reader.read_full(x[i])
        t2 = time.time()
        d = max([abs(u - v) for u, v in zip(a["pfirrmann_logits"], b["pfirrmann_logits"])] + [abs(a[k] - b[k]) for k in QUESTIONS])
        print(f"crop {i}: max |difference| {d:.4f}; cached {t1 - t0:.2f}s, full {t2 - t1:.2f}s", flush=True)


def run(split):
    from spider import guard_test
    guard_test(split)
    from grader import load_split_crops
    reader = Reader()
    rows = []
    for loc in ("tss", "spider_masks"):
        x, idx = load_split_crops(split, loc)
        for i, r in enumerate(idx.itertuples()):
            rows.append({"patient": r.patient, "ivd": r.ivd, "localisation": loc, "fold": r.fold, **reader.read(x[i])})
            if i % 50 == 0:
                print(split, loc, i, "/", len(idx), flush=True)
    json.dump(rows, open(os.path.join(EVAL, f"medgemma_raw_{split}.json"), "w"))


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def calibrated(raw, fit_rows):
    """Calibrated MedGemma probabilities for `raw` rows with temperatures fitted on `fit_rows`."""
    from grader import fit_temperature
    zp = np.array([r["pfirrmann_logits"] for r in fit_rows])
    yp = np.array([r["t_pfirrmann"] for r in fit_rows])
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 121))
    nll = [-np.mean(np.log(softmax(zp / t)[np.arange(len(yp)), yp - 1] + 1e-9)) for t in grid]
    tp = grid[int(np.argmin(nll))]
    zb = np.array([[r[b] for b in ITEMS] for r in fit_rows])
    yb = np.array([[r["t_" + b] for b in ITEMS] for r in fit_rows])
    tb = fit_temperature(zb, yb)
    cls = softmax(np.array([r["pfirrmann_logits"] for r in raw]) / tp)
    pb = 1 / (1 + np.exp(-np.array([[r[b] for b in ITEMS] for r in raw]) / tb))
    return cls, pb


def with_truth(raw):
    from spider import grades
    g = grades()
    g["modic"] = (g["modic"] > 0).astype(int)
    truth = {(int(r.patient), int(r.ivd)): r for r in g.itertuples()}
    for r in raw:
        t = truth[(r["patient"], r["ivd"])]
        r["t_pfirrmann"] = int(t.pfirrmann)
        for b in ITEMS:
            r["t_" + b] = int(getattr(t, b))
    return raw


def rows_for(test, fit, gprobs, name, loc):
    """Prediction rows for `test` with temperatures and thresholds fitted on `fit` (development rows)."""
    from grader import youden
    cls, pb = calibrated(test, fit)
    fcls, fpb = calibrated(fit, fit)
    if name == "combo":
        g = gprobs[gprobs.localisation == loc].set_index(["patient", "ivd"])
        gf = gprobs[gprobs.localisation == "tss"].set_index(["patient", "ivd"])
        gc = np.array([[g.loc[(r["patient"], r["ivd"]), f"c{c}"] for c in range(1, 6)] for r in test])
        gb = np.array([[g.loc[(r["patient"], r["ivd"]), f"{b}_score"] for b in ITEMS] for r in test])
        cls, pb = (cls + gc) / 2, (pb + gb) / 2
        fpb = (fpb + np.array([[gf.loc[(r["patient"], r["ivd"]), f"{b}_score"] for b in ITEMS] for r in fit])) / 2
    thr = [youden(fpb[:, j], np.array([f["t_" + b] for f in fit])) for j, b in enumerate(ITEMS)]
    out = []
    for r, cc, pp in zip(test, cls, pb):
        above = 1 - np.cumsum(cc)[:4]          # P(grade > k), k = 1..4
        row = {"patient": r["patient"], "ivd": r["ivd"], "localisation": loc, "pfirrmann_pred": int(1 + (above > 0.5).sum())}
        for j, b in enumerate(ITEMS):
            row[f"{b}_score"], row[f"{b}_pred"] = float(pp[j]), int(pp[j] >= thr[j])
        out.append(row)
    return out


def predict(split):
    """pred_medgemma_<split>.csv and pred_combo_<split>.csv. Development rows are cross-fitted
    (fold k calibrated on the other folds' tss rows); test rows use all development rows."""
    from grader import write_preds
    dev = with_truth(json.load(open(os.path.join(EVAL, "medgemma_raw_dev.json"))))
    raw = dev if split == "dev" else with_truth(json.load(open(os.path.join(EVAL, f"medgemma_raw_{split}.json"))))
    gprobs = pd.read_csv(os.path.join(EVAL, f"grader_probs_{split}.csv"))
    gprobs_dev = pd.read_csv(os.path.join(EVAL, "grader_probs_dev.csv"))
    for name in ("medgemma", "combo"):
        rows = []
        for loc in ("tss", "spider_masks"):
            rr = [r for r in raw if r["localisation"] == loc]
            if split == "dev":
                for k in range(5):
                    test = [r for r in rr if int(r["fold"]) == k]
                    fit = [r for r in dev if r["localisation"] == "tss" and int(r["fold"]) != k]
                    if test:
                        rows += rows_for(test, fit, gprobs if name != "combo" else pd.concat([gprobs, gprobs_dev]).drop_duplicates(), name, loc)
            else:
                fit = [r for r in dev if r["localisation"] == "tss"]
                if name == "combo":
                    rows += rows_for(rr, fit, pd.concat([gprobs, gprobs_dev]), name, loc)
                else:
                    rows += rows_for(rr, fit, gprobs, name, loc)
        write_preds(split, pd.DataFrame(rows), method=name)


def qualify():
    """Whether MedGemma and the combination enter the final evaluation: the combination's development
    Pfirrmann kappa must beat the grader's."""
    import score
    k = {}
    for m in ("grader", "medgemma", "combo"):
        df = score.load(os.path.join(EVAL, f"pred_{m}_dev.csv"), score.path("radiological_gradings.csv"))
        sel = df[(df.localisation == "tss") & (df.in_scope == 1)]
        t, y, _ = score.filled(sel, "pfirrmann")
        k[m] = score.m_qwk(t, y)
    verdict = {"dev_pfirrmann_kappa": k, "qualifies": k["combo"] > k["grader"]}
    json.dump(verdict, open(os.path.join(EVAL, "medgemma_qualify.json"), "w"), indent=1)
    print(verdict)


if __name__ == "__main__":
    if sys.argv[1] == "qualify":
        qualify()
    elif sys.argv[1] == "selfcheck":
        selfcheck()
    else:
        run(sys.argv[1])
        predict(sys.argv[1])

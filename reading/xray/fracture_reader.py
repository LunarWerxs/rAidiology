"""Read one radiograph with the X-ray fracture detector pair (v140 and v141), on the processor.

    python reading/xray/fracture_reader.py PICTURE --profile wrist|other [--weights DIR] [--onnx]

Prints the pair's score from 0 to 1 (the average of the ten fold networks), the yes or no at that profile's
cut-off, and the research-model line. Weights: --weights DIR (holding v140/ and v141/), else
models/xray-fracture-detector/ inside the project, else ../Models/xray-fracture-detector-v140 and -v141 beside it.
The pictures are prepared by modalities/xray/eval/prepare.py, the code the scores were measured with. Without --onnx
the .safetensors files are used and PyTorch is needed; with --onnx the .onnx files are used and PyTorch is not.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("RAIDIOLOGY_DEVICE", "cpu")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "modalities", "xray", "eval"), ROOT]

VERSIONS = (140, 141)
FOLDS = 5
# profile -> (the collection the networks know it as, the pair's cut-off fitted on development data, as the card says)
PROFILES = {"wrist": ("grazpedwri", 0.5208339011424208), "other": ("fracatlas", 0.126139112968734)}
NOTICE = "Research model. Not a medical device. For education and research, not for diagnosis."


def say_and_exit(message):
    print(message)
    sys.exit(2)


def weight_folders(given):
    """The two version folders, from --weights, the project's models/ folder, or the folder beside the project."""
    homes = []
    if given:
        homes.append([os.path.join(given, "v%d" % v) for v in VERSIONS])
        homes.append([os.path.join(given, "xray-fracture-detector-v%d" % v) for v in VERSIONS])
    else:
        inside = os.path.join(ROOT, "models", "xray-fracture-detector")
        beside = os.path.join(os.path.dirname(ROOT), "Models")
        homes.append([os.path.join(inside, "v%d" % v) for v in VERSIONS])
        homes.append([os.path.join(beside, "xray-fracture-detector-v%d" % v) for v in VERSIONS])
    for home in homes:
        if all(os.path.isdir(d) for d in home):
            return home
    where = given or ("models/xray-fracture-detector/ (with v140/ and v141/ inside) "
                      "or ../Models/xray-fracture-detector-v140 and -v141")
    say_and_exit("The model files were not found in %s. Put the v140 and v141 folders of the rAidiology "
                 "fracture detector there, or point --weights at the folder that holds them." % where)


def load_meta(folder):
    path = os.path.join(folder, "meta.json")
    if not os.path.exists(path):
        say_and_exit("%s has no meta.json." % folder)
    return json.load(open(path, encoding="utf-8"))


def fold_file(folder, k, ext):
    path = os.path.join(folder, "fold%d%s" % (k, ext))
    if not os.path.exists(path):
        say_and_exit("%s is missing." % path)
    return path


def torch_scorers(folder, dataset_index):
    """One function per fold: pixels in [0, 1] as (n, 1, 384, 384) -> probabilities (n), from the .safetensors files."""
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    import detector
    meta = load_meta(folder)
    c = meta["cfg"]
    n_datasets = len(meta["datasets"])
    out = []
    for k in range(FOLDS):
        state = load_file(fold_file(folder, k, ".safetensors"))
        net = detector.Net(c["widths"], c["dropout"], c["stem_stride"], n_datasets, loc="loc.weight" in state)
        net.load_state_dict(state)
        net.eval()
        mean, sd = meta["folds"][k]["mean"], meta["folds"][k]["sd"]

        def run(x, net=net, mean=mean, sd=sd):
            which = F.one_hot(torch.full((len(x),), dataset_index), n_datasets).float()
            with torch.inference_mode():
                return torch.sigmoid(net((torch.from_numpy(x) - mean) / sd, which).float()).double().numpy()
        out.append(run)
    return out


def onnx_scorers(folder, dataset_index):
    import onnxruntime as ort
    meta = load_meta(folder)
    which = np.zeros((1, len(meta["datasets"])), np.float32)
    which[0, dataset_index] = 1.0
    out = []
    for k in range(FOLDS):
        s = ort.InferenceSession(fold_file(folder, k, ".onnx"), providers=["CPUExecutionProvider"])
        out.append(lambda x, s=s: s.run(None, {"pixels": x, "which": np.repeat(which, len(x), 0)})[0].astype(np.float64))
    return out


def read_picture(path):
    """The picture as the detector was trained and scored on it: prepare.decode's steps (read_grey, scaled,
    letterbox), which keep a 16-bit PNG's full range. prepare.read_any would turn such a picture into
    an all-black one (its convert("L") clips at 255), so it is used only for a DICOM file."""
    import prepare
    from pydicom.misc import is_dicom
    try:
        if is_dicom(path):
            pix = prepare.read_any(path, prepare.SIZE)[0]
        else:
            pix = prepare.letterbox(prepare.scaled(prepare.read_grey(path)), prepare.SIZE)
    except Exception:
        say_and_exit("%s could not be read as a picture. Give a PNG or JPEG radiograph (or a DICOM file)." % path)
    return (pix.astype(np.float32) / np.float32(255.0))[None, None]


def score(picture, profile, folders, onnx):
    """The pair's score: the mean of the two versions' scores, each the mean of its five folds."""
    dataset = PROFILES[profile][0]
    index = load_meta(folders[0])["datasets"].index(dataset)
    x = read_picture(picture)
    make = onnx_scorers if onnx else torch_scorers
    versions = [sum(float(f(x)[0]) for f in make(folder, index)) / FOLDS for folder in folders]
    return sum(versions) / len(versions)


def main():
    ap = argparse.ArgumentParser(description="Read one radiograph with the research fracture detector pair.")
    ap.add_argument("picture")
    ap.add_argument("--profile", required=True, choices=sorted(PROFILES), help="wrist, or other (hand, leg, hip, shoulder)")
    ap.add_argument("--weights", help="a folder holding v140/ and v141/")
    ap.add_argument("--onnx", action="store_true", help="read with the .onnx files and ONNX Runtime (no PyTorch)")
    a = ap.parse_args()
    if not os.path.isfile(a.picture):
        say_and_exit("%s is not a file." % a.picture)
    folders = weight_folders(a.weights)
    s = score(a.picture, a.profile, folders, a.onnx)
    cutoff = PROFILES[a.profile][1]
    print("score: %.4f" % s)
    print("%s: %s (cut-off %.4f)" % (a.profile, "yes" if s >= cutoff else "no", cutoff))
    print(NOTICE)


if __name__ == "__main__":
    main()

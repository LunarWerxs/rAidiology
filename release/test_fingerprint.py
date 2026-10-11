import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest
import torch
from torch import nn

import fingerprint as fp

RATE = "8"


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(1024, 32)

    def forward(self, x):
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("fp")
    key = d / "k.key"
    fp.main(["keygen", "--out", str(key)])
    torch.manual_seed(0)
    net = Net().eval()
    x = torch.randn(2, 3, 18, 18)
    orig = d / "orig.pt"
    torch.save(net.state_dict(), orig)
    marked = d / "m.pt"
    fp.main(["mark", "--mark-id", "licA", "--in", str(orig), "--out", str(marked),
             "--key-file", str(key), "--rate", RATE])
    other = d / "o.pt"
    fp.main(["mark", "--mark-id", "licB", "--in", str(orig), "--out", str(other),
             "--key-file", str(key), "--rate", RATE])
    return dict(d=d, key=key, net=net, x=x, orig=orig, marked=marked, other=other)


def verdict(env, path, mark_id="licA"):
    return fp.verify_one(fp.Weights(path), fp.load_key(env["key"]), mark_id, int(RATE))


def marked_net(env):
    n = Net().eval()
    n.load_state_dict(torch.load(env["marked"], weights_only=True))
    return n


def test_output_unchanged(env):
    with torch.no_grad():
        a, b = env["net"](env["x"]), marked_net(env)(env["x"])
    assert (a - b).abs().max() / a.abs().max() < 1e-5


def test_record_written(env):
    rec = json.loads(Path(str(env["marked"]) + ".mark.json").read_text())
    assert rec["tensors_marked"] == 2 and rec["mark_id"] == "licA"


def test_finds_mark_all_formats(env):
    from safetensors.torch import save_file
    assert verdict(env, env["marked"])["marked"]
    st = env["d"] / "m.safetensors"
    save_file({k: v.contiguous() for k, v in marked_net(env).state_dict().items()}, st)
    assert verdict(env, st)["marked"]
    ox = env["d"] / "m.onnx"
    torch.onnx.export(marked_net(env), env["x"], str(ox), opset_version=17, dynamo=False)
    r = verdict(env, ox)
    assert r["marked"], r


def test_unmarked_and_other_id(env):
    assert not verdict(env, env["orig"])["marked"]
    assert not verdict(env, env["other"])["marked"]
    assert verdict(env, env["other"], "licB")["marked"]


def test_float16_erases_mark(env):
    h = env["d"] / "h.pt"
    torch.save({k: v.half() for k, v in marked_net(env).state_dict().items()}, h)
    assert not verdict(env, h)["marked"]


def test_pt_wrapper_structure_kept(env):
    sd = torch.load(env["orig"], weights_only=True)
    w = env["d"] / "wrap.pt"
    torch.save({"state_dict": sd, "epoch": 7}, w)
    out = env["d"] / "wrap_m.pt"
    fp.main(["mark", "--mark-id", "licA", "--in", str(w), "--out", str(out),
             "--key-file", str(env["key"]), "--rate", RATE])
    obj = torch.load(out, weights_only=True)
    assert obj["epoch"] == 7 and verdict(env, out)["marked"]


def test_keygen_silent_and_no_overwrite(tmp_path, capsys):
    k = tmp_path / "sub" / "x.key"
    fp.main(["keygen", "--out", str(k)])
    out = capsys.readouterr().out.strip()
    raw = k.read_bytes()
    assert len(raw) == 32 and out == fp.key_id(raw) and len(out) == 12
    assert raw.hex() not in out
    with pytest.raises(SystemExit):
        fp.main(["keygen", "--out", str(k)])
    assert k.read_bytes() == raw

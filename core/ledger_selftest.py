"""The ledger's promises, checked in a throwaway folder: python core/ledger_selftest.py

- a run whose process is killed is written in as failed, with its numbers, settings, curve and the
  time it ran, even when it recorded nothing for a while (the heartbeat, not the last number, says
  it was alive);
- a run that is still running is never swept, and one that has closed leaves nothing behind that a
  later sweep could write in a second time;
- a process that exits with 259, the code that otherwise means "still running", reads as ended;
- a run whose record cannot be written does not fail, and is left for the next sweep;
- time spent waiting for the graphics card is recorded as card_wait_s and kept out of the slow flags;
- runs made with different settings, or an older run whose note says so, are not compared.

The real ledger is never touched: everything goes into a temporary folder that is removed after.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core import ledger  # noqa: E402  arkitect-allow: test-floor-loosened - the import must follow the sys.path line above it, which is how every script here reaches core

QUIET = 0x08000000 if os.name == "nt" else 0          # CREATE_NO_WINDOW
CHILD = r"""
import sys, time
sys.path.insert(0, %r)
from core import ledger
ledger.BEAT_S = 0.1
with ledger.run("t", "killed on purpose") as r:
    r.setting("lr", 0.001)
    for step in range(1, 4):
        r.point("loss", step, 1.0 / step)
    r.number("scans", 12, "count")
    time.sleep(600)
"""


def killed_run_is_kept():
    child = subprocess.Popen([sys.executable, "-c", CHILD % ROOT], creationflags=QUIET,
                             env=dict(os.environ, RAIDIOLOGY_LEDGER=ledger.DIR))
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            live = ledger.open_runs()
            if live and live[0].get("numbers", {}).get("scans"):
                break
            time.sleep(0.2)
        live = ledger.open_runs()
        assert len(live) == 1 and live[0]["alive"], live
        assert ledger.sweep() == [], "a live run must not be swept"
        time.sleep(1.5)                       # nothing recorded now: only the heartbeat speaks
    finally:
        child.kill()
        child.wait(10)
    time.sleep(0.5)
    assert len(ledger.sweep()) == 1
    rec = ledger.records("t")[0]
    assert rec["ok"] == 0 and rec["ended"] == ledger.NO_RECORD, rec
    assert rec["seconds"] >= 1.2, "a killed run keeps the time it ran: %s" % rec["seconds"]
    assert rec["numbers"]["scans"]["value"] == 12.0, rec
    assert rec["series"]["loss"] == [[1.0, 1.0], [2.0, 0.5], [3.0, 1.0 / 3]], rec
    assert rec["settings"] == {"lr": "0.001"}, rec
    assert ledger.open_runs() == [] and ledger.sweep() == [], "swept once, and only once"


def closed_runs_leave_nothing():
    ledger.BEAT_S = 0.0
    for i in range(30):
        with ledger.run("t", "quick %d" % i) as r:
            r.number("i", i)
            time.sleep(0.01)
    ledger.BEAT_S = 20.0
    assert ledger.open_runs() == [], "a heartbeat landing as a run closes must not leave a marker"


def exit_code_259_is_ended():
    odd = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(259)"], creationflags=QUIET)
    odd.wait(30)
    assert odd.returncode == 259 and ledger._born(odd.pid) is None


def unwritable_record_waits_for_the_sweep():
    real = ledger.write

    def broken(*_a, **_k):
        raise OSError("disk full")
    ledger.write = broken
    try:
        with ledger.run("t", "record fails") as r:
            r.number("x", 1)
    finally:
        ledger.write = real
    left = ledger.open_runs()
    assert len(left) == 1 and left[0]["alive"], "the marker stays for the next sweep"
    os.remove(left[0]["path"])


def unlike_runs_are_not_compared():
    old = {"note": "jobs=8; first build", "numbers": {}}
    assert not ledger.comparable(old, {"note": "jobs=1", "numbers": {}})
    assert ledger.comparable(old, {"settings": {"jobs": "8"}, "numbers": {}})
    assert not ledger.comparable({"numbers": {"checks": {"value": 0}}},
                                 {"numbers": {"checks": {"value": 310}}})


class FakeCard:
    """Stands in for core.card: waited_seconds() answers from a list, then keeps the last."""
    def __init__(self, *waits):
        self.waits = list(waits)

    def waited_seconds(self):
        return self.waits.pop(0) if len(self.waits) > 1 else self.waits[0]


def card_wait_stays_out_of_the_limits():
    assert "core.card" not in sys.modules and "card" not in sys.modules, "the selftest must not import card"
    with ledger.run("t", "no card module") as r:
        r.number("x", 1)
    assert "card_wait_s" not in ledger.records("t")[-1]["numbers"], "no card, no wait"
    assert "core.card" not in sys.modules, "the ledger must not import card"
    try:
        sys.modules["core.card"] = FakeCard(0, 4000)
        with ledger.run("t", "waited four thousand") as r:
            r.t0 -= 4300                      # the clock: the run took 4300 s
        rec = ledger.records("t")[-1]
    finally:
        sys.modules.pop("core.card", None)
    assert abs(rec["seconds"] - 4300) < 5 and rec["numbers"]["card_wait_s"] == {"value": 4000.0, "unit": "s"}, rec
    flagged = ledger.slow_lines({("t", "waited four thousand"): [rec]}, 5)
    assert flagged == [], "300 s of work is inside the target: %s" % flagged
    said = _printed(lambda: ledger.write("t", "direct, waited", 4300, card_wait=4000))
    assert "OVER THE ONE-HOUR LIMIT" not in said and "TARGET" not in said, said
    assert "4000.0 s of it waiting for the card" in said, said
    assert "OVER THE ONE-HOUR LIMIT" in _printed(lambda: ledger.write("t", "direct, short wait", 4300,
                                                                      card_wait=100))
    # the direct path takes the wait accrued since the last record, never more than the run's seconds
    sys.modules["core.card"] = FakeCard(9000)
    try:
        ledger._MARK = 0.0
        rec = ledger.write("t", "direct, capped", 50)
        assert rec["numbers"]["card_wait_s"]["value"] == 50.0, rec
        assert ledger.write("t", "direct, already counted", 50)["numbers"].get("card_wait_s") is None
    finally:
        sys.modules.pop("core.card", None)
        ledger._MARK = 0.0


def _printed(call):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        call()
    return buf.getvalue()


def main():
    tmp = tempfile.mkdtemp(prefix="ledger-selftest-")
    ledger.DIR, ledger.OPEN = tmp, os.path.join(tmp, ".open")
    try:
        for check in (killed_run_is_kept, closed_runs_leave_nothing, exit_code_259_is_ended,
                      unwritable_record_waits_for_the_sweep, unlike_runs_are_not_compared,
                      card_wait_stays_out_of_the_limits):
            check()
            print("ok  %s" % check.__name__.replace("_", " "), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ledger selftest: all passed")


if __name__ == "__main__":
    main()

"""A long run on somebody's own computer gives way to the person sitting at it.

Training a segmenter is hours of a graphics card at full tilt, and this machine is also the one its
owner works on. A job pinned at 100% makes the desktop stutter for exactly as long as he is trying
to use it, and is no faster for him once he has walked away. So a long loop asks, after each step,
how long it should rest before the next one:

- **somebody is at the keyboard or mouse** (input in the last two minutes) and another program is
  using the card (nvidia-smi says so; when it cannot say, we assume it is): rest long enough that
  the job uses about half the card. One blip (a desktop app redrawing) does not count; two readings
  in a row, 15 s apart, do;
- **nobody is**, or **nothing else is using the card**: rest hardly at all. Changed 2026-10-09 on
  the owner's decision: while he is at the keyboard the job keeps the whole card, unless another
  program uses it.

`RAIDIOLOGY_SHARE` fixes the share instead (0.25 to 1.0), for a machine with nobody to yield to or a
run that has to be slow regardless. Off Windows there is no input clock to read, so the share is the
fixed one, 1.0 by default.

    from core import polite
    pace = polite.Pace()
    ...
    pace.rest(seconds_the_step_took)
"""
from collections import deque
import os
import subprocess
import sys
import time

BUSY_SHARE = 0.5              # of the device, while its owner is using the machine
AWAY_SHARE = 0.95
AWAY_AFTER_S = 120.0          # this long without input counts as away
FOREIGN_MIN_SM = 10           # another program under this share of the card is not using it
FOREIGN_EVERY_S = 15.0        # nvidia-smi is asked at most this often


def idle_seconds():
    """Seconds since the last keyboard or mouse input, or None where that cannot be read."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes  # noqa: PLC0415

        class LastInput(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        info = LastInput()
        info.cbSize = ctypes.sizeof(info)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return None
        now = ctypes.windll.kernel32.GetTickCount()
        return ((now - info.dwTime) & 0xFFFFFFFF) / 1000.0
    except Exception:  # noqa: BLE001  not knowing is an answer; a crash in a courtesy is not
        return None


def _foreign_sm(pmon_text):
    """The highest sm % of a program on the card that is not ours, from `nvidia-smi pmon` text."""
    best = 0.0
    for line in pmon_text.splitlines():
        cols = line.split()
        if len(cols) < 4 or cols[0].startswith("#"):
            continue
        sm = 0.0 if cols[3] == "-" else float(cols[3])
        name = cols[-1].lower()
        if name.startswith("python") or name == "dwm.exe":
            continue
        if sm > best:
            best = sm
    return best


_foreign_seen = {"at": None, "value": None}
_foreign_recent = deque(maxlen=2)   # the last two readings nvidia-smi took; None where it could not say


def _ask_nvidia_smi():
    """One fresh nvidia-smi reading: the highest sm % of another program on the card, or None."""
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        done = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1", "-s", "u"],
            capture_output=True, text=True, timeout=5, creationflags=flags,
        )
        if done.returncode == 0:
            return _foreign_sm(done.stdout)
    except Exception:  # noqa: BLE001  no nvidia-smi, or it hung: not knowing is an answer
        pass
    return None


def foreign_gpu_use():
    """The highest sm % of another program on the graphics card, or None where that is not known.

    nvidia-smi is asked at most once per FOREIGN_EVERY_S; the answer in between is the last one.
    """
    now = time.monotonic()
    if _foreign_seen["at"] is not None and now - _foreign_seen["at"] < FOREIGN_EVERY_S:
        return _foreign_seen["value"]
    value = _ask_nvidia_smi()
    _foreign_recent.append(value)
    _foreign_seen["at"] = now
    _foreign_seen["value"] = value
    return value


def _foreign_busy():
    """Last two fresh readings in a row both at FOREIGN_MIN_SM or more."""
    return len(_foreign_recent) == 2 and all(r is not None and r >= FOREIGN_MIN_SM
                                             for r in _foreign_recent)


class Pace:
    def __init__(self):
        fixed = os.environ.get("RAIDIOLOGY_SHARE", "").strip()
        self.fixed = min(1.0, max(0.25, float(fixed))) if fixed else None
        self.rested = 0.0
        self.worked = 0.0

    def share(self):
        if self.fixed is not None:
            return self.fixed
        idle = idle_seconds()
        if idle is None:
            return 1.0
        if idle >= AWAY_AFTER_S:
            return AWAY_SHARE
        foreign = foreign_gpu_use()
        if foreign is None or _foreign_busy():
            return BUSY_SHARE
        return AWAY_SHARE

    def rest(self, step_seconds):
        """Sleep so that this step's work was `share` of the time it took, work plus rest."""
        share = self.share()
        pause = step_seconds * (1.0 - share) / share
        self.worked += step_seconds
        if pause > 0.001:
            time.sleep(pause)
            self.rested += pause
        return pause

    def used(self):
        """The share of wall time actually spent working so far."""
        total = self.worked + self.rested
        return self.worked / total if total else 1.0


PMON_OWN = """# gpu        pid  type    sm   mem   enc   dec   jpg   ofa  command
# Idx          #   C/G     %     %     %     %     %     %   name
    0       4312     C    90    40     -     -     -     -  python.exe
    0       1180     G     3     1     -     -     -     -  dwm.exe
    0       2208     G     -     -     -     -     -     -  explorer.exe
"""

PMON_GAME = """# gpu        pid  type    sm   mem   enc   dec   jpg   ofa  command
# Idx          #   C/G     %     %     %     %     %     %   name
    0       4312     C    60    30     -     -     -     -  pythonw.exe
    0       7720     G    45    20     -     -     -     -  game.exe
"""


def selftest():
    if _foreign_sm(PMON_OWN) != 0:
        raise AssertionError("own python and dwm should not count as foreign: %r" % _foreign_sm(PMON_OWN))
    if _foreign_sm(PMON_GAME) != 45:
        raise AssertionError("game at 45 should read 45, got %r" % _foreign_sm(PMON_GAME))

    global idle_seconds, _ask_nvidia_smi
    real_idle, real_ask = idle_seconds, _ask_nvidia_smi
    # each case: the idle seconds, then the fresh nvidia-smi readings in the order they are taken
    cases = [
        ("away, a game is on the card", 500.0, [45.0, 45.0], AWAY_SHARE),
        ("at keyboard, nothing foreign", 5.0, [0.0, 0.0], AWAY_SHARE),
        ("at keyboard, foreign at 45 twice", 5.0, [45.0, 45.0], BUSY_SHARE),
        ("at keyboard, foreign unknown", 5.0, [None], BUSY_SHARE),
        ("at keyboard, one blip at 35 then 0", 5.0, [35.0, 0.0], AWAY_SHARE),
        ("at keyboard, two readings at 35", 5.0, [35.0, 35.0], BUSY_SHARE),
        ("at keyboard, a None reading after a 35", 5.0, [35.0, None], BUSY_SHARE),
    ]
    try:
        for name, idle, readings, want in cases:
            idle_seconds = lambda idle=idle: idle
            answers = iter(readings)
            _ask_nvidia_smi = lambda: next(answers)
            _foreign_recent.clear()
            pace = Pace()
            pace.fixed = None
            for _ in readings:
                _foreign_seen["at"] = None  # each reading is a fresh nvidia-smi ask, not a cached repeat
                foreign_gpu_use()
            got = pace.share()
            if got != want:
                raise AssertionError("%s: share %r, want %r" % (name, got, want))
    finally:
        idle_seconds, _ask_nvidia_smi = real_idle, real_ask
        _foreign_recent.clear()
        _foreign_seen.update(at=None, value=None)
    print("polite selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        raise SystemExit("usage: python core/polite.py selftest")

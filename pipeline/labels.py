"""TotalSpineSeg step-2 label ids -> names."""

NAMES = {1: "cord", 2: "canal", 50: "sacrum"}
for i in range(1, 8):
    NAMES[10 + i] = f"C{i}"
for i in range(1, 13):
    NAMES[20 + i] = f"T{i}"
for i in range(1, 6):
    NAMES[40 + i] = f"L{i}"
# discs are numbered by the vertebra BELOW them
_below = {63: "C3", 64: "C4", 65: "C5", 66: "C6", 67: "C7", 71: "T1"}
for i in range(2, 13):
    _below[70 + i] = f"T{i}"
_below.update({91: "L1", 92: "L2", 93: "L3", 94: "L4", 95: "L5", 100: "S1"})
ORDER = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"] + [f"T{i}" for i in range(1, 13)] + \
        ["L1", "L2", "L3", "L4", "L5", "S1"]
for lab, lower in _below.items():
    upper = ORDER[ORDER.index(lower) - 1]
    NAMES[lab] = f"disc_{upper}_{lower}"


def kind(name):
    if name.startswith("disc_"):
        return "disc"
    if name in ("cord", "canal", "sacrum"):
        return name
    return "vertebra"


def level_of(name):
    return name[5:].replace("_", "-") if name.startswith("disc_") else name

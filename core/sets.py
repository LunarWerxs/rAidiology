"""One reading of `--set key=value` for every trainer.

Six trainers each parsed `--set` their own way. Three were strict JSON; three fell back to the bare word as
a string. 2026-10-04: a queue passed `--set distill=work/eval/chest/teacher-v192-fold0.npz`, the shell had
no JSON quotes to keep, and reader.py died only after waiting 54 minutes for the graphics card. In the
lenient three, `--set amp=False` silently became the string 'False', which is truthy.

The rule: the value is JSON. If it is not, it is kept as a plain string only when it plainly is a path or a
name whose quotes the shell removed (work/x.npz, batch). Anything that looks like a mistyped boolean, null,
number, list or object is refused when the trainer starts. That is before its card lease when the trainer
leases for itself, but after the wait when a queue runs it through `core/card.py hold`.
"""

import json

_WORDS = {"true", "false", "null", "none", "nan", "inf", "-inf"}
_NOT_FIRST = set("0123456789-+.[{\"'")


def parse(items, known=None):
    """{key: value} from a list of 'key=value' strings; later items override earlier ones.

    `known`, a dict or any iterable of keys, limits the keys that may be set."""
    keys = None if known is None else set(known)
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit("--set wants key=value, got %r" % item)
        key, val = item.split("=", 1)
        if keys is not None and key not in keys:
            raise SystemExit("unknown setting %r; known: %s" % (key, ", ".join(sorted(keys))))
        try:
            out[key] = json.loads(val)
        except ValueError:
            if val and val.lower() not in _WORDS and val[0] not in _NOT_FIRST:
                out[key] = val
            else:
                raise SystemExit(
                    "--set %s=%s: not a value. Write true/false/null in lower case, numbers as numbers, "
                    "and a string that starts like a number or bracket in JSON quotes (%s='\"%s\"')"
                    % (key, val, key, val))
    return out

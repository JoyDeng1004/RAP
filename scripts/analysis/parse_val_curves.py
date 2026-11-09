"""Per-epoch training-val metrics from Lightning progress-bar logs (qsub_outputs/*.o<job>).

The bar of epoch N shows the val metrics of epoch N after its validation, so the last bar line
tagged "Epoch N:" carries that epoch's val values.

    python scripts/analysis/parse_val_curves.py qsub_outputs/rap-dk.o9062117 [...]
"""
import re
import sys
from pathlib import Path

KEYS = ("val/score", "val/best_score", "val/min_loss", "val/ade_real")
EPOCH = re.compile(r"Epoch (\d+):")


def parse(path):
    out = {}
    text = Path(path).read_text(errors="replace")
    for line in re.split(r"[\r\n]", text):
        m = EPOCH.search(line)
        if not m or "val/score=" not in line:
            continue
        row = {}
        for k in KEYS:
            v = re.search(re.escape(k) + r"=([-\d.e]+)", line)
            if v:
                row[k] = float(v.group(1))
        out[int(m.group(1))] = row
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        curve = parse(p)
        print(p)
        for e in sorted(curve):
            print(f"  e{e:02d} " + " ".join(f"{k.split('/')[1]}={v:.3f}" for k, v in curve[e].items()))

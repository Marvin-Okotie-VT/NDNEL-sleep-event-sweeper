"""load, filter, derive eye components, plot chosen epochs.

usage: python scripts/check_epochs.py <edf> <map: sample_psg|t03_seeg> <epoch[:note]> ...
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from sleep_event_sweeper import config
from sleep_event_sweeper.edf_io import load
from sleep_event_sweeper.preprocess import apply_filters, eye_components
from sleep_event_sweeper.plots import epoch_plot

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")


def folder(var: str) -> Path:
    value = os.environ.get(var)
    if not value:
        sys.exit(f"{var} is not set: make sure .env is in {REPO} and set up correctly")
    return Path(value)


if len(sys.argv) < 4:
    sys.exit(__doc__)
edf_arg, mapname, *eps = sys.argv[1:]
maps = {"sample_psg": config.SAMPLE_PSG, "t03_seeg": config.T03_SEEG}
if mapname not in maps:
    sys.exit(f"unknown map {mapname!r}; choose from {list(maps)}")
cmap = maps[mapname]

edf = Path(edf_arg)
if not edf.is_absolute():
    edf = folder("SLEEP_DATA_DIR") / edf
if not edf.is_file():
    sys.exit(f"EDF not found: {edf}")
outdir = folder("SLEEP_OUTPUT_DIR") / edf.stem.strip() / "check_epochs"
outdir.mkdir(parents=True, exist_ok=True)

rec = load(str(edf), cmap)
print(f"{edf.name}: start {rec.start}  duration {rec.duration_s:.0f}s  fs {rec.fs}  "
      f"epochs {rec.duration_s / config.EPOCH_S:.2f}")
for k, v in rec.labels.items():
    print(f"  {k:14s} <- {v}")
sig = apply_filters(rec)
sig.update(eye_components(sig))
for e in eps:
    n, _, note = e.partition(":")
    p = outdir / f"epoch_{int(n):04d}.png"
    epoch_plot(sig, rec.labels, rec.fs, rec.start, int(n), note, str(p), cmap.display_uv)
    print("wrote", p)

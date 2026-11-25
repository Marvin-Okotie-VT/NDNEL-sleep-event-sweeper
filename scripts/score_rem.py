"""Score stage R (milestone: REM + low chin EMG) and write the Excel sheet.

usage: python scripts/score_rem.py <edf> <map: sample_psg|t03_seeg>

<edf> is a file name inside SLEEP_DATA_DIR, or a full path. The sheet goes to
SLEEP_OUTPUT_DIR/<edf name>/<edf name>_REM_scores.xlsx (overwritten on each run).
example: python scripts/score_rem.py "Scored SP sample .edf" sample_psg
"""
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from sleep_event_sweeper import config
from sleep_event_sweeper.edf_io import load
from sleep_event_sweeper.preprocess import apply_filters
from sleep_event_sweeper.detectors import rapid_eye_movements, chin_emg_level
from sleep_event_sweeper.scoring import score_rem_milestone
from sleep_event_sweeper.score_results import write_scores

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")


def folder(var: str) -> Path:
    value = os.environ.get(var)
    if not value:
        sys.exit(f"{var} is not set: make sure .env is in {REPO} and set up correctly")
    return Path(value)


def code_version() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return sha + (" + uncommitted changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"


if len(sys.argv) != 3:
    sys.exit(__doc__)
edf_arg, mapname = sys.argv[1:]
maps = {"sample_psg": config.SAMPLE_PSG, "t03_seeg": config.T03_SEEG}
if mapname not in maps:
    sys.exit(f"unknown map {mapname!r}; choose from {list(maps)}")
cmap = maps[mapname]

edf = Path(edf_arg)
if not edf.is_absolute():
    edf = folder("SLEEP_DATA_DIR") / edf
if not edf.is_file():
    sys.exit(f"EDF not found: {edf}")
outdir = folder("SLEEP_OUTPUT_DIR") / edf.stem.strip()
outdir.mkdir(parents=True, exist_ok=True)
out = outdir / f"{edf.stem.strip()}_REM_scores.xlsx"

tune, manual = config.DetectorTuning(), config.ManualCriteria()
t0 = time.time()
rec = load(str(edf), cmap)
print(f"{edf.name}: start {rec.start}  duration {rec.duration_s:.0f}s  fs {rec.fs}")
for k, v in rec.labels.items():
    print(f"  {k:14s} <- {v}")
sig = apply_filters(rec)
print(f"loaded and filtered in {time.time() - t0:.0f}s")

rems, eye_info = rapid_eye_movements(sig["eog_right"], sig["eog_left"], rec.fs, tune, manual)
emg_rms, emg_pct = chin_emg_level(sig["chin_emg"], rec.fs, config.EPOCH_S)
results = score_rem_milestone(rems, emg_rms, emg_pct, config.EPOCH_S, tune)

n_r = sum(r.score == "R" for r in results)
n_u = sum(r.unsure == "R?" for r in results)
n_rem_ep = sum(bool(r.rems) for r in results)
print(f"eye noise {eye_info['sigma_uv']} uV -> REM size >= {eye_info['min_amp_uv']} uV | {len(rems)} REMs")
print(f"{len(results)} epochs: {n_rem_ep} with REMs | R {n_r} | R? {n_u}")

run_info = {
    "file": str(edf),
    "channel map": mapname,
    "signals": "; ".join(f"{k} = {v}" for k, v in rec.labels.items()),
    "recording start": rec.start,
    "scored at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "code version (git)": code_version(),
    "method": "milestone: R = >= 1 REM and low chin EMG (no LAMF, K-complex/spindle, continuation or look-back yet)",
    "eye noise sigma (uV)": eye_info["sigma_uv"],
    "REM min size (uV)": eye_info["min_amp_uv"],
    "REM count": len(rems),
    "epochs R / R?": f"{n_r} / {n_u}",
    **{f"tuning.{k}": v for k, v in asdict(tune).items()},
    **{f"manual.{k}": v for k, v in asdict(manual).items()},
}
try:
    write_scores(out, rec.start, results, run_info)
except PermissionError:
    sys.exit(f"cannot write {out}: close it in Excel and run again")
print(f"wrote {out}  ({time.time() - t0:.0f}s total)")
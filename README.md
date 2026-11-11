# NDNEL-sleep-event-sweeper

Explainable sleep event detection and scoring(per AASM manual) for lab EDF recordings. 
Stage R (REM) first; to be extended to all stages.

## Layout

```
src/            the package, imported as sleep_event_sweeper
  config.py     channel maps per recording type, filter bands, plot scales
  edf_io.py     reads only the needed signals, builds derivations (plus - minus)
  preprocess.py 60 Hz notch, band-pass per signal, eye_diff (R-L) / eye_sum ((R+L)/2)
  plots.py      30-s epoch plot, fixed scale per row, EDFbrowser-style overflow
scripts/
  check_epochs.py   load -> filter -> plot chosen epochs
tests/
```

## Setup (once per PC)

```
git clone 
cd NDNEL-sleep-event-sweeper
python -m venv .venv
.venv\Scripts\activate            (Windows)
pip install -e .
```

## Run

```
python scripts/check_epochs.py "Scored SP sample .edf" sample_psg "680:REM" "113:K-complex"
```

Epoch n starts at (n - 1) * 30 s, the same numbering as the Excel rows.
For SEEG files use the map `t03_seeg` (T03 files only); first confirm in EDFbrowser that signals #215/#216
are the two "POL EMG" signals.

"""EDF loading: reads the signals specified in config.py, and returns a Recording object with the signals and metadata."""
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pyedflib

from .config import ChannelMap, Derivation


@dataclass
class Recording:
    start: datetime
    duration_s: float
    fs: float
    signals: dict            # 1-D float array (uV), all at fs
    labels: dict             # derivations, e.g. "POL RUE - EEG A2-Ref"


def _resolve(reader: pyedflib.EdfReader, sig) -> int:
    if isinstance(sig, int):
        return sig
    labels = reader.getSignalLabels()
    hits = [i for i, l in enumerate(labels) if l.strip() == sig]
    if len(hits) != 1:
        raise ValueError(f"label {sig!r} matched {len(hits)} signals; use an index")
    return hits[0]


def _read(reader, idx: int) -> tuple[np.ndarray, float, str]:
    return reader.readSignal(idx), reader.getSampleFrequency(idx), reader.getLabel(idx).strip()


def load(path: str, cmap: ChannelMap) -> Recording:
    reader = pyedflib.EdfReader(path)
    try:
        roles = {k: getattr(cmap, k) for k in
                 ("eog_right", "eog_left", "chin_emg", "eeg_frontal", "eeg_central", "eeg_occipital")}
        signals, labels, rates = {}, {}, set()
        for role, d in roles.items():
            if d is None:
                continue
            ip = _resolve(reader, d.plus)  #index of the plus signal
            x, fs, lp = _read(reader, ip)
            if isinstance(d.plus, int):        # picked by index: show EDFbrowser number (1-based)
                lp = f"{lp} (#{ip + 1})"
            label = lp
            if d.minus is not None:
                im = _resolve(reader, d.minus) 
                y, fs2, lm = _read(reader, im)
                if isinstance(d.minus, int):
                    lm = f"{lm} (#{im + 1})"
                if fs2 != fs:
                    raise ValueError(f"{role}: sample rates differ ({fs} vs {fs2})")
                x = x - y                      # drop common reference
                label = f"{lp} - {lm}"
            signals[role] = x.astype(np.float64)
            labels[role] = label
            rates.add(fs)
        if len(rates) != 1:
            raise ValueError(f"mixed sample rates {rates}; resampling not implemented yet")
        return Recording(reader.getStartdatetime(), reader.getFileDuration(),
                         rates.pop(), signals, labels)
    finally:
        reader.close()
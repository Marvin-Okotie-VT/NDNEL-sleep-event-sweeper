"""filters + two derived eye signals(markers/ scoring hints).

Zero-phase (forward-backward) filtering keeps event timing aligned with the raw
trace, so annotations are marked where the deflection starts in EDFbrowser.
note: magnitude response is squared (order 2 behaves like order 4 in
roll-off).
"""

import numpy as np
from scipy import signal as ss

from .config import FilterSettings
from .edf_io import Recording


def _band(x, fs, lo, hi, order, role):
    nyquist = 0.5 * fs
    if not 0 < lo < hi < nyquist:
        raise ValueError(f"{role}: band {lo}-{hi} Hz is not inside 0-{nyquist} Hz "
                         f"(Nyquist for fs = {fs} Hz); change the band or the recording")
    sos = ss.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return ss.sosfiltfilt(sos, x)
 
 
def _notch(x, fs, f0, q):
    if f0 >= fs / 2:
        return x
    b, a = ss.iirnotch(f0, q, fs=fs)
    return ss.filtfilt(b, a, x)
 
 
def apply_filters(rec: Recording, fs_cfg: FilterSettings = FilterSettings()) -> dict:
    out = {}
    for role, x in rec.signals.items():
        x = _notch(x, rec.fs, fs_cfg.notch_hz, fs_cfg.notch_q)
        if role not in fs_cfg.bands:
            raise KeyError(f"no filter band configured for role {role!r} (FilterSettings.bands)")
        out[role] = _band(x, rec.fs, *fs_cfg.bands[role], fs_cfg.order, role)
    return out


def eye_components(filt: dict) -> dict:
    """RUE and LLE move in opposite directions for any eye movement (horizontal,
    vertical, blinks) and in the same direction for frontal EEG.

    eye_diff = R - L : eye movements add, shared EEG cancels  -> REM detector input
    eye_sum  = (R+L)/2: eye movements cancel, frontal EEG stays -> LAMF proxy
    """
    r, l = filt["eog_right"], filt["eog_left"]
    return {"eye_diff": r - l, "eye_sum": 0.5 * (r + l)}
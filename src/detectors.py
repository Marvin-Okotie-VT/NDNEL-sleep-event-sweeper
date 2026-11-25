"""Event detectors. Each returns Event objects that carry their measurements and metadata. 

Sections: eye movements (REM) | chin EMG | EEG / proxy | arousals
"""
from dataclasses import dataclass, field
import numpy as np
from scipy import signal as ss

from .config import DetectorTuning, ManualCriteria


@dataclass
class Event:
    kind: str                 # e.g. "REM"
    onset_s: float            # seconds from recording start
    duration_s: float
    values: dict = field(default_factory=dict)    # measurements, e.g. {"amp_uv": 280, "rise_ms": 176}


#eye movements (EOG)

def _zigzag(x: np.ndarray, h: float) -> np.ndarray:
    """Turning points of x. A reversal only counts once x has moved back by at least h,
    so small wiggles inside one eye movement don't split it into several swings.

    ext stores the farthest point reached in the current swing
    pts stores the turning points (the peaks/troughs that define each swing)
    """
    pts, ext, direction = [0], 0, 0
    for i in range(1, len(x)):
        if direction == 0:  #no direction yet, look for the first swing; direction does not become 0 again
            if x[i] - x[0] >= h:
                direction, ext = 1, i
            elif x[0] - x[i] >= h:      
                direction, ext = -1, i
        elif direction == 1:
            if x[i] > x[ext]:
                ext = i
            elif x[ext] - x[i] >= h:
                pts.append(ext); direction, ext = -1, i
        else:
            if x[i] < x[ext]:
                ext = i
            elif x[i] - x[ext] >= h:
                pts.append(ext); direction, ext = 1, i
    pts.append(ext)
    return np.array(pts)


def rapid_eye_movements(eog_right, eog_left, fs, tune=DetectorTuning(), manual=ManualCriteria()):
    """Find rapid eye movements on R - L. Returns (events, info).

    A swing of R - L (turning point to turning point) is a REM when:
      1. amplitude >= tune.rem_min_amp_sigma x the night's noise level of R - L
      2. conjugate: R and L move in opposite directions, and the smaller of the two moves
         is >= tune.rem_min_conj_ratio x the larger (both eyes take part)
      3. initial deflection (10% of the swing to its peak) < manual.rem_max_rise_s
    Every REM keeps its amplitude, rise time and R / L changes, whatever the rules later decide.
    """
    # Smooth (keeps eye movements, removes EEG/EMG detail), then thin to ~100 Hz for speed.
    sos = ss.butter(2, tune.eye_lowpass_hz, "lowpass", fs=fs, output="sos")
    step = max(1, int(fs // 100))
    r = ss.sosfiltfilt(sos, eog_right)[::step]
    l = ss.sosfiltfilt(sos, eog_left)[::step]
    ds_freq = fs / step
    eog_diff = r - l

    # Noise level of R - L over the session: median absolute deviation, scaled to a standard
    # deviation. Robust to the minority of time spent in eye movements.
    sigma = float(np.median(np.abs(eog_diff - np.median(eog_diff))) / 0.6745)
    min_amp = tune.rem_min_amp_sigma * sigma

    tp = _zigzag(eog_diff, tune.eye_hysteresis_sigma * sigma)
    events = []
    for a, b in zip(tp[:-1], tp[1:]):
        amp = eog_diff[b] - eog_diff[a]
        if abs(amp) < min_amp:
            continue
        seg = eog_diff[a:b + 1]
        start = a + int(np.argmax((seg - (eog_diff[a] + 0.1 * amp)) * np.sign(amp) >= 0))   # 10% point
        rise = (b - start) / ds_freq
        if rise >= manual.rem_max_rise_s:
            continue
        d_r, d_l = r[b] - r[start], l[b] - l[start]
        if np.sign(d_r) == np.sign(d_l):
            continue                                                   # same direction: not an eye movement
        ratio = min(abs(d_r), abs(d_l)) / max(abs(d_r), abs(d_l))
        if ratio < tune.rem_min_conj_ratio:
            continue                                                   # only one channel moved
        events.append(Event("REM", start / ds_freq, rise, {
            "amp_uv": round(float(abs(amp)), 1),
            "rise_ms": round(rise * 1000),
            "right_uv": round(float(d_r), 1),
            "left_uv": round(float(d_l), 1),
            "conj_ratio": round(float(ratio), 2),
        }))
    info = {"sigma_uv": round(sigma, 1), "min_amp_uv": round(min_amp, 1)}
    return events, info


def chin_emg_level(emg, fs, epoch_s: float):
    """Chin EMG level per epoch, relative to the whole night.
    "low" chin EMG is low relative to the rest of the
    recording (usually its lowest level). So each epoch gets:
      rms_uv: root mean square of the filtered EMG over the epoch
      pct:    percentile of that RMS among all epochs of the night (0 = quietest epoch)
    The last epoch may be shorter than epoch_s; its RMS uses the samples it has.
    """
    n = int(round(epoch_s * fs))
    n_epochs = int(np.ceil(len(emg) / n))
    rms = np.array([np.sqrt(np.mean(emg[i * n:(i + 1) * n] ** 2)) for i in range(n_epochs)])
    rank = np.argsort(np.argsort(rms))                 # 0 = lowest RMS
    pct = 100.0 * rank / max(n_epochs - 1, 1)
    return rms, pct

#other detectors (EEG / proxy, arousals, k2, splindles) ...
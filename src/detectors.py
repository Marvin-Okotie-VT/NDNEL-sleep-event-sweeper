"""Event detectors. Each returns Event objects that carry their measurements and metadata. 

Sections: eye movements (REM) | chin EMG | EEG / proxy | arousals
"""
from dataclasses import dataclass, field
import numpy as np
from scipy import signal as ss
from numpy.lib.stride_tricks import sliding_window_view

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

#EEG (frontal SEEG or scalp)

def _night_pct(v: np.ndarray) -> np.ndarray:
    """Percentile of each value among the night's values (0 = lowest, 100 = highest). NaN stays NaN."""
    out = np.full(len(v), np.nan)
    ok = ~np.isnan(v)
    n = int(ok.sum())
    if n:
        rank = np.argsort(np.argsort(v[ok]))
        out[ok] = 100.0 * rank / max(n - 1, 1)
    return out


@dataclass
class LamfResult:
    slot_s: float               # slot j covers j*slot_s .. (j+1)*slot_s from recording start
    rms_uv: np.ndarray          # per slot: RMS of the total band (from the spectrum)
    entropy: np.ndarray         # per slot: spectral entropy, 0 (one bin) .. 1 (flat)
    theta_delta: np.ndarray     # per slot: theta power / delta power
    rms_pct: np.ndarray         # per slot: night percentiles of the three features
    entropy_pct: np.ndarray
    theta_delta_pct: np.ndarray
    is_lamf: np.ndarray         # per slot: all three criteria met (False where no data)
    n_lamf: np.ndarray          # per epoch: LAMF slots
    n_valid: np.ndarray         # per epoch: slots with data
    lamf: np.ndarray            # per epoch: n_lamf > majority x n_valid


def lamf_measure(x, fs, epoch_s: float, tune=DetectorTuning(), manual=ManualCriteria()) -> LamfResult:
    """Low amplitude, mixed frequency activity on one filtered EEG channel (uV).
      low amplitude:    rms_uv      = sqrt(sum P df) over the total band 
      mixed frequency:   entropy     = spectral entropy over the total band 
      predominantly 4-7 Hz:     theta_delta = 4-7 Hz power / 0.5-2 Hz power
    A slot is LAMF when all three pass their percentile limits; an epoch has LAMF when
    more than half of its slots do.
    """
    per_epoch = int(round(epoch_s / tune.lamf_slot_s))
    if abs(per_epoch * tune.lamf_slot_s - epoch_s) > 1e-9:
        raise ValueError(f"epoch_s ({epoch_s}) is not a whole number of slots ({tune.lamf_slot_s} s)")
    x = np.asarray(x, dtype=float)
    n_epochs = int(np.ceil(len(x) / int(round(epoch_s * fs))))    

    # 1. Decimate: order-8 Chebyshev low-pass at 0.8 x new Nyquist (zero phase), keep every q-th sample.
    q = max(1, int(round(fs / tune.lamf_ds_hz)))
    xd = ss.decimate(x, q, ftype="iir", zero_phase=True) if q > 1 else x
    fsd = fs / q
    n = len(xd)

    # 2. Slot grid aligned to epochs. Window = slot plus equal margins, shifted inward at the recording edges.
    n_slots = n_epochs * per_epoch
    nslot = int(round(tune.lamf_slot_s * fsd))
    nwin = int(round(tune.lamf_win_s * fsd))
    if n < nwin:
        raise ValueError(f"recording shorter than one LAMF window ({tune.lamf_win_s} s)")
    s0 = np.round(np.arange(n_slots) * tune.lamf_slot_s * fsd).astype(int)    # slot start samples
    valid = s0 + nslot // 2 <= n                                             # at least half a slot of data
    starts = np.clip(s0 - (nwin - nslot) // 2, 0, n - nwin)
    win = sliding_window_view(xd, nwin)[starts[valid]]                       # (valid slots, nwin)

    # 3. Welch per window: Hann segments of seg_s at 50% overlap, mean removed, spectra averaged (uV^2/Hz).
    nseg = int(round(tune.lamf_seg_s * fsd))
    f, P = ss.welch(win, fs=fsd, window="hann", nperseg=nseg, noverlap=nseg // 2,
                    detrend="constant", axis=-1)
    df = f[1] - f[0]

    def band(lo_hi):
        return (f >= lo_hi[0]) & (f <= lo_hi[1])

    # 4. Features from the total-band bins.
    tot = band(tune.lamf_total_band)
    p_tot = P[:, tot].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = P[:, tot] / p_tot[:, None]                                      # spectrum as a distribution
        ent = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1) / np.log(tot.sum())
        ent[p_tot == 0] = np.nan                                            # flat line: no spectrum
        td = P[:, band(manual.lamf_band)].sum(axis=1) / P[:, band(manual.slow_wave_band)].sum(axis=1)
    td[p_tot == 0] = np.nan

    def per_slot(v):
        out = np.full(n_slots, np.nan)
        out[valid] = v
        return out

    rms_uv, entropy, theta_delta = per_slot(np.sqrt(p_tot * df)), per_slot(ent), per_slot(td)

    # 5. Rank against the night.  6. Slot rule (NaN compares False).
    rms_pct, entropy_pct, td_pct = _night_pct(rms_uv), _night_pct(entropy), _night_pct(theta_delta)
    is_lamf = ((rms_pct <= tune.lamf_rms_max_pct) & (entropy_pct >= tune.lamf_entropy_min_pct)
               & (td_pct >= tune.lamf_theta_delta_min_pct))

    # 7. Epoch count: more than half of the epoch's slots with data.
    n_lamf = is_lamf.reshape(n_epochs, per_epoch).sum(axis=1)
    n_valid = valid.reshape(n_epochs, per_epoch).sum(axis=1)
    lamf = n_lamf > manual.majority * n_valid

    return LamfResult(tune.lamf_slot_s, rms_uv, entropy, theta_delta, rms_pct, entropy_pct, td_pct,
                      is_lamf, n_lamf, n_valid, lamf)

#other detectors (EEG / proxy, arousals, k2, splindles) ...
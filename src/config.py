"""Channel maps and filter settings."""
from dataclasses import dataclass, field
from typing import Optional, Union

Sig = Union[int, str]           # EDF signal index or exact label


@dataclass
class Derivation:
    plus: Sig
    minus: Optional[Sig] = None  # None = signal is already referenced


@dataclass
class ChannelMap:
    name: str
    eog_right: Derivation        # right upper eye (E2 / RUE)
    eog_left: Derivation         # left lower eye  (E1 / LLE)
    chin_emg: Derivation
    # SEEG frontal contacts (superior frontal gyrus)
    eeg_frontal_right: Optional[Derivation] = None
    eeg_frontal_left: Optional[Derivation] = None
    #scalp EEG - check if present in the recording before using
    eeg_frontal: Optional[Derivation] = None
    eeg_central: Optional[Derivation] = None
    eeg_occipital: Optional[Derivation] = None
    # Plot scale per row: half-range in uV (row spans -h..+h around its baseline).
    display_uv: dict = field(default_factory=dict)


# Sample recording
SAMPLE_PSG = ChannelMap(
    name="sample_psg",
    eog_right=Derivation("E2 M1"),
    eog_left=Derivation("E1 M2"),
    chin_emg=Derivation("EMG sm"),
    eeg_frontal=Derivation("F4 M1"),
    eeg_central=Derivation("C4 M1"),
    eeg_occipital=Derivation("O2 M1"),
    display_uv={"eog_right": 150, "eog_left": 150, "eye_diff": 250, "eye_sum": 100,
                "eeg_frontal": 100, "eeg_central": 100, "eeg_occipital": 100, "chin_emg": 20},
)

# SEEG recordings
T03_SEEG = ChannelMap(
    name="t03_seeg",
    eog_right=Derivation("POL RUE", "EEG A2-Ref"),
    eog_left=Derivation("POL LLE", "EEG A2-Ref"),
    chin_emg=Derivation(214, 215),   # EDFbrowser #215 - #216, both "POL EMG", pyedflib indexes from 0 so 214 = #215
    eeg_frontal_right=Derivation("POL RSFG9", "EEG A2-Ref"),
    eeg_frontal_left=Derivation("POL LSFG1", "EEG A2-Ref"),
    display_uv={"eog_right": 30, "eog_left": 30, "eye_diff": 60, "eye_sum": 30,
                "eeg_frontal_right": 100, "eeg_frontal_left": 100, "chin_emg": 21},
)

EEG_EOG_BAND = (0.2, 35.0)    
EMG_BAND = (10.0, 100.0)       

@dataclass
class FilterSettings:
    
    notch_hz: float = 60.0
    notch_q: float = 20.0                     # EDFbrowser default Q-factor
    order: int = 2                            # Butterworth order
    bands: dict = field(default_factory=lambda: {       #`bands` gives every role its band-pass. 
        "eog_right": EEG_EOG_BAND,
        "eog_left": EEG_EOG_BAND,
        "eeg_frontal_right": EEG_EOG_BAND,
        "eeg_frontal_left": EEG_EOG_BAND,
        "eeg_frontal": EEG_EOG_BAND,
        "eeg_central": EEG_EOG_BAND,
        "eeg_occipital": EEG_EOG_BAND,
        "chin_emg": EMG_BAND,
    })

#scoring settings
@dataclass
class RemRules:
    # Default (True) = AASM manual; False = binder flow chart.
    # Epoch with REMs and low chin EMG is R even if it has a K-complex or spindle 
    # False: a K-complex/spindle makes it N1-3 not R.
    rem_beats_k_spindle: bool = True
    # Look-back stops at the first epoch that fails the continuation conditions
    # (LAMF, low chin EMG, no K-complex/spindle) 
    # False: look-back stops only at a K-complex/spindle.
    lookback_stops_at_continuation_fail: bool = True
    # An arousal ends R only when slow eye movements follow it 
    # False: any arousal ends R.
    arousal_needs_sem_to_end_r: bool = True


@dataclass
class ManualCriteria:
    """Values defined by the AASM manual(p.25-37)"""
    rem_max_rise_s: float = 0.5            # REM initial deflection usually < 500 ms ; SEM > 500 ms 
    majority: float = 0.5                  # "more than 50% of the epoch" (alpha -> W , LAMF -> N1)
    first_half_s: float = 15.0             # K-complex/spindle in the first half ends R 
    alpha_band: tuple = (8.0, 13.0)        # alpha rhythm 
    lamf_band: tuple = (4.0, 7.0)          # LAMF: low amplitude, predominantly 4-7 Hz 
    slow_wave_band: tuple = (0.5, 2.0)     # slow wave activity 
    slow_wave_min_ptp_uv: float = 75.0     # slow waves > 75 uV peak-to-peak 
    n3_min_s: float = 6.0                  # slow waves in >= 20% of the epoch = 6 s 
    spindle_band: tuple = (11.0, 16.0)     # sleep spindle 
    spindle_min_s: float = 0.5             # spindle duration >= 0.5 s 
    kcomplex_min_s: float = 0.5            # K-complex total duration >= 0.5 s 
    kcomplex_arousal_gap_s: float = 1.0    # K-complex is arousal-associated if the arousal starts <= 1 s after it 
    arousal_min_s: float = 3.0             # EEG frequency shift lasting >= 3 s 
    arousal_stable_sleep_s: float = 10.0   # preceded by >= 10 s of stable sleep 
    arousal_fast_hz: float = 16.0          # shift to alpha, theta and/or > 16 Hz, not spindles 
    arousal_emg_min_s: float = 1.0         # arousal in R also needs a chin EMG increase >= 1 s 
    movement_min_fraction: float = 0.5     # major body movement obscures the EEG for > half the epoch 

@dataclass
class DetectorTuning:
    """Detection settings that can be adjusted to the sessions's noise level or the subject's physiology."""
    eye_lowpass_hz: float = 8.0            # smoothing before finding eye movements
    eye_hysteresis_sigma: float = 2.0      # a swing only reverses after moving back >= 2 x noise
    rem_min_amp_sigma: float = 4.5         # REM size on R - L >= 4.5 x noise 
    rem_min_conj_ratio: float = 0.3        # smaller eye change >= 30% of the larger (both eyes move)
    emg_low_pct: float = 25.0              # chin EMG "low": epoch RMS at or below the night's 25th percentile
    emg_unsure_pct: float = 35.0           # 25th-35th percentile: borderline -> R?


EPOCH_S = 30.0
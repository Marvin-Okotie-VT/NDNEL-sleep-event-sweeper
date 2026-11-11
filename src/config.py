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
    display_uv={"eog_right": 30, "eog_left": 30, "eye_diff": 60, "eye_sum": 30, "chin_emg": 21},
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
        "eeg_frontal": EEG_EOG_BAND,
        "eeg_central": EEG_EOG_BAND,
        "eeg_occipital": EEG_EOG_BAND,
        "chin_emg": EMG_BAND,
    })


EPOCH_S = 30.0
"""Epoch check plot: stacked traces on a 1-s grid, similar to EDFbrowser page.

All rows share one canvas. Each signal is drawn around its own baseline at a
fixed scale (ChannelMap.display_uv): +/-h uV fills one row. A deflection larger
than that is not clipped; it runs into the neighbouring rows, as in EDFbrowser.
"""
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt

INK, MUTED, GRID, ACCENT, GAP = "#1f2328", "#6e7781", "#d0d7de", "#0969da", "#eaeef2"

# role -> (row title, derived from the eye channels?)
ROWS = [
    ("eog_right", "Right eye (E2 / RUE)", False),
    ("eog_left", "Left eye (E1 / LLE)", False),
    ("eye_diff", "R - L  (eye movements add)", True),
    ("eye_sum", "(R + L)/2  (frontal EEG proxy)", True),
    ("eeg_frontal_right", "Right superior frontal (SEEG)", False),
    ("eeg_frontal_left", "Left superior frontal (SEEG)", False),
    ("eeg_frontal", "Frontal EEG", False),
    ("eeg_central", "Central EEG", False),
    ("eeg_occipital", "Occipital EEG", False),
    ("chin_emg", "Chin EMG", False),
]


def epoch_plot(sig: dict, labels: dict, fs: float, start, epoch_no: int,
               title_note: str, path: str, display_uv: dict, epoch_s: float = 30.0):
    n_total = len(next(iter(sig.values())))
    i0 = int(round((epoch_no - 1) * epoch_s * fs))
    if epoch_no < 1 or i0 >= n_total:
        last = int(np.ceil(n_total / (epoch_s * fs)))
        raise ValueError(f"epoch {epoch_no} is outside the recording (epochs 1-{last})")
    i1 = min(int(round(i0 + epoch_s * fs)), n_total)       # last epoch may be partial
    t = np.arange(i1 - i0) / fs

    rows = [r for r in ROWS if r[0] in sig]
    missing = [r[0] for r in rows if r[0] not in display_uv]
    if missing:
        raise KeyError(f"no plot scale for {missing} (ChannelMap.display_uv)")

    fig, ax = plt.subplots(figsize=(13, 1.15 * len(rows) + 0.9))
    for k, (role, name, derived) in enumerate(rows):
        base, h = -k, display_uv[role]                      # row k centred at y = -k
        ax.axhline(base, color=GRID, lw=0.6)
        for edge in (base + 0.5, base - 0.5):               # row edges = +/-h uV
            ax.axhline(edge, color=GRID, lw=0.4, ls=":")
        ax.plot(t, base + 0.5 * sig[role][i0:i1] / h, lw=0.8,
                color=ACCENT if derived else INK)
        sub = labels.get(role, "derived from the two eye channels")
        ax.text(-0.012, base, f"{name}\n{sub}\n±{h:g} µV per row",
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8, color=INK)

    if i1 - i0 < epoch_s * fs:                               # shade the part past the end
        ax.axvspan(t[-1], epoch_s, color=GAP, zorder=0)
        ax.text((t[-1] + epoch_s) / 2, 0.5, "end of recording", ha="center",
                va="center", fontsize=8, color=MUTED)

    ax.set_xlim(0, epoch_s)
    ax.set_ylim(-len(rows) + 0.5 - 0.5, 0.5 + 0.5)          # half row of headroom top/bottom
    ax.set_xticks(np.arange(0, epoch_s + 1, 1))
    ax.set_xticklabels([str(int(x)) if x % 5 == 0 else "" for x in np.arange(0, epoch_s + 1, 1)])
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_yticks([])
    ax.tick_params(labelsize=7, colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("seconds into epoch (grid = 1 s)", fontsize=8, color=MUTED)

    clock = start + timedelta(seconds=(epoch_no - 1) * epoch_s)
    el = timedelta(seconds=int((epoch_no - 1) * epoch_s))
    fig.suptitle(f"Epoch {epoch_no}  |  {clock:%H:%M:%S}  |  elapsed {el}  |  {title_note}",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.2, right=0.99, top=0.93, bottom=0.07)
    fig.savefig(path, dpi=110)
    plt.close(fig)
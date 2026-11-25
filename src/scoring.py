"""Per-epoch scoring.

Milestone rule (REM + low chin EMG only). LAMF, K-complex/spindle checks, continuation
and look-back are not implemented yet, so epochs of R without REMs are not scored.
"""
from dataclasses import dataclass, field

from .config import DetectorTuning


@dataclass
class EpochResult:
    epoch: int                 # 1-based, same numbering as the Excel rows
    start_s: float             # seconds from recording start
    rems: list = field(default_factory=list)   # REM events whose onset falls in this epoch
    emg_rms_uv: float = 0.0
    emg_pct: float = 0.0
    score: str = ""
    unsure: str = ""
    notes: str = ""


def _rem_text(rems, epoch_start_s, max_listed=5):
    parts = [f"{e.onset_s - epoch_start_s:.1f}s ({e.values['amp_uv']:.0f}uV, {e.values['rise_ms']}ms)"
             for e in rems[:max_listed]]
    more = f" +{len(rems) - max_listed} more" if len(rems) > max_listed else ""
    return f"{len(rems)} REM: " + ", ".join(parts) + more


def score_rem_milestone(rem_events, emg_rms, emg_pct, epoch_s, tune=DetectorTuning()):
    """One result per epoch.

    Score "R"         : >= 1 REM and chin EMG <= tune.emg_low_pct percentile
    Unsure_Score "R?" : >= 1 REM and chin EMG between emg_low_pct and emg_unsure_pct
    blank             : everything else. Epochs with REMs but higher chin EMG get a note
                        saying why they are not R (REMs with normal/high chin EMG = W).
    """
    results = [EpochResult(i + 1, i * epoch_s, emg_rms_uv=float(emg_rms[i]), emg_pct=float(emg_pct[i]))
               for i in range(len(emg_rms))]
    
    for ev in rem_events:
        i = int(ev.onset_s // epoch_s)
        if i < len(results):
            results[i].rems.append(ev)

    for r in results:
        if not r.rems:
            continue
        emg = f"chin EMG {r.emg_rms_uv:.2f}uV ({r.emg_pct:.0f}th pct)"
        rem = _rem_text(r.rems, r.start_s)
        if r.emg_pct <= tune.emg_low_pct:
            r.score = "R"
            r.notes = f"R: {rem} | {emg}, low"
        elif r.emg_pct <= tune.emg_unsure_pct:
            r.unsure = "R?"
            r.notes = f"R?: {rem} | {emg}, borderline"
        else:
            r.notes = f"not R: {rem} | {emg}, not low"
    return results
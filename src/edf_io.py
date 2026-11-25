"""EDF loading: reads the signals specified in config.py, and returns a Recording object with the signals and metadata.

Single-pass reader: the file is read front to back in large blocks (default 32 MB)
"""
from dataclasses import dataclass, fields
from datetime import datetime
import os
import numpy as np
import pyedflib

from .config import ChannelMap, Derivation

BLOCK_BYTES = 32 * 2**20          # bytes read per call; memory per block is about this much


@dataclass
class Recording:
    start: datetime
    duration_s: float
    fs: float
    signals: dict            # 1-D float array (uV), all at fs
    labels: dict             # derivations, e.g. "POL RUE - EEG A2-Ref"


@dataclass
class _Layout:
    """Where each signal lives inside a data record, from the raw EDF header."""
    header_bytes: int        # data starts here
    n_records: int
    record_samples: int      # samples per record, all signals together (2 bytes each)
    offset: list             # per raw signal: first sample of its block inside a record
    spr: list                # per raw signal: samples per record
    bitvalue: list           # per raw signal: uV per digital step
    dig_offset: list         # per raw signal: uV = bitvalue * (digital + dig_offset)
    raw_index: list          # pyedflib / EDFbrowser index -> raw header index


def _layout(path: str) -> _Layout:
    """Parse the fixed-width ASCII header. Unlike pyedflib's signal list, this
    includes EDF+ annotation signals: they are hidden from the user but still
    occupy space in every data record, so the offsets must count them."""
    with open(path, "rb") as fh:
        head = fh.read(256)
        ns = int(head[252:256])
        sig = fh.read(256 * ns)

    def field(start, width):                       # one fixed-width column, ns entries
        block = sig[start * ns:(start + width) * ns]
        return [block[i * width:(i + 1) * width].decode("latin-1").strip() for i in range(ns)]

    # signal header columns, in order: label 16, transducer 80, dimension 8,
    # phys min 8, phys max 8, dig min 8, dig max 8, prefilter 80, samples/record 8, reserved 32
    labels = field(0, 16)
    pmin = [float(v) for v in field(104, 8)]
    pmax = [float(v) for v in field(112, 8)]
    dmin = [int(v) for v in field(120, 8)]
    dmax = [int(v) for v in field(128, 8)]
    spr = [int(v) for v in field(216, 8)]

    header_bytes = 256 * (ns + 1)
    record_samples = sum(spr)
    n_records = int(head[236:244])
    if n_records < 0:                               # -1 = "unknown", allowed while recording
        n_records = (os.path.getsize(path) - header_bytes) // (2 * record_samples)

    # Same scaling as edflib (what pyedflib uses), so values match it exactly.
    bitvalue = [(pmax[i] - pmin[i]) / (dmax[i] - dmin[i]) for i in range(ns)]
    dig_offset = [pmax[i] / bitvalue[i] - dmax[i] for i in range(ns)]

    offset = list(np.cumsum([0] + spr[:-1]))
    raw_index = [i for i, l in enumerate(labels) if l not in ("EDF Annotations", "BDF Annotations")]
    return _Layout(header_bytes, n_records, record_samples, offset, spr,
                   bitvalue, dig_offset, raw_index)


def _read_pass(path: str, lay: _Layout, wanted: list) -> dict:
    """One front-to-back pass. wanted = pyedflib indices. Returns index -> uV array."""
    raw = {i: lay.raw_index[i] for i in wanted}
    record_bytes = 2 * lay.record_samples
    per_block = max(1, BLOCK_BYTES // record_bytes)            # whole records per read
    parts = {i: [] for i in wanted}
    done, last_pct = 0, -10
    with open(path, "rb") as fh:
        fh.seek(lay.header_bytes)
        while done < lay.n_records:
            k = min(per_block, lay.n_records - done)
            buf = fh.read(k * record_bytes)
            k = len(buf) // record_bytes                         # a truncated file ends early
            if k == 0:
                break
            # rows = records, columns = every sample of every signal in that record
            grid = np.frombuffer(buf[:k * record_bytes], dtype="<i2").reshape(k, lay.record_samples)
            for i, r in raw.items():
                o = lay.offset[r]
                parts[i].append(grid[:, o:o + lay.spr[r]].reshape(-1))   # this signal's blocks, end to end
            done += k
            pct = 100 * done // lay.n_records
            if pct >= last_pct + 10:
                print(f"  reading {os.path.basename(path)}: {pct}%", flush=True)
                last_pct = pct
    out = {}
    for i, r in raw.items():
        d = np.concatenate(parts[i]).astype(np.float64)
        out[i] = lay.bitvalue[r] * (d + lay.dig_offset[r])      # digital -> uV
    return out


def _resolve(reader: pyedflib.EdfReader, sig) -> int:
    if isinstance(sig, int):
        return sig
    labels = reader.getSignalLabels()
    hits = [i for i, l in enumerate(labels) if l.strip() == sig]
    if len(hits) != 1:
        raise ValueError(f"label {sig!r} matched {len(hits)} signals; use an index")
    return hits[0]


def load(path: str, cmap: ChannelMap) -> Recording:
    # pyedflib is used for the header only (labels, rates, start time); samples come from _read_pass.
    reader = pyedflib.EdfReader(path, annotations_mode=0)   # skip EDF+ annotations
    try:
        # every ChannelMap field holding a Derivation is a role to load (unset optional roles are None)
        roles = {f.name: getattr(cmap, f.name) for f in fields(cmap)
                 if isinstance(getattr(cmap, f.name), Derivation)}
        picks = {}                                   # role -> (plus index, minus index or None)
        for role, d in roles.items():
            ip = _resolve(reader, d.plus)  #index of the plus signal
            im = _resolve(reader, d.minus) if d.minus is not None else None
            picks[role] = (ip, im)
        wanted = sorted({i for p in picks.values() for i in p if i is not None})
        n_sig = reader.signals_in_file
        bad = [i for i in wanted if not 0 <= i < n_sig]
        if bad:
            raise ValueError(f"signal index {bad} outside this file (0-{n_sig - 1}, "
                             f"EDFbrowser #1-#{n_sig})")
        fs_of = {i: reader.getSampleFrequency(i) for i in wanted}
        name_of = {i: reader.getLabel(i).strip() for i in wanted}
        start, duration = reader.getStartdatetime(), reader.getFileDuration()
    finally:
        reader.close()

    data = _read_pass(path, _layout(path), wanted)

    signals, labels, rates = {}, {}, set()
    for role, (ip, im) in picks.items():
        d = roles[role]
        lp = name_of[ip]
        if isinstance(d.plus, int):        # picked by index: show EDFbrowser number (1-based)
            lp = f"{lp} (#{ip + 1})"
        x, label = data[ip], lp
        if im is not None:
            lm = name_of[im]
            if isinstance(d.minus, int):
                lm = f"{lm} (#{im + 1})"
            if fs_of[im] != fs_of[ip]:
                raise ValueError(f"{role}: sample rates differ ({fs_of[ip]} vs {fs_of[im]})")
            x = x - data[im]                      # drop common reference
            label = f"{lp} - {lm}"
        signals[role] = x
        labels[role] = label
        rates.add(fs_of[ip])
    if len(rates) != 1:
        raise ValueError(f"mixed sample rates {rates}; resampling not implemented yet")
    return Recording(start, duration, rates.pop(), signals, labels)
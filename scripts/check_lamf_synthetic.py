"""Checks for lamf_measure using synthetic signals for verification.  
python scripts/check_lamf_synthetic.py
"""
import time
import numpy as np
from scipy import signal as ss
from sleep_event_sweeper.detectors import lamf_measure
from sleep_event_sweeper.config import EPOCH_S

FS = 999.9999999999999          # the T03 header value: q must round to 10
rng = np.random.default_rng(0)


def pink(n, sd):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    X[1:] /= np.sqrt(f[1:])
    X[0] = 0
    x = np.fft.irfft(X, n)
    return sd * x / x.std()


def bandnoise(n, lo, hi, sd):
    x = ss.sosfiltfilt(ss.butter(4, [lo, hi], "bandpass", fs=FS, output="sos"), rng.standard_normal(n))
    return sd * x / x.std()


def sine(n, hz, amp):
    return amp * np.sin(2 * np.pi * hz * np.arange(n) / FS)


def prefilter(x):                               # what apply_filters does: 0.2-35 Hz, order 2, zero phase
    return ss.sosfiltfilt(ss.butter(2, [0.2, 35.0], "bandpass", fs=FS, output="sos"), x)


ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# 1. RMS from the spectrum equals the band's true RMS (Parseval), decimation q = 10
print("1. RMS via Parseval, white noise sd 10 uV at fs 999.99...")
n = int(20 * 60 * FS)
r = lamf_measure(rng.standard_normal(n) * 10, FS, EPOCH_S)
expected = 10 * np.sqrt(30.0 / (FS / 2))       # 60 bins x 0.5 Hz = 30 Hz of a flat 0-500 Hz spectrum
med = np.nanmedian(r.rms_uv)
print(f"   median rms {med:.3f} uV, expected {expected:.3f} uV")
check("within 5%", abs(med / expected - 1) < 0.05)

# 2. Each feature moves the right way (raw values, not percentiles)
print("2. Feature direction on pure signals (medians over 10 min)")
n = int(10 * 60 * FS)
cases = {
    "white noise":         rng.standard_normal(n) * 20,
    "10 Hz sine + noise":  sine(n, 10, 50) + rng.standard_normal(n) * 2,
    "1 Hz sine + pink":    sine(n, 1, 100) + pink(n, 10),
    "pink + 4-7 Hz noise": pink(n, 10) + bandnoise(n, 4, 7, 15),
    "pink only":           pink(n, 10),
}
med = {}
for k, x in cases.items():
    rr = lamf_measure(prefilter(x), FS, EPOCH_S)
    med[k] = (np.nanmedian(rr.entropy), np.nanmedian(rr.theta_delta))
    print(f"   {k:22s} entropy {med[k][0]:.3f}  theta/delta {med[k][1]:7.3f}")
check("white noise entropy > 0.9", med["white noise"][0] > 0.9)
check("10 Hz sine entropy < 0.3", med["10 Hz sine + noise"][0] < 0.3)
check("10 Hz sine entropy < white noise", med["10 Hz sine + noise"][0] < med["white noise"][0])
check("1 Hz wave: lowest theta/delta", med["1 Hz sine + pink"][1] == min(v[1] for v in med.values()))
check("4-7 Hz mix: theta/delta > pink only", med["pink + 4-7 Hz noise"][1] > med["pink only"][1])

# 3. End-to-end night: N3-like | LAMF | alpha | spindles, 10 epochs each, + artifact + 9 ms tail
print("3. End-to-end synthetic night (40 epochs + 9 ms partial epoch)")
ne = int(round(EPOCH_S * FS))
blocks = [
    sine(10 * ne, 1, 150) + pink(10 * ne, 20),                        # epochs 1-10: slow waves
    pink(10 * ne, 12) + bandnoise(10 * ne, 4, 7, 10),                  # epochs 11-20: LAMF
    sine(10 * ne, 10, 60) + pink(10 * ne, 15),                         # epochs 21-30: alpha
    pink(10 * ne, 30),                                                 # epochs 31-40: spindles (below)
]
t = np.arange(10 * ne) / FS
blocks[3] += 50 * np.sin(2 * np.pi * 12 * t) * ((t % 5) < 1.0)         # 1-s 12 Hz burst every 5 s
x = np.concatenate(blocks + [pink(9, 12)])                              # 9 samples = 9 ms tail
a = 15 * ne + int(10 * FS)
x[a:a + int(FS)] += 600 * np.sign(np.sin(2 * np.pi * 3 * np.arange(int(FS)) / FS))   # 1-s artifact, epoch 16
t0 = time.perf_counter()
r = lamf_measure(prefilter(x), FS, EPOCH_S)
print(f"   epochs {len(r.lamf)}, n_valid of last two: {r.n_valid[-2:]}")
for lo, hi, name in [(0, 10, "slow waves"), (10, 20, "LAMF"), (20, 30, "alpha"), (30, 40, "spindles")]:
    print(f"   epochs {lo + 1:2d}-{hi:2d} {name:10s} n_lamf {r.n_lamf[lo:hi]}  lamf {r.lamf[lo:hi].sum()}/10")
print(f"   epoch 16 (1-s artifact): n_lamf {r.n_lamf[15]}/15, lamf {r.lamf[15]}")
check("41 epochs (40 + partial)", len(r.lamf) == 41)
check("partial 9 ms epoch: 0 valid slots, not LAMF", r.n_valid[-1] == 0 and not r.lamf[-1])
check("all LAMF epochs detected", r.lamf[10:20].all())
check("no slow-wave or alpha epoch called LAMF", not r.lamf[:10].any() and not r.lamf[20:30].any())
# Spindle block: the pink background between bursts IS LAMF-like (N2 background is LAMF per the handbook);
# spindles are for the spindle detector. LAMF's job: windows that contain a burst should fail more often.
j = np.arange(30 * 15, 40 * 15)                          # slots of epochs 31-40
w0 = j * 2.0 - 1.0 - 30 * EPOCH_S                        # window start (s) relative to the block
has_burst = np.array([any((b < w + 4) and (b + 1 > w) for b in np.arange(0, 300, 5)) for w in w0])
rate_b, rate_n = r.is_lamf[j][has_burst].mean(), r.is_lamf[j][~has_burst].mean()
print(f"   spindle block: LAMF rate, windows with a burst {rate_b:.2f} ({has_burst.sum()}), without {rate_n:.2f} ({(~has_burst).sum()})")
check("windows with a spindle burst rejected more often than burst-free ones", rate_b < rate_n)
check("artifact epoch still LAMF", r.lamf[15])

# 4. Timing stuff
print("4. Runtime, 8 h at fs 999.99...")
x = rng.standard_normal(int(8 * 3600 * FS)).astype(float)
t0 = time.perf_counter()
r = lamf_measure(x, FS, EPOCH_S)
print(f"   {time.perf_counter() - t0:.1f} s for {len(r.lamf)} epochs / {len(r.rms_uv)} slots")

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
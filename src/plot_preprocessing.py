"""
Sinal CSI antes vs depois do pré-processamento — estilo linha limpa.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter

NPZ_FILE   = "Data_DS2_raspberry_npz/001/01_2023_10_30_-_12_20_05_bw_80_ch_36.npz"
OUT_FILE   = "charts_output/preprocessing_before_after.png"
SUBCARRIER = 128
N_SUB_MEAN = 10
BANDPASS   = (0.8, 2.17)
TRIM_S     = 10

# ── Carregamento ──────────────────────────────────────────────────────────────
z = np.load(NPZ_FILE, allow_pickle=True)
csi_raw = z["csi"]
ts_raw  = z["ts"] if "ts" in z.files else None

X = np.abs(csi_raw).astype(np.float32) if np.iscomplexobj(csi_raw) else csi_raw.astype(np.float32)

if ts_raw is not None:
    ts = np.array(ts_raw, dtype=float).ravel()
    ts = ts - ts[0]
    fs = 1.0 / np.mean(np.diff(ts))
else:
    fs = 500 / 60.0
    ts = np.arange(X.shape[0]) / fs

# ── Pipeline completo ─────────────────────────────────────────────────────────
X_proc = X - np.mean(X, axis=0)

b, a = butter(3, BANDPASS, btype="band", fs=fs)
X_proc = np.column_stack([filtfilt(b, a, X_proc[:, i]) for i in range(X_proc.shape[1])])

X_proc = np.column_stack([savgol_filter(X_proc[:, i], 15, 3) for i in range(X_proc.shape[1])])

cut = int(TRIM_S * fs)
X_proc = X_proc[cut:]
ts_proc = ts[cut:] - ts[cut]
ts_raw_trim = ts[cut:] - ts[cut]
X_raw_trim  = X[cut:]

# ── Sinal de referência: média das subportadoras centrais ─────────────────────
lo = max(0, SUBCARRIER - N_SUB_MEAN // 2)
hi = min(X.shape[1], SUBCARRIER + N_SUB_MEAN // 2)

sig_before = X_raw_trim[:, lo:hi].mean(axis=1)
sig_after  = X_proc[:, lo:hi].mean(axis=1)

# Z-score para colocar na mesma escala visual
def norm(s):
    return (s - s.mean()) / (s.std() + 1e-8)

sig_before_n = norm(sig_before)
sig_after_n  = norm(sig_after)

# ── Figura ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
fig.patch.set_facecolor("white")
fig.subplots_adjust(hspace=0.35, left=0.06, right=0.98, top=0.88, bottom=0.12)

pairs = [
    (ts_raw_trim, sig_before_n, "#1f77b4", "Raw signal  (CSI amplitude, normalized)"),
    (ts_proc,     sig_after_n,  "#d62728", "Preprocessed signal  (DC removal + bandpass + Savitzky-Golay, normalized)"),
]

for ax, (t, s, color, label) in zip(axes, pairs):
    ax.plot(t, s, color=color, lw=1.0, label=label)
    ax.set_facecolor("white")
    ax.grid(True, color="#e0e0e0", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_ylabel("Amplitude (a.u.)", fontsize=9)
    ax.tick_params(labelsize=8.5, colors="#444444")
    ax.set_xlim(t[0], t[-1])

axes[1].set_xlabel("Time (s)", fontsize=9.5)

Path("charts_output").mkdir(exist_ok=True)
plt.savefig(OUT_FILE, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Salvo: {OUT_FILE}")
plt.show()

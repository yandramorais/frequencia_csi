"""
compare_models.py
Generates publication-quality comparison charts: GRU vs LSTM.
Also produces improved individual GRU charts.

Usage
-----
  python src/compare_models.py
  python src/compare_models.py --out results/figures
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import uniform_filter1d
from scipy.stats import norm as sp_norm
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "saida_full"
GRU_DIR  = ROOT / "output" / "gru"
LSTM_DIR = ROOT / "output" / "lstm"

# ─── Style ────────────────────────────────────────────────────────────────────
C_GRU    = "#6B0000"       # dark red — GRU
C_GRU_L  = "#D4919B"       # light rose — GRU secondary
C_LSTM   = "#1565C0"       # dark blue — LSTM
C_LSTM_L = "#90CAF9"       # light blue — LSTM secondary
ACCENT   = "#A23B72"
BG_FIG   = "#F2F2F7"
BG_AX    = "#FFFFFF"
GRID     = "#E0E0E0"

RC = {
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlepad":     12,
    "axes.labelsize":    12,
    "axes.titlesize":    14,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
}


def _save(fig, name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  [OK] {p.name}")
    plt.close(fig)


def _int_axes(ax, x=True, y=True):
    if x: ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if y: ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def _red_bar(fig, title):
    bar = fig.add_axes([0.0, 0.955, 1.0, 0.045])
    bar.set_facecolor(C_GRU)
    bar.axis("off")
    bar.text(0.5, 0.5, title, ha="center", va="center",
             fontsize=12, fontweight="bold", color="white",
             transform=bar.transAxes)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model definitions
# ═══════════════════════════════════════════════════════════════════════════════

class PulseFiGRU(nn.Module):
    def __init__(self, input_dim, hidden=256, layers=2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True,
                          dropout=0.3, bidirectional=True)
        h = hidden * 2
        self.norm = nn.LayerNorm(h)
        self.attn = nn.Linear(h, 1)
        self.regressor = nn.Sequential(
            nn.Linear(h, 128), nn.LayerNorm(128),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):
        x      = self.input_proj(x)
        out, _ = self.gru(x)
        out    = self.norm(out)
        w      = torch.softmax(self.attn(out), dim=1)
        return self.regressor((out * w).sum(1)).squeeze(-1)


class PulseFiLSTM(nn.Module):
    def __init__(self, input_dim, hidden=256, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                            dropout=0.3, bidirectional=True)
        h = hidden * 2
        self.regressor = nn.Sequential(
            nn.Linear(h, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.regressor(out[:, -1, :]).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data / inference helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_device():
    if torch.backends.mps.is_available():  return torch.device("mps")
    if torch.cuda.is_available():           return torch.device("cuda")
    return torch.device("cpu")


def run_inference(model, X_np, device, batch=64):
    dl    = DataLoader(TensorDataset(torch.from_numpy(X_np)),
                       batch_size=batch, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for (bx,) in dl:
            preds.append(model(bx.to(device)).cpu().numpy())
    return np.concatenate(preds)


def measure_inference_time(model, X_np, device, batch=64, n_runs=3):
    dl = DataLoader(TensorDataset(torch.from_numpy(X_np)),
                    batch_size=batch, shuffle=False)
    times = []
    model.eval()
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            for (bx,) in dl:
                model(bx.to(device))
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_all():
    device = get_device()

    with open(GRU_DIR  / "history_gru.json")  as f: h_gru  = json.load(f)
    with open(LSTM_DIR / "history_lstm.json") as f: h_lstm = json.load(f)

    X_val  = np.load(DATA_DIR / "X_val.npz")["X"].astype(np.float32)
    y_val  = np.load(DATA_DIR / "y_val.npy").astype(np.float32)
    pos    = np.load(DATA_DIR / "positions_val.npy")
    subj   = np.load(DATA_DIR / "subject_val.npy")

    input_dim = h_gru["input_dim"]

    model_gru  = PulseFiGRU(input_dim).to(device)
    model_lstm = PulseFiLSTM(input_dim).to(device)
    model_gru.load_state_dict(
        torch.load(GRU_DIR  / "best_model_gru.pt",  map_location=device, weights_only=True))
    model_lstm.load_state_dict(
        torch.load(LSTM_DIR / "best_model_lstm.pt", map_location=device, weights_only=True))

    print("  Running GRU  inference …")
    y_gru  = run_inference(model_gru,  X_val, device)
    print("  Running LSTM inference …")
    y_lstm = run_inference(model_lstm, X_val, device)

    print("  Measuring inference times …")
    t_gru  = measure_inference_time(model_gru,  X_val, device)
    t_lstm = measure_inference_time(model_lstm, X_val, device)

    n_gru  = count_params(model_gru)
    n_lstm = count_params(model_lstm)

    print(f"  GRU  params={n_gru:,}  |  inference={t_gru:.3f}s")
    print(f"  LSTM params={n_lstm:,}  |  inference={t_lstm:.3f}s")

    return {
        "h_gru": h_gru, "h_lstm": h_lstm,
        "y_true": y_val, "y_gru": y_gru, "y_lstm": y_lstm,
        "pos": pos, "subj": subj,
        "t_gru": t_gru, "t_lstm": t_lstm,
        "n_gru": n_gru, "n_lstm": n_lstm,
    }


def metrics(y_true, y_pred):
    err     = y_true - y_pred
    abs_err = np.abs(err)
    return {
        "mae":   float(mean_absolute_error(y_true, y_pred)),
        "rmse":  float(np.sqrt(np.mean(err ** 2))),
        "r2":    float(r2_score(y_true, y_pred)),
        "bias":  float(np.mean(err)),
        "sigma": float(np.std(err)),
        "abs":   abs_err,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPARISON CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 01. Learning Curves (side by side) ─────────────────────────────────────────
def plot_01_learning_curves(d, out_dir):
    def _series(h, color, color_l, label):
        train = np.array(h["train_loss"])
        val   = np.array(h["val_mae"])
        sm    = uniform_filter1d(val, size=15)
        best  = int(np.argmin(val))
        return train, val, sm, best, color, color_l, label

    series = [
        _series(d["h_gru"],  C_GRU,  C_GRU_L,  "GRU"),
        _series(d["h_lstm"], C_LSTM, C_LSTM_L, "LSTM"),
    ]

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharey=False)
        fig.patch.set_facecolor(BG_FIG)
        fig.suptitle("Learning Curves — GRU vs. LSTM",
                     fontsize=15, fontweight="bold", color="#1A1A2E", y=1.01)

        for ax, (train, val, sm, best, clr, clr_l, lbl) in zip(axes, series):
            ax.set_facecolor(BG_AX)
            eps = np.arange(1, len(train) + 1)
            ax.plot(eps, train, color=clr_l, lw=1.2, alpha=0.6, label="Train Loss")
            ax.plot(eps, val,   color=clr,   lw=1.4, alpha=0.5, ls="--")
            ax.plot(eps, sm,    color=clr,   lw=2.5, label="Val MAE (smoothed)")
            ax.fill_between(eps, sm, alpha=0.08, color=clr)
            ax.axvline(best + 1, color="#BDBDBD", ls=":", lw=1.3)
            ax.scatter([best + 1], [val[best]], s=90, color=clr, zorder=5)
            ax.annotate(
                f"Best ep. {best + 1}\nMAE = {val[best]:.2f}",
                xy=(best + 1, val[best]),
                xytext=(best + 1 + max(4, len(eps)//10),
                        val[best] + (val.max() - val[best]) * 0.30 + 0.3),
                fontsize=8.5, color="#333",
                arrowprops=dict(arrowstyle="->", color="#999", lw=1),
            )
            ax.set_title(f"{lbl} — Learning Curve", fontweight="bold")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss / MAE (BPM)")
            _int_axes(ax); ax.legend()
            ax.grid(True, alpha=0.4, ls="--", color=GRID)

        fig.tight_layout()
        _save(fig, "comp_01_learning_curves", out_dir)


# ── 01b. Learning Curves overlaid ──────────────────────────────────────────────
def plot_01b_learning_curves_overlay(d, out_dir):
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(17, 6))
        fig.patch.set_facecolor(BG_FIG)
        fig.suptitle("Training Loss and Validation MAE — GRU vs. LSTM",
                     fontsize=15, fontweight="bold", color="#1A1A2E", y=1.01)

        for h, clr, lbl in [(d["h_gru"], C_GRU, "GRU"), (d["h_lstm"], C_LSTM, "LSTM")]:
            train = np.array(h["train_loss"])
            val   = np.array(h["val_mae"])
            eps   = np.arange(1, len(train) + 1)
            sm    = uniform_filter1d(val, size=15)

            axes[0].plot(eps, train, lw=2.0, color=clr, label=lbl)
            axes[1].plot(eps, sm,    lw=2.5, color=clr, label=lbl)
            axes[1].fill_between(eps, sm, alpha=0.07, color=clr)
            best = int(np.argmin(val))
            axes[1].scatter([best + 1], [val[best]], s=90, color=clr, zorder=5)

        for ax, title, ylabel in zip(
            axes,
            ["Training Loss (Huber)", "Validation MAE — Smoothed (w=15)"],
            ["Huber Loss", "MAE (BPM)"],
        ):
            ax.set_facecolor(BG_AX)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
            _int_axes(ax); ax.legend()
            ax.grid(True, alpha=0.4, ls="--", color=GRID)

        fig.tight_layout()
        _save(fig, "comp_01b_learning_curves_overlay", out_dir)


# ── 02. Real vs Predicted side by side ─────────────────────────────────────────
def plot_02_scatter(d, out_dir):
    y_true = d["y_true"]
    pairs  = [("GRU", d["y_gru"], C_GRU), ("LSTM", d["y_lstm"], C_LSTM)]
    lo = int(min(y_true.min(), d["y_gru"].min(), d["y_lstm"].min())) - 3
    hi = int(max(y_true.max(), d["y_gru"].max(), d["y_lstm"].max())) + 3

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor(BG_FIG)
        fig.suptitle("Real vs. Predicted Heart Rate — GRU vs. LSTM",
                     fontsize=15, fontweight="bold", color="#1A1A2E", y=1.01)

        for ax, (lbl, y_pred, clr) in zip(axes, pairs):
            m_val = metrics(y_true, y_pred)
            ax.set_facecolor(BG_AX)
            ax.scatter(y_true, y_pred, alpha=0.18, s=10, color=clr,
                       edgecolors="none", rasterized=True)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.8, label="Identity (y=x)")
            m, b = np.polyfit(y_true, y_pred, 1)
            xs   = np.linspace(lo, hi, 300)
            ax.plot(xs, m*xs+b, color=ACCENT, lw=2,
                    label=f"Linear fit (slope={m:.2f})")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            _int_axes(ax)
            ax.text(0.04, 0.97,
                    f"MAE  = {m_val['mae']:.2f} BPM\n"
                    f"RMSE = {m_val['rmse']:.2f} BPM\n"
                    f"R²    = {m_val['r2']:.3f}",
                    transform=ax.transAxes, fontsize=10, va="top",
                    bbox=dict(boxstyle="round,pad=0.5", fc="white",
                              ec=GRID, alpha=0.95))
            ax.set_title(f"{lbl} — Real vs. Predicted", fontweight="bold")
            ax.set_xlabel("Ground Truth — Smartwatch (BPM)")
            ax.set_ylabel(f"Predicted — {lbl} (BPM)")
            ax.legend(); ax.grid(True, alpha=0.30, ls="--", color=GRID)

        fig.tight_layout()
        _save(fig, "comp_02_scatter_real_vs_pred", out_dir)


# ── 03. Bland-Altman side by side ──────────────────────────────────────────────
def plot_03_bland_altman(d, out_dir):
    y_true = d["y_true"]
    pairs  = [("GRU", d["y_gru"], C_GRU), ("LSTM", d["y_lstm"], C_LSTM)]

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(17, 7))
        fig.patch.set_facecolor(BG_FIG)
        fig.suptitle("Bland-Altman Agreement Analysis — GRU vs. LSTM",
                     fontsize=15, fontweight="bold", color="#1A1A2E", y=1.01)

        for ax, (lbl, y_pred, clr) in zip(axes, pairs):
            m_val = metrics(y_true, y_pred)
            err   = y_true - y_pred
            mean_ = (y_true + y_pred) / 2.0
            bias  = m_val["bias"]
            sigma = m_val["sigma"]
            loa_u = bias + 1.96 * sigma
            loa_l = bias - 1.96 * sigma
            xlo   = float(mean_.min()) - 2
            xhi   = float(mean_.max()) + 2

            ax.set_facecolor(BG_AX)
            ax.scatter(mean_, err, alpha=0.15, s=10, color=clr,
                       edgecolors="none", rasterized=True)
            ax.axhline(bias,  color=clr,     lw=2.2, label=f"Bias = {bias:.2f} BPM")
            ax.axhline(loa_u, color="#757575", lw=1.5, ls="--",
                       label=f"+1.96σ = {loa_u:.2f}")
            ax.axhline(loa_l, color="#757575", lw=1.5, ls="--",
                       label=f"−1.96σ = {loa_l:.2f}")
            ax.fill_between([xlo, xhi], loa_l, loa_u, alpha=0.07, color=clr)
            ax.set_xlim(xlo, xhi)
            ax.text(xhi - 4, loa_u + 0.3, f"+1.96σ={loa_u:.2f}", fontsize=8.5, color="#555")
            ax.text(xhi - 4, loa_l - 0.8, f"−1.96σ={loa_l:.2f}", fontsize=8.5, color="#555")
            _int_axes(ax)
            ax.set_title(f"{lbl} — Bland-Altman", fontweight="bold")
            ax.set_xlabel("Mean (Smartwatch + Predicted) / 2 (BPM)")
            ax.set_ylabel("Difference: GT − Predicted (BPM)")
            ax.legend(); ax.grid(True, alpha=0.30, ls="--", color=GRID)

        fig.tight_layout()
        _save(fig, "comp_03_bland_altman", out_dir)


# ── 04. CDF overlaid ──────────────────────────────────────────────────────────
def plot_04_cdf(d, out_dir):
    y_true = d["y_true"]
    pairs  = [("GRU", d["y_gru"], C_GRU, "-"), ("LSTM", d["y_lstm"], C_LSTM, "--")]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        for lbl, y_pred, clr, ls in pairs:
            abs_err   = np.abs(y_true - y_pred)
            sorted_ae = np.sort(abs_err)
            cdf_vals  = np.arange(1, len(sorted_ae) + 1) / len(sorted_ae) * 100
            ax.plot(sorted_ae, cdf_vals, color=clr, lw=2.8, ls=ls, label=lbl)
            ax.fill_between(sorted_ae, 0, cdf_vals, alpha=0.07, color=clr)

        for thr, ls_s in [(5, ":"), (10, ":")]:
            ax.axvline(thr, color="#BDBDBD", ls=ls_s, lw=1.2)
            for lbl, y_pred, clr, _ in pairs:
                pct = int(round(np.mean(np.abs(y_true - y_pred) <= thr) * 100))
                ax.text(thr + 0.15, 5 if lbl == "LSTM" else 20,
                        f"{lbl}: {pct}%\n≤{thr} BPM",
                        fontsize=8.5, color=clr, va="bottom")

        _int_axes(ax)
        ax.set_ylim(0, 103)
        ax.set_title("Cumulative Distribution of Absolute Error — GRU vs. LSTM",
                     fontweight="bold")
        ax.set_xlabel("|Error| (BPM)")
        ax.set_ylabel("Cumulative Samples (%)")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.35, ls="--", color=GRID)
        fig.tight_layout()
        _save(fig, "comp_04_cdf_comparison", out_dir)


# ── 05. Metrics bar chart ──────────────────────────────────────────────────────
def plot_05_metrics_bar(d, out_dir):
    y_true = d["y_true"]
    mg = metrics(y_true, d["y_gru"])
    ml = metrics(y_true, d["y_lstm"])

    # sub-figures: one per metric group
    fig_data = [
        ("MAE (BPM)",            [mg["mae"],    ml["mae"]],    True),
        ("RMSE (BPM)",           [mg["rmse"],   ml["rmse"]],   True),
        ("R²",                   [mg["r2"],     ml["r2"]],     False),
        ("Parameters (M)",       [d["n_gru"]/1e6, d["n_lstm"]/1e6], False),
        ("Inference Time (s)",   [d["t_gru"],   d["t_lstm"]],  True),
    ]
    # True = lower is better; False = higher is better

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 5, figsize=(18, 6))
        fig.patch.set_facecolor(BG_FIG)
        fig.suptitle("GRU vs. LSTM — Performance Comparison",
                     fontsize=15, fontweight="bold", color="#1A1A2E", y=1.01)

        for ax, (ylabel, vals, lower_better) in zip(axes, fig_data):
            ax.set_facecolor(BG_AX)
            clrs = [C_GRU, C_LSTM]
            if lower_better:
                best = int(np.argmin(vals))
            else:
                best = int(np.argmax(vals))
            alphas = [0.90, 0.90]
            bars = ax.bar(["GRU", "LSTM"], vals, color=clrs, alpha=0.88,
                          edgecolor=["#3A0000", "#0D3B7A"], linewidth=0.8)
            for i, (bar, v) in enumerate(zip(bars, vals)):
                star = " ★" if i == best else ""
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.015,
                        f"{v:.3f}{star}", ha="center", fontsize=9.5,
                        color="#111", fontweight="bold" if i == best else "normal")

            ax.set_title(ylabel, fontweight="bold", fontsize=11)
            ax.set_ylabel(ylabel)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["GRU", "LSTM"])
            ax.grid(True, alpha=0.35, ls="--", color=GRID, axis="y")
            # no integer formatting here — metrics need decimals on y-axis

        fig.tight_layout()
        _save(fig, "comp_05_metrics_bar", out_dir)


# ── 06. MAE per Subject — grouped bars ──────────────────────────────────────────
def plot_06_mae_per_subject(d, out_dir):
    y_true   = d["y_true"]
    subj     = d["subj"]
    subj_ids = np.unique(subj)

    mae_gru  = [float(np.mean(np.abs(y_true[subj == s] - d["y_gru"][subj == s])))
                for s in subj_ids]
    mae_lstm = [float(np.mean(np.abs(y_true[subj == s] - d["y_lstm"][subj == s])))
                for s in subj_ids]

    x      = np.arange(len(subj_ids))
    width  = 0.40
    ov_gru  = float(np.mean(np.abs(y_true - d["y_gru"])))
    ov_lstm = float(np.mean(np.abs(y_true - d["y_lstm"])))

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(20, 7))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        ax.bar(x - width/2, mae_gru,  width, color=C_GRU,  alpha=0.85,
               edgecolor="#3A0000", linewidth=0.3, label=f"GRU  (mean={ov_gru:.2f})")
        ax.bar(x + width/2, mae_lstm, width, color=C_LSTM, alpha=0.85,
               edgecolor="#0D3B7A", linewidth=0.3, label=f"LSTM (mean={ov_lstm:.2f})")

        ax.axhline(ov_gru,  color=C_GRU,  ls="--", lw=1.5, alpha=0.7)
        ax.axhline(ov_lstm, color=C_LSTM, ls="--", lw=1.5, alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in subj_ids],
                           fontsize=6.5, rotation=45, ha="right")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title("MAE per Subject — GRU vs. LSTM", fontweight="bold")
        ax.set_xlabel("Subject ID")
        ax.set_ylabel("MAE (BPM)")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.35, ls="--", color=GRID, axis="y")
        fig.tight_layout()
        _save(fig, "comp_06_mae_per_subject_grouped", out_dir)


# ── 07. MAE per Position — grouped bars ─────────────────────────────────────────
def plot_07_mae_per_position(d, out_dir):
    y_true  = d["y_true"]
    pos     = d["pos"]
    pos_ids = np.unique(pos)

    mae_gru  = [float(np.mean(np.abs(y_true[pos == p] - d["y_gru"][pos == p])))
                for p in pos_ids]
    mae_lstm = [float(np.mean(np.abs(y_true[pos == p] - d["y_lstm"][pos == p])))
                for p in pos_ids]

    x     = np.arange(len(pos_ids))
    width = 0.38

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        b1 = ax.bar(x - width/2, mae_gru,  width, color=C_GRU,  alpha=0.88,
                    edgecolor="#3A0000", linewidth=0.5, label="GRU")
        b2 = ax.bar(x + width/2, mae_lstm, width, color=C_LSTM, alpha=0.88,
                    edgecolor="#0D3B7A", linewidth=0.5, label="LSTM")

        for bar, v in zip(list(b1) + list(b2), mae_gru + mae_lstm):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.04,
                    str(round(v)), ha="center", fontsize=7.5, color="#333")

        ov_gru  = float(np.mean(np.abs(y_true - d["y_gru"])))
        ov_lstm = float(np.mean(np.abs(y_true - d["y_lstm"])))
        ax.axhline(ov_gru,  color=C_GRU,  ls="--", lw=1.5, alpha=0.7,
                   label=f"GRU  overall = {ov_gru:.2f} BPM")
        ax.axhline(ov_lstm, color=C_LSTM, ls="--", lw=1.5, alpha=0.7,
                   label=f"LSTM overall = {ov_lstm:.2f} BPM")

        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in pos_ids], fontsize=10)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title("MAE per Body Position — GRU vs. LSTM", fontweight="bold")
        ax.set_xlabel("Body Position")
        ax.set_ylabel("MAE (BPM)")
        ax.legend(); ax.grid(True, alpha=0.35, ls="--", color=GRID, axis="y")
        fig.tight_layout()
        _save(fig, "comp_07_mae_per_position_grouped", out_dir)


# ── 08. HR per Position Real vs Pred — both models ──────────────────────────────
def plot_08_hr_position_both(d, out_dir):
    y_true  = d["y_true"]
    pos     = d["pos"]
    pos_ids = np.unique(pos)

    gt_m    = [float(np.mean(y_true[pos == p]))       for p in pos_ids]
    gru_m   = [float(np.mean(d["y_gru"][pos == p]))   for p in pos_ids]
    lstm_m  = [float(np.mean(d["y_lstm"][pos == p]))  for p in pos_ids]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(14, 7))
        _red_bar(fig, "Mean Heart Rate per Position  —  Ground Truth vs. GRU vs. LSTM")
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        # connector lines
        for x, gt, gr, ls in zip(pos_ids, gt_m, gru_m, lstm_m):
            ax.plot([x, x], [gt, gr], color="#CCCCCC", lw=1.0, zorder=1)
            ax.plot([x, x], [gt, ls], color="#AACCEE", lw=1.0, zorder=1)

        ax.scatter(pos_ids, gt_m,   s=160, color="#F0B8C0", edgecolors=C_GRU,
                   linewidths=1.5, zorder=4, label="Smartwatch (Ground Truth)")
        ax.scatter(pos_ids, gru_m,  s=160, color=C_GRU,    edgecolors="#3A0000",
                   linewidths=1.5, zorder=4, label="GRU (Predicted)")
        ax.scatter(pos_ids, lstm_m, s=140, color=C_LSTM,   edgecolors="#0D3B7A",
                   linewidths=1.5, zorder=3, marker="D", label="LSTM (Predicted)")

        ax.set_xlabel("Body Position")
        ax.set_ylabel("Mean HR (BPM)")
        ax.set_xticks(pos_ids)
        _int_axes(ax)
        ax.legend()
        ax.grid(True, alpha=0.30, ls="--", color=GRID)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, "comp_08_hr_position_gt_gru_lstm", out_dir)


# ── 09. Metrics comparison table — GRU vs LSTM (publication style) ───────────
def plot_09_metrics_table(d, out_dir):
    y_true = d["y_true"]
    mg = metrics(y_true, d["y_gru"])
    ml = metrics(y_true, d["y_lstm"])
    ag = mg["abs"]; al = ml["abs"]
    pearson_g = float(np.corrcoef(y_true, d["y_gru"])[0, 1])
    pearson_l = float(np.corrcoef(y_true, d["y_lstm"])[0, 1])

    # (label, gru_value_str, lstm_value_str, gru_is_best)
    def _row(label, vg, vl, fmt, lower_is_better=True):
        sg = fmt.format(vg)
        sl = fmt.format(vl)
        gru_best = (vg <= vl) if lower_is_better else (vg >= vl)
        return label, sg, sl, gru_best

    row_defs = [
        _row("MAE (BPM)",         mg["mae"],                    ml["mae"],                    "{:.2f}", True),
        _row("RMSE (BPM)",        mg["rmse"],                   ml["rmse"],                   "{:.2f}", True),
        _row("R²",                mg["r2"],                     ml["r2"],                     "{:.4f}", False),
        _row("Pearson r",         pearson_g,                    pearson_l,                    "{:.4f}", False),
        _row("Bias (BPM)",        mg["bias"],                   ml["bias"],                   "{:.2f}", False),
        _row("Std Dev σ (BPM)",   mg["sigma"],                  ml["sigma"],                  "{:.2f}", True),
        _row("LoA upper (BPM)",   mg["bias"]+1.96*mg["sigma"],  ml["bias"]+1.96*ml["sigma"],  "{:.2f}", True),
        _row("LoA lower (BPM)",   mg["bias"]-1.96*mg["sigma"],  ml["bias"]-1.96*ml["sigma"],  "{:.2f}", False),
        _row("% ≤ 5 BPM",         np.mean(ag<=5)*100,           np.mean(al<=5)*100,           "{:.1f}%", False),
        _row("% ≤ 10 BPM",        np.mean(ag<=10)*100,          np.mean(al<=10)*100,          "{:.1f}%", False),
        _row("% ≤ 15 BPM",        np.mean(ag<=15)*100,          np.mean(al<=15)*100,          "{:.1f}%", False),
        _row("Parameters",        d["n_gru"],                   d["n_lstm"],                  "{:,}",   True),
        _row("Inference Time (s)", d["t_gru"],                  d["t_lstm"],                  "{:.3f}", True),
        _row("N samples",         len(y_true),                  len(y_true),                  "{:,}",   True),
    ]

    # build cell text — mark best with ★
    cell_text = []
    best_cells = []   # (row_idx, col_idx) where col 1=GRU, 2=LSTM
    for ri, (label, sg, sl, gru_best) in enumerate(row_defs):
        if sg == sl:                      # tie (e.g. N samples)
            cell_text.append([label, sg, sl])
        elif gru_best:
            cell_text.append([label, f"★  {sg}", sl])
            best_cells.append((ri + 1, 1))
        else:
            cell_text.append([label, sg, f"★  {sl}"])
            best_cells.append((ri + 1, 2))

    ROW_H   = 0.068          # fraction of figure height per data row
    N_ROWS  = len(cell_text)
    FIG_H   = max(7.5, N_ROWS * ROW_H * 14)  # ~14-inch scale

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(10, FIG_H))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_FIG)
        ax.axis("off")

        tbl = ax.table(
            cellText=cell_text,
            colLabels=["Metric", "GRU", "LSTM"],
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.30, 2.00)

        # ── column widths ──
        tbl.auto_set_column_width([0, 1, 2])

        # ── header row ─────────────────────────────────────────────────────────
        HDR_CLR    = C_GRU          # same dark-red header as single-model table
        EVEN_CLR   = "#F5E9EC"      # very light rose — even data rows
        ODD_CLR    = "#FFFFFF"      # white — odd data rows
        BEST_CLR   = "#E8F5E9"      # light green — best-value cells
        BEST_FONT  = "bold"
        EDGE_CLR   = "#E0E0E0"

        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor(EDGE_CLR)
            cell.set_linewidth(0.6)

            if r == 0:
                # header
                cell.set_facecolor(HDR_CLR)
                cell.set_text_props(color="white", fontweight="bold", fontsize=12)
            else:
                # data rows — zebra stripe
                base = EVEN_CLR if r % 2 == 0 else ODD_CLR
                cell.set_facecolor(base)
                cell.set_text_props(color="#1A1A1A", fontsize=12)

        # ── highlight best values ───────────────────────────────────────────────
        for (r, c) in best_cells:
            cell = tbl[r, c]
            cell.set_facecolor(BEST_CLR)
            cell.set_text_props(fontweight=BEST_FONT, color="#1B5E20")

        # ── left-align metric column ───────────────────────────────────────────
        for r in range(N_ROWS + 1):
            tbl[r, 0].set_text_props(ha="left")
            # small left padding via text position
            tbl[r, 0].get_text().set_position((0.04, 0.5))

        ax.set_title(
            "GRU vs. LSTM — Evaluation Metrics Comparison\n"
            "(★ = best value per metric)",
            fontsize=14, fontweight="bold", pad=20, color="#1A1A2E",
        )
        fig.tight_layout()
        _save(fig, "comp_09_metrics_table", out_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  IMPROVED GRU CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_gru_01_learning(d, out_dir):
    h     = d["h_gru"]
    train = np.array(h["train_loss"])
    val   = np.array(h["val_mae"])
    eps   = np.arange(1, len(train) + 1)
    sm    = uniform_filter1d(val, size=15)
    best  = int(np.argmin(val))

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(12, 6))
        _red_bar(fig, "GRU — Learning Curve")
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        ax.plot(eps, train, color=C_LSTM_L, lw=1.5, alpha=0.7, label="Training Loss (Huber)")
        ax.plot(eps, val,   color=C_GRU_L,  lw=1.0, alpha=0.4)
        ax.plot(eps, sm,    color=C_GRU,    lw=2.8, label="Val MAE (smoothed w=15)")
        ax.fill_between(eps, sm, alpha=0.09, color=C_GRU)
        ax.axvline(best + 1, color="#BDBDBD", ls=":", lw=1.3)
        ax.scatter([best + 1], [val[best]], s=110, color=C_GRU, zorder=5)
        ax.annotate(
            f"Best  Epoch {best + 1}\nVal MAE = {val[best]:.2f} BPM",
            xy=(best + 1, val[best]),
            xytext=(best + 1 + 20, val[best] + 2.5),
            fontsize=9.5, color="#222",
            arrowprops=dict(arrowstyle="->", color="#999", lw=1.1),
        )
        _int_axes(ax); ax.legend()
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss / MAE (BPM)")
        ax.grid(True, alpha=0.4, ls="--", color=GRID)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, "gru_01_learning_curve", out_dir)


def plot_gru_02_hr_position(d, out_dir):
    y_true  = d["y_true"]
    y_pred  = d["y_gru"]
    pos     = d["pos"]
    pos_ids = np.unique(pos)
    gt_m    = [float(np.mean(y_true[pos == p])) for p in pos_ids]
    pr_m    = [float(np.mean(y_pred[pos == p])) for p in pos_ids]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(14, 7))
        _red_bar(fig, "Mean Heart Rate per Body Position  —  Real vs. Estimated  (GRU)")
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        for x, gt, pr in zip(pos_ids, gt_m, pr_m):
            ax.plot([x, x], [gt, pr], color="#DDDDDD", lw=1.3, zorder=1)

        ax.scatter(pos_ids, gt_m, s=200, color="#F0B8C0", edgecolors=C_GRU,
                   linewidths=1.8, zorder=3,
                   label="Smartwatch (Ground Truth)", alpha=0.95)
        ax.scatter(pos_ids, pr_m, s=200, color=C_GRU, edgecolors="#3A0000",
                   linewidths=1.8, zorder=3,
                   label="Wi-Cardio / GRU (Predicted)")

        ax.set_xticks(pos_ids)
        _int_axes(ax); ax.legend(fontsize=11)
        ax.set_xlabel("Body Position"); ax.set_ylabel("Mean HR (BPM)")
        ax.grid(True, alpha=0.28, ls="--", color=GRID)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, "gru_02_hr_per_position", out_dir)


def plot_gru_03_scatter(d, out_dir):
    y_true = d["y_true"]; y_pred = d["y_gru"]
    m_val  = metrics(y_true, y_pred)
    lo = int(min(y_true.min(), y_pred.min())) - 3
    hi = int(max(y_true.max(), y_pred.max())) + 3

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, 8))
        fig.patch.set_facecolor(BG_FIG); ax.set_facecolor(BG_AX)
        ax.scatter(y_true, y_pred, alpha=0.18, s=12, color=C_GRU,
                   edgecolors="none", rasterized=True, label="Samples")
        ax.plot([lo,hi],[lo,hi],"k--",lw=1.8,label="Identity (y=x)")
        m,b = np.polyfit(y_true, y_pred, 1); xs = np.linspace(lo,hi,300)
        ax.plot(xs, m*xs+b, color=ACCENT, lw=2, label=f"Linear fit (slope={m:.2f})")
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.set_aspect("equal","box")
        _int_axes(ax)
        ax.text(0.04,0.97,
                f"MAE  = {m_val['mae']:.2f} BPM\n"
                f"RMSE = {m_val['rmse']:.2f} BPM\n"
                f"R²    = {m_val['r2']:.4f}",
                transform=ax.transAxes,fontsize=10.5,va="top",
                bbox=dict(boxstyle="round,pad=0.5",fc="white",ec=GRID,alpha=0.95))
        ax.set_title("GRU — Real vs. Predicted Heart Rate", fontweight="bold")
        ax.set_xlabel("Ground Truth — Smartwatch (BPM)")
        ax.set_ylabel("Predicted — Wi-Cardio / GRU (BPM)")
        ax.legend(); ax.grid(True,alpha=0.30,ls="--",color=GRID)
        fig.tight_layout()
        _save(fig, "gru_03_scatter", out_dir)


def plot_gru_04_bland_altman(d, out_dir):
    y_true = d["y_true"]; y_pred = d["y_gru"]
    m_val  = metrics(y_true, y_pred)
    err    = y_true - y_pred
    mean_  = (y_true + y_pred) / 2.0
    bias   = m_val["bias"]; sigma = m_val["sigma"]
    loa_u  = bias + 1.96*sigma; loa_l = bias - 1.96*sigma
    xlo = float(mean_.min())-2; xhi = float(mean_.max())+2

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.patch.set_facecolor(BG_FIG); ax.set_facecolor(BG_AX)
        ax.scatter(mean_, err, alpha=0.18, s=12, color=C_GRU,
                   edgecolors="none", rasterized=True)
        ax.axhline(bias,  color=C_GRU,    lw=2.2, label=f"Bias = {bias:.2f} BPM")
        ax.axhline(loa_u, color="#757575", lw=1.6, ls="--",
                   label=f"+1.96σ = {loa_u:.2f} BPM")
        ax.axhline(loa_l, color="#757575", lw=1.6, ls="--",
                   label=f"−1.96σ = {loa_l:.2f} BPM")
        ax.fill_between([xlo,xhi], loa_l, loa_u, alpha=0.07, color=C_GRU)
        ax.set_xlim(xlo,xhi)
        ax.text(xhi-4, loa_u+0.3, f"+1.96σ={loa_u:.2f}", fontsize=9, color="#555")
        ax.text(xhi-4, loa_l-0.8, f"−1.96σ={loa_l:.2f}", fontsize=9, color="#555")
        _int_axes(ax)
        ax.set_title("GRU — Bland-Altman Agreement Analysis", fontweight="bold")
        ax.set_xlabel("Mean (Smartwatch + GRU) / 2 (BPM)")
        ax.set_ylabel("Difference: GT − Predicted (BPM)")
        ax.legend(); ax.grid(True,alpha=0.30,ls="--",color=GRID)
        fig.tight_layout()
        _save(fig, "gru_04_bland_altman", out_dir)


def plot_gru_05_cdf(d, out_dir):
    y_true = d["y_true"]; y_pred = d["y_gru"]
    abs_err   = np.abs(y_true - y_pred)
    sorted_ae = np.sort(abs_err)
    cdf_vals  = np.arange(1, len(sorted_ae)+1) / len(sorted_ae) * 100

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(BG_FIG); ax.set_facecolor(BG_AX)
        ax.plot(sorted_ae, cdf_vals, color=C_GRU, lw=2.8)
        ax.fill_between(sorted_ae, 0, cdf_vals, alpha=0.11, color=C_GRU)
        for thr, ls_s, y_txt in [(5,"--",12),(10,":",28),(15,"-.",44)]:
            pct = int(round(np.mean(abs_err<=thr)*100))
            ax.axvline(thr, color="#9E9E9E", ls=ls_s, lw=1.3)
            ax.text(thr+0.25, y_txt, f"{pct}%\n≤{thr} BPM",
                    fontsize=9, color="#555", va="bottom")
        _int_axes(ax); ax.set_ylim(0,103)
        ax.set_title("GRU — Cumulative Distribution of Absolute Error",
                     fontweight="bold")
        ax.set_xlabel("|Error| (BPM)"); ax.set_ylabel("Cumulative Samples (%)")
        ax.grid(True,alpha=0.35,ls="--",color=GRID)
        fig.tight_layout()
        _save(fig, "gru_05_cdf", out_dir)


def plot_gru_06_mae_position_bar(d, out_dir):
    y_true  = d["y_true"]; y_pred = d["y_gru"]
    pos     = d["pos"]; pos_ids = np.unique(pos)
    pos_mae = [float(np.mean(np.abs(y_true[pos==p]-y_pred[pos==p]))) for p in pos_ids]
    overall = float(np.mean(np.abs(y_true - y_pred)))

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(13, 6))
        fig.patch.set_facecolor(BG_FIG); ax.set_facecolor(BG_AX)
        clrs = [C_GRU if m > overall else C_GRU_L for m in pos_mae]
        bars = ax.bar(pos_ids, pos_mae, color=clrs,
                      edgecolor="#3A0000", linewidth=0.6, alpha=0.88)
        ax.axhline(overall, color="#757575", ls="--", lw=1.8,
                   label=f"Overall MAE = {overall:.2f} BPM")
        for bar, v in zip(bars, pos_mae):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.04,
                    f"{v:.1f}", ha="center", fontsize=9, color="#222")
        ax.set_xticks(pos_ids); _int_axes(ax)
        ax.set_title("GRU — MAE per Body Position", fontweight="bold")
        ax.set_xlabel("Body Position"); ax.set_ylabel("MAE (BPM)")
        ax.legend(); ax.grid(True,alpha=0.35,ls="--",color=GRID,axis="y")
        fig.tight_layout()
        _save(fig, "gru_06_mae_per_position_bar", out_dir)


def plot_gru_07_metrics_table(d, out_dir):
    y_true = d["y_true"]; y_pred = d["y_gru"]
    m_val  = metrics(y_true, y_pred)
    abs_err = m_val["abs"]
    bias = m_val["bias"]; sigma = m_val["sigma"]
    pearson = float(np.corrcoef(y_true, y_pred)[0,1])

    rows = [
        ["MAE",                f"{m_val['mae']:.2f} BPM"],
        ["RMSE",               f"{m_val['rmse']:.2f} BPM"],
        ["R²",                 f"{m_val['r2']:.4f}"],
        ["Pearson r",          f"{pearson:.4f}"],
        ["Bias (mean error)",  f"{bias:.2f} BPM"],
        ["Std Dev (σ)",        f"{sigma:.2f} BPM"],
        ["LoA upper",          f"+{bias+1.96*sigma:.2f} BPM"],
        ["LoA lower",          f"{bias-1.96*sigma:.2f} BPM"],
        ["% ≤ 5 BPM",          f"{np.mean(abs_err<=5)*100:.1f}%"],
        ["% ≤ 10 BPM",         f"{np.mean(abs_err<=10)*100:.1f}%"],
        ["% ≤ 15 BPM",         f"{np.mean(abs_err<=15)*100:.1f}%"],
        ["Parameters",         f"{d['n_gru']:,}"],
        ["Inference Time",     f"{d['t_gru']:.3f} s"],
        ["N samples",          f"{len(y_true):,}"],
    ]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(8, 7))
        fig.patch.set_facecolor(BG_FIG); ax.set_facecolor(BG_FIG); ax.axis("off")
        tbl = ax.table(cellText=rows, colLabels=["Metric","Value"],
                       loc="center", cellLoc="left")
        tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.25, 1.80)
        for (r,c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#E0E0E0")
            if r == 0:
                cell.set_facecolor(C_GRU)
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#F5E9EC")
            else:
                cell.set_facecolor("white")
        ax.set_title("GRU — Evaluation Metrics Summary",
                     fontsize=14, fontweight="bold", pad=16)
        fig.tight_layout()
        _save(fig, "gru_07_metrics_table", out_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGNAL COMPARISON CHARTS (single subject × position)
# ═══════════════════════════════════════════════════════════════════════════════

def _pick_cases(d, min_samples=15):
    """Return (subject, position) for best / typical / worst GRU MAE cases."""
    y_true = d["y_true"]; y_gru = d["y_gru"]
    subj   = d["subj"];   pos   = d["pos"]

    candidates = []
    for s in np.unique(subj):
        for p in np.unique(pos):
            mask = (subj == s) & (pos == p)
            if mask.sum() < min_samples:
                continue
            mae = float(np.mean(np.abs(y_true[mask] - y_gru[mask])))
            candidates.append((mae, int(s), int(p), int(mask.sum())))

    candidates.sort()
    best    = candidates[0]
    worst   = candidates[-1]
    mid_idx = len(candidates) // 2
    typical = candidates[mid_idx]
    return best, typical, worst


def _signal_axes(ax, samples, y_true_seg, y_gru_seg, y_lstm_seg,
                 subj_id, pos_id, mae_g, mae_l, title_prefix=""):
    """Draw one time-series panel onto ax."""
    x = np.arange(len(samples))

    # shaded error band around GRU
    ax.fill_between(x, y_gru_seg, y_true_seg,
                    alpha=0.12, color=C_GRU, label="_gru_band")

    ax.plot(x, y_true_seg,  color="#D4919B", lw=2.2, alpha=0.9,
            label="Smartwatch (Ground Truth)")
    ax.plot(x, y_gru_seg,   color=C_GRU,    lw=2.0,
            label=f"GRU  (MAE={mae_g:.2f})")
    ax.plot(x, y_lstm_seg,  color=C_LSTM,   lw=1.8, ls="--",
            label=f"LSTM (MAE={mae_l:.2f})")

    ax.set_title(
        f"{title_prefix}Subject {subj_id} — Position {pos_id}  "
        f"(n={len(samples)} windows)",
        fontweight="bold", fontsize=11,
    )
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Heart Rate (BPM)")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35, ls="--", color=GRID)
    ax.set_facecolor(BG_AX)


def plot_signal_case_studies(d, out_dir):
    """
    3-panel chart: best / typical / worst GRU case, all with GT vs GRU vs LSTM.
    """
    y_true = d["y_true"]; y_gru = d["y_gru"]; y_lstm = d["y_lstm"]
    subj   = d["subj"];   pos   = d["pos"]

    best, typical, worst = _pick_cases(d)

    cases = [
        (best,    "Best Case — "),
        (typical, "Typical Case — "),
        (worst,   "Worst Case — "),
    ]

    with plt.rc_context(RC):
        fig, axes = plt.subplots(3, 1, figsize=(14, 15))
        _red_bar(fig,
            "Heart Rate Signal — Ground Truth vs. GRU vs. LSTM  (Best / Typical / Worst)")
        fig.patch.set_facecolor(BG_FIG)

        for ax, ((mae_g, s, p, n), prefix) in zip(axes, cases):
            mask      = (subj == s) & (pos == p)
            idx_order = np.where(mask)[0]          # keep original order
            seg_true  = y_true[idx_order]
            seg_gru   = y_gru[idx_order]
            seg_lstm  = y_lstm[idx_order]
            mae_l     = float(np.mean(np.abs(seg_true - seg_lstm)))
            _signal_axes(ax, idx_order, seg_true, seg_gru, seg_lstm,
                         s, p, mae_g, mae_l, prefix)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, "signal_01_case_studies", out_dir)


def plot_signal_single(d, out_dir, subject=None, position=None):
    """
    Single subject × position: two panels — top = time series, bottom = error.
    If subject/position are None, auto-picks the best GRU case.
    """
    y_true = d["y_true"]; y_gru = d["y_gru"]; y_lstm = d["y_lstm"]
    subj   = d["subj"];   pos   = d["pos"]

    if subject is None or position is None:
        best, _, _ = _pick_cases(d)
        _, subject, position, _ = best
        print(f"  Auto-selected Subject {subject}, Position {position}")

    mask      = (subj == subject) & (pos == position)
    idx_order = np.where(mask)[0]
    seg_true  = y_true[idx_order]
    seg_gru   = y_gru[idx_order]
    seg_lstm  = y_lstm[idx_order]
    err_gru   = seg_true - seg_gru
    err_lstm  = seg_true - seg_lstm
    x         = np.arange(len(idx_order))

    mae_g = float(np.mean(np.abs(err_gru)))
    mae_l = float(np.mean(np.abs(err_lstm)))

    with plt.rc_context(RC):
        fig, (ax_sig, ax_err) = plt.subplots(
            2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2.2, 1]}
        )
        _red_bar(fig,
            f"HR Signal Comparison — Subject {subject}, Position {position}  "
            f"(Ground Truth vs. GRU vs. LSTM)")
        fig.patch.set_facecolor(BG_FIG)

        # ── Top: signal ────────────────────────────────────────────────────────
        ax_sig.fill_between(x, seg_gru, seg_true, alpha=0.10, color=C_GRU)
        ax_sig.plot(x, seg_true,  color="#D4919B", lw=2.5, alpha=0.95,
                    zorder=3, label="Smartwatch (Ground Truth)")
        ax_sig.plot(x, seg_gru,   color=C_GRU,    lw=2.2, zorder=4,
                    label=f"GRU  — MAE={mae_g:.2f} BPM")
        ax_sig.plot(x, seg_lstm,  color=C_LSTM,   lw=2.0, ls="--", zorder=3,
                    label=f"LSTM — MAE={mae_l:.2f} BPM")

        ax_sig.set_ylabel("Heart Rate (BPM)")
        ax_sig.set_title(
            f"Subject {subject} — Position {position}  ({len(idx_order)} windows)",
            fontweight="bold",
        )
        ax_sig.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax_sig.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_sig.legend(fontsize=10)
        ax_sig.grid(True, alpha=0.35, ls="--", color=GRID)
        ax_sig.set_facecolor(BG_AX)

        # ── Bottom: error ──────────────────────────────────────────────────────
        ax_err.axhline(0, color="#555", ls="--", lw=1.2)
        ax_err.fill_between(x, err_gru,  alpha=0.18, color=C_GRU)
        ax_err.fill_between(x, err_lstm, alpha=0.12, color=C_LSTM)
        ax_err.plot(x, err_gru,  color=C_GRU,  lw=1.8,
                    label=f"GRU error  (σ={np.std(err_gru):.2f})")
        ax_err.plot(x, err_lstm, color=C_LSTM, lw=1.6, ls="--",
                    label=f"LSTM error (σ={np.std(err_lstm):.2f})")

        ax_err.set_xlabel("Window Index")
        ax_err.set_ylabel("Error: GT − Predicted (BPM)")
        ax_err.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax_err.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_err.legend(fontsize=9)
        ax_err.grid(True, alpha=0.35, ls="--", color=GRID)
        ax_err.set_facecolor(BG_AX)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        tag = f"s{subject}_p{position}"
        _save(fig, f"signal_02_single_{tag}", out_dir)


def plot_signal_grid(d, out_dir, n_subjects=6):
    """
    Grid of n_subjects panels (GRU only), one per subject, all at the same
    position. Picks the position where the most subjects have ≥15 samples.
    """
    y_true = d["y_true"]; y_gru = d["y_gru"]
    subj   = d["subj"];   pos   = d["pos"]

    # find position with most qualifying subjects
    best_pos, best_subjs = None, []
    for p in np.unique(pos):
        p_mask = pos == p
        ok = [s for s in np.unique(subj[p_mask])
              if np.sum((subj == s) & p_mask) >= 15]
        if len(ok) > len(best_subjs):
            best_pos, best_subjs = p, ok

    chosen = best_subjs[:n_subjects]
    ncols  = 3
    nrows  = int(np.ceil(len(chosen) / ncols))

    with plt.rc_context(RC):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 6, nrows * 4),
                                 squeeze=False)
        _red_bar(fig,
            f"HR Signal per Subject — Position {best_pos}  "
            f"(Ground Truth vs. GRU)")
        fig.patch.set_facecolor(BG_FIG)

        for idx, s in enumerate(chosen):
            ax   = axes[idx // ncols][idx % ncols]
            mask = (subj == s) & (pos == best_pos)
            seg_true = y_true[mask]
            seg_gru  = y_gru[mask]
            x        = np.arange(len(seg_true))
            mae_g    = float(np.mean(np.abs(seg_true - seg_gru)))

            ax.fill_between(x, seg_gru, seg_true, alpha=0.12, color=C_GRU)
            ax.plot(x, seg_true, color="#D4919B", lw=2.0, alpha=0.9,
                    label="Ground Truth")
            ax.plot(x, seg_gru,  color=C_GRU,    lw=1.8,
                    label=f"GRU  MAE={mae_g:.2f}")
            ax.set_title(f"Subject {s}", fontweight="bold", fontsize=10)
            ax.set_xlabel("Window"); ax.set_ylabel("HR (BPM)")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(fontsize=8); ax.set_facecolor(BG_AX)
            ax.grid(True, alpha=0.30, ls="--", color=GRID)

        # hide unused axes
        for idx in range(len(chosen), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, f"signal_03_grid_position_{best_pos}", out_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ROOT / "charts_output",
                        help="Output directory")
    parser.add_argument("--subject",  type=int, default=None,
                        help="Subject ID for signal chart (default: auto best)")
    parser.add_argument("--position", type=int, default=None,
                        help="Body position for signal chart (default: auto best)")
    args = parser.parse_args()

    print("\nLoading models and running inference …")
    d = load_all()

    print("\n── Comparison charts ─────────────────────────────────────────────")
    plot_01_learning_curves(d, args.out)
    plot_01b_learning_curves_overlay(d, args.out)
    plot_02_scatter(d, args.out)
    plot_03_bland_altman(d, args.out)
    plot_04_cdf(d, args.out)
    plot_05_metrics_bar(d, args.out)
    plot_06_mae_per_subject(d, args.out)
    plot_07_mae_per_position(d, args.out)
    plot_08_hr_position_both(d, args.out)
    plot_09_metrics_table(d, args.out)

    print("\n── Improved GRU charts ──────────────────────────────────────────")
    plot_gru_01_learning(d, args.out)
    plot_gru_02_hr_position(d, args.out)
    plot_gru_03_scatter(d, args.out)
    plot_gru_04_bland_altman(d, args.out)
    plot_gru_05_cdf(d, args.out)
    plot_gru_06_mae_position_bar(d, args.out)
    plot_gru_07_metrics_table(d, args.out)

    print("\n── Signal comparison charts ─────────────────────────────────────")
    plot_signal_case_studies(d, args.out)
    plot_signal_single(d, args.out, subject=args.subject, position=args.position)
    plot_signal_grid(d, args.out)

    print(f"\nDone — 20 charts saved to {args.out}/")


if __name__ == "__main__":
    main()

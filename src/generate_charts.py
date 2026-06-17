"""
generate_charts.py
Generates publication-quality figures from GRU training history and dataset.

Usage
-----
  # All charts that don't need the model (always works):
  python src/generate_charts.py

  # Full set including prediction-based charts — with checkpoint:
  python src/generate_charts.py --ckpt output/gru/best_model_gru.pt

  # Full set using pre-saved predictions:
  python src/generate_charts.py --preds path/to/y_pred_val.npy

ALWAYS generated (history + ground-truth data):
  01  Learning Curve (Train Loss + Val MAE)
  02  Validation MAE Curve (smoothed + convergence zones)
  03  Training Loss Curve (smoothed + phases)
  04  Convergence Analysis (rolling avg + generalization gap)
  05  Mean HR per Position — Ground Truth dots  ← dot chart like example image
  06  Best Participant per Position — dots       ← most stable subject per position
  07  Mean HR per Position — Bar chart
  08  Sample Count per Position — Bar chart
  09  HR Distribution per Position — Violin plot
  10  HR Distribution per Position — Box plot
  11  Sample Count per Subject — Bar chart

GENERATED when predictions are available:
  12  Mean HR per Position — Real vs. Predicted  ← full example-image style
  13  Best Collection per Position — Real vs. Predicted dots
  14  MAE per Position — Bar chart
  15  MAE per Position — Dot + band chart
  16  MAE per Subject — Bar chart
  17  MAE by HR Range — Bar chart
  18  Scatter: Real vs. Predicted
  19  Bland-Altman Agreement Analysis
  20  Residual Distribution
  21  CDF of Absolute Error
  22  Error Box Plot per Position
  23  Error Heatmap — Subject × Position
  24  Metrics Summary Table

Recommended for scientific publication: 05, 12, 13, 18, 19, 21, 22, 23, 24.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import uniform_filter1d

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
HISTORY_F    = ROOT / "history_gru.json"
DATA_DIR     = ROOT / "saida_full"
DEFAULT_CKPT = ROOT / "output" / "gru" / "best_model_gru.pt"

# ─── Style ────────────────────────────────────────────────────────────────────
DARK_RED   = "#6B0000"
LIGHT_ROSE = "#D4919B"
ROSE_FILL  = "#F0B8C0"
ACCENT     = "#A23B72"
BLUE       = "#1565C0"
BLUE_LIGHT = "#90CAF9"
BG_FIG     = "#F2F2F7"
BG_AX      = "#FFFFFF"
GRID_CLR   = "#E0E0E0"

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


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _save(fig, name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  [OK] {path.name}")
    plt.close(fig)


def _int_axes(ax, x=True, y=True):
    if x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if y:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def _red_title_bar(fig, title):
    """Dark red header bar like the example image."""
    bar = fig.add_axes([0.0, 0.955, 1.0, 0.045])
    bar.set_facecolor(DARK_RED)
    bar.axis("off")
    bar.text(0.5, 0.5, title, ha="center", va="center",
             fontsize=12, fontweight="bold", color="white",
             transform=bar.transAxes)


# ═══════════════════════════════════════════════════════════════════════════════
#  GROUP A — History-based charts (no model, no data needed)
# ═══════════════════════════════════════════════════════════════════════════════

def chart_01_learning_curve(history, out_dir):
    train = np.array(history["train_loss"])
    val   = np.array(history["val_mae"])
    eps   = np.arange(1, len(train) + 1)
    best  = int(np.argmin(val))

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        ax.plot(eps, train, color=BLUE,      lw=2.2, label="Training Loss (Huber)")
        ax.plot(eps, val,   color=DARK_RED,  lw=2.2, ls="--", label="Validation MAE (BPM)")
        ax.axvline(best + 1, color="#BDBDBD", ls=":", lw=1.4)
        ax.scatter([best + 1], [val[best]], s=100, color=DARK_RED, zorder=5)
        offset = max(5, len(eps) // 10)
        ax.annotate(
            f"Best  Epoch {best + 1}\nMAE = {round(val[best])} BPM",
            xy=(best + 1, val[best]),
            xytext=(best + 1 + offset, val[best] + (val.max() - val[best]) * 0.3 + 0.5),
            fontsize=9, color="#333",
            arrowprops=dict(arrowstyle="->", color="#9E9E9E", lw=1),
        )
        ax.set_title("GRU — Learning Curve", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss / MAE (BPM)")
        _int_axes(ax)
        ax.legend()
        ax.grid(True, alpha=0.4, ls="--", color=GRID_CLR)
        fig.tight_layout()
        _save(fig, "chart_01_learning_curve", out_dir)


def chart_02_val_mae(history, out_dir):
    val  = np.array(history["val_mae"])
    eps  = np.arange(1, len(val) + 1)
    sm10 = uniform_filter1d(val, size=10)
    sm30 = uniform_filter1d(val, size=30)
    best = int(np.argmin(val))

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        ax.plot(eps, val,  color=ROSE_FILL,  lw=1.0, alpha=0.6, label="Val MAE (raw)")
        ax.plot(eps, sm10, color=LIGHT_ROSE, lw=2.0, label="Smoothed (w=10)")
        ax.plot(eps, sm30, color=DARK_RED,   lw=2.5, label="Smoothed (w=30)")
        ax.fill_between(eps, sm30, alpha=0.08, color=DARK_RED)
        ax.axhline(val[best], color="#BDBDBD", ls=":", lw=1.2)
        ax.scatter([best + 1], [val[best]], s=110, color=DARK_RED, zorder=5,
                   label=f"Best MAE = {round(val[best])} BPM (ep. {best + 1})")
        ax.set_title("GRU — Validation MAE over Epochs", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MAE (BPM)")
        _int_axes(ax)
        ax.legend()
        ax.grid(True, alpha=0.4, ls="--", color=GRID_CLR)
        fig.tight_layout()
        _save(fig, "chart_02_val_mae_curve", out_dir)


def chart_03_training_loss(history, out_dir):
    train = np.array(history["train_loss"])
    eps   = np.arange(1, len(train) + 1)
    sm    = uniform_filter1d(train, size=8)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        ax.plot(eps, train, color=BLUE_LIGHT, lw=1.0, alpha=0.5, label="Train Loss (raw)")
        ax.plot(eps, sm,    color=BLUE,       lw=2.5, label="Smoothed (w=8)")
        ax.fill_between(eps, sm, alpha=0.10, color=BLUE)

        for lo, hi, clr, lbl in [
            (1,   50,        "#FFCDD2", "Phase 1\nRapid"),
            (51,  150,       "#FFF9C4", "Phase 2\nSteady"),
            (151, len(eps),  "#C8E6C9", "Phase 3\nFine-tune"),
        ]:
            ax.axvspan(lo, min(hi, len(eps)), alpha=0.10, color=clr)
            mid = (lo + min(hi, len(eps))) / 2
            ax.text(mid, sm[0] * 0.90, lbl, ha="center", fontsize=8,
                    color="#666", style="italic")

        ax.set_title("GRU — Training Loss (Huber) over Epochs", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Huber Loss")
        _int_axes(ax)
        ax.legend()
        ax.grid(True, alpha=0.4, ls="--", color=GRID_CLR)
        fig.tight_layout()
        _save(fig, "chart_03_training_loss", out_dir)


def chart_04_convergence(history, out_dir):
    val   = np.array(history["val_mae"])
    train = np.array(history["train_loss"])
    eps   = np.arange(1, len(val) + 1)
    w20   = uniform_filter1d(val, size=20)
    w50   = uniform_filter1d(val, size=50)
    gap   = train - val

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor(BG_FIG)
        fig.suptitle("GRU — Convergence Analysis",
                     fontsize=15, fontweight="bold", color="#1A1A2E", y=1.01)

        ax = axes[0]; ax.set_facecolor(BG_AX)
        ax.plot(eps, val,  color=ROSE_FILL,  lw=1.0, alpha=0.4, label="Val MAE (raw)")
        ax.plot(eps, w20,  color=LIGHT_ROSE, lw=2.0, label="Rolling avg (w=20)")
        ax.plot(eps, w50,  color=DARK_RED,   lw=2.5, label="Rolling avg (w=50)")
        ax.axvspan(1,   50,       alpha=0.08, color="#E53935", label="Phase 1 (1–50)")
        ax.axvspan(50,  150,      alpha=0.06, color="#FFA726", label="Phase 2 (51–150)")
        ax.axvspan(150, len(eps), alpha=0.05, color="#66BB6A", label="Phase 3 (151+)")
        ax.set_title("Validation MAE — Training Phases", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (BPM)")
        _int_axes(ax); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4, ls="--", color=GRID_CLR)

        ax = axes[1]; ax.set_facecolor(BG_AX)
        ax.plot(eps, gap, color=ACCENT, lw=1.8, label="Train Loss − Val MAE")
        ax.axhline(0, color="#9E9E9E", ls="--", lw=1.3)
        ax.fill_between(eps, gap, where=(gap > 0), alpha=0.12, color=ACCENT,
                        label="Generalization gap")
        ax.fill_between(eps, gap, where=(gap < 0), alpha=0.12, color="#E53935",
                        label="Underfitting zone")
        ax.set_title("Generalization Gap (Train − Val)", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Difference (BPM)")
        _int_axes(ax); ax.legend()
        ax.grid(True, alpha=0.4, ls="--", color=GRID_CLR)

        fig.tight_layout()
        _save(fig, "chart_04_convergence", out_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  GROUP B — Ground-truth data charts (no predictions needed)
# ═══════════════════════════════════════════════════════════════════════════════

def chart_05_hr_position_dots_gt(y_true, positions, out_dir):
    """Mean HR per position — dot chart (Ground Truth only, example-image style)."""
    pos_ids  = np.unique(positions)
    gt_means = np.array([float(np.mean(y_true[positions == p])) for p in pos_ids])
    gt_stds  = np.array([float(np.std(y_true[positions == p]))  for p in pos_ids])

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(13, 7))
        _red_title_bar(fig, "Mean Heart Rate per Body Position  —  Smartwatch (Ground Truth)")
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        ax.errorbar(pos_ids, gt_means, yerr=gt_stds, fmt="none",
                    color="#AAAAAA", lw=1.2, capsize=4, zorder=1)
        ax.scatter(pos_ids, gt_means, s=180, color=ROSE_FILL,
                   edgecolors=DARK_RED, linewidths=1.8, zorder=3,
                   label="Smartwatch (Ground Truth)")

        ax.set_xlabel("Body Position")
        ax.set_ylabel("Mean HR (BPM)")
        ax.set_xticks(pos_ids)
        _int_axes(ax)
        ax.legend()
        ax.grid(True, alpha=0.30, ls="--", color=GRID_CLR)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, "chart_05_hr_position_dots_gt", out_dir)


def chart_06_best_participant_position_dots(y_true, positions, subjects, out_dir):
    """
    Best participant per position: subject with lowest HR variability (std dev).
    Dot chart showing that participant's mean HR per position.
    """
    pos_ids    = np.unique(positions)
    best_means = []
    best_subjs = []

    for p in pos_ids:
        p_mask   = positions == p
        p_subjs  = np.unique(subjects[p_mask])
        best_std = np.inf
        best_mu  = 0.0
        best_s   = -1
        for s in p_subjs:
            mask = p_mask & (subjects == s)
            if mask.sum() < 3:
                continue
            sd = float(np.std(y_true[mask]))
            if sd < best_std:
                best_std = sd
                best_mu  = float(np.mean(y_true[mask]))
                best_s   = int(s)
        best_means.append(best_mu)
        best_subjs.append(best_s)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(13, 7))
        _red_title_bar(fig,
            "Most Consistent Participant per Position  —  Mean HR (Ground Truth)")
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        # overall population mean per position for reference
        pop_means = [float(np.mean(y_true[positions == p])) for p in pos_ids]
        ax.scatter(pos_ids, pop_means, s=120, color=ROSE_FILL,
                   edgecolors=DARK_RED, linewidths=1.2, zorder=2,
                   alpha=0.55, label="Population Mean (all subjects)")

        for x, pm, bm in zip(pos_ids, pop_means, best_means):
            ax.plot([x, x], [pm, bm], color="#CCCCCC", lw=1.2, zorder=1)

        ax.scatter(pos_ids, best_means, s=200, color=DARK_RED,
                   edgecolors="#3A0000", linewidths=1.5, zorder=3,
                   label="Best Subject (lowest HR std dev)")

        for x, bm, s in zip(pos_ids, best_means, best_subjs):
            ax.text(x, bm + 0.6, f"S{s}", ha="center", fontsize=7.5,
                    color="#3A0000", fontweight="bold")

        ax.set_xlabel("Body Position")
        ax.set_ylabel("Mean HR (BPM)")
        ax.set_xticks(pos_ids)
        _int_axes(ax)
        ax.legend()
        ax.grid(True, alpha=0.30, ls="--", color=GRID_CLR)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, "chart_06_best_participant_per_position", out_dir)


def chart_07_mean_hr_bar(y_true, positions, out_dir):
    """Bar chart of mean HR ± std per body position."""
    pos_ids  = np.unique(positions)
    means    = np.array([float(np.mean(y_true[positions == p])) for p in pos_ids])
    stds     = np.array([float(np.std(y_true[positions == p]))  for p in pos_ids])
    overall  = float(np.mean(y_true))

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(13, 6))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        bars = ax.bar(pos_ids, means, yerr=stds, color=DARK_RED,
                      edgecolor="#3A0000", linewidth=0.6, alpha=0.85,
                      error_kw=dict(ecolor="#555", capsize=4, lw=1.2))
        ax.axhline(overall, color="#757575", ls="--", lw=1.8,
                   label=f"Overall mean = {round(overall)} BPM")

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    str(round(m)), ha="center", fontsize=8.5, color="#222")

        ax.set_title("Mean Heart Rate per Body Position (Ground Truth)",
                     fontweight="bold")
        ax.set_xlabel("Body Position")
        ax.set_ylabel("Mean HR (BPM)")
        ax.set_xticks(pos_ids)
        _int_axes(ax)
        ax.legend()
        ax.grid(True, alpha=0.35, ls="--", color=GRID_CLR, axis="y")
        fig.tight_layout()
        _save(fig, "chart_07_mean_hr_per_position_bar", out_dir)


def chart_08_sample_count_bar(positions, subjects, out_dir):
    """Bar chart: sample count per body position, stacked by split."""
    pos_ids = np.unique(positions)
    counts  = [int(np.sum(positions == p)) for p in pos_ids]
    n_subjs = [int(len(np.unique(subjects[positions == p]))) for p in pos_ids]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(13, 6))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        bars = ax.bar(pos_ids, counts, color=DARK_RED,
                      edgecolor="#3A0000", linewidth=0.6, alpha=0.85)
        for bar, c, ns in zip(bars, counts, n_subjs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 8,
                    f"n={c}\n({ns} subj)", ha="center", fontsize=7.5, color="#333")

        ax.set_title("Sample Count per Body Position (Validation Set)",
                     fontweight="bold")
        ax.set_xlabel("Body Position")
        ax.set_ylabel("Number of Samples")
        ax.set_xticks(pos_ids)
        _int_axes(ax)
        ax.grid(True, alpha=0.35, ls="--", color=GRID_CLR, axis="y")
        fig.tight_layout()
        _save(fig, "chart_08_sample_count_per_position", out_dir)


def chart_09_hr_violin(y_true, positions, out_dir):
    """HR distribution per position — violin plot."""
    pos_ids = np.unique(positions)
    data    = [y_true[positions == p] for p in pos_ids]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        vp = ax.violinplot(data, positions=pos_ids,
                           showmedians=True, showextrema=True)
        for body in vp["bodies"]:
            body.set_facecolor(ROSE_FILL)
            body.set_edgecolor(DARK_RED)
            body.set_alpha(0.65)
        for part in ("cmedians", "cbars", "cmins", "cmaxes"):
            vp[part].set_color(DARK_RED)
            vp[part].set_linewidth(1.8)

        ax.set_xticks(pos_ids)
        _int_axes(ax)
        ax.set_title("HR Distribution per Body Position — Ground Truth",
                     fontweight="bold")
        ax.set_xlabel("Body Position")
        ax.set_ylabel("Heart Rate (BPM)")
        ax.grid(True, alpha=0.35, ls="--", color=GRID_CLR, axis="y")
        fig.tight_layout()
        _save(fig, "chart_09_hr_violin_per_position", out_dir)


def chart_10_hr_boxplot_gt(y_true, positions, out_dir):
    """HR distribution per position — box plot (Ground Truth)."""
    pos_ids = np.unique(positions)
    data    = [y_true[positions == p] for p in pos_ids]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        bp = ax.boxplot(
            data, positions=pos_ids, widths=0.55, patch_artist=True,
            medianprops=dict(color=DARK_RED, lw=2.2),
            whiskerprops=dict(color=DARK_RED, lw=1.4),
            capprops=dict(color=DARK_RED, lw=1.4),
            flierprops=dict(marker="o", color=DARK_RED,
                            alpha=0.25, markersize=3, markeredgewidth=0),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(ROSE_FILL)
            patch.set_alpha(0.65)

        ax.set_xticks(pos_ids)
        _int_axes(ax)
        ax.set_title("HR Distribution per Body Position — Ground Truth (Box Plot)",
                     fontweight="bold")
        ax.set_xlabel("Body Position")
        ax.set_ylabel("Heart Rate (BPM)")
        ax.grid(True, alpha=0.30, ls="--", color=GRID_CLR, axis="y")
        fig.tight_layout()
        _save(fig, "chart_10_hr_boxplot_per_position", out_dir)


def chart_11_samples_per_subject(subjects, out_dir):
    """Sample count per subject — bar chart."""
    subj_ids = np.unique(subjects)
    counts   = [int(np.sum(subjects == s)) for s in subj_ids]
    mean_c   = float(np.mean(counts))

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(18, 6))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        colors = [DARK_RED if c < mean_c else LIGHT_ROSE for c in counts]
        ax.bar(range(len(subj_ids)), counts, color=colors,
               edgecolor="#3A0000", linewidth=0.4, alpha=0.88)
        ax.axhline(mean_c, color="#757575", ls="--", lw=1.8,
                   label=f"Mean = {round(mean_c)} samples")
        ax.set_xticks(range(len(subj_ids)))
        ax.set_xticklabels([str(s) for s in subj_ids],
                           fontsize=7, rotation=45, ha="right")
        _int_axes(ax)
        ax.set_title("Sample Count per Subject (Validation Set)", fontweight="bold")
        ax.set_xlabel("Subject ID")
        ax.set_ylabel("Number of Samples")
        ax.legend()
        ax.grid(True, alpha=0.35, ls="--", color=GRID_CLR, axis="y")
        fig.tight_layout()
        _save(fig, "chart_11_sample_count_per_subject", out_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  GROUP C — Prediction-based charts (require y_pred)
# ═══════════════════════════════════════════════════════════════════════════════

def chart_12_hr_position_real_vs_pred(y_true, y_pred, positions, out_dir):
    """Mean HR per position — Real vs. Predicted dots (example-image style)."""
    pos_ids    = np.unique(positions)
    gt_means   = [float(np.mean(y_true[positions == p])) for p in pos_ids]
    pred_means = [float(np.mean(y_pred[positions == p])) for p in pos_ids]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(13, 7))
        _red_title_bar(fig,
            "Mean Heart Rate per Position  —  Real vs. Predicted")
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        for x, gt, pr in zip(pos_ids, gt_means, pred_means):
            ax.plot([x, x], [gt, pr], color="#CCCCCC", lw=1.2, zorder=1)

        ax.scatter(pos_ids, gt_means,   s=170, color=ROSE_FILL,
                   edgecolors=DARK_RED, linewidths=1.5, zorder=3,
                   label="Smartwatch (Ground Truth)", alpha=0.92)
        ax.scatter(pos_ids, pred_means, s=170, color=DARK_RED,
                   edgecolors="#3A0000", linewidths=1.5, zorder=3,
                   label="Wi-Cardio / GRU (Predicted)")

        ax.set_xlabel("Body Position")
        ax.set_ylabel("Mean HR (BPM)")
        ax.set_xticks(pos_ids)
        _int_axes(ax)
        ax.legend()
        ax.grid(True, alpha=0.30, ls="--", color=GRID_CLR)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, "chart_12_hr_position_real_vs_pred", out_dir)


def chart_13_best_collection_real_vs_pred(y_true, y_pred, positions, subjects, out_dir):
    """
    Best collection per position: subject with lowest MAE.
    Dot chart — Real vs. Predicted (example-image style).
    """
    pos_ids       = np.unique(positions)
    best_gt_means = []
    best_pd_means = []
    best_subjs    = []

    for p in pos_ids:
        p_mask   = positions == p
        p_subjs  = np.unique(subjects[p_mask])
        best_mae = np.inf
        best_gt = best_pd = 0.0
        best_s  = -1
        for s in p_subjs:
            mask = p_mask & (subjects == s)
            if mask.sum() < 3:
                continue
            m = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
            if m < best_mae:
                best_mae = m
                best_gt  = float(np.mean(y_true[mask]))
                best_pd  = float(np.mean(y_pred[mask]))
                best_s   = int(s)
        best_gt_means.append(best_gt)
        best_pd_means.append(best_pd)
        best_subjs.append(best_s)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(13, 7))
        _red_title_bar(fig,
            "Best Collection per Position  —  Real vs. Predicted  (lowest MAE subject)")
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        for x, gt, pr in zip(pos_ids, best_gt_means, best_pd_means):
            ax.plot([x, x], [gt, pr], color="#CCCCCC", lw=1.2, zorder=1)

        ax.scatter(pos_ids, best_gt_means, s=180, color=ROSE_FILL,
                   edgecolors=DARK_RED, linewidths=1.5, zorder=3,
                   label="Smartwatch — Best Subject (GT)", alpha=0.95)
        ax.scatter(pos_ids, best_pd_means, s=180, color=DARK_RED,
                   edgecolors="#3A0000", linewidths=1.5, zorder=3,
                   label="Wi-Cardio / GRU — Best Subject (Predicted)")

        for x, pd, s in zip(pos_ids, best_pd_means, best_subjs):
            ax.text(x, pd + 0.6, f"S{s}", ha="center", fontsize=7.5,
                    color="#3A0000", fontweight="bold")

        ax.set_xlabel("Body Position")
        ax.set_ylabel("Mean HR (BPM)")
        ax.set_xticks(pos_ids)
        _int_axes(ax)
        ax.legend()
        ax.grid(True, alpha=0.35, ls="--", color=GRID_CLR)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, "chart_13_best_collection_per_position", out_dir)


def chart_14_mae_per_position_bar(y_true, y_pred, positions, out_dir):
    pos_ids = np.unique(positions)
    pos_mae = [float(np.mean(np.abs(y_true[positions == p] - y_pred[positions == p])))
               for p in pos_ids]
    overall = float(np.mean(np.abs(y_true - y_pred)))

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(13, 6))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        colors = [DARK_RED if m > overall else LIGHT_ROSE for m in pos_mae]
        bars   = ax.bar(pos_ids, pos_mae, color=colors,
                        edgecolor="#3A0000", linewidth=0.6, alpha=0.88)
        ax.axhline(overall, color="#757575", ls="--", lw=1.8,
                   label=f"Overall MAE = {round(overall)} BPM")

        for bar, val in zip(bars, pos_mae):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    str(round(val)), ha="center", fontsize=8.5, color="#333")

        ax.set_title("GRU — MAE per Body Position", fontweight="bold")
        ax.set_xlabel("Body Position")
        ax.set_ylabel("MAE (BPM)")
        ax.set_xticks(pos_ids)
        _int_axes(ax)
        ax.legend()
        ax.grid(True, alpha=0.35, ls="--", color=GRID_CLR, axis="y")
        fig.tight_layout()
        _save(fig, "chart_14_mae_per_position_bar", out_dir)


def chart_15_mae_position_dot_band(y_true, y_pred, positions, out_dir):
    """MAE per position — dot connected by line + ±1 std band."""
    pos_ids = np.unique(positions)
    mae_arr = [np.abs(y_true[positions == p] - y_pred[positions == p]) for p in pos_ids]
    means   = np.array([float(np.mean(m)) for m in mae_arr])
    stds    = np.array([float(np.std(m))  for m in mae_arr])

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(13, 6))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        ax.fill_between(pos_ids, means - stds, means + stds,
                        alpha=0.15, color=DARK_RED, label="±1 std dev")
        ax.plot(pos_ids, means, color=DARK_RED, lw=2.2, zorder=3)
        ax.scatter(pos_ids, means, s=130, color=ROSE_FILL,
                   edgecolors=DARK_RED, linewidths=1.5, zorder=4,
                   label="Mean MAE per position")

        overall = float(np.mean(means))
        ax.axhline(overall, color="#757575", ls="--", lw=1.6,
                   label=f"Overall avg = {round(overall)} BPM")

        ax.set_xticks(pos_ids)
        _int_axes(ax)
        ax.set_title("GRU — MAE per Position with Variability Band", fontweight="bold")
        ax.set_xlabel("Body Position")
        ax.set_ylabel("MAE (BPM)")
        ax.legend()
        ax.grid(True, alpha=0.35, ls="--", color=GRID_CLR)
        fig.tight_layout()
        _save(fig, "chart_15_mae_position_dot_band", out_dir)


def chart_16_mae_per_subject_bar(y_true, y_pred, subjects, out_dir):
    subj_ids = np.unique(subjects)
    subj_mae = [float(np.mean(np.abs(y_true[subjects == s] - y_pred[subjects == s])))
                for s in subj_ids]
    overall  = float(np.mean(np.abs(y_true - y_pred)))

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(18, 6))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        colors = [DARK_RED if m > overall else LIGHT_ROSE for m in subj_mae]
        ax.bar(range(len(subj_ids)), subj_mae, color=colors,
               edgecolor="#3A0000", linewidth=0.4, alpha=0.88)
        ax.axhline(overall, color="#757575", ls="--", lw=1.8,
                   label=f"Overall MAE = {round(overall)} BPM")
        ax.set_xticks(range(len(subj_ids)))
        ax.set_xticklabels([str(s) for s in subj_ids],
                           fontsize=7, rotation=45, ha="right")
        _int_axes(ax)
        ax.set_title("GRU — MAE per Subject", fontweight="bold")
        ax.set_xlabel("Subject ID")
        ax.set_ylabel("MAE (BPM)")
        ax.legend()
        ax.grid(True, alpha=0.35, ls="--", color=GRID_CLR, axis="y")
        fig.tight_layout()
        _save(fig, "chart_16_mae_per_subject", out_dir)


def chart_17_mae_hr_range_bar(y_true, y_pred, out_dir):
    abs_err = np.abs(y_true - y_pred)
    bins    = [40, 60, 80, 100, 120, 200]
    lbls    = ["40–60", "60–80", "80–100", "100–120", "120+"]
    maes, ns = [], []
    for i in range(len(bins) - 1):
        m = (y_true >= bins[i]) & (y_true < bins[i + 1])
        if m.sum() > 0:
            maes.append(float(np.mean(abs_err[m])))
            ns.append(int(m.sum()))

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        bars = ax.bar(range(len(maes)), maes, color=DARK_RED,
                      edgecolor="#3A0000", linewidth=0.7, alpha=0.88)
        for rect, n, val in zip(bars, ns, maes):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.08,
                    f"n={n}", ha="center", fontsize=9, color="#555")

        ax.set_xticks(range(len(maes)))
        ax.set_xticklabels(lbls[:len(maes)], fontsize=11)
        _int_axes(ax, x=False)
        ax.set_title("GRU — MAE by Heart Rate Range", fontweight="bold")
        ax.set_xlabel("HR Range (BPM)")
        ax.set_ylabel("MAE (BPM)")
        ax.grid(True, alpha=0.35, ls="--", color=GRID_CLR, axis="y")
        fig.tight_layout()
        _save(fig, "chart_17_mae_per_hr_range", out_dir)


def chart_18_scatter(y_true, y_pred, out_dir):
    from sklearn.metrics import mean_absolute_error, r2_score
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2   = float(r2_score(y_true, y_pred))
    lo   = int(min(y_true.min(), y_pred.min())) - 3
    hi   = int(max(y_true.max(), y_pred.max())) + 3

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, 8))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        ax.scatter(y_true, y_pred, alpha=0.20, s=12, color=DARK_RED,
                   edgecolors="none", rasterized=True, label="Samples")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.8, label="Identity (y=x)")
        m, b = np.polyfit(y_true, y_pred, 1)
        xs   = np.linspace(lo, hi, 300)
        ax.plot(xs, m * xs + b, color=ACCENT, lw=2,
                label=f"Linear fit (slope = {m:.2f})")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        _int_axes(ax)
        ax.text(0.04, 0.97,
                f"MAE  = {round(mae)} BPM\nRMSE = {round(rmse)} BPM\nR²    = {r2:.3f}",
                transform=ax.transAxes, fontsize=10.5, va="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=GRID_CLR, alpha=0.95))
        ax.set_title("GRU — Real vs. Predicted Heart Rate", fontweight="bold")
        ax.set_xlabel("Ground Truth — Smartwatch (BPM)")
        ax.set_ylabel("Predicted — Wi-Cardio / GRU (BPM)")
        ax.legend()
        ax.grid(True, alpha=0.30, ls="--", color=GRID_CLR)
        fig.tight_layout()
        _save(fig, "chart_18_scatter_real_vs_pred", out_dir)


def chart_19_bland_altman(y_true, y_pred, out_dir):
    err   = y_true - y_pred
    mean_ = (y_true + y_pred) / 2.0
    bias  = float(np.mean(err))
    sigma = float(np.std(err))
    loa_u = bias + 1.96 * sigma
    loa_l = bias - 1.96 * sigma
    xlo   = float(mean_.min()) - 2
    xhi   = float(mean_.max()) + 2

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        ax.scatter(mean_, err, alpha=0.18, s=12, color=DARK_RED,
                   edgecolors="none", rasterized=True)
        ax.axhline(bias,  color=DARK_RED,  lw=2.2, label=f"Bias = {round(bias)} BPM")
        ax.axhline(loa_u, color="#757575", lw=1.6, ls="--",
                   label=f"+1.96σ = {round(loa_u)} BPM")
        ax.axhline(loa_l, color="#757575", lw=1.6, ls="--",
                   label=f"−1.96σ = {round(loa_l)} BPM")
        ax.fill_between([xlo, xhi], loa_l, loa_u, alpha=0.06, color=DARK_RED)
        ax.set_xlim(xlo, xhi)
        ax.text(xhi - 3, loa_u + 0.4, f"+1.96σ = {round(loa_u)}", fontsize=9, color="#555")
        ax.text(xhi - 3, loa_l - 0.9, f"−1.96σ = {round(loa_l)}", fontsize=9, color="#555")
        _int_axes(ax)
        ax.set_title("GRU — Bland-Altman Agreement Analysis", fontweight="bold")
        ax.set_xlabel("Mean (Smartwatch + GRU) / 2 (BPM)")
        ax.set_ylabel("Difference: GT − Predicted (BPM)")
        ax.legend()
        ax.grid(True, alpha=0.30, ls="--", color=GRID_CLR)
        fig.tight_layout()
        _save(fig, "chart_19_bland_altman", out_dir)


def chart_20_error_dist(y_true, y_pred, out_dir):
    from scipy.stats import norm as sp_norm
    err   = y_true - y_pred
    bias  = float(np.mean(err))
    sigma = float(np.std(err))

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        ax.hist(err, bins=55, color=DARK_RED, edgecolor="white",
                alpha=0.75, density=True, label="Residuals")
        xr = np.linspace(err.min() - 3, err.max() + 3, 500)
        ax.plot(xr, sp_norm.pdf(xr, bias, sigma), color=ACCENT, lw=2.5,
                label=f"Normal(μ={round(bias)}, σ={round(sigma)})")
        ax.axvline(0,    color="#212121", ls="--", lw=1.8, label="Zero error")
        ax.axvline(bias, color=DARK_RED,  ls=":",  lw=1.5, alpha=0.85, label="Bias")
        _int_axes(ax, y=False)
        ax.set_title("GRU — Residual Distribution", fontweight="bold")
        ax.set_xlabel("Error = GT − Predicted (BPM)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.35, ls="--", color=GRID_CLR)
        fig.tight_layout()
        _save(fig, "chart_20_error_distribution", out_dir)


def chart_21_cdf(y_true, y_pred, out_dir):
    abs_err   = np.abs(y_true - y_pred)
    sorted_ae = np.sort(abs_err)
    cdf_vals  = np.arange(1, len(sorted_ae) + 1) / len(sorted_ae) * 100

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        ax.plot(sorted_ae, cdf_vals, color=DARK_RED, lw=2.5)
        ax.fill_between(sorted_ae, 0, cdf_vals, alpha=0.10, color=DARK_RED)

        for thr, ls_s, y_txt in [(5, "--", 10), (10, ":", 24), (15, "-.", 38)]:
            pct = int(round(float(np.mean(abs_err <= thr) * 100)))
            ax.axvline(thr, color="#9E9E9E", ls=ls_s, lw=1.3)
            ax.text(thr + 0.25, y_txt, f"{pct}%\n≤{thr} BPM",
                    fontsize=9, color="#555", va="bottom")

        _int_axes(ax)
        ax.set_ylim(0, 103)
        ax.set_title("GRU — Cumulative Distribution of Absolute Error", fontweight="bold")
        ax.set_xlabel("|Error| (BPM)")
        ax.set_ylabel("Cumulative Samples (%)")
        ax.grid(True, alpha=0.35, ls="--", color=GRID_CLR)
        fig.tight_layout()
        _save(fig, "chart_21_cdf_abs_error", out_dir)


def chart_22_error_boxplot_position(y_true, y_pred, positions, out_dir):
    pos_ids = np.unique(positions)
    errs    = [y_true[positions == p] - y_pred[positions == p] for p in pos_ids]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)

        bp = ax.boxplot(
            errs, positions=pos_ids, widths=0.55, patch_artist=True,
            medianprops=dict(color=DARK_RED, lw=2.2),
            whiskerprops=dict(color=DARK_RED, lw=1.4),
            capprops=dict(color=DARK_RED, lw=1.4),
            flierprops=dict(marker="o", color=DARK_RED,
                            alpha=0.25, markersize=3, markeredgewidth=0),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(ROSE_FILL)
            patch.set_alpha(0.65)

        ax.axhline(0, color="#212121", ls="--", lw=1.5, label="Zero Error")
        ax.set_xticks(pos_ids)
        _int_axes(ax)
        ax.set_title("GRU — Error Distribution per Body Position", fontweight="bold")
        ax.set_xlabel("Body Position")
        ax.set_ylabel("Error: GT − Predicted (BPM)")
        ax.legend()
        ax.grid(True, alpha=0.30, ls="--", color=GRID_CLR, axis="y")
        fig.tight_layout()
        _save(fig, "chart_22_error_boxplot_per_position", out_dir)


def chart_23_heatmap(y_true, y_pred, positions, subjects, out_dir):
    """MAE heatmap: subjects (rows) × body positions (cols)."""
    pos_ids  = np.unique(positions)
    subj_ids = np.unique(subjects)
    matrix   = np.full((len(subj_ids), len(pos_ids)), np.nan)

    for si, s in enumerate(subj_ids):
        for pi, p in enumerate(pos_ids):
            mask = (subjects == s) & (positions == p)
            if mask.sum() > 0:
                matrix[si, pi] = float(
                    np.mean(np.abs(y_true[mask] - y_pred[mask]))
                )

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(16, max(8, len(subj_ids) // 5)))
        fig.patch.set_facecolor(BG_FIG)

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        plt.colorbar(im, ax=ax, label="MAE (BPM)")
        ax.set_xticks(range(len(pos_ids)))
        ax.set_xticklabels([str(p) for p in pos_ids])
        ax.set_yticks(range(len(subj_ids)))
        ax.set_yticklabels([str(s) for s in subj_ids], fontsize=6)
        ax.set_title("GRU — MAE Heatmap: Subject × Body Position",
                     fontweight="bold", fontsize=14)
        ax.set_xlabel("Body Position")
        ax.set_ylabel("Subject ID")
        fig.tight_layout()
        _save(fig, "chart_23_mae_heatmap_subject_position", out_dir)


def chart_24_metrics_table(y_true, y_pred, out_dir):
    from sklearn.metrics import mean_absolute_error, r2_score
    abs_err = np.abs(y_true - y_pred)
    err     = y_true - y_pred
    mae     = float(mean_absolute_error(y_true, y_pred))
    rmse    = float(np.sqrt(np.mean(err ** 2)))
    r2      = float(r2_score(y_true, y_pred))
    bias    = float(np.mean(err))
    sigma   = float(np.std(err))
    loa_u   = bias + 1.96 * sigma
    loa_l   = bias - 1.96 * sigma
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1])

    rows = [
        ["MAE",                f"{round(mae)} BPM"],
        ["RMSE",               f"{round(rmse)} BPM"],
        ["R²",                 f"{r2:.4f}"],
        ["Pearson r",          f"{pearson:.4f}"],
        ["Bias (mean error)",  f"{round(bias)} BPM"],
        ["Std Dev (σ error)",  f"{round(sigma)} BPM"],
        ["LoA upper",          f"+{round(loa_u)} BPM"],
        ["LoA lower",          f"{round(loa_l)} BPM"],
        ["% ≤ 5 BPM",          f"{int(round(np.mean(abs_err <= 5) * 100))}%"],
        ["% ≤ 10 BPM",         f"{int(round(np.mean(abs_err <= 10) * 100))}%"],
        ["% ≤ 15 BPM",         f"{int(round(np.mean(abs_err <= 15) * 100))}%"],
        ["N samples",          f"{len(y_true):,}"],
    ]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_FIG)
        ax.axis("off")

        tbl = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                       loc="center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.25, 1.90)

        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#E0E0E0")
            if r == 0:
                cell.set_facecolor(DARK_RED)
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#F5E9EC")
            else:
                cell.set_facecolor("white")

        ax.set_title("GRU — Evaluation Metrics Summary",
                     fontsize=14, fontweight="bold", pad=16)
        fig.tight_layout()
        _save(fig, "chart_24_metrics_table", out_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model inference
# ═══════════════════════════════════════════════════════════════════════════════

def load_predictions(ckpt_path, history):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    input_dim  = history["input_dim"]
    hidden_dim = 256
    n_layers   = 2

    class _GRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers,
                              batch_first=True, dropout=0.3, bidirectional=True)
            h_out = hidden_dim * 2
            self.norm = nn.LayerNorm(h_out)
            self.attn = nn.Linear(h_out, 1)
            self.regressor = nn.Sequential(
                nn.Linear(h_out, 128), nn.LayerNorm(128),
                nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1),
            )

        def forward(self, x):
            x      = self.input_proj(x)
            out, _ = self.gru(x)
            out    = self.norm(out)
            w      = torch.softmax(self.attn(out), dim=1)
            ctx    = (out * w).sum(dim=1)
            return self.regressor(ctx).squeeze(-1)

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available()          else
              torch.device("cpu"))

    model = _GRU().to(device)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model.eval()

    X  = np.load(DATA_DIR / "X_val.npz")["X"].astype(np.float32)
    dl = DataLoader(TensorDataset(torch.from_numpy(X)),
                    batch_size=64, shuffle=False)
    preds = []
    with torch.no_grad():
        for (bx,) in dl:
            preds.append(model(bx.to(device)).cpu().numpy())
    return np.concatenate(preds)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate GRU result charts")
    parser.add_argument("--ckpt",    type=Path, default=None,
                        help="Model checkpoint (.pt)")
    parser.add_argument("--preds",   type=Path, default=None,
                        help="Pre-saved predictions (.npy)")
    parser.add_argument("--out",     type=Path, default=ROOT / "charts_output",
                        help="Output directory")
    parser.add_argument("--history", type=Path, default=HISTORY_F,
                        help="Path to history_gru.json")
    args = parser.parse_args()

    print(f"\nLoading history: {args.history}")
    with open(args.history) as f:
        history = json.load(f)
    print(f"  Epochs: {len(history['train_loss'])}  |  Output: {args.out}\n")

    # ── Group A: history charts ────────────────────────────────────────────────
    print("── Group A: history charts ──────────────────────────────────────")
    chart_01_learning_curve(history, args.out)
    chart_02_val_mae(history, args.out)
    chart_03_training_loss(history, args.out)
    chart_04_convergence(history, args.out)

    # ── Group B: ground-truth data charts ─────────────────────────────────────
    print("\n── Group B: ground-truth data charts ───────────────────────────")
    y_true    = np.load(DATA_DIR / "y_val.npy").astype(np.float32)
    positions = np.load(DATA_DIR / "positions_val.npy")
    subjects  = np.load(DATA_DIR / "subject_val.npy")

    chart_05_hr_position_dots_gt(y_true, positions, args.out)
    chart_06_best_participant_position_dots(y_true, positions, subjects, args.out)
    chart_07_mean_hr_bar(y_true, positions, args.out)
    chart_08_sample_count_bar(positions, subjects, args.out)
    chart_09_hr_violin(y_true, positions, args.out)
    chart_10_hr_boxplot_gt(y_true, positions, args.out)
    chart_11_samples_per_subject(subjects, args.out)

    # ── Group C: prediction-based charts ──────────────────────────────────────
    y_pred = None

    if args.preds is not None:
        print(f"\nLoading predictions: {args.preds}")
        y_pred = np.load(args.preds).astype(np.float32)
    elif args.ckpt is not None:
        print(f"\nRunning inference: {args.ckpt}")
        y_pred = load_predictions(args.ckpt, history)
    elif DEFAULT_CKPT.exists():
        print(f"\nFound default checkpoint: {DEFAULT_CKPT}")
        y_pred = load_predictions(DEFAULT_CKPT, history)

    if y_pred is None:
        print("\n[INFO] Charts 12–24 need predictions. Re-run with:")
        print("         --ckpt  output/gru/best_model_gru.pt")
        print("         --preds path/to/y_pred_val.npy")
        print(f"\nDone — 11 charts saved to {args.out}/")
        return

    assert len(y_pred) == len(y_true), (
        f"Length mismatch: pred={len(y_pred)}, true={len(y_true)}"
    )

    print("\n── Group C: prediction-based charts ─────────────────────────────")
    chart_12_hr_position_real_vs_pred(y_true, y_pred, positions, args.out)
    chart_13_best_collection_real_vs_pred(y_true, y_pred, positions, subjects, args.out)
    chart_14_mae_per_position_bar(y_true, y_pred, positions, args.out)
    chart_15_mae_position_dot_band(y_true, y_pred, positions, args.out)
    chart_16_mae_per_subject_bar(y_true, y_pred, subjects, args.out)
    chart_17_mae_hr_range_bar(y_true, y_pred, args.out)
    chart_18_scatter(y_true, y_pred, args.out)
    chart_19_bland_altman(y_true, y_pred, args.out)
    chart_20_error_dist(y_true, y_pred, args.out)
    chart_21_cdf(y_true, y_pred, args.out)
    chart_22_error_boxplot_position(y_true, y_pred, positions, args.out)
    chart_23_heatmap(y_true, y_pred, positions, subjects, args.out)
    chart_24_metrics_table(y_true, y_pred, args.out)

    print(f"\nDone — 24 charts saved to {args.out}/")


if __name__ == "__main__":
    main()

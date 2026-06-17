"""
Gera os dois gráficos de resultados para o artigo:
  1. MAE por posição corporal — barras agrupadas GRU vs LSTM
  2. FC média por posição — Smartwatch vs GRU (+ variante de participante único por tempo)
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.utils.data import DataLoader, TensorDataset


# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE    = get_device()
INPUT_DIR = Path("saida_full")
OUT_DIR   = Path("charts_output")
OUT_DIR.mkdir(exist_ok=True)
BATCH     = 64


# ── Architectures ─────────────────────────────────────────────────────────────
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
        x = self.input_proj(x)
        out, _ = self.gru(x)
        out = self.norm(out)
        w = torch.softmax(self.attn(out), dim=1)
        return self.regressor((out * w).sum(1)).squeeze(-1)


class PulseFiLSTM(nn.Module):
    def __init__(self, input_dim, hidden=256, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                            dropout=0.3, bidirectional=True)
        h = hidden * 2
        self.regressor = nn.Sequential(
            nn.Linear(h, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.regressor(out[:, -1, :]).squeeze(-1)


# ── Data helpers ──────────────────────────────────────────────────────────────
def load_val():
    X   = np.load(INPUT_DIR / "X_val.npz")["X"].astype(np.float32)
    y   = np.load(INPUT_DIR / "y_val.npy").astype(np.float32)
    pos = np.load(INPUT_DIR / "positions_val.npy")
    sub = np.load(INPUT_DIR / "subject_val.npy")
    dl  = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                     batch_size=BATCH, shuffle=False)
    return dl, y, pos, sub


def infer(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in loader:
            preds.append(model(bx.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)


def load_model(cls, ckpt, input_dim):
    m = cls(input_dim).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    return m


# ── Style helpers ─────────────────────────────────────────────────────────────
def clean_ax(ax):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7, zorder=0)
    ax.tick_params(colors="#444444")


# ── Chart 1: MAE per position — grouped bars GRU vs LSTM ─────────────────────
def chart_mae_per_position(y_true, pos, y_gru, y_lstm):
    pos_ids  = np.sort(np.unique(pos))
    mae_gru  = [np.mean(np.abs(y_true[pos == p] - y_gru[pos == p]))  for p in pos_ids]
    mae_lstm = [np.mean(np.abs(y_true[pos == p] - y_lstm[pos == p])) for p in pos_ids]

    C_GRU  = "#6B0000"
    C_LSTM = "#1565C0"
    w = 0.38
    x = np.arange(len(pos_ids))

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")
    clean_ax(ax)

    bars_g = ax.bar(x - w/2, mae_gru,  w, color=C_GRU,  label="GRU",  zorder=3)
    bars_l = ax.bar(x + w/2, mae_lstm, w, color=C_LSTM, label="LSTM", zorder=3)

    # dashed mean lines with label in legend
    mean_gru  = float(np.mean(mae_gru))
    mean_lstm = float(np.mean(mae_lstm))
    ax.axhline(mean_gru,  color=C_GRU,  lw=1.4, ls="--", alpha=0.8,
               label=f"GRU  overall = {mean_gru:.2f} BPM")
    ax.axhline(mean_lstm, color=C_LSTM, lw=1.4, ls="--", alpha=0.8,
               label=f"LSTM overall = {mean_lstm:.2f} BPM")

    # value labels — 2 decimal places, rotated 60° to avoid overlap
    top = max(max(mae_gru), max(mae_lstm))
    for bar in bars_g:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + top * 0.015,
                f"{h:.2f}", ha="left", va="bottom",
                fontsize=7.2, color=C_GRU, rotation=60, rotation_mode="anchor")
    for bar in bars_l:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + top * 0.015,
                f"{h:.2f}", ha="left", va="bottom",
                fontsize=7.2, color=C_LSTM, rotation=60, rotation_mode="anchor")

    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in pos_ids], fontsize=9)
    ax.set_xlabel("Body Position", fontsize=10.5, labelpad=10)
    ax.set_ylabel("MAE (BPM)", fontsize=10.5)
    ax.legend(frameon=True, fontsize=9, framealpha=0.9, edgecolor="#cccccc")
    ax.set_ylim(0, top * 1.35)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.14)
    out = OUT_DIR / "mae_per_position_gru_lstm.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Salvo: {out}")
    plt.show()


# ── Chart 2a: Mean HR per position — dots Smartwatch vs GRU ──────────────────
def chart_hr_per_position(y_true, pos, y_gru):
    C_GT_EDGE  = "#6B0000"
    C_GT_FACE  = "#D4919B"   # light pink
    C_GRU_FACE = "#6B0000"   # dark red

    pos_ids = np.sort(np.unique(pos))
    hr_gt  = np.array([np.mean(y_true[pos == p]) for p in pos_ids])
    hr_gru = np.array([np.mean(y_gru[pos == p])  for p in pos_ids])

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")
    clean_ax(ax)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7, zorder=0)
    ax.grid(axis="x", visible=False)

    ax.scatter(pos_ids, hr_gt,  s=120, color=C_GT_FACE,  edgecolors=C_GT_EDGE,
               linewidths=1.2, zorder=4, label="Smartwatch (Ground Truth)")
    ax.scatter(pos_ids, hr_gru, s=120, color=C_GRU_FACE, edgecolors=C_GT_EDGE,
               linewidths=1.2, zorder=4, label="Wi-Cardio / GRU (Predicted)")

    # all integer x ticks
    ax.set_xticks(pos_ids)
    ax.set_xticklabels([str(p) for p in pos_ids], fontsize=9)
    ax.set_xlabel("Body Position", fontsize=10.5, labelpad=10)
    ax.set_ylabel("Mean HR (BPM)", fontsize=10.5)

    # integer y-axis ticks
    all_vals = np.concatenate([hr_gt, hr_gru])
    y_lo = int(all_vals.min()) - 3
    y_hi = int(all_vals.max()) + 3
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))

    ax.legend(frameon=True, fontsize=10, framealpha=0.9, edgecolor="#cccccc")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.14)
    out = OUT_DIR / "hr_per_position_gt_vs_gru.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Salvo: {out}")
    plt.show()


# ── Chart 2b: Single participant HR over time — dots Smartwatch vs GRU ────────
def chart_hr_time_single(y_true, pos, sub, y_gru, target_pos=5):
    C_GT_EDGE  = "#6B0000"
    C_GT_FACE  = "#D4919B"
    C_GRU_FACE = "#6B0000"

    mask_pos = pos == target_pos
    if mask_pos.sum() == 0:
        target_pos = np.unique(pos)[0]
        mask_pos   = pos == target_pos

    subs_at_pos, counts = np.unique(sub[mask_pos], return_counts=True)
    best_sub = subs_at_pos[np.argmax(counts)]

    mask = mask_pos & (sub == best_sub)
    gt   = y_true[mask]
    gru  = y_gru[mask]

    # x-axis: time in seconds (step = 0.5 s per window)
    t = np.arange(len(gt)) * 0.5

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")
    clean_ax(ax)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7, zorder=0)
    ax.grid(axis="x", visible=False)

    ax.scatter(t, gt,  s=35, color=C_GT_FACE,  edgecolors=C_GT_EDGE,
               linewidths=0.8, zorder=4, label="Smartwatch (Ground Truth)")
    ax.scatter(t, gru, s=35, color=C_GRU_FACE, edgecolors=C_GT_EDGE,
               linewidths=0.8, zorder=4, label="Wi-Cardio / GRU (Predicted)")

    ax.set_xlabel("Time (s)", fontsize=10.5, labelpad=10)
    ax.set_ylabel("Mean HR (BPM)", fontsize=10.5)

    # all integer y ticks
    all_v = np.concatenate([gt, gru])
    y_lo = int(all_v.min()) - 3
    y_hi = int(all_v.max()) + 3
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))

    ax.legend(frameon=True, fontsize=10, framealpha=0.9, edgecolor="#cccccc")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.14)

    out = OUT_DIR / f"hr_time_subject{best_sub}_pos{target_pos:02d}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Salvo: {out}  (sujeito {best_sub}, posição {target_pos})")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dl, y_true, pos, sub = load_val()
    input_dim = int(np.load(INPUT_DIR / "X_val.npz")["X"].shape[2])

    gru_ckpt  = Path("output/gru/best_model_gru.pt")
    lstm_ckpt = Path("output/lstm/best_model_lstm.pt")

    if not gru_ckpt.exists() or not lstm_ckpt.exists():
        print("Checkpoints não encontrados. Execute train_gru.py e train_lstm.py primeiro.")
        raise SystemExit

    print("Rodando inferência GRU...")
    y_gru  = infer(load_model(PulseFiGRU,  gru_ckpt,  input_dim), dl)
    print("Rodando inferência LSTM...")
    y_lstm = infer(load_model(PulseFiLSTM, lstm_ckpt, input_dim), dl)

    print("\n[1/3] MAE por posição (GRU vs LSTM)...")
    chart_mae_per_position(y_true, pos, y_gru, y_lstm)

    print("[2/3] FC média por posição (Smartwatch vs GRU)...")
    chart_hr_per_position(y_true, pos, y_gru)

    print("[3/3] FC ao longo do tempo — participante único...")
    chart_hr_time_single(y_true, pos, sub, y_gru, target_pos=5)

    print("\nConcluído.")

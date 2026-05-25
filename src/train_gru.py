import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import norm as sp_norm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0005
HIDDEN_SIZE = 128
NUM_LAYERS = 2
SEED = 42
INPUT_DIR = Path("saida_full")


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data():
    X_train = np.load(INPUT_DIR / "X_train.npz")["X"].astype(np.float32)
    X_val = np.load(INPUT_DIR / "X_val.npz")["X"].astype(np.float32)
    y_train = np.load(INPUT_DIR / "y_train.npy").astype(np.float32)
    y_val = np.load(INPUT_DIR / "y_val.npy").astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(val_ds, batch_size=BATCH_SIZE), y_val)

class PulseFiModelGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(PulseFiModelGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.3)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.regressor(gru_out[:, -1, :]).squeeze(-1)

def plot_metrics(history, y_true, y_pred):
    MODEL   = "GRU"
    COLOR   = "#2E86AB"
    ACCENT  = "#A23B72"
    HDR_CLR = "#1565C0"

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    erro     = y_true - y_pred
    abs_erro = np.abs(erro)

    mae   = float(mean_absolute_error(y_true, y_pred))
    r2    = float(r2_score(y_true, y_pred))
    rmse  = float(np.sqrt(np.mean(erro ** 2)))
    bias  = float(np.mean(erro))
    sigma = float(np.std(erro))
    loa_u = bias + 1.96 * sigma
    loa_l = bias - 1.96 * sigma

    rc = {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 10,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 3, figsize=(19, 11))
        fig.patch.set_facecolor("#F7F9FC")
        fig.suptitle(f"Avaliação do Modelo  —  {MODEL}",
                     fontsize=17, fontweight="bold", y=1.01, color="#1A1A2E")

        BG = "#FFFFFF"

        # ── 1. Curva de aprendizado ──────────────────────────────────────────
        ax = axes[0, 0]
        ax.set_facecolor(BG)
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], color="#1976D2", lw=2, label="Treino")
        ax.plot(epochs, history["val_loss"],   color="#E53935", lw=2, ls="--", label="Validação")
        best_ep  = int(np.argmin(history["val_loss"])) + 1
        best_val = history["val_loss"][best_ep - 1]
        ax.axvline(best_ep, color="#BDBDBD", ls=":", lw=1.4)
        offset = max(3, len(epochs) // 12)
        ax.annotate(
            f"Melhor val\nMAE={best_val:.2f}\n(ép. {best_ep})",
            xy=(best_ep, best_val),
            xytext=(best_ep + offset, best_val + (max(history["val_loss"]) - best_val) * 0.25 + 0.2),
            fontsize=8.5, color="#424242",
            arrowprops=dict(arrowstyle="->", color="#9E9E9E", lw=1),
        )
        ax.set_title("Curva de Aprendizado", fontsize=13, fontweight="bold")
        ax.set_xlabel("Época", fontsize=11)
        ax.set_ylabel("MAE (BPM)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.35, ls="--", color="#CFD8DC")

        # ── 2. Real vs Predito ───────────────────────────────────────────────
        ax = axes[0, 1]
        ax.set_facecolor(BG)
        lo = min(y_true.min(), y_pred.min()) - 3
        hi = max(y_true.max(), y_pred.max()) + 3
        ax.scatter(y_true, y_pred, alpha=0.30, s=14, color=COLOR,
                   edgecolors="none", rasterized=True)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.6, label="Ideal (y=x)")
        m_fit, b_fit = np.polyfit(y_true, y_pred, 1)
        xs = np.linspace(lo, hi, 300)
        ax.plot(xs, m_fit * xs + b_fit, color=ACCENT, lw=1.8, alpha=0.9,
                label=f"Regressão (m={m_fit:.2f})")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.text(
            0.04, 0.97,
            f"MAE  = {mae:.2f} BPM\nRMSE = {rmse:.2f} BPM\nR²    = {r2:.3f}",
            transform=ax.transAxes, fontsize=9.5, va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#CFD8DC", alpha=0.95),
        )
        ax.set_title("Real vs Predito", fontsize=13, fontweight="bold")
        ax.set_xlabel("Ground Truth — Smartwatch (BPM)", fontsize=11)
        ax.set_ylabel("Predição — CSI (BPM)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.30, ls="--", color="#CFD8DC")

        # ── 3. Histograma de resíduos ────────────────────────────────────────
        ax = axes[0, 2]
        ax.set_facecolor(BG)
        ax.hist(erro, bins=40, color=COLOR, edgecolor="white", alpha=0.82,
                density=True, label="Resíduos")
        xr = np.linspace(erro.min() - 2, erro.max() + 2, 400)
        ax.plot(xr, sp_norm.pdf(xr, bias, sigma), color="#E53935", lw=2,
                label=f"Normal(μ={bias:.2f}, σ={sigma:.2f})")
        ax.axvline(0,    color="#212121", ls="--", lw=1.5, label="Zero")
        ax.axvline(bias, color="#E53935", ls=":",  lw=1.5, alpha=0.85)
        ax.set_title("Distribuição dos Resíduos", fontsize=13, fontweight="bold")
        ax.set_xlabel("Erro  =  GT − Predição (BPM)", fontsize=11)
        ax.set_ylabel("Densidade", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.35, ls="--", color="#CFD8DC")

        # ── 4. Bland-Altman ──────────────────────────────────────────────────
        ax = axes[1, 0]
        ax.set_facecolor(BG)
        mean_ba = (y_true + y_pred) / 2.0
        xlo, xhi = float(mean_ba.min()) - 2, float(mean_ba.max()) + 2
        ax.scatter(mean_ba, erro, alpha=0.30, s=14, color=COLOR,
                   edgecolors="none", rasterized=True)
        ax.axhline(bias,  color="#E53935", lw=2.0, label=f"Viés = {bias:.2f}")
        ax.axhline(loa_u, color="#757575", lw=1.5, ls="--",
                   label=f"+1.96σ = {loa_u:.2f}")
        ax.axhline(loa_l, color="#757575", lw=1.5, ls="--",
                   label=f"−1.96σ = {loa_l:.2f}")
        ax.fill_between([xlo, xhi], loa_l, loa_u, alpha=0.08, color="#9E9E9E")
        ax.set_xlim(xlo, xhi)
        ax.set_title("Bland-Altman — Concordância", fontsize=13, fontweight="bold")
        ax.set_xlabel("Média (Smartwatch + CSI) / 2 (BPM)", fontsize=11)
        ax.set_ylabel("Diferença (GT − Predição) (BPM)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.30, ls="--", color="#CFD8DC")

        # ── 5. CDF do erro absoluto ──────────────────────────────────────────
        ax = axes[1, 1]
        ax.set_facecolor(BG)
        sorted_ae = np.sort(abs_erro)
        cdf_vals  = np.arange(1, len(sorted_ae) + 1) / len(sorted_ae) * 100
        ax.plot(sorted_ae, cdf_vals, color=COLOR, lw=2.5)
        ax.fill_between(sorted_ae, 0, cdf_vals, alpha=0.13, color=COLOR)
        for thr, ls_style, ytxt in [(5, "--", 10), (10, ":", 22), (15, "-.", 34)]:
            pct = float(np.mean(abs_erro <= thr) * 100)
            ax.axvline(thr, color="#9E9E9E", ls=ls_style, lw=1.3)
            ax.text(thr + 0.25, ytxt, f"{pct:.0f}%\n≤{thr} BPM",
                    fontsize=8.5, color="#555", va="bottom")
        ax.set_title("CDF do Erro Absoluto", fontsize=13, fontweight="bold")
        ax.set_xlabel("|Erro| (BPM)", fontsize=11)
        ax.set_ylabel("Amostras acumuladas (%)", fontsize=11)
        ax.set_ylim(0, 103)
        ax.grid(True, alpha=0.35, ls="--", color="#CFD8DC")

        # ── 6. Tabela de métricas ────────────────────────────────────────────
        ax = axes[1, 2]
        ax.set_facecolor("#F7F9FC")
        ax.axis("off")
        rows = [
            ["MAE",              f"{mae:.3f} BPM"],
            ["RMSE",             f"{rmse:.3f} BPM"],
            ["R²",               f"{r2:.4f}"],
            ["Viés (μ erro)",    f"{bias:.3f} BPM"],
            ["Desvio (σ erro)",  f"{sigma:.3f} BPM"],
            ["LoA superior",     f"+{loa_u:.3f} BPM"],
            ["LoA inferior",     f"{loa_l:.3f} BPM"],
            ["% ≤ 5 BPM",        f"{np.mean(abs_erro <= 5)*100:.1f}%"],
            ["% ≤ 10 BPM",       f"{np.mean(abs_erro <= 10)*100:.1f}%"],
            ["N amostras",       f"{len(y_true):,}"],
        ]
        tbl = ax.table(cellText=rows, colLabels=["Métrica", "Valor"],
                       loc="center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.15, 1.72)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#E0E0E0")
            if r == 0:
                cell.set_facecolor(HDR_CLR)
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#EEF4FB")
            else:
                cell.set_facecolor("white")
        ax.set_title(f"Resumo — {MODEL}", fontsize=13, fontweight="bold", pad=14)

        plt.tight_layout()
        out = "resultado_gru.png"
        plt.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Figura salva: {out}")
        plt.show()

def train():
    set_seed()
    train_loader, val_loader, y_val_true = load_data()
    example_x, _ = next(iter(train_loader))

    model = PulseFiModelGRU(example_x.shape[2], HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )

    history = {'train_loss': [], 'val_loss': []}

    print(f"Iniciando Treino com GRU no dispositivo: {DEVICE}")

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0
        preds_all = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                p = model(bx)
                v_loss += criterion(p, by).item()
                preds_all.append(p.cpu().numpy())

        history['train_loss'].append(t_loss/len(train_loader))
        history['val_loss'].append(v_loss/len(val_loader))
        scheduler.step(history['val_loss'][-1])

        if (epoch + 1) % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} | Val MAE: {history['val_loss'][-1]:.2f} | LR: {lr_now:.2e}")

    y_val_pred = np.concatenate(preds_all)
    
    print("\n--- Métricas Finais (GRU) ---")
    print(f"MAE Final: {mean_absolute_error(y_val_true, y_val_pred):.2f} BPM")
    print(f"R² Score: {r2_score(y_val_true, y_val_pred):.2f}")

    plot_metrics(history, y_val_true, y_val_pred)

if __name__ == "__main__":
    train()
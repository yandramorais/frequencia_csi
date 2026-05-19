from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple, List
from scipy.signal import savgol_filter

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
import logging
from collections import Counter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("preprocess")

DEFAULT_FS = 500 / 60.0
BANDPASS = (0.8, 2.17)
FILTER_ORDER = 3
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

INVALID_SUBJECTS = {
    59, 61, 62, 63, 64, 65, 66, 67, 68, 69,
    70, 71, 98, 203, 90, 106, 81, 10, 9, 35
}


def load_fault_intervals(fp: Path):

    if not fp.exists():
        return []

    intervals = []

    with open(fp, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 2:
                continue

            start = float(parts[0])
            end = float(parts[1])

            intervals.append((start, end))

    return intervals


def is_valid_window(t_mean, fault_intervals):

    for start, end in fault_intervals:

        if start <= t_mean <= end:
            return False

    return True


def savgol_smooth(x: np.ndarray, window: int = 9, poly: int = 3):

    if x.shape[0] < window:
        return x

    if x.ndim == 1:
        return savgol_filter(x, window, poly)

    y = np.empty_like(x)

    for i in range(x.shape[1]):
        y[:, i] = savgol_filter(x[:, i], window, poly)

    return y


def extract_position_from_filename(fp: Path) -> Optional[int]:
    m = re.match(r"^(\d+)_", fp.name)
    if not m:
        return None
    return int(m.group(1))


def bandpass_filter(
    x: np.ndarray, fs: float, low: float, high: float, order: int = FILTER_ORDER
):
    b, a = butter(order, [low, high], btype="band", fs=fs)
    y = np.empty_like(x)
    for i in range(x.shape[1]):
        y[:, i] = filtfilt(b, a, x[:, i])
    return y


def csi_to_amplitude(x: np.ndarray):
    if np.iscomplexobj(x):
        return np.abs(x)
    return x


def remove_dc(x: np.ndarray):
    return x - np.mean(x, axis=0)


def zscore(x: np.ndarray, eps=1e-8):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + eps)


def apply_pca(x: np.ndarray, n_components: int = 1):

    x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8)

    pca = PCA(n_components=n_components)

    return pca.fit_transform(x)


def load_smartwatch_gt(fp: Path) -> Optional[pd.DataFrame]:
    with open(fp, "r") as f:
        js = json.load(f)

    rows = []

    if isinstance(js, dict) and "Data" in js:
        for rec in js["Data"]:
            hr = rec.get("HeartRate") or rec.get("Value")
            t = rec.get("StartTime") or rec.get("Time")
            if hr is None or t is None:
                continue
            ts = pd.to_datetime(t.replace(" ", "T"))
            rows.append((ts, float(hr)))

    elif isinstance(js, dict) and "heart_rate" in js and "start_time" in js:
        for t, hr in zip(js["start_time"], js["heart_rate"]):
            ts = pd.to_datetime(t.replace(" ", "T"))
            rows.append((ts, float(hr)))

    elif isinstance(js, list):
        for rec in js:
            hr = rec.get("HeartRate") or rec.get("Value")
            t = rec.get("StartTime") or rec.get("Time")
            if hr is None or t is None:
                continue
            ts = pd.to_datetime(t.replace(" ", "T"))
            rows.append((ts, float(hr)))

    else:
        return None

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["datetime", "hr_bpm"])
    df["time"] = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds()
    return df[["time", "hr_bpm"]]


def find_matching_gt(gt_root: Path, base_name: str) -> Optional[Path]:
    base_clean = re.sub(r"_bw_.*$", "", base_name)
    for f in gt_root.rglob(f"{base_clean}_HeartRateData.json"):
        return f
    return None


def extract_subject_from_gt(gt_path: Path) -> int:

    return int(gt_path.parent.name)


def load_one_npz(fp: Path):
    z = np.load(fp, allow_pickle=True)

    csi = z["csi"]
    if csi.ndim == 1:
        csi = csi[:, :]
    elif csi.ndim > 2:
        csi = csi.reshape(csi.shape[0], -1)

    ts = z["ts"] if "ts" in z.files else None
    csi = csi_to_amplitude(csi)
    return csi.astype(np.float32), ts
    


def infer_fs(ts):
    if ts is None:
        return None

    try:
        ts = np.array(ts, dtype=float).ravel()
    except Exception:
        return None

    if ts.ndim == 0:
        return None

    if ts.size < 2:
        return None

    dt = np.mean(np.diff(ts))

    if not np.isfinite(dt) or dt <= 0:
        return None

    return 1.0 / dt


def sliding_window_with_gt(
    X,
    ts,
    gt_df,
    fs,
    window,
    step,
    fault_intervals,
    *,
    max_gt_gap_s: float = 0.75,
    hr_jump_bpm: float = 25.0,
    local_hr_window_s: float = 3.0,
    debug_every: int = 2000,
):
    X_out, y_out = [], []

    stats = Counter()
    dt_list = []
    try:
        ts = np.array(ts, dtype=float).ravel()
        if ts.size != len(X):
            raise ValueError
        if not np.all(np.isfinite(ts)):
            raise ValueError
    except Exception:
        ts = np.arange(len(X)) / fs
        stats["ts_fallback_used"] += 1

    gt_t = gt_df["time"].to_numpy(dtype=float)
    gt_hr = gt_df["hr_bpm"].to_numpy(dtype=float)

    if gt_t.size < 2:
        return None, None, {"reason": "gt_t_too_small"}

    gt_t_min, gt_t_max = float(gt_t.min()), float(gt_t.max())
    local_half = float(local_hr_window_s)

    accepted = 0

    for start in range(0, len(X) - window + 1, step):
        end = start + window
        win = X[start:end]

        win = (win - np.mean(win)) / (np.std(win) + 1e-8)

        t_mean = float(np.mean(ts[start:end]))

        if not is_valid_window(t_mean, fault_intervals):
            stats["drop_fault_interval"] += 1
            continue

        # Sempre pega o valor mais próximo (mesmo fora do range)
        idx = int(np.argmin(np.abs(gt_t - t_mean)))

        # Só loga se estiver fora
        if t_mean < gt_t_min or t_mean > gt_t_max:
             stats["out_of_range_but_used"] += 1
        dt = float(gt_t[idx] - t_mean)
        abs_dt = abs(dt)

        if abs_dt > max_gt_gap_s:
            stats["drop_gt_gap_too_large"] += 1
            if stats["drop_gt_gap_too_large"] <= 5:
                log.warning(
                    f"GT GAP LARGE | t_mean={t_mean:.3f}s gt_t={gt_t[idx]:.3f}s dt={dt:+.3f}s "
                    f"(max={max_gt_gap_s}s)"
                )
            continue

        t0 = t_mean - local_half
        t1 = t_mean + local_half
        mask = (gt_t >= t0) & (gt_t <= t1)
        if mask.sum() >= 3:
            local_hr = gt_hr[mask]
            if np.nanmax(local_hr) - np.nanmin(local_hr) > hr_jump_bpm:
                stats["drop_hr_unstable_local"] += 1
                continue

        hr = float(gt_hr[idx])

        X_out.append(win)
        y_out.append(hr)
        dt_list.append(dt)
        accepted += 1

        if accepted % debug_every == 0:
            log.info(
                f"WIN OK sample | accepted={accepted} | mean_abs_dt={np.mean(np.abs(dt_list)):.3f}s"
            )

    if not X_out:
        return None, None, {"reason": "no_windows", "stats": dict(stats)}

    diag = {
        "stats": dict(stats),
        "accepted": int(len(X_out)),
        "mean_dt": float(np.mean(dt_list)) if dt_list else None,
        "mean_abs_dt": float(np.mean(np.abs(dt_list))) if dt_list else None,
        "p95_abs_dt": float(np.percentile(np.abs(dt_list), 95)) if dt_list else None,
    }

    return np.stack(X_out).astype(np.float32), np.array(y_out), diag


def build_windows(
    dataset_path: Path, gt_dir: Path, fs: float, window_sec: float, step_sec: float
):

    all_X, all_y, all_p, all_s = [], [], [], []

    files = sorted(dataset_path.rglob("*.npz"))
    print("Arquivos CSI encontrados:", len(files))

    for fp in files:

        pos = extract_position_from_filename(fp)
        if pos is None:
            continue

        gt_fp = find_matching_gt(gt_dir, fp.stem)

        if gt_fp is None:
            continue

        fault_fp = gt_dir.parent / "faltas.txt"

        fault_intervals = load_fault_intervals(fault_fp)

        gt_df = load_smartwatch_gt(gt_fp)
        if gt_df is None or len(gt_df) == 0:
            continue

        # Agora sim pode mexer
        gt_df["time"] = gt_df["time"] - gt_df["time"].iloc[0]

        subject = extract_subject_from_gt(gt_fp)
        
        if subject in INVALID_SUBJECTS:
            continue

        X_raw, ts = load_one_npz(fp)
        
        if ts is not None:
            try:
                ts = np.array(ts,dtype=float).ravel()
                ts = ts - ts[0]
            except:
                ts = None
        
        fs_eff = infer_fs(ts)

        if fs_eff is None:
            fs_eff = fs

        X = csi_to_amplitude(X_raw)

        X = remove_dc(X)

        X = bandpass_filter(X, fs_eff, 0.8, 2.17)

        X = savgol_smooth(X, window=15, poly=3)

        cut = int(10 * fs_eff)

        if X.shape[0] <= cut:
            continue

        X = X[cut:]

        if ts is not None:

            try:
                ts = np.array(ts, dtype=float).ravel()

                if ts.size == X_raw.shape[0]:
                    ts = ts[cut:]
                else:
                    ts = None

            except:
                ts = None

        window = int(window_sec * fs_eff)
        step = max(1, int(step_sec * fs_eff))
        
        #print(f"\nDEBUG FILE: {fp.name}")
        #print(f"GT FILE: {gt_fp.name}")
        #print("CSI time range:", ts.min() if ts is not None else "None", ts.max() if ts is not None else "None")
        #print("GT time range:", gt_df["time"].min(), gt_df["time"].max())

        if X.shape[0] < window:
            continue

        Xw, yw, diag = sliding_window_with_gt(
            X, ts, gt_df, fs_eff, window, step, fault_intervals, max_gt_gap_s=0.75
        )

        if Xw is None:
            log.warning(f"NO WINDOWS | file={fp.name} | diag={diag}")
            continue

        log.info(
            f"FILE OK | {fp.name} | pos={pos} subj={subject} | fs={fs_eff:.3f} "
            f"| windows={len(Xw)} | mean_abs_dt={diag.get('mean_abs_dt', None)} "
            f"| p95_abs_dt={diag.get('p95_abs_dt', None)} | drops={diag.get('stats', {})}"
        )

        if Xw is None:
            continue
        
        TARGET_WINDOW = int(window_sec * DEFAULT_FS)

        if Xw.shape[1] != TARGET_WINDOW:
            log.warning(f"Corrigindo window size de {Xw.shape[1]} para {TARGET_WINDOW}")
            Xw = Xw[:, :TARGET_WINDOW, :]

        pw = np.full(len(Xw), pos)
        sw = np.full(len(Xw), subject)

        all_X.append(Xw)
        all_y.append(yw)
        all_p.append(pw)
        all_s.append(sw)

    if not all_X:
        raise RuntimeError("Nenhuma janela gerada.")

    X = np.concatenate(all_X).astype(np.float32)
    y = np.concatenate(all_y).astype(np.float32)
    p = np.concatenate(all_p).astype(np.int32)
    s = np.concatenate(all_s).astype(np.int32)

    return X, y, p, s


def split_and_save(X, y, p, s, out_dir: Path):
    """Split by subject to prevent data leakage between sets."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_STATE)
    subjects = np.unique(s)
    rng.shuffle(subjects)

    n = len(subjects)
    n_test = max(1, round(n * TEST_SIZE))
    n_val  = max(1, round(n * VAL_SIZE))

    test_subjs  = subjects[:n_test]
    val_subjs   = subjects[n_test : n_test + n_val]
    train_subjs = subjects[n_test + n_val :]

    log.info(f"Subject split — train: {len(train_subjs)} | val: {len(val_subjs)} | test: {len(test_subjs)}")
    log.info(f"Train subjects: {sorted(train_subjs)}")
    log.info(f"Val   subjects: {sorted(val_subjs)}")
    log.info(f"Test  subjects: {sorted(test_subjs)}")

    def _split(subj_set):
        mask = np.isin(s, subj_set)
        idx  = np.where(mask)[0]
        return X[mask], y[mask], p[mask], s[mask], idx

    X_train, y_train, p_train, s_train, idx_train = _split(train_subjs)
    X_val,   y_val,   p_val,   s_val,   idx_val   = _split(val_subjs)
    X_test,  y_test,  p_test,  s_test,  idx_test  = _split(test_subjs)

    np.savez_compressed(out_dir / "X_train.npz", X=X_train)
    np.savez_compressed(out_dir / "X_val.npz",   X=X_val)
    np.savez_compressed(out_dir / "X_test.npz",  X=X_test)

    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_val.npy",   y_val)
    np.save(out_dir / "y_test.npy",  y_test)

    np.save(out_dir / "positions_train.npy", p_train)
    np.save(out_dir / "positions_val.npy",   p_val)
    np.save(out_dir / "positions_test.npy",  p_test)

    np.save(out_dir / "subject_train.npy", s_train)
    np.save(out_dir / "subject_val.npy",   s_val)
    np.save(out_dir / "subject_test.npy",  s_test)

    np.save(out_dir / "idx_train.npy", idx_train)
    np.save(out_dir / "idx_val.npy",   idx_val)
    np.save(out_dir / "idx_test.npy",  idx_test)

    np.save(out_dir / "subjects_train_ids.npy", train_subjs)
    np.save(out_dir / "subjects_val_ids.npy",   val_subjs)
    np.save(out_dir / "subjects_test_ids.npy",  test_subjs)

    print(f"Split salvo — janelas: train={len(y_train)} | val={len(y_val)} | test={len(y_test)}")


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--fs", type=float, default=DEFAULT_FS)
    ap.add_argument("--window_sec", type=float, default=20.0)
    ap.add_argument("--step_sec", type=float, default=0.5)

    args = ap.parse_args()

    X, y, p, s = build_windows(
        Path(args.dataset_path),
        Path(args.gt_dir),
        args.fs,
        args.window_sec,
        args.step_sec,
    )

    print("Dataset final:", X.shape, y.shape, p.shape, s.shape)

    X = X.astype(np.float16)

    out_path = Path(args.out_dir)

    split_and_save(X, y, p, s, out_path)

    print("Valor da janela:", X.shape)

    print(y.min(), y.max(), np.mean(y), np.std(y))


if __name__ == "__main__":
    main()

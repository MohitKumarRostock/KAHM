#!/usr/bin/env python3
"""
benchmark_kahm_vs_baselines.py

Benchmark KAHM-based multivariate regression (kahm_regression.py) against strong
scikit-learn baselines (and optional external "SOTA" gradient boosting libs if installed).

This script is intentionally self-contained: it generates an example multi-output dataset,
trains KAHM (with soft-parameter tuning), trains several baselines, and prints a metrics table.

KAHM API expectations
---------------------
The KAHM implementation in kahm_regression.py expects matrices shaped (D, N),
i.e., features/targets are column-major with samples along columns.

Most scikit-learn estimators expect (N, D) and (N, T).

Usage
-----
python benchmark_kahm_vs_baselines.py

Optional arguments:
  --n-samples 20000 --n-features 20 --n-targets 6 --n-regimes 32
  --kahm-clusters 256 --kahm-subspace-dim 20 --kahm-Nb 50
  --do-nlms-centers  (runs NLMS tuning of cluster centers)
  --skip-optional-gbdt (skip XGBoost/LightGBM/CatBoost even if installed)

Notes
-----
- If KAHM is disk-backed (default in kahm_regression.py), it will create an AE cache directory.
- If your dataset is large or you use many clusters, training can be heavy.
  Start with smaller --kahm-clusters (e.g., 64..256) and increase gradually.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from typing import Dict, Any, Tuple, Optional, List

import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge, MultiTaskElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor

try:
    # Available in modern scikit-learn; if unavailable we'll skip.
    from sklearn.ensemble import HistGradientBoostingRegressor
    _HAS_HGBR = True
except Exception:
    _HAS_HGBR = False

# --- Import KAHM functions (file must be on PYTHONPATH or in same directory) ---
from kahm_regression import (
    train_kahm_regressor,
    kahm_regress,
    tune_soft_params,
    tune_cluster_centers_nlms,
)


# ---------------------------
# Data generation (example)
# ---------------------------

def generate_regime_dataset(
    n_samples: int,
    n_features: int,
    n_targets: int,
    n_regimes: int,
    *,
    noise_std: float = 0.05,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a multi-output regression dataset with regime structure.

    Intuition:
      - Each sample belongs to a latent regime z in {0..K-1}.
      - Outputs are a mixture of (regime-specific center) + (small regime-specific linear term)
        + mild nonlinearity + noise.
      - This produces clustered outputs (helpful for KAHM’s output clustering) while still being
        a non-trivial regression task.

    Returns:
      X: (N, D)
      Y: (N, T)
      z: (N,) regime labels
    """
    rng = np.random.default_rng(random_state)

    # Regime label per sample
    z = rng.integers(0, n_regimes, size=n_samples, endpoint=False)

    # Features
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)

    # Regime-specific output centers (create clear cluster structure in Y)
    centers = rng.normal(scale=2.0, size=(n_regimes, n_targets)).astype(np.float32)

    # Regime-specific small linear maps to add within-cluster variation
    W = rng.normal(scale=0.25, size=(n_regimes, n_targets, n_features)).astype(np.float32)
    b = rng.normal(scale=0.1, size=(n_regimes, n_targets)).astype(np.float32)

    # Mild global nonlinearity shared across regimes
    # (adds complexity that linear models often struggle with)
    phi = np.tanh(X[:, : min(6, n_features)])  # (N, <=6)
    U = rng.normal(scale=0.4, size=(phi.shape[1], n_targets)).astype(np.float32)

    Y = np.empty((n_samples, n_targets), dtype=np.float32)
    for k in range(n_regimes):
        idx = np.where(z == k)[0]
        if idx.size == 0:
            continue
        Xk = X[idx]
        Y[idx] = (
            centers[k]
            + (Xk @ W[k].transpose(1, 0)).astype(np.float32)  # (N_k, T)
            + b[k]
            + (phi[idx] @ U).astype(np.float32)
        )

    if noise_std > 0:
        Y += rng.normal(scale=float(noise_std), size=Y.shape).astype(np.float32)

    return X, Y, z


# ---------------------------
# Metrics helpers
# ---------------------------

def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Metrics for multi-output regression.
    y_* are (N, T).
    """
    mse = mean_squared_error(y_true, y_pred, multioutput="uniform_average")
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred, multioutput="uniform_average")
    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def pretty_seconds(s: float) -> str:
    if s < 1e-3:
        return f"{s*1e6:.0f} µs"
    if s < 1:
        return f"{s*1e3:.1f} ms"
    if s < 60:
        return f"{s:.2f} s"
    return f"{s/60:.1f} min"


# ---------------------------
# Baselines
# ---------------------------

def build_baselines(seed: int, *, include_optional_gbdt: bool = True) -> Dict[str, Any]:
    """
    Return a dictionary of strong baselines. All must support multi-output.
    """
    models: Dict[str, Any] = {}

    models["Ridge"] = Ridge(alpha=1.0, random_state=seed)
    models["MultiTaskElasticNet"] = MultiTaskElasticNet(
        alpha=1e-3,
        l1_ratio=0.5,
        max_iter=5000,
        random_state=seed,
    )

    models["RandomForest"] = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=seed,
    )

    models["ExtraTrees"] = ExtraTreesRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=seed,
    )

    # A strong general-purpose neural baseline (can be competitive for smooth nonlinearities)
    models["MLP"] = MLPRegressor(
        hidden_layer_sizes=(256, 256),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=3e-4,
        max_iter=200,
        early_stopping=True,
        random_state=seed,
    )

    if _HAS_HGBR:
        # Histogram GBRT is strong, but single-output; wrap for multi-output.
        models["HistGBRT (MultiOutput)"] = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.08,
                max_depth=None,
                max_iter=400,
                random_state=seed,
            ),
            n_jobs=-1,
        )

    if include_optional_gbdt:
        # Optional external SOTA libraries if installed.
        # We keep conservative defaults; feel free to tune for your use case.
        try:
            from xgboost import XGBRegressor  # type: ignore

            models["XGBoost (MultiOutput)"] = MultiOutputRegressor(
                XGBRegressor(
                    n_estimators=1200,
                    learning_rate=0.05,
                    max_depth=8,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=seed,
                    n_jobs=-1,
                ),
                n_jobs=-1,
            )
        except Exception:
            pass

        try:
            from lightgbm import LGBMRegressor  # type: ignore

            models["LightGBM (MultiOutput)"] = MultiOutputRegressor(
                LGBMRegressor(
                    n_estimators=5000,
                    learning_rate=0.03,
                    num_leaves=127,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=seed,
                    n_jobs=-1,
                ),
                n_jobs=-1,
            )
        except Exception:
            pass

        try:
            from catboost import CatBoostRegressor  # type: ignore

            models["CatBoost (MultiOutput)"] = MultiOutputRegressor(
                CatBoostRegressor(
                    loss_function="RMSE",
                    iterations=4000,
                    learning_rate=0.03,
                    depth=8,
                    l2_leaf_reg=3.0,
                    random_seed=seed,
                    verbose=False,
                ),
                n_jobs=-1,
            )
        except Exception:
            pass

    return models


# ---------------------------
# Main benchmark
# ---------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)

    # Dataset
    p.add_argument("--n-samples", type=int, default=20000)
    p.add_argument("--n-features", type=int, default=50)
    p.add_argument("--n-targets", type=int, default=50)
    p.add_argument("--n-regimes", type=int, default=50)
    p.add_argument("--noise-std", type=float, default=0.05)

    # KAHM params (kept modest by default)
    p.add_argument("--kahm-clusters", type=int, default=500)
    p.add_argument("--kahm-subspace-dim", type=int, default=20)
    p.add_argument("--kahm-Nb", type=int, default=100)
    p.add_argument("--kahm-input-scale", type=float, default=1.0)
    p.add_argument("--kahm-max-train-per-cluster", type=int, default=1000)
    p.add_argument("--kahm-model-dtype", type=str, default="float32")
    p.add_argument("--kahm-batch-size", type=int, default=1024)

    # Soft tuning grid (small but meaningful)
    p.add_argument("--alpha-grid", type=str, default="2.0, 5.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 18.0, 20.0, 25.0, 50.0")
    p.add_argument("--topk-grid", type=str, default="2, 5, 10, 11, 12, 13, 14, 15, 20, 25, 50, 100, 200, 300, 400, 500")

    # Optional KAHM NLMS tuning
    p.add_argument("--do-nlms-centers", type=bool, default=True)
    p.add_argument("--nlms-mu", type=float, default=0.1)
    p.add_argument("--nlms-epochs", type=int, default=10)

    # Optional external baselines
    p.add_argument("--skip-optional-gbdt", action="store_true")

    args = p.parse_args()
    seed = int(args.seed)

    # --- Generate example data ---
    X, Y, z = generate_regime_dataset(
        n_samples=int(args.n_samples),
        n_features=int(args.n_features),
        n_targets=int(args.n_targets),
        n_regimes=int(args.n_regimes),
        noise_std=float(args.noise_std),
        random_state=seed,
    )

    # Split train/val/test (70/15/15)
    X_train, X_tmp, Y_train, Y_tmp = train_test_split(
        X, Y, test_size=0.30, random_state=seed, shuffle=True
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_tmp, Y_tmp, test_size=0.50, random_state=seed, shuffle=True
    )

    # Feature scaling (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # ---------------------------
    # Train KAHM
    # ---------------------------
    print("\n=== KAHM training ===")

    # Convert to KAHM's expected shape: (D, N)
    X_train_k = X_train_s.T
    Y_train_k = Y_train.T
    X_val_k = X_val_s.T
    Y_val_k = Y_val.T
    X_test_k = X_test_s.T
    Y_test_k = Y_test.T

    t0 = time.perf_counter()
    model = train_kahm_regressor(
        X_train_k,
        Y_train_k,
        n_clusters=int(args.kahm_clusters),
        subspace_dim=int(args.kahm_subspace_dim),
        Nb=int(args.kahm_Nb),
        random_state=seed,
        verbose=True,
        input_scale=float(args.kahm_input_scale),
        max_train_per_cluster=int(args.kahm_max_train_per_cluster) if int(args.kahm_max_train_per_cluster) > 0 else None,
        model_dtype=str(args.kahm_model_dtype),
        # Keep defaults for disk-backed AEs (recommended for many clusters)
        save_ae_to_disk=False,
    )
    t_train_kahm = time.perf_counter() - t0

    # Tune alpha/topk on validation and store in model
    alpha_grid = []
    for s in str(args.alpha_grid).split(","):
        s = s.strip()
        if not s:
            continue
        alpha_grid.append(float(s))

    topk_grid: List[Optional[int]] = []
    for s in str(args.topk_grid).split(","):
        s = s.strip()
        if not s:
            continue
        if s.lower() == "none":
            topk_grid.append(None)
        else:
            topk_grid.append(int(s))

    print("\n=== KAHM soft-parameter tuning (val set) ===")
    t0 = time.perf_counter()
    tune_res = tune_soft_params(
        model,
        X_val_k,
        Y_val_k,
        alphas=tuple(alpha_grid),
        topks=tuple(topk_grid),
        n_jobs=-1,
        verbose=True,
    )
    t_tune = time.perf_counter() - t0

    if args.do_nlms_centers:
        print("\n=== KAHM NLMS cluster-center tuning (train set) ===")
        t0 = time.perf_counter()
        nlms_res = tune_cluster_centers_nlms(
            model,
            X_train_k,
            Y_train_k,
            mu=float(args.nlms_mu),
            epsilon=1.0,
            epochs=int(args.nlms_epochs),
            batch_size=int(args.kahm_batch_size),
            shuffle=True,
            random_state=seed,
            anchor_lambda=0.0,
            n_jobs=-1,
            preload_classifier=True,
            verbose=True,
            alpha=tune_res.best_alpha,
            topk=tune_res.best_topk,
        )
        t_nlms = time.perf_counter() - t0
        print(f"NLMS summary: {asdict(nlms_res)}")
    else:
        t_nlms = 0.0

    # Evaluate KAHM on test (soft and hard)
    print("\n=== KAHM inference (test set) ===")
    t0 = time.perf_counter()
    Y_pred_soft_k = kahm_regress(
        model,
        X_test_k,
        n_jobs=-1,
        mode="soft",
        batch_size=int(args.kahm_batch_size),
        show_progress=False,
    )
    t_pred_soft = time.perf_counter() - t0

    t0 = time.perf_counter()
    Y_pred_hard_k = kahm_regress(
        model,
        X_test_k,
        n_jobs=-1,
        mode="hard",
        batch_size=int(args.kahm_batch_size),
        show_progress=False,
    )
    t_pred_hard = time.perf_counter() - t0

    # Convert back to sklearn shape (N, T)
    Y_pred_soft = Y_pred_soft_k.T
    Y_pred_hard = Y_pred_hard_k.T

    kahm_soft_metrics = eval_metrics(Y_test, Y_pred_soft)
    kahm_hard_metrics = eval_metrics(Y_test, Y_pred_hard)

    results: List[Dict[str, Any]] = []
    results.append(
        {
            "model": "KAHM (soft)",
            **kahm_soft_metrics,
            "train_time": t_train_kahm,
            "tune_time": t_tune + t_nlms,
            "pred_time": t_pred_soft,
        }
    )
    results.append(
        {
            "model": "KAHM (hard)",
            **kahm_hard_metrics,
            "train_time": t_train_kahm,
            "tune_time": t_tune + t_nlms,
            "pred_time": t_pred_hard,
        }
    )

    # ---------------------------
    # Train baselines
    # ---------------------------
    print("\n=== Baselines (scikit-learn; optional external GBDT if installed) ===")
    baselines = build_baselines(seed, include_optional_gbdt=not bool(args.skip_optional_gbdt))

    for name, est in baselines.items():
        print(f"\n-- {name} --")
        t0 = time.perf_counter()
        est.fit(X_train_s, Y_train)
        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        Yp = est.predict(X_test_s)
        t_pred = time.perf_counter() - t0

        m = eval_metrics(Y_test, Yp)
        results.append(
            {
                "model": name,
                **m,
                "train_time": t_train,
                "tune_time": 0.0,
                "pred_time": t_pred,
            }
        )

    # ---------------------------
    # Report
    # ---------------------------
    # Sort by MSE (lower is better)
    results_sorted = sorted(results, key=lambda r: r["mse"])

    print("\n=== Summary (sorted by test MSE) ===")
    header = (
        f"{'Model':30s}  {'MSE':>12s}  {'RMSE':>10s}  {'MAE':>10s}  {'R2':>8s}  "
        f"{'Train':>10s}  {'Tune':>10s}  {'Predict':>10s}"
    )
    print(header)
    print("-" * len(header))

    for r in results_sorted:
        print(
            f"{r['model'][:30]:30s}  "
            f"{r['mse']:12.6g}  {r['rmse']:10.6g}  {r['mae']:10.6g}  {r['r2']:8.4f}  "
            f"{pretty_seconds(r['train_time']):>10s}  {pretty_seconds(r['tune_time']):>10s}  {pretty_seconds(r['pred_time']):>10s}"
        )

    print("\nKAHM soft params stored in model:")
    print(f"  soft_alpha = {model.get('soft_alpha')}")
    print(f"  soft_topk  = {model.get('soft_topk')}")

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Open KAHM → QUBO Demo + Multi-Solver Comparison
====================================================================

This is a token-free, open-solvers-only demo of the KAHM→scenario evidence→QUBO pipeline,
with a **--compare** mode that runs multiple solvers on the same QUBO and prints a ranked table.

Improvements vs earlier builds
------------------------------
- Prints per-solver progress lines so you can see what is running (useful when QAOA is slow).
- Adds --qaoa_maxiter to cap QAOA optimizer iterations.
- Suppresses SciPy SparseEfficiencyWarning by default (those warnings are benign).

Solvers (token-free)
-------------------
- bruteforce   (exact; small M)
- local        (1-bit-flip local search)
- neal_sa      (simulated annealing via dwave-neal; local)
- tabu         (tabu search via dwave-tabu; local; import via `tabu` module)
- qiskit_qaoa  (QAOA via Qiskit on local primitive/simulator; no IBM creds)

Typical installs (inside your active venv)
------------------------------------------
python -m pip install dimod dwave-neal dwave-tabu
python -m pip install qiskit qiskit-optimization qiskit-algorithms qiskit-aer
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import kahm_classification as kc


# =============================================================================
# Warnings
# =============================================================================

def configure_warnings(suppress_sparse_warnings: bool = True) -> None:
    if not suppress_sparse_warnings:
        return
    try:
        from scipy.sparse import SparseEfficiencyWarning  # type: ignore
        warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
    except Exception:
        # SciPy not installed or warning class moved; ignore.
        pass


# =============================================================================
# Objective helpers
# =============================================================================

def qubo_energy_Qq(u: np.ndarray, Q: np.ndarray, q: np.ndarray, *, constant: float = 0.0) -> float:
    """Energy E(u) = u^T Q u + q^T u + constant for binary u∈{0,1}^M."""
    u = np.asarray(u, dtype=np.int64).reshape(-1)
    Q = np.asarray(Q, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    return float(u @ (Q @ u) + q @ u + constant)


def Qq_to_qubo_dict(Q: np.ndarray, q: np.ndarray, *, tol: float = 0.0) -> Dict[Tuple[int, int], float]:
    """
    Convert objective u^T Q u + q^T u to a standard QUBO dict A:
      A_ii = Q_ii + q_i
      A_ij = Q_ij + Q_ji  for i<j
    """
    Q = np.asarray(Q, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    M = int(Q.shape[0])
    if Q.shape != (M, M) or q.shape != (M,):
        raise ValueError("Expected Q:(M,M) and q:(M,)")

    A: Dict[Tuple[int, int], float] = {}
    for i in range(M):
        v = float(Q[i, i] + q[i])
        if abs(v) > tol:
            A[(i, i)] = v
    for i in range(M):
        for j in range(i + 1, M):
            v = float(Q[i, j] + Q[j, i])
            if abs(v) > tol:
                A[(i, j)] = v
    return A


# =============================================================================
# Optional: cardinality penalty λ(Σ u - K)^2
# =============================================================================

def add_cardinality_penalty(Q: np.ndarray, q: np.ndarray, *, K: int, lam: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Add λ(Σ_i u_i - K)^2 to u^TQ u + q^T u. Returns (Q_new, q_new, constant).
    """
    Q = np.asarray(Q, dtype=np.float64).copy()
    q = np.asarray(q, dtype=np.float64).reshape(-1).copy()
    M = int(Q.shape[0])

    K = int(K)
    lam = float(lam)
    if K < 0 or K > M:
        raise ValueError(f"K must be in [0,M]; got K={K}, M={M}")
    if lam < 0:
        raise ValueError("lam must be non-negative")

    Q += lam * (np.ones((M, M), dtype=np.float64) - np.eye(M, dtype=np.float64))
    Q[np.diag_indices(M)] += lam * (1.0 - 2.0 * K)

    constant = lam * (K ** 2)
    return Q, q, constant


# =============================================================================
# Classical solvers
# =============================================================================

def solve_qubo_bruteforce(Q: np.ndarray, q: np.ndarray, *, constant: float = 0.0) -> Tuple[np.ndarray, float]:
    M = int(Q.shape[0])
    best_u = None
    best_E = float("inf")
    for bits in itertools.product([0, 1], repeat=M):
        u = np.fromiter(bits, dtype=np.int64, count=M)
        E = qubo_energy_Qq(u, Q, q, constant=constant)
        if E < best_E:
            best_E, best_u = E, u
    assert best_u is not None
    return best_u, best_E


def solve_qubo_local_search(
    Q: np.ndarray,
    q: np.ndarray,
    *,
    seed: int = 0,
    n_restarts: int = 60,
    n_steps: int = 10000,
    constant: float = 0.0,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    M = int(Q.shape[0])

    def improve(u: np.ndarray) -> Tuple[np.ndarray, float]:
        E = qubo_energy_Qq(u, Q, q, constant=constant)
        for _ in range(n_steps):
            i = int(rng.integers(0, M))
            u2 = u.copy()
            u2[i] = 1 - u2[i]
            E2 = qubo_energy_Qq(u2, Q, q, constant=constant)
            if E2 < E:
                u, E = u2, E2
        return u, E

    best_u = None
    best_E = float("inf")
    for _ in range(n_restarts):
        u0 = rng.integers(0, 2, size=M, dtype=np.int64)
        u, E = improve(u0)
        if E < best_E:
            best_u, best_E = u.copy(), E
    assert best_u is not None
    return best_u, best_E


# =============================================================================
# Open local samplers
# =============================================================================

def solve_qubo_neal_sa(qubo: Dict[Tuple[int, int], float], *, num_reads: int = 2000, seed: int = 0) -> Tuple[np.ndarray, float]:
    """Local simulated annealing via dwave-neal."""
    try:
        import dimod  # type: ignore
        import neal  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "neal_sa import failed.\n"
            "Note: If you see 'SeedlessSequence' AttributeError, upgrade NumPy to >=2.4.1 (or downgrade to <2.4.0).\n"
            f"Python executable: {sys.executable}\n"
            f"Underlying error: {type(e).__name__}: {e}\n\n"
            "Fix (inside the SAME venv):\n"
            "  python -m pip install --upgrade numpy\n"
            "  python -m pip install --upgrade --force-reinstall dimod dwave-neal\n"
            "Verify:\n"
            "  python -c \"import dimod, neal; print('dimod', dimod.__version__, 'neal', neal.__version__)\"\n"
        ) from e

    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
    ss = neal.SimulatedAnnealingSampler().sample(bqm, num_reads=int(num_reads), seed=int(seed))
    best = ss.first
    u = np.array([best.sample[v] for v in sorted(best.sample.keys())], dtype=np.int64)
    return u, float(best.energy)


def solve_qubo_tabu(qubo: Dict[Tuple[int, int], float], *, num_reads: int = 100, timeout: int = 200) -> Tuple[np.ndarray, float]:
    """Local tabu search via dwave-tabu (import is `tabu.TabuSampler`)."""
    try:
        import dimod  # type: ignore
        try:
            from tabu import TabuSampler  # type: ignore
        except Exception:
            # Alternate packaging (rare): dwave-samplers
            from dwave.samplers import TabuSampler  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "tabu import failed.\n"
            f"Python executable: {sys.executable}\n"
            f"Underlying error: {type(e).__name__}: {e}\n\n"
            "Fix (inside the SAME venv):\n"
            "  python -m pip install dimod dwave-tabu\n"
            "Verify:\n"
            "  python -c \"from tabu import TabuSampler; print('tabu ok')\"\n"
        ) from e

    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
    ss = TabuSampler().sample(bqm, num_reads=int(num_reads), timeout=int(timeout))
    best = ss.first
    u = np.array([best.sample[v] for v in sorted(best.sample.keys())], dtype=np.int64)
    return u, float(best.energy)


def solve_qubo_qiskit_qaoa(
    qubo: Dict[Tuple[int, int], float],
    *,
    reps: int = 1,
    shots: int = 2000,
    seed: int = 0,
    maxiter: int = 80,
) -> Tuple[np.ndarray, float]:
    """
    Token-free QAOA via Qiskit Optimization using a **V2 primitive**.

    We prefer `qiskit.primitives.StatevectorSampler` for maximum robustness (offline, no Aer compilation path).
    For M>~20, statevector simulation can become expensive.

    `maxiter` caps the classical optimizer iterations to avoid long runs during comparisons.
    """
    try:
        from qiskit_optimization import QuadraticProgram  # type: ignore
        from qiskit_optimization.algorithms import MinimumEigenOptimizer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "qiskit_optimization import failed.\n"
            f"Python executable: {sys.executable}\n"
            f"Underlying error: {type(e).__name__}: {e}\n\n"
            "Fix:\n"
            "  python -m pip install qiskit-optimization\n"
        ) from e

    try:
        from qiskit_algorithms.minimum_eigensolvers import QAOA  # type: ignore
        from qiskit_algorithms.optimizers import COBYLA  # type: ignore
    except Exception:
        try:
            from qiskit.algorithms.minimum_eigensolvers import QAOA  # type: ignore
            from qiskit.algorithms.optimizers import COBYLA  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "QAOA/optimizer imports failed.\n"
                f"Python executable: {sys.executable}\n"
                f"Underlying error: {type(e).__name__}: {e}\n\n"
                "Fix:\n"
                "  python -m pip install qiskit qiskit-algorithms\n"
            ) from e

    try:
        from qiskit.primitives import StatevectorSampler  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "StatevectorSampler import failed.\n"
            f"Python executable: {sys.executable}\n"
            f"Underlying error: {type(e).__name__}: {e}\n\n"
            "Fix:\n"
            "  python -m pip install qiskit\n"
        ) from e

    # Build QuadraticProgram
    max_idx = max((max(i, j) for (i, j) in qubo.keys()), default=-1)
    M = max_idx + 1

    qp = QuadraticProgram("qubo")
    for i in range(M):
        qp.binary_var(f"x{i}")

    linear = {f"x{i}": float(qubo.get((i, i), 0.0)) for i in range(M)}
    quadratic = {(f"x{i}", f"x{j}"): float(v) for (i, j), v in qubo.items() if i < j}
    qp.minimize(linear=linear, quadratic=quadratic)

    sampler = StatevectorSampler(default_shots=int(shots), seed=int(seed))
    optimizer = COBYLA(maxiter=int(maxiter))
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=int(reps))
    res = MinimumEigenOptimizer(qaoa).solve(qp)

    u = np.array(res.x, dtype=np.int64).reshape(-1)
    return u, float(res.fval)


# =============================================================================
# KAHM scenario scoring helpers
# =============================================================================

def compute_folding_distances(model: dict, X_new: np.ndarray, *, n_jobs: int = -1) -> np.ndarray:
    X_new = np.asarray(X_new)
    if X_new.ndim != 2:
        raise ValueError("X_new must be shaped (D_in, N_new).")

    input_scale = float(model.get("input_scale", 1.0))
    if input_scale != 1.0:
        X_new = kc._scale_like(kc._as_float_ndarray(X_new), input_scale, inplace=False)
    else:
        X_new = kc._as_float_ndarray(X_new)

    AE_arr = model.get("_classifier_cache", model.get("classifier", None))
    if not isinstance(AE_arr, (list, tuple)) or len(AE_arr) == 0:
        raise TypeError("Expected model['classifier'] (or model['_classifier_cache']) to be a non-empty list/tuple.")
    C = int(model.get("n_classes", len(AE_arr)))
    C = min(C, len(AE_arr))

    N_new = int(X_new.shape[1])
    T = np.empty((C, N_new), dtype=np.float32)

    base_dir = model.get("classifier_dir", None)

    def _resolve(ae_ref):
        if isinstance(ae_ref, (str, bytes, np.str_, np.bytes_)):
            from pathlib import Path
            p = Path(str(ae_ref))
            if (not p.is_absolute()) and base_dir is not None:
                p = Path(base_dir) / p
            return str(p), True
        return ae_ref, False

    for c in range(C):
        ae_ref = AE_arr[c]
        ae_obj, is_path = _resolve(ae_ref)
        if is_path:
            from joblib import load
            AE_c = load(ae_obj)
        else:
            AE_c = ae_obj

        d = kc._call_combine_multiple_autoencoders_extended(X_new, kc._ae_as_list(AE_c), "folding", n_jobs=n_jobs)
        d = np.asarray(d, dtype=np.float32).reshape(-1)
        if d.size != N_new:
            raise ValueError(f"Distance vector from class {c} has shape {d.shape}; expected ({N_new},).")
        T[c, :] = d

        if is_path:
            del AE_c

    return T


def folding_to_weights(T: np.ndarray, *, alpha: float = 10.0, topk: Optional[int] = 3, eps: float = 1e-12) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    w = 1.0 - T
    np.clip(w, 0.0, 1.0, out=w)

    if float(alpha) != 1.0:
        w = np.power(w, float(alpha))

    if topk is not None:
        k = int(topk)
        if 0 < k < w.size:
            idx = np.argpartition(w, kth=w.size - k)[: w.size - k]
            w[idx] = 0.0

    s = float(w.sum())
    if s <= eps:
        w[:] = 1.0 / w.size
    else:
        w /= s
    return w


# =============================================================================
# Synthetic data + synthetic scenario-QUBOs (replace in your project)
# =============================================================================

def make_synthetic_scenario_data(*, D_in: int, N_per: int, C: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []
    for c in range(C):
        mean = (c - (C - 1) / 2.0) * (1.25 / max(1, C - 1))
        Xc = rng.normal(loc=mean, scale=1.0, size=(D_in, N_per)).astype(np.float32)
        Xc = np.tanh(Xc)
        X_parts.append(Xc)
        y_parts.append(np.full((N_per,), c, dtype=np.int64))
    X = np.concatenate(X_parts, axis=1)
    y = np.concatenate(y_parts, axis=0)
    perm = rng.permutation(X.shape[1])
    return X[:, perm], y[perm]


def make_scenario_qubos(*, M: int, C: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Qs = np.zeros((C, M, M), dtype=np.float64)
    qs = np.zeros((C, M), dtype=np.float64)
    for c in range(C):
        A = rng.normal(size=(M, M))
        Q = (A + A.T) / 2.0
        Q *= 0.25 / M
        diag_bias = (c - (C - 1) / 2.0) * (0.8 / max(1, C - 1))
        Q += np.eye(M) * (0.5 + diag_bias)
        q = rng.normal(size=(M,)) * 0.1 + (0.2 * diag_bias)
        Qs[c], qs[c] = Q, q
    return Qs, qs


def assemble_context_qubo(Qs: np.ndarray, qs: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    Q = np.tensordot(w, Qs, axes=(0, 0))
    q = np.tensordot(w, qs, axes=(0, 0))
    return np.asarray(Q, dtype=np.float64), np.asarray(q, dtype=np.float64)


# =============================================================================
# Comparison utilities
# =============================================================================

@dataclass(frozen=True)
class SolverRun:
    solver: str
    ok: bool
    energy: float
    ones: int
    elapsed_s: float
    u: Optional[np.ndarray]
    note: str


def _parse_compare_list(s: str) -> List[str]:
    items = [t.strip() for t in s.split(",") if t.strip()]
    out: List[str] = []
    seen = set()
    for it in items:
        if it not in seen:
            out.append(it)
            seen.add(it)
    return out


def run_one_solver(
    solver: str,
    *,
    Q: np.ndarray,
    q: np.ndarray,
    qubo: Dict[Tuple[int, int], float],
    constant: float,
    seed: int,
    num_reads: int,
    tabu_timeout: int,
    shots: int,
    reps: int,
    qaoa_maxiter: int,
) -> SolverRun:
    t0 = time.perf_counter()
    try:
        if solver == "bruteforce":
            M = int(Q.shape[0])
            if M > 22:
                return SolverRun(solver, False, float("inf"), 0, 0.0, None, "skipped (M>22)")
            u, E = solve_qubo_bruteforce(Q, q, constant=constant)

        elif solver == "local":
            u, E = solve_qubo_local_search(Q, q, seed=seed, constant=constant)

        elif solver == "neal_sa":
            u, E = solve_qubo_neal_sa(qubo, num_reads=num_reads, seed=seed)
            E += constant

        elif solver == "tabu":
            u, E = solve_qubo_tabu(qubo, num_reads=max(1, num_reads // 20), timeout=tabu_timeout)
            E += constant

        elif solver == "qiskit_qaoa":
            u, E = solve_qubo_qiskit_qaoa(qubo, reps=reps, shots=shots, seed=seed, maxiter=qaoa_maxiter)
            E += constant

        else:
            return SolverRun(solver, False, float("inf"), 0, 0.0, None, "unknown solver")

        elapsed = time.perf_counter() - t0
        return SolverRun(solver, True, float(E), int(np.sum(u)), elapsed, u, "")

    except Exception as e:
        elapsed = time.perf_counter() - t0
        return SolverRun(solver, False, float("inf"), 0, elapsed, None, f"{type(e).__name__}: {e}")


def print_comparison_table(runs: List[SolverRun]) -> None:
    ok = [r for r in runs if r.ok]
    bad = [r for r in runs if not r.ok]

    print("\n[compare] solver results (ranked by energy):")
    if not ok:
        print("  No solvers succeeded.")
    else:
        ok_sorted = sorted(ok, key=lambda r: r.energy)
        best_E = ok_sorted[0].energy

        header = f"{'solver':<12} {'energy':>14} {'Δ vs best':>12} {'ones':>6} {'time(s)':>9}"
        print("  " + header)
        print("  " + "-" * len(header))
        for r in ok_sorted:
            print(f"  {r.solver:<12} {r.energy:>14.6f} {r.energy - best_E:>12.6f} {r.ones:>6d} {r.elapsed_s:>9.3f}")

        best = ok_sorted[0]
        if best.u is not None:
            print("\n  best solution vector (from lowest-energy solver):")
            print("  u* =", best.u.astype(int).tolist())

    if bad:
        print("\n[compare] skipped/failed solvers:")
        for r in bad:
            print(f"  - {r.solver}: {r.note}")


# =============================================================================
# Main
# =============================================================================

@dataclass(frozen=True)
class DemoConfig:
    seed: int
    D_in: int
    C: int
    N_per: int
    M: int
    subspace_dim: int
    Nb: int
    input_scale: float
    alpha: float
    topk: Optional[int]
    qubo_solver: str
    num_reads: int
    tabu_timeout: int
    shots: int
    reps: int
    target_ones: int
    penalty_lambda: float
    compare: bool
    compare_solvers: List[str]
    qaoa_maxiter: int
    suppress_sparse_warnings: bool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--D_in", type=int, default=8)
    ap.add_argument("--C", type=int, default=3)
    ap.add_argument("--N_per", type=int, default=400)
    ap.add_argument("--M", type=int, default=12)
    ap.add_argument("--subspace_dim", type=int, default=20)
    ap.add_argument("--Nb", type=int, default=10)
    ap.add_argument("--input_scale", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--topk", type=int, default=3, help="0 disables top-k")

    ap.add_argument(
        "--qubo_solver",
        type=str,
        default="local",
        choices=["bruteforce", "local", "neal_sa", "tabu", "qiskit_qaoa"],
    )
    ap.add_argument("--num_reads", type=int, default=2000)
    ap.add_argument("--tabu_timeout", type=int, default=200)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--qaoa_maxiter", type=int, default=80, help="cap classical optimizer iterations for qiskit_qaoa")

    ap.add_argument("--target_ones", type=int, default=0, help="K in λ(Σu-K)^2; 0 disables")
    ap.add_argument("--penalty_lambda", type=float, default=1.0, help="λ in λ(Σu-K)^2")

    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--compare_solvers", type=str, default="local,neal_sa,tabu,qiskit_qaoa,bruteforce")

    ap.add_argument("--no_suppress_sparse_warnings", action="store_true", help="show SciPy sparse efficiency warnings")

    args = ap.parse_args()

    cfg = DemoConfig(
        seed=int(args.seed),
        D_in=int(args.D_in),
        C=int(args.C),
        N_per=int(args.N_per),
        M=int(args.M),
        subspace_dim=int(args.subspace_dim),
        Nb=int(args.Nb),
        input_scale=float(args.input_scale),
        alpha=float(args.alpha),
        topk=None if int(args.topk) <= 0 else int(args.topk),
        qubo_solver=str(args.qubo_solver),
        num_reads=int(args.num_reads),
        tabu_timeout=int(args.tabu_timeout),
        shots=int(args.shots),
        reps=int(args.reps),
        target_ones=int(args.target_ones),
        penalty_lambda=float(args.penalty_lambda),
        compare=bool(args.compare),
        compare_solvers=_parse_compare_list(str(args.compare_solvers)),
        qaoa_maxiter=int(args.qaoa_maxiter),
        suppress_sparse_warnings=not bool(args.no_suppress_sparse_warnings),
    )

    configure_warnings(cfg.suppress_sparse_warnings)

    rng = np.random.default_rng(cfg.seed)

    X, y = make_synthetic_scenario_data(D_in=cfg.D_in, N_per=cfg.N_per, C=cfg.C, seed=cfg.seed)
    N = X.shape[1]
    N_train = int(0.8 * N)
    X_train, y_train = X[:, :N_train], y[:N_train]
    X_val, y_val = X[:, N_train:], y[N_train:]
    print(f"[data] X_train={X_train.shape}, X_val={X_val.shape}, C={cfg.C}")

    model = kc.train_kahm_classifier(
        X_train,
        y_train,
        subspace_dim=cfg.subspace_dim,
        Nb=cfg.Nb,
        random_state=cfg.seed,
        verbose=True,
        input_scale=cfg.input_scale,
        save_ae_to_disk=False,
    )

    j = int(rng.integers(0, X_val.shape[1]))
    x = X_val[:, j:j+1]
    true_scenario = int(y_val[j])
    print(f"[query] picked one context x from validation set; true scenario label={true_scenario}")

    Tmat = compute_folding_distances(model, x, n_jobs=-1)
    T = Tmat[:, 0]
    w = folding_to_weights(T, alpha=cfg.alpha, topk=cfg.topk)

    c_hat = int(np.argmin(T))
    print("\n[scenario scoring]")
    for c in range(cfg.C):
        print(f"  scenario {c}: T_c(x)={float(T[c]):.6f}  w_c(x)={float(w[c]):.6f}")
    print(f"  predicted scenario (argmin T) = {c_hat}")

    Qs, qs = make_scenario_qubos(M=cfg.M, C=cfg.C, seed=cfg.seed + 123)
    Qx, qx = assemble_context_qubo(Qs, qs, w)

    constant = 0.0
    if cfg.target_ones and cfg.penalty_lambda > 0:
        Qx, qx, cst = add_cardinality_penalty(Qx, qx, K=cfg.target_ones, lam=cfg.penalty_lambda)
        constant += cst
        print(f"[constraint] added cardinality penalty λ(Σu-{cfg.target_ones})^2 with λ={cfg.penalty_lambda} (constant={cst})")

    qubo = Qq_to_qubo_dict(Qx, qx, tol=0.0)
    print(f"\n[qubo] M={cfg.M}, nonzeros={len(qubo)}")

    if cfg.compare:
        runs: List[SolverRun] = []
        solvers = cfg.compare_solvers or ["local", "neal_sa", "tabu", "qiskit_qaoa"]
        for s in solvers:
            print(f"[compare] running {s} ...")
            r = run_one_solver(
                s,
                Q=Qx,
                q=qx,
                qubo=qubo,
                constant=constant,
                seed=cfg.seed,
                num_reads=cfg.num_reads,
                tabu_timeout=cfg.tabu_timeout,
                shots=cfg.shots,
                reps=cfg.reps,
                qaoa_maxiter=cfg.qaoa_maxiter,
            )
            if r.ok:
                print(f"[compare] done {s}: energy={r.energy:.6f}, ones={r.ones}, time={r.elapsed_s:.3f}s")
            else:
                print(f"[compare] failed {s}: {r.note}")
            runs.append(r)

        print_comparison_table(runs)
        print("\nDone.")
        return

    # Single solver mode
    s = cfg.qubo_solver
    r = run_one_solver(
        s,
        Q=Qx,
        q=qx,
        qubo=qubo,
        constant=constant,
        seed=cfg.seed,
        num_reads=cfg.num_reads,
        tabu_timeout=cfg.tabu_timeout,
        shots=cfg.shots,
        reps=cfg.reps,
        qaoa_maxiter=cfg.qaoa_maxiter,
    )
    if not r.ok or r.u is None:
        raise RuntimeError(r.note)

    print("\n[result]")
    print(f"  solution ones={r.ones}: {r.u.astype(int).tolist()}")
    print(f"  energy = {r.energy:.6f}")
    print("Done.")


if __name__ == "__main__":
    main()

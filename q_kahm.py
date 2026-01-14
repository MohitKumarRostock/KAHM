#!/usr/bin/env python3
"""
KAHM → Scenario Evidence → QUBO (Flyer-aligned) Demo
===================================================

This script demonstrates the flyer pipeline end-to-end and adds **quantum solver interfaces**:

  1) Train scenario/regime models with a KAHM-style scorer (here: per-class OTFL autoencoders,
     as implemented in kahm_classification.py).
  2) For a new operational context x, compute per-scenario space-folding measures T_c(x) in [0,1].
  3) Convert folding measures into scenario evidence weights w_c(x) = 1 - T_c(x)
     (optional sharpening and top-k pruning).
  4) Assemble a context-conditioned QUBO:
         Q(x) = Σ_c w_c(x) Q_c,   q(x) = Σ_c w_c(x) q_c
     and solve:
         min_{u∈{0,1}^M}  u^T Q(x) u + q(x)^T u
  5) Solve the resulting QUBO with one of:
       - classical bruteforce (small M)
       - classical local search (larger M)
       - D-Wave Leap hybrid (cloud)
       - D-Wave QPU (cloud annealer; embedding required)
       - Qiskit QAOA (local primitive / simulator)
       - IBM Runtime QAOA (hardware; template)

Install & configure (in your environment)
-----------------------------------------
D-Wave Ocean (quantum annealing / hybrid):
    pip install dwave-ocean-sdk
    dwave setup

Qiskit (gate-model QAOA):
    pip install qiskit qiskit-optimization qiskit-algorithms
    # to run on IBM hardware:
    pip install qiskit-ibm-runtime

Usage examples
--------------
# Classical:
python kahm_flyer_demo.py --qubo_solver bruteforce --M 14
python kahm_flyer_demo.py --qubo_solver local --M 60

# D-Wave:
python kahm_flyer_demo.py --qubo_solver dwave_hybrid
python kahm_flyer_demo.py --qubo_solver dwave_qpu --num_reads 2000

# Qiskit:
python kahm_flyer_demo.py --qubo_solver qiskit_qaoa --reps 2 --shots 2000
python kahm_flyer_demo.py --qubo_solver qiskit_ibm --ibm_backend ibm_brisbane --reps 1 --shots 1000

Notes
-----
- This demo uses *synthetic* scenario data and *synthetic* scenario-conditioned QUBO templates.
  Replace `make_scenario_qubos(...)` with your domain template generator.
- Data matrices follow kahm_classification.py convention: (D_in, N).

"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# Your existing implementation
import kahm_classification as kc


# =============================================================================
# QUBO helpers
# =============================================================================

def qubo_energy_Qq(u: np.ndarray, Q: np.ndarray, q: np.ndarray) -> float:
    """Energy E(u) = u^T Q u + q^T u for binary u∈{0,1}^M."""
    u = np.asarray(u, dtype=np.int64).reshape(-1)
    Q = np.asarray(Q, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    return float(u @ (Q @ u) + q @ u)


def solve_qubo_bruteforce(Q: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, float]:
    """Exact solve by enumeration (feasible up to ~20–22 bits)."""
    M = int(Q.shape[0])
    best_u = None
    best_E = float("inf")
    for bits in itertools.product([0, 1], repeat=M):
        u = np.fromiter(bits, dtype=np.int64, count=M)
        E = qubo_energy_Qq(u, Q, q)
        if E < best_E:
            best_E = E
            best_u = u
    assert best_u is not None
    return best_u, best_E


def solve_qubo_local_search(
    Q: np.ndarray,
    q: np.ndarray,
    *,
    seed: int = 0,
    n_restarts: int = 50,
    n_steps: int = 8000
) -> Tuple[np.ndarray, float]:
    """Simple randomized 1-bit-flip local search (good fallback for larger M)."""
    rng = np.random.default_rng(seed)
    M = int(Q.shape[0])

    def improve(u: np.ndarray) -> Tuple[np.ndarray, float]:
        E = qubo_energy_Qq(u, Q, q)
        for _ in range(n_steps):
            i = int(rng.integers(0, M))
            u2 = u.copy()
            u2[i] = 1 - u2[i]
            E2 = qubo_energy_Qq(u2, Q, q)
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


def Qq_to_qubo_dict(Q: np.ndarray, q: np.ndarray, *, tol: float = 0.0) -> Dict[Tuple[int, int], float]:
    """
    Convert objective u^T Q u + q^T u into a standard QUBO dict A for D-Wave/Qiskit.

    D-Wave's dict represents:
      E(u)= Σ_{i<j} A_ij u_i u_j + Σ_i A_ii u_i

    Our objective is:
      E(u)= u^T Q u + q^T u = Σ_{i,j} Q_ij u_i u_j + Σ_i q_i u_i

    Exact mapping (energy-preserving):
      A_ii = Q_ii + q_i
      A_ij = Q_ij + Q_ji   for i<j
    """
    Q = np.asarray(Q, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    M = int(Q.shape[0])
    if Q.shape != (M, M) or q.shape != (M,):
        raise ValueError("Expected Q:(M,M) and q:(M,)")

    A: Dict[Tuple[int, int], float] = {}
    for i in range(M):
        val = float(Q[i, i] + q[i])
        if abs(val) > tol:
            A[(i, i)] = val
    for i in range(M):
        for j in range(i + 1, M):
            val = float(Q[i, j] + Q[j, i])
            if abs(val) > tol:
                A[(i, j)] = val
    return A


# =============================================================================
# Quantum backends (lazy imports)
# =============================================================================

def solve_qubo_dwave_hybrid(qubo: Dict[Tuple[int, int], float], *, time_limit: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """D-Wave Leap Hybrid solver (cloud). Requires D-Wave credentials (dwave setup)."""
    try:
        import dimod
        from dwave.system import LeapHybridSampler
    except Exception as e:
        raise RuntimeError("D-Wave Ocean not available. Install with: pip install dwave-ocean-sdk") from e

    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
    sampler = LeapHybridSampler()
    sampleset = sampler.sample(bqm, time_limit=time_limit)
    best = sampleset.first
    u = np.array([best.sample[v] for v in sorted(best.sample.keys())], dtype=np.int64)
    return u, float(best.energy)


def solve_qubo_dwave_qpu(qubo: Dict[Tuple[int, int], float], *, num_reads: int = 2000) -> Tuple[np.ndarray, float]:
    """D-Wave QPU (annealer). Requires D-Wave credentials and QPU access."""
    try:
        import dimod
        from dwave.system import DWaveSampler, EmbeddingComposite
    except Exception as e:
        raise RuntimeError("D-Wave Ocean not available. Install with: pip install dwave-ocean-sdk") from e

    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
    sampler = EmbeddingComposite(DWaveSampler())
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    best = sampleset.first
    u = np.array([best.sample[v] for v in sorted(best.sample.keys())], dtype=np.int64)
    return u, float(best.energy)


def solve_qubo_qiskit_qaoa(qubo: Dict[Tuple[int, int], float], *, reps: int = 1, shots: int = 2000, seed: int = 0) -> Tuple[np.ndarray, float]:
    """QAOA via Qiskit Optimization (Sampler primitive)."""
    try:
        from qiskit.algorithms.minimum_eigensolvers import QAOA
        from qiskit.algorithms.optimizers import COBYLA
        from qiskit.primitives import Sampler
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
    except Exception as e:
        raise RuntimeError(
            "Qiskit stack not available. Install with: pip install qiskit qiskit-optimization qiskit-algorithms"
        ) from e

    max_idx = max((max(i, j) for (i, j) in qubo.keys()), default=-1)
    M = max_idx + 1

    qp = QuadraticProgram("qubo")
    for i in range(M):
        qp.binary_var(f"x{i}")

    linear = {f"x{i}": float(qubo.get((i, i), 0.0)) for i in range(M)}
    quadratic = {(f"x{i}", f"x{j}"): float(v) for (i, j), v in qubo.items() if i < j}
    qp.minimize(linear=linear, quadratic=quadratic)

    sampler = Sampler(options={"shots": int(shots), "seed": int(seed)})
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=int(reps))
    meo = MinimumEigenOptimizer(qaoa)
    res = meo.solve(qp)

    u = np.array(res.x, dtype=np.int64).reshape(-1)
    return u, float(res.fval)


def solve_qubo_qiskit_ibm_runtime(
    qubo: Dict[Tuple[int, int], float],
    *,
    reps: int,
    shots: int,
    seed: int,
    instance: str,
    backend_name: str
) -> Tuple[np.ndarray, float]:
    """
    Template: QAOA via IBM Runtime primitives (hardware).
    Requires: pip install qiskit-ibm-runtime
    """
    try:
        from qiskit.algorithms.minimum_eigensolvers import QAOA
        from qiskit.algorithms.optimizers import COBYLA
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
    except Exception as e:
        raise RuntimeError(
            "IBM Runtime stack not available. Install with: pip install qiskit-ibm-runtime"
        ) from e

    if not backend_name:
        raise ValueError("backend_name is required for IBM Runtime execution (e.g., ibm_brisbane).")

    max_idx = max((max(i, j) for (i, j) in qubo.keys()), default=-1)
    M = max_idx + 1

    qp = QuadraticProgram("qubo")
    for i in range(M):
        qp.binary_var(f"x{i}")

    linear = {f"x{i}": float(qubo.get((i, i), 0.0)) for i in range(M)}
    quadratic = {(f"x{i}", f"x{j}"): float(v) for (i, j), v in qubo.items() if i < j}
    qp.minimize(linear=linear, quadratic=quadratic)

    service = QiskitRuntimeService(instance=instance)
    backend = service.backend(backend_name)

    with Session(service=service, backend=backend) as session:
        sampler = Sampler(session=session)
        sampler.options.default_shots = int(shots)
        # seeding support varies by version; ignore if unavailable
        try:
            sampler.options.seed = int(seed)
        except Exception:
            pass

        qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=int(reps))
        meo = MinimumEigenOptimizer(qaoa)
        res = meo.solve(qp)

    u = np.array(res.x, dtype=np.int64).reshape(-1)
    return u, float(res.fval)


# =============================================================================
# KAHM scenario scoring helpers
# =============================================================================

def compute_folding_distances(model: dict, X_new: np.ndarray, *, n_jobs: int = -1) -> np.ndarray:
    """Return T with shape (C, N_new): per-class folding costs T_c(x) in [0,1]."""
    X_new = np.asarray(X_new)
    if X_new.ndim != 2:
        raise ValueError("X_new must be shaped (D_in, N_new).")

    # Apply the same input scaling used during training (kahm_classification convention)
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
    """
    Convert folding costs T_c(x) into scenario evidence weights w_c(x).

    Default:
      w_raw = (1 - T) ** alpha, optional top-k, normalized to sum=1.
    """
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
    """Synthetic context signals for C scenarios. Output X:(D_in,N), y:(N,)."""
    rng = np.random.default_rng(seed)

    X_parts = []
    y_parts = []
    for c in range(C):
        mean = (c - (C - 1) / 2.0) * (1.25 / max(1, C - 1))
        Xc = rng.normal(loc=mean, scale=1.0, size=(D_in, N_per)).astype(np.float32)
        Xc = np.tanh(Xc)
        X_parts.append(Xc)
        y_parts.append(np.full((N_per,), c, dtype=np.int64))

    X = np.concatenate(X_parts, axis=1)
    y = np.concatenate(y_parts, axis=0)

    perm = rng.permutation(X.shape[1])
    X = X[:, perm]
    y = y[perm]
    return X, y


def make_scenario_qubos(*, M: int, C: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic per-scenario (Q_c, q_c). Replace this with your domain encoding.
    """
    rng = np.random.default_rng(seed)

    Qs = np.zeros((C, M, M), dtype=np.float64)
    qs = np.zeros((C, M), dtype=np.float64)

    for c in range(C):
        A = rng.normal(size=(M, M))
        Q = (A + A.T) / 2.0
        Q *= 0.25 / M

        diag_bias = (c - (C - 1) / 2.0) * (0.8 / max(1, C - 1))
        Q += np.eye(M) * (0.5 + diag_bias)

        q = rng.normal(size=(M,)) * 0.1
        q += (0.2 * diag_bias)

        Qs[c] = Q
        qs[c] = q

    return Qs, qs


def assemble_context_qubo(Qs: np.ndarray, qs: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Q(x)=Σ w_c Q_c, q(x)=Σ w_c q_c"""
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    Q = np.tensordot(w, Qs, axes=(0, 0))
    q = np.tensordot(w, qs, axes=(0, 0))
    return np.asarray(Q, dtype=np.float64), np.asarray(q, dtype=np.float64)


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
    # quantum params
    num_reads: int
    time_limit: Optional[int]
    shots: int
    reps: int
    ibm_instance: str
    ibm_backend: str
    # optional soft tuning
    tune_soft: bool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--D_in", type=int, default=8, help="context dimension n")
    ap.add_argument("--C", type=int, default=3, help="number of scenarios/classes")
    ap.add_argument("--N_per", type=int, default=400, help="training samples per scenario")
    ap.add_argument("--M", type=int, default=12, help="number of QUBO bits")
    ap.add_argument("--subspace_dim", type=int, default=20, help="OTFL AE subspace_dim")
    ap.add_argument("--Nb", type=int, default=10, help="OTFL AE Nb")
    ap.add_argument("--input_scale", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=10.0, help="soft evidence sharpening")
    ap.add_argument("--topk", type=int, default=3, help="keep only top-k scenario weights (0 disables)")

    # New unified solver switch (classical + quantum)
    ap.add_argument(
        "--qubo_solver",
        type=str,
        default="local",
        choices=["bruteforce", "local", "dwave_hybrid", "dwave_qpu", "qiskit_qaoa", "qiskit_ibm"],
        help="QUBO solver backend"
    )

    # Backwards compatibility: accept --solver from the older demo (maps to bruteforce/local)
    ap.add_argument("--solver", type=str, default="", choices=["", "bruteforce", "local"], help=argparse.SUPPRESS)

    # D-Wave params
    ap.add_argument("--num_reads", type=int, default=2000, help="D-Wave QPU reads")
    ap.add_argument("--time_limit", type=int, default=0, help="D-Wave hybrid time limit seconds (0=default)")

    # Qiskit params
    ap.add_argument("--shots", type=int, default=2000, help="Qiskit sampling shots")
    ap.add_argument("--reps", type=int, default=1, help="QAOA depth")

    # IBM Runtime params
    ap.add_argument("--ibm_instance", type=str, default="ibm-q/open/main")
    ap.add_argument("--ibm_backend", type=str, default="", help="IBM backend name, e.g., ibm_brisbane")

    # Optional soft-parameter tuning (uses functions from kahm_classification.py)
    ap.add_argument("--tune_soft", action="store_true", help="tune alpha/topk on held-out validation set")

    args = ap.parse_args()

    # If user used legacy --solver, let it override (only if --qubo_solver left at default)
    qubo_solver = str(args.qubo_solver)
    if args.solver and args.qubo_solver == "local":
        qubo_solver = str(args.solver)

    cfg = DemoConfig(
        seed=args.seed,
        D_in=args.D_in,
        C=args.C,
        N_per=args.N_per,
        M=args.M,
        subspace_dim=args.subspace_dim,
        Nb=args.Nb,
        input_scale=args.input_scale,
        alpha=args.alpha,
        topk=None if int(args.topk) <= 0 else int(args.topk),
        qubo_solver=qubo_solver,
        num_reads=int(args.num_reads),
        time_limit=None if int(args.time_limit) <= 0 else int(args.time_limit),
        shots=int(args.shots),
        reps=int(args.reps),
        ibm_instance=str(args.ibm_instance),
        ibm_backend=str(args.ibm_backend),
        tune_soft=bool(args.tune_soft),
    )

    rng = np.random.default_rng(cfg.seed)

    # 1) Synthetic scenario data (context signals)
    X, y = make_synthetic_scenario_data(D_in=cfg.D_in, N_per=cfg.N_per, C=cfg.C, seed=cfg.seed)

    # Split train/val (also used for optional tuning)
    N = X.shape[1]
    N_train = int(0.8 * N)
    X_train, y_train = X[:, :N_train], y[:N_train]
    X_val, y_val = X[:, N_train:], y[N_train:]
    print(f"[data] X_train={X_train.shape}, X_val={X_val.shape}, C={cfg.C}")

    # 2) Train KAHM scenario scorer (per-class AEs)
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

    # Optional: tune soft mapping alpha/topk using your implementation
    alpha = cfg.alpha
    topk = cfg.topk
    if cfg.tune_soft:
        print("[tune] tuning alpha/topk via cross-entropy on validation set...")
        kc.tune_soft_params_ce(
            model,
            X_val,
            y_val,
            alphas=(2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0),
            topks=(None, 1, 2, 3, 5, 10),
            n_jobs=-1,
            verbose=True,
            show_progress=True,
        )
        # Use tuned params unless user explicitly set CLI overrides
        alpha = float(model.get("soft_alpha") or alpha)
        tuned_topk = model.get("soft_topk")
        if cfg.topk is None and tuned_topk is not None:
            topk = int(tuned_topk)

    # 3) Choose a new operational context x to score
    j = int(rng.integers(0, X_val.shape[1]))
    x = X_val[:, j:j+1]  # shape (D_in,1)
    true_scenario = int(y_val[j])
    print(f"[query] picked one context x from validation set; true scenario label={true_scenario}")

    # 4) Compute folding costs T_c(x) and evidence weights w_c(x)
    Tmat = compute_folding_distances(model, x, n_jobs=-1)  # shape (C,1)
    T = Tmat[:, 0]
    w = folding_to_weights(T, alpha=alpha, topk=topk)

    c_hat = int(np.argmin(T))
    print("\n[scenario scoring]")
    for c in range(cfg.C):
        print(f"  scenario {c}: T_c(x)={float(T[c]):.6f}  w_c(x)={float(w[c]):.6f}")
    print(f"  predicted scenario (argmin T) = {c_hat}")

    # 5) Build scenario-conditioned QUBOs and assemble Q(x), q(x)
    Qs, qs = make_scenario_qubos(M=cfg.M, C=cfg.C, seed=cfg.seed + 123)
    Qx, qx = assemble_context_qubo(Qs, qs, w)
    qubo = Qq_to_qubo_dict(Qx, qx, tol=0.0)

    print(f"\n[qubo] M={cfg.M}, nonzeros={len(qubo)}, solver={cfg.qubo_solver}")

    # 6) Solve the context-conditioned QUBO
    if cfg.qubo_solver == "bruteforce":
        if cfg.M > 22:
            raise ValueError("bruteforce is too expensive for M > ~22; use --qubo_solver local/dwave/qiskit.")
        u_star, E_star = solve_qubo_bruteforce(Qx, qx)
        origin = "classical bruteforce"
    elif cfg.qubo_solver == "local":
        u_star, E_star = solve_qubo_local_search(Qx, qx, seed=cfg.seed)
        origin = "classical local-search"
    elif cfg.qubo_solver == "dwave_hybrid":
        u_star, E_star = solve_qubo_dwave_hybrid(qubo, time_limit=cfg.time_limit)
        origin = "D-Wave LeapHybridSampler"
    elif cfg.qubo_solver == "dwave_qpu":
        u_star, E_star = solve_qubo_dwave_qpu(qubo, num_reads=cfg.num_reads)
        origin = "D-Wave QPU (EmbeddingComposite(DWaveSampler))"
    elif cfg.qubo_solver == "qiskit_qaoa":
        u_star, E_star = solve_qubo_qiskit_qaoa(qubo, reps=cfg.reps, shots=cfg.shots, seed=cfg.seed)
        origin = "Qiskit QAOA (Sampler primitive)"
    elif cfg.qubo_solver == "qiskit_ibm":
        u_star, E_star = solve_qubo_qiskit_ibm_runtime(
            qubo,
            reps=cfg.reps,
            shots=cfg.shots,
            seed=cfg.seed,
            instance=cfg.ibm_instance,
            backend_name=cfg.ibm_backend,
        )
        origin = "IBM Runtime QAOA (Sampler)"
    else:
        raise RuntimeError(f"Unknown solver: {cfg.qubo_solver}")

    print(f"\n[result] origin={origin}")
    print(f"  solution u* (M={cfg.M}): {u_star.astype(int).tolist()}")
    print(f"  solver energy = {E_star:.6f}")

    # Always report the original objective energy for consistency
    E_check = qubo_energy_Qq(u_star, Qx, qx)
    print(f"  objective check (u^TQ u + q^T u) = {E_check:.6f}  (Δ={E_check - E_star:+.6e})")

    # 7) Interpretability: per-scenario optimal energies (avoid quantum calls here by default)
    print("\n[interpretability] per-scenario optimal energies (classical local-search baseline):")
    for c in range(cfg.C):
        Qc, qc = Qs[c], qs[c]
        if cfg.M <= 22:
            uc, Ec = solve_qubo_bruteforce(Qc, qc)
        else:
            uc, Ec = solve_qubo_local_search(Qc, qc, seed=cfg.seed + c + 1, n_restarts=25, n_steps=5000)
        print(f"  scenario {c}: best E={Ec:.6f}  u={uc.astype(int).tolist()}")

    print("\nDone.")


if __name__ == "__main__":
    main()

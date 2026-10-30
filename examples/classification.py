#!/usr/bin/env python3
"""Run the fixed Digits classification example and write inspectable artifacts.

Install the repository first: ``python -m pip install -e '.[examples]'``.
Then run ``python examples/classification.py --output-dir artifacts/digits``.
The evaluation protocol is defined in the adjacent PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_info, threadpool_limits

import kahm
from kahm import KAHMClassifier


SEED = 42
PROTOCOL_ID = "digits-classification-v1"
PROTOCOL_PATH = Path(__file__).with_name("PROTOCOL.md")
TUNED_METHODS = ("KAHM", "Logistic regression", "RBF SVM", "k-nearest neighbors")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_value(value: Any) -> Any:
    """Convert results to strict, readable JSON; failed scores become null."""
    if isinstance(value, dict):
        return {str(key): json_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(val) for val in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, StandardScaler):
        return "StandardScaler"
    if isinstance(value, MinMaxScaler):
        return "MinMaxScaler(-1,1; clip=False)"
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_splits(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Return original dataset IDs and CV indices relative to development."""
    development, test = train_test_split(
        np.arange(len(y)), test_size=0.25, stratify=y, random_state=SEED
    )
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    folds = list(splitter.split(np.zeros((len(development), 1)), y[development]))
    return development, test, folds


def model_specs() -> list[tuple[str, Pipeline, list[dict[str, Any]]]]:
    """Six declared candidates per tuned method, plus one dummy reference."""
    models = [
        (
            "KAHM",
            KAHMClassifier(block_size=64, alpha=10.0, random_state=SEED),
            {"model__subspace_dim": [4, 8, 16]},
        ),
        (
            "Logistic regression",
            LogisticRegression(solver="lbfgs", max_iter=5000, random_state=SEED),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        ("RBF SVM", SVC(kernel="rbf", gamma="scale"), {"model__C": [1.0, 10.0, 100.0]}),
        (
            "k-nearest neighbors",
            KNeighborsClassifier(weights="distance", metric="minkowski", p=2, n_jobs=1),
            {"model__n_neighbors": [1, 3, 5]},
        ),
    ]
    specs = []
    for name, estimator, model_grid in models:
        pipeline = Pipeline([("scale", StandardScaler()), ("model", estimator)])
        grid = [
            {"scale": [StandardScaler()], **model_grid},
            {"scale": [MinMaxScaler(feature_range=(-1, 1), clip=False)], **model_grid},
        ]
        specs.append((name, pipeline, grid))
    specs.append(
        (
            "Dummy (most frequent)",
            Pipeline([("scale", "passthrough"), ("model", DummyClassifier(strategy="most_frequent"))]),
            [{}],
        )
    )
    return specs


def source_provenance() -> dict[str, Any]:
    """Hash executed source files; a Git revision alone can omit local edits."""
    files = {"examples/classification.py": Path(__file__).resolve()}
    for path in Path(kahm.__file__).parent.glob("*.py"):
        files[f"kahm/{path.name}"] = path
    legacy_modules = [
        "kahm_classification", "autoencoder", "parallel_autoencoders", "dim_reduce",
        "kxx_matrix", "kxa_matrix", "distance_matrix", "kernel_regularized_least_squares",
        "autoencoder_filtering_extended", "combine_multiple_autoencoders_extended",
    ]
    for name in legacy_modules:
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.origin:
            files[f"{name}.py"] = Path(spec.origin)
    provenance: dict[str, Any] = {
        "sha256": {name: sha256_file(path) for name, path in sorted(files.items())}
    }
    root = Path(__file__).resolve().parent.parent
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True, timeout=5
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True, timeout=5
        )
        provenance.update(git_commit=head.stdout.strip(), working_tree_dirty=bool(status.stdout.strip()))
    except (OSError, subprocess.SubprocessError):
        provenance.update(git_commit=None, working_tree_dirty=None)
    return provenance


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "kahm": getattr(kahm, "__version__", "unknown"),
        "search_n_jobs": 1,
        "numerical_thread_limit": 1,
        "threadpools_during_evaluation": [
            {key: item.get(key) for key in ("internal_api", "prefix", "version", "num_threads", "architecture")}
            for item in threadpool_info()
        ],
    }


def warning_records(caught: list[Any]) -> list[dict[str, str]]:
    return [{"category": item.category.__name__, "message": str(item.message)} for item in caught]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(json_value(row))


def write_report(output: Path, results: dict[str, Any]) -> None:
    complete = results["complete"]
    lines = [
        "# KAHM classification example: Digits", "",
        "**Status: " + ("complete" if complete else "INCOMPLETE — inspect failures before using these results") + ".**", "",
        "This is a tutorial benchmark on one fixed held-out split, not a claim of general superiority.", "",
        f"The dataset has {results['dataset']['n_samples']:,} examples, 64 raw image features, and 10 classes. "
        f"There are {results['split']['n_development']:,} development and {results['split']['n_test']:,} test examples. "
        "Every tuned method receives the same three development folds and six candidate pipelines. "
        "All searches finish before any held-out prediction is evaluated. The dummy has one candidate.", "",
        "## Held-out results", "",
        "| Method | Accuracy | Balanced accuracy | Macro F1 | Search + refit (s) | Predict 450 examples (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results["methods"]:
        if row["status"] != "ok":
            lines.append(f"| {row['method']} | FAILED | — | — | {row['search_refit_seconds']:.4f} | — |")
            continue
        metrics = row["test_metrics"]
        lines.append(
            f"| {row['method']} | {metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | "
            f"{metrics['macro_f1']:.4f} | {row['search_refit_seconds']:.4f} | {row['test_predict_seconds']:.6f} |"
        )
    lines.extend([
        "", "![Held-out scores and measured computation](comparison.svg)", "",
        "Selection used development balanced accuracy only. Equal candidate counts do not equate model "
        "capacity or computation, and these small grids are not exhaustive tuning. There are no significance "
        "tests or population confidence intervals. Test examples are from one dataset, not independent domains.",
        "", "## Selected configurations", "",
        "| Method | Mean development balanced accuracy | Selected parameters |",
        "|---|---:|---|",
    ])
    for row in results["methods"]:
        if "selected_parameters" in row:
            params = json.dumps(json_value(row["selected_parameters"]), sort_keys=True)
            lines.append(f"| {row['method']} | {row['best_cv_balanced_accuracy']:.4f} | `{params}` |")
    kahm_row = next(row for row in results["methods"] if row["method"] == "KAHM")
    if kahm_row["status"] == "ok":
        metrics = kahm_row["test_metrics"]
        successful_baselines = [
            row for row in results["methods"] if row["method"] in TUNED_METHODS[1:] and row["status"] == "ok"
        ]
        lines.extend(["", "## Interpretation", ""])
        lines.append(
            f"KAHM obtains {metrics['accuracy']:.2%} accuracy and {metrics['balanced_accuracy']:.2%} "
            "balanced accuracy on the held-out images. "
            "The complete table includes favorable and unfavorable comparisons."
        )
        if successful_baselines:
            strongest = max(successful_baselines, key=lambda row: row["test_metrics"]["balanced_accuracy"])
            difference = 100 * (metrics["balanced_accuracy"] - strongest["test_metrics"]["balanced_accuracy"])
            lines.append(
                f"The highest observed baseline balanced accuracy belongs to {strongest['method']}; "
                f"KAHM's difference is {difference:+.2f} percentage points. This is a descriptive comparison "
                "after evaluation, not another model-selection step or evidence across datasets."
            )
        if kahm_row.get("model_stats"):
            stats = kahm_row["model_stats"]
            lines.append(
                f"The refitted KAHM contains {stats['local_models']} local reference models and "
                f"{stats['reference_samples']} retained reference observations."
            )
        lines.extend(["", "![KAHM confusion matrix](kahm_confusion.svg)"])
    lines.extend([
        "", "## Timing scope and warnings", "",
        "Search time includes fold-local preprocessing, validation fits, model selection, and the final refit. "
        "Prediction time includes the selected scaler and classifier for all 450 test examples. "
        "Imports, shared data loading, plotting, and artifact writing are excluded. "
        "Each duration is one wall-clock measurement, with serial search and one numerical-library thread; "
        "it depends on this implementation and environment.", "",
    ])
    warned = False
    for row in results["methods"]:
        if row.get("error"):
            warned = True
            lines.append(f"- **{row['method']} failed:** `{row['error']}`")
        for warning in row.get("warnings", []):
            warned = True
            lines.append(f"- {row['method']}: {warning['category']}: {warning['message']}")
    if not warned:
        lines.append("No fitting or prediction warnings were captured.")
    env = results["environment"]
    lines.extend([
        "", "## Provenance", "",
        f"- Protocol: `{PROTOCOL_ID}`.",
        f"- Protocol SHA-256: `{results['protocol_sha256']}`.",
        f"- Dataset SHA-256: `{results['dataset']['sha256']}`.",
        f"- Run started (UTC): `{results['started_at_utc']}`.",
        f"- Python {env['python']}; NumPy {env['numpy']}; scikit-learn {env['scikit_learn']}; KAHM {env['kahm']}.",
        f"- Platform: {env['platform']} ({env['machine']}).", "",
        "See `results.json` for the complete environment and executed source hashes, `splits.json` for "
        "the exact indices, `cv_results.csv` for every validation candidate, and `predictions.csv` for all "
        "selected-model test predictions. Protocol details are in `examples/PROTOCOL.md` in the repository.",
        "", "KAHM scores are not assumed to be calibrated probabilities. This example evaluates class labels only.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(output: Path, results: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    rows = [row for row in results["methods"] if row["status"] == "ok"]
    if not rows:
        return
    names = [row["method"] for row in rows]
    names = ["Dummy reference" if name.startswith("Dummy") else name for name in names]
    colors = ["#d8792c" if row["method"] == "KAHM" else "#8a939b" if row["method"].startswith("Dummy") else "#35566c" for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.set_facecolor("#fbfaf7")
    for axis in axes:
        axis.set_facecolor("#fbfaf7")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.set_axisbelow(True)
        axis.grid(axis="x", alpha=0.18)
    scores = [row["test_metrics"]["balanced_accuracy"] for row in rows]
    axes[0].barh(names, scores, color=colors, height=0.58)
    axes[0].set_xlim(0, 1.12)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Held-out balanced accuracy · higher is better")
    axes[0].set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    for index, score in enumerate(scores):
        axes[0].text(score + .014, index, f"{score:.1%}", va="center", fontsize=10)
    times = [row["search_refit_seconds"] for row in rows]
    axes[1].barh(names, times, color=colors, height=0.58)
    axes[1].set_xscale("log")
    axes[1].invert_yaxis()
    axes[1].set_yticklabels([])
    axes[1].set_xlim(max(min(times) * .6, 1e-6), max(times) * 5)
    axes[1].set_xlabel("Search + refit seconds · log scale · lower is better")
    for index, elapsed in enumerate(times):
        axes[1].text(elapsed * 1.10, index, f"{elapsed:.3f}s", va="center", fontsize=10)
    fig.suptitle("KAHM classification example: Digits", x=.03, ha="left", y=.98, fontsize=18, fontweight="bold", color="#233845")
    fig.text(.03, .89, "450 held-out images • common development folds • six candidates per tuned method", color="#52616b")
    fig.text(.03, .025, "One fixed split; descriptive results. Dummy reference uses one candidate. Timings include selection and refit.", fontsize=9, color="#52616b")
    fig.subplots_adjust(left=.18, right=.97, top=.79, bottom=.18, wspace=.16)
    for extension in ("svg", "png"):
        fig.savefig(output / f"comparison.{extension}", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    kahm_result = next(row for row in rows if row["method"] == "KAHM") if any(row["method"] == "KAHM" for row in rows) else None
    if kahm_result is not None:
        counts = np.asarray(kahm_result["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(7.4, 6.5))
        im = ax.imshow(counts, cmap="YlOrBr", vmin=0)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("Predicted digit")
        ax.set_ylabel("True digit")
        ax.set_title("KAHM · held-out confusion matrix\nCounts on the same 450 test images", pad=15)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if counts[i, j] > counts.max() * .55 else "#45352a")
        fig.colorbar(im, ax=ax, shrink=.8, label="Examples")
        fig.tight_layout()
        for extension in ("svg", "png"):
            fig.savefig(output / f"kahm_confusion.{extension}", dpi=160)
        plt.close(fig)


def run(output: Path) -> dict[str, Any]:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(
            f"Output path must be a new or empty directory: {output}. "
            "Choose a fresh --output-dir to preserve previous results and avoid stale artifacts."
        )
    output.mkdir(parents=True, exist_ok=True)
    X, y = load_digits(return_X_y=True)
    development, test, folds = make_splits(y)
    dataset_hash = hashlib.sha256(
        np.asarray(X, dtype="<f8").tobytes(order="C") + np.asarray(y, dtype="<i8").tobytes(order="C")
    ).hexdigest()
    results: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "dataset": {"name": "sklearn.datasets.load_digits", "n_samples": len(y), "n_features": X.shape[1], "n_classes": len(np.unique(y)), "sha256": dataset_hash},
        "split": {"n_development": len(development), "n_test": len(test), "cv_folds": 3},
        "source_provenance": source_provenance(),
        "methods": [],
        "complete": True,
    }
    split_record = {
        "development_indices": development, "test_indices": test,
        "cv_folds_original_indices": [
            {"train": development[train], "validation": development[valid]} for train, valid in folds
        ],
    }
    write_json(output / "splits.json", split_record)
    fitted: dict[str, Any] = {}
    cv_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    with threadpool_limits(limits=1):
        results["environment"] = environment()
        # Deliberately finish all model selection before evaluating test predictions.
        for name, pipeline, grid in model_specs():
            print(f"Selecting {name} on development data ({len(ParameterGrid(grid))} candidates)...", flush=True)
            record: dict[str, Any] = {"method": name, "candidate_count": len(ParameterGrid(grid)), "status": "pending", "warnings": []}
            search = GridSearchCV(
                pipeline, grid, scoring="balanced_accuracy", cv=folds, refit=True,
                n_jobs=1, error_score=np.nan, return_train_score=False,
            )
            started = perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    search.fit(X[development], y[development])
                    record["search_refit_seconds"] = perf_counter() - started
                    record["selected_parameters"] = json_value(search.best_params_)
                    record["best_cv_balanced_accuracy"] = float(search.best_score_)
                    record["best_candidate_index"] = int(search.best_index_)
                    scores = search.cv_results_["mean_test_score"]
                    failed = np.flatnonzero(~np.isfinite(scores))
                    record["failed_candidate_indices"] = failed.tolist()
                    if len(failed):
                        results["complete"] = False
                    for i, params in enumerate(search.cv_results_["params"]):
                        cv_rows.append({
                            "method": name, "candidate": i,
                            "parameters": json.dumps(json_value(params), sort_keys=True),
                            **{f"fold_{k}_balanced_accuracy": search.cv_results_[f"split{k}_test_score"][i] for k in range(3)},
                            "mean_balanced_accuracy": scores[i],
                            "std_balanced_accuracy": search.cv_results_["std_test_score"][i],
                            "mean_fit_seconds": search.cv_results_["mean_fit_time"][i],
                            "mean_score_seconds": search.cv_results_["mean_score_time"][i],
                            "selected": i == search.best_index_,
                        })
                    fitted[name] = search.best_estimator_
                    record["status"] = "selected"
                except Exception as error:
                    record.update(status="failed", error=f"{type(error).__name__}: {error}", search_refit_seconds=perf_counter() - started)
                    results["complete"] = False
                    if not any(row["method"] == name for row in cv_rows):
                        for i, params in enumerate(ParameterGrid(grid)):
                            cv_rows.append({
                                "method": name, "candidate": i,
                                "parameters": json.dumps(json_value(params), sort_keys=True),
                                **{f"fold_{k}_balanced_accuracy": None for k in range(3)},
                                "mean_balanced_accuracy": None, "std_balanced_accuracy": None,
                                "mean_fit_seconds": None, "mean_score_seconds": None,
                                "selected": False,
                            })
                record["warnings"].extend(warning_records(caught))
            results["methods"].append(record)

        print("All development searches finished. Evaluating selected models on the holdout...", flush=True)
        prediction_rows = [{"sample_index": int(index), "true_label": int(y[index])} for index in test]
        for record in results["methods"]:
            name = record["method"]
            if name not in fitted:
                continue
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    start = perf_counter()
                    prediction = fitted[name].predict(X[test])
                    record["test_predict_seconds"] = perf_counter() - start
                    record["test_metrics"] = {
                        "accuracy": float(accuracy_score(y[test], prediction)),
                        "balanced_accuracy": float(balanced_accuracy_score(y[test], prediction)),
                        "macro_f1": float(f1_score(y[test], prediction, average="macro", zero_division=0)),
                    }
                    record["confusion_matrix"] = confusion_matrix(y[test], prediction, labels=np.arange(10)).tolist()
                    for row, label in zip(prediction_rows, prediction):
                        row[name] = int(label)
                    if name == "KAHM":
                        model = fitted[name].named_steps["model"]
                        state = getattr(model, "model_", None)
                        if isinstance(state, dict):
                            banks = [bank for group in state["classifier"] for bank in group]
                            record["model_stats"] = {"local_models": len(banks), "reference_samples": sum(bank["y_data"].shape[1] for bank in banks)}
                    record["status"] = "ok"
                except Exception as error:
                    record.update(status="failed", error=f"{type(error).__name__}: {error}")
                    results["complete"] = False
                record["warnings"].extend(warning_records(caught))

    results["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "results.json", results)
    cv_fields = ["method", "candidate", "parameters", "fold_0_balanced_accuracy", "fold_1_balanced_accuracy", "fold_2_balanced_accuracy", "mean_balanced_accuracy", "std_balanced_accuracy", "mean_fit_seconds", "mean_score_seconds", "selected"]
    write_csv(output / "cv_results.csv", cv_rows, cv_fields)
    write_csv(output / "predictions.csv", prediction_rows, ["sample_index", "true_label", *[row["method"] for row in results["methods"]]])
    write_figures(output, results)
    write_report(output, results)
    for row in results["methods"]:
        if row["status"] == "ok":
            print(f"{row['method']:<24} balanced accuracy={row['test_metrics']['balanced_accuracy']:.4f}; search+refit={row['search_refit_seconds']:.3f}s")
        else:
            print(f"{row['method']:<24} FAILED (see results.json)")
    print(f"Artifacts: {output.resolve()}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/digits"), help="New or empty directory for reports, predictions, CV records, and figures.")
    args = parser.parse_args()
    results = run(args.output_dir)
    return 0 if results["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())

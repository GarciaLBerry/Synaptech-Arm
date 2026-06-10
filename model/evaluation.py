from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    confusion_matrix
)
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from .dimensionality_visualization import plot_dimensionality_results
from .config import LABEL_MAPPING, PACKET_SIZE, PACKET_STRIDE, DROP_TRANSITIONS
from . import utils

LABELS = [key for key in LABEL_MAPPING]

def print_train_results(x_test, y_test, tuner, version: int = 0):
    print_model_results(x_test, y_test, tuner.best_estimator_, version)
    
def print_results_by_version(target_model: int = 6):
    _, x_test, _, y_test = utils.get_data("data")
    estimator = utils.load_pipeline(target_model)
    
    print_model_results(x_test, y_test, estimator, version=target_model)

def print_model_results(x_test, y_test, estimator, version=0):
    
    y_pred = estimator.predict(x_test)
    y_proba = estimator.predict_proba(x_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    test_log_loss = log_loss(y_test, y_proba, labels=LABELS)
    true_class_prob = np.exp(-test_log_loss)

    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    )
    
    plot_dimensionality_results(
        x_test,
        y_test,
        estimator,
        out_dir=f"outputs/fa_v{version}",
        reducer_step="reduce",
        prefix=f"fa_v{version}",
    )

    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Log loss: {test_log_loss:.4f}")
    print(f"True-class assigned probability: {true_class_prob:.4f}")
    print("Confusion matrix:")
    print(cm_df)
    
def save_cv_training_report(tuner, version: int, out_dir: str = "outputs"):
    out_path = Path(out_dir) / f"fa_v{version}"
    out_path.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(tuner.cv_results_)
    results = results.sort_values("rank_test_balanced_accuracy").reset_index(drop=True)

    # Save full raw CV results.
    cv_results_path = out_path / f"fa_v{version}_cv_results.csv"
    results.to_csv(cv_results_path, index=False)

    # Build a smaller readable summary table.
    summary_cols = [
        "rank_test_balanced_accuracy",
        "mean_train_balanced_accuracy",
        "std_train_balanced_accuracy",
        "mean_test_balanced_accuracy",
        "std_test_balanced_accuracy",
        "mean_train_f1_macro",
        "mean_test_f1_macro",
        "mean_train_neg_log_loss",
        "mean_test_neg_log_loss",
        "mean_fit_time",
        "mean_score_time",
    ]

    param_cols = [
        col for col in results.columns
        if col.startswith("param_")
    ]

    available_summary_cols = [
        col for col in summary_cols + param_cols
        if col in results.columns
    ]

    summary = results[available_summary_cols].copy()

    if (
        "mean_train_balanced_accuracy" in summary.columns
        and "mean_test_balanced_accuracy" in summary.columns
    ):
        summary["balanced_accuracy_gap"] = (
            summary["mean_train_balanced_accuracy"]
            - summary["mean_test_balanced_accuracy"]
        )

    summary_path = out_path / f"fa_v{version}_cv_summary.csv"
    summary.to_csv(summary_path, index=False)

    _plot_train_vs_validation_score(results, version, out_path)
    _plot_generalization_gap(results, version, out_path)
    _plot_best_score_so_far(results, version, out_path)
    #_plot_param_effects(results, version, out_path)
    _save_final_estimator_stats(tuner.best_estimator_, version, out_path)

    print(f"Saved CV results: {cv_results_path}")
    print(f"Saved CV summary: {summary_path}")
    
def _plot_train_vs_validation_score(results, version, out_path):
    if "mean_train_balanced_accuracy" not in results.columns:
        return

    ordered = results.sort_values("mean_test_balanced_accuracy").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ordered))

    ax.plot(
        x,
        ordered["mean_train_balanced_accuracy"],
        marker="o",
        linewidth=1,
        label="train balanced accuracy",
    )
    ax.plot(
        x,
        ordered["mean_test_balanced_accuracy"],
        marker="o",
        linewidth=1,
        label="validation balanced accuracy",
    )

    ax.set_title("Train vs validation balanced accuracy across CV candidates")
    ax.set_xlabel("hyperparameter candidate, sorted by validation score")
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path / f"fa_v{version}_train_vs_validation_balanced_accuracy.png", dpi=160)
    plt.close(fig)


def _plot_generalization_gap(results, version, out_path):
    if "mean_train_balanced_accuracy" not in results.columns:
        return

    ordered = results.sort_values("mean_test_balanced_accuracy").reset_index(drop=True)
    gap = ordered["mean_train_balanced_accuracy"] - ordered["mean_test_balanced_accuracy"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(ordered)), gap)

    ax.set_title("Generalization gap across CV candidates")
    ax.set_xlabel("hyperparameter candidate, sorted by validation score")
    ax.set_ylabel("train balanced accuracy - validation balanced accuracy")
    ax.axhline(0, color="black", linewidth=1)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path / f"fa_v{version}_generalization_gap.png", dpi=160)
    plt.close(fig)


def _plot_best_score_so_far(results, version, out_path):
    scores = results["mean_test_balanced_accuracy"].to_numpy()
    best_so_far = np.maximum.accumulate(scores)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(scores, marker="o", linewidth=1, label="candidate validation score")
    ax.plot(best_so_far, marker="o", linewidth=1.5, label="best validation score so far")

    ax.set_title("Random search progress")
    ax.set_xlabel("random-search candidate")
    ax.set_ylabel("validation balanced accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path / f"fa_v{version}_random_search_progress.png", dpi=160)
    plt.close(fig)
    
def _plot_param_effects(results, version, out_path):
    param_names = [
        "param_reduce__n_components",
        "param_reduce__append_original",
        "param_reduce__rotation",
        "param_wave__level",
        "param_wave__wavelet",
    ]

    for param_name in param_names:
        if param_name not in results.columns:
            continue

        grouped = (
            results
            .groupby(param_name, dropna=False)["mean_test_balanced_accuracy"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(grouped))

        ax.bar(x, grouped["mean"], yerr=grouped["std"].fillna(0))
        ax.set_title(f"Validation balanced accuracy by {param_name.replace('param_', '')}")
        ax.set_xlabel(param_name.replace("param_", ""))
        ax.set_ylabel("mean validation balanced accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped[param_name].astype(str), rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.25)

        for i, count in enumerate(grouped["count"]):
            ax.text(i, grouped["mean"].iloc[i], f"n={count}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        safe_name = param_name.replace("param_", "").replace("__", "_")
        fig.savefig(out_path / f"fa_v{version}_cv_effect_{safe_name}.png", dpi=160)
        plt.close(fig)
        
def _save_final_estimator_stats(estimator, version, out_path):
    rows = []

    reducer = estimator.named_steps.get("reduce")
    model = estimator.named_steps.get("model")

    if reducer is not None:
        rows.append({
            "component": "reduce",
            "stat": "n_iter_",
            "value": getattr(reducer, "n_iter_", None),
        })
        rows.append({
            "component": "reduce",
            "stat": "n_components",
            "value": getattr(reducer, "n_components", None),
        })

        loglike = getattr(reducer, "loglike_", None)
        if loglike is not None and len(loglike) > 0:
            rows.append({
                "component": "reduce",
                "stat": "final_loglike",
                "value": float(loglike[-1]),
            })
            rows.append({
                "component": "reduce",
                "stat": "n_loglike_steps",
                "value": len(loglike),
            })

    if model is not None:
        n_iter = getattr(model, "n_iter_", None)
        if n_iter is not None:
            rows.append({
                "component": "model",
                "stat": "n_iter_",
                "value": np.asarray(n_iter).tolist(),
            })

        rows.append({
            "component": "model",
            "stat": "C",
            "value": getattr(model, "C", None),
        })
        rows.append({
            "component": "model",
            "stat": "l1_ratio",
            "value": getattr(model, "l1_ratio", None),
        })
        rows.append({
            "component": "model",
            "stat": "max_iter",
            "value": getattr(model, "max_iter", None),
        })
    
    rows.append({
        "component": "config",
        "stat": "PACKET_SIZE",
        "value": PACKET_SIZE,
    })
    
    rows.append({
        "component": "config",
        "stat": "PACKET_STRIDE",
        "value": PACKET_STRIDE,
    })
    
    rows.append({
        "component": "config",
        "stat": "DROP_TRANSITIONS",
        "value": DROP_TRANSITIONS,
    })

    stats = pd.DataFrame(rows)
    stats.to_csv(out_path / f"fa_v{version}_final_estimator_stats.csv", index=False)
"""
Probe training logic: Difference-in-Means (Mass-Mean) baseline,
Logistic Regression with Stratified K-Fold CV, and layer accuracy visualization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


@dataclass
class LayerResult:
    layer: int
    cv_mean_acc: float
    cv_std_acc: float
    mass_mean_acc: float
    probe: LogisticRegression
    scaler: StandardScaler
    direction: np.ndarray = field(repr=False)
    mu_truth: np.ndarray = field(repr=False)
    mu_lie: np.ndarray = field(repr=False)
    avg_activation_norm: float = 0.0


def mass_mean_direction(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Difference-in-Means / Mass-Mean probing (Marks & Tegmark, 2023).

    Vector connecting Truth-centroid to Lie-centroid in activation space, normalized to unit length.
    """
    mu_truth = X[y == 0].mean(axis=0)
    mu_lie = X[y == 1].mean(axis=0)
    direction = mu_lie - mu_truth
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return direction, mu_truth, mu_lie


def mass_mean_accuracy(X: np.ndarray, y: np.ndarray, direction: np.ndarray, mu_truth: np.ndarray, mu_lie: np.ndarray) -> float:
    """
    Classify by nearest centroid along mass-mean direction as baseline sanity check.
    """
    midpoint = (mu_truth + mu_lie) / 2
    projections = (X - midpoint) @ direction
    preds = (projections > 0).astype(int)  # >0 => closer to Lie side
    return float((preds == y).mean())


def train_probes(
    X_by_layer: Dict[int, np.ndarray],
    y: np.ndarray,
    layers: List[int],
    seed: int = 42,
    n_cv_folds: int = 5,
) -> List[LayerResult]:
    """
    Trains Logistic Regression probes per layer with K-Fold CV and computes mass-mean baselines.
    """
    results = []
    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)

    for l in layers:
        X_raw = X_by_layer[l]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
        clf.fit(X_scaled, y)  # Fit on full data for downstream steering

        direction, mu_truth, mu_lie = mass_mean_direction(X_raw, y)
        mm_acc = mass_mean_accuracy(X_raw, y, direction, mu_truth, mu_lie)
        avg_norm = float(np.linalg.norm(X_raw, axis=1).mean())

        result = LayerResult(
            layer=l,
            cv_mean_acc=cv_scores.mean(),
            cv_std_acc=cv_scores.std(),
            mass_mean_acc=mm_acc,
            probe=clf,
            scaler=scaler,
            direction=direction,
            mu_truth=mu_truth,
            mu_lie=mu_lie,
            avg_activation_norm=avg_norm,
        )
        results.append(result)

        print(
            f"Layer {l:>2}: logreg CV acc = {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}  "
            f"| mass-mean acc = {mm_acc:.3f}  | avg |resid| = {avg_norm:.1f}"
        )

    return results


def plot_layer_accuracy(
    results: List[LayerResult],
    save_path: str = "layer_accuracy.png",
    model_name: str = "model",
    show_plot: bool = False,
):
    """
    Generates and saves plot comparing probe accuracy across layers against chance baseline.
    """
    df = pd.DataFrame([{
        "layer": r.layer,
        "logreg_acc": r.cv_mean_acc,
        "logreg_std": r.cv_std_acc,
        "mass_mean_acc": r.mass_mean_acc,
    } for r in results])

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))

    plt.plot(df["layer"], df["logreg_acc"], marker="o", linewidth=2, label="Logistic Regression (5-fold CV)")
    plt.fill_between(
        df["layer"],
        df["logreg_acc"] - df["logreg_std"],
        df["logreg_acc"] + df["logreg_std"],
        alpha=0.2,
    )
    plt.plot(
        df["layer"],
        df["mass_mean_acc"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Mass-Mean (Difference-in-Means)",
    )
    plt.axhline(0.5, color="red", linestyle=":", label="Chance (50%)")

    plt.title(f"Truth vs. Deception: Linear Separability by Layer\n({model_name})")
    plt.xlabel("Layer")
    plt.ylabel("Classification Accuracy")
    plt.ylim(0.4, 1.05)
    plt.xticks(df["layer"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close()
    print(f"Saved accuracy plot to {save_path}")

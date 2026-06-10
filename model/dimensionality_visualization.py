from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import core_cols, LABEL_MAPPING

def plot_dimensionality_results(
    x,
    y,
    estimator,
    out_dir="model/dimensionality_outputs",
    reducer_step="reduce",
    prefix=None,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y)

    if prefix is None:
        prefix = reducer_step

    reducer = estimator.named_steps.get(reducer_step)
    
    paths = {}
    if reducer is not None:
        x_features = _transform_until_step(estimator, reducer_step, x, include_step=False)
        
        if hasattr(reducer, "transform_latent"):
            x_latent = reducer.transform_latent(x_features)
        else:
            x_latent = _transform_until_step(estimator, reducer_step, x, include_step=True)
            
        active_core_cols = _active_core_cols(estimator, core_cols)
        feature_names = _wavelet_feature_names(estimator, x_features.shape[1], active_core_cols)
        components = _get_reducer_attr(reducer, "components_")
        noise_variance = _get_reducer_attr(reducer, "noise_variance_")
        reducer_feature_count = _reducer_feature_count(components, noise_variance)
        reducer_feature_names = _reducer_feature_names(
            reducer,
            feature_names,
            reducer_feature_count,
        )

        # for i, name in enumerate(feature_names):
        #     print(i, name)

        paths["latent_2d"] = _plot_latent_2d(
            x_latent,
            y,
            out_path / f"{prefix}_latent_2d.png",
        )

        if x_latent.shape[1] >= 3:
            paths["latent_3d"] = _plot_latent_3d(
                x_latent,
                y,
                out_path / f"{prefix}_latent_3d.png",
            )

        paths["variance_breakdown"] = _plot_variance_breakdown(
            reducer,
            x_latent,
            out_path / f"{prefix}_variance_breakdown.png",
        )
        paths["latent_variance_by_label"] = _plot_latent_variance_by_label(
            x_latent,
            y,
            out_path / f"{prefix}_latent_variance_by_label.png",
        )

        if components is not None:
            paths["loadings_heatmap"] = _plot_loadings_heatmap(
                components,
                reducer_feature_names,
                out_path / f"{prefix}_loadings_heatmap.png",
            )

        if noise_variance is not None:
            paths["noise_variance"] = _plot_noise_variance(
                noise_variance,
                estimator,
                reducer_feature_names,
                out_path / f"{prefix}_noise_variance.png",
            )
            
            if components is not None:
                paths["noise_fraction_table"] = _save_fa_noise_fraction_table(
                    components,
                    noise_variance,
                    reducer_feature_names,
                    out_path / f"{prefix}_noise_fraction_table.csv",
                )

    if hasattr(estimator, "predict_proba"):
        paths["classifier_confidence"] = _plot_classifier_confidence(
            estimator,
            x,
            y,
            out_path / f"{prefix}_classifier_confidence.png",
        )
        
    #df, X_features = diagnose_wavelet_features(estimator, x)

    return paths
    
def diagnose_wavelet_features(estimator, X, reducer_step="reduce"):
    # Features going into FA/reducer
    X_features = _transform_until_step(
        estimator,
        reducer_step,
        X,
        include_step=False,
    )

    reducer = estimator.named_steps[reducer_step]
    noise = getattr(reducer, "noise_variance_", None)
    if noise is None and hasattr(reducer, "model_"):
        noise = getattr(reducer.model_, "noise_variance_", None)

    print("X_features shape:", X_features.shape)

    feature_var = np.var(X_features, axis=0)
    feature_std = np.std(X_features, axis=0)
    feature_min = np.min(X_features, axis=0)
    feature_max = np.max(X_features, axis=0)

    df = pd.DataFrame({
        "feature_idx": np.arange(X_features.shape[1]),
        "var": feature_var,
        "std": feature_std,
        "min": feature_min,
        "max": feature_max,
    })

    if noise is not None and len(noise) == len(df):
        df["fa_noise_variance"] = noise

    print("\nLowest raw/input feature variance:")
    print(df.sort_values("var").head(15).to_string(index=False))

    print("\nLowest FA noise variance:")
    if noise is not None:
        print(df.sort_values("fa_noise_variance").head(15).to_string(index=False))

    # Detect exact or near-exact duplicate feature columns
    print("\nNear-duplicate feature pairs:")
    n_features = X_features.shape[1]
    found = False
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if np.allclose(X_features[:, i], X_features[:, j], atol=1e-8, rtol=1e-6):
                print(f"feature {i} and feature {j} are nearly identical")
                found = True
    if not found:
        print("No exact/near-exact duplicate columns found.")

    # Correlation sanity check
    corr = np.corrcoef(X_features, rowvar=False)
    np.fill_diagonal(corr, 0)
    strongest = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    print(
        "\nStrongest off-diagonal correlation:",
        strongest,
        corr[strongest],
    )

    return df, X_features


def _save_fa_noise_fraction_table(components, noise_variance, feature_names, path):
    components = np.asarray(components)
    noise_variance = np.asarray(noise_variance)
    n_features = components.shape[1]
    feature_names = _fit_feature_names(feature_names, n_features)

    if len(noise_variance) != n_features:
        raise ValueError(
            "FA components and noise variance describe different feature counts: "
            f"components={n_features}, noise_variance={len(noise_variance)}."
        )

    common_variance = np.sum(components ** 2, axis=0)
    total_variance = common_variance + noise_variance
    noise_fraction = noise_variance / np.maximum(total_variance, 1e-12)

    table = pd.DataFrame({
        "feature": feature_names,
        "common_variance": common_variance,
        "noise_variance": noise_variance,
        "noise_fraction": noise_fraction,
    })

    table = table.sort_values("noise_fraction", ascending=False)
    table.to_csv(path, index=False)
    return path

def _transform_until_step(estimator, step_name, x, include_step):
    step_names = [name for name, _ in estimator.steps]
    stop_index = step_names.index(step_name)
    if include_step:
        stop_index += 1
    return Pipeline(estimator.steps[:stop_index]).transform(x)


def _wavelet_feature_names(estimator, n_features: int, core_cols):
    """
    Mirrors dimensionality_visualization._wavelet_feature_names.
    """
    wave = estimator.named_steps.get("wave")
    if wave is None:
        return [f"feature_{i}" for i in range(n_features)]

    if hasattr(wave, "feature_names") and hasattr(wave, "n_samples_in_"):
        names = wave.feature_names(core_cols, wave.n_samples_in_)
    else:
        names = []
        for channel_name in core_cols:
            for detail_level in wave.detail_levels():
                names.append(f"{channel_name} cD{detail_level} energy")

    if len(names) != n_features:
        print(
            "[WARN] Generated wavelet feature names do not match feature count. "
            f"Generated={len(names)}, actual={n_features}. Falling back to generic names."
        )
        return [f"feature_{i}" for i in range(n_features)]

    return names


def _get_reducer_attr(reducer, attr_name):
    if hasattr(reducer, attr_name):
        return getattr(reducer, attr_name)
    if hasattr(reducer, "model_") and hasattr(reducer.model_, attr_name):
        return getattr(reducer.model_, attr_name)
    return None


def _reducer_feature_count(components, noise_variance):
    if components is not None:
        return np.asarray(components).shape[1]
    if noise_variance is not None:
        return len(noise_variance)
    return None


def _reducer_feature_names(reducer, feature_names, n_features):
    if n_features is None or len(feature_names) == n_features:
        return feature_names

    n_energy_details = getattr(reducer, "n_energy_details", -1)
    if (
        n_energy_details > 0
        and len(feature_names) - n_energy_details == n_features
    ):
        return feature_names[n_energy_details:]

    print(
        "[WARN] Reducer feature count does not match wavelet feature names. "
        f"Generated={len(feature_names)}, reducer={n_features}. "
        "Falling back to generic reducer feature names."
    )
    return [f"reducer_feature_{i}" for i in range(n_features)]


def _fit_feature_names(feature_names, n_features):
    if len(feature_names) == n_features:
        return feature_names
    return [f"feature_{i}" for i in range(n_features)]


def _plot_latent_2d(x_latent, y, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in np.unique(y):
        mask = y == label
        ax.scatter(
            x_latent[mask, 0],
            x_latent[mask, 1],
            s=24,
            alpha=0.75,
            label=_label_text(label),
        )
        ax.scatter(
            np.mean(x_latent[mask, 0]),
            np.mean(x_latent[mask, 1]),
            s=120,
            marker="x",
            linewidths=2,
            color="black",
        )

    ax.set_title("Top 2 latent dimensions")
    ax.set_xlabel("latent dimension 1")
    ax.set_ylabel("latent dimension 2")
    ax.legend(title="label")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_latent_3d(x_latent, y, path):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    view_elev, view_azim = _best_3d_view_angles(x_latent[:, :3])

    for label in np.unique(y):
        mask = y == label
        ax.scatter(
            x_latent[mask, 0],
            x_latent[mask, 1],
            x_latent[mask, 2],
            s=22,
            alpha=0.75,
            label=_label_text(label),
        )

    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_box_aspect(_axis_span(x_latent[:, :3]))
    ax.set_title(f"Top 3 latent dimensions, auto view elev={view_elev}, azim={view_azim}")
    ax.set_xlabel("latent dimension 1")
    ax.set_ylabel("latent dimension 2")
    ax.set_zlabel("latent dimension 3")
    ax.legend(title="label")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _best_3d_view_angles(points):
    centered = points - np.mean(points, axis=0)
    best_score = (-np.inf, -np.inf)
    best_angles = (30, -60)

    for elev in range(-80, 81, 5):
        elev_rad = np.deg2rad(elev)
        cos_elev = np.cos(elev_rad)
        sin_elev = np.sin(elev_rad)

        for azim in range(0, 360, 5):
            azim_rad = np.deg2rad(azim)
            normal = np.array([
                cos_elev * np.cos(azim_rad),
                cos_elev * np.sin(azim_rad),
                sin_elev,
            ])
            projected = centered @ _view_basis(normal)
            covariance = np.cov(projected, rowvar=False)
            area = max(0.0, np.linalg.det(covariance))
            spread = np.trace(covariance)

            if (area, spread) > best_score:
                best_score = (area, spread)
                best_angles = (elev, azim)

    return best_angles


def _view_basis(normal):
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, reference)) > 0.95:
        reference = np.array([0.0, 1.0, 0.0])

    horizontal = np.cross(reference, normal)
    horizontal = horizontal / np.linalg.norm(horizontal)
    vertical = np.cross(normal, horizontal)
    return np.column_stack([horizontal, vertical])


def _axis_span(points):
    span = np.ptp(points, axis=0)
    span[span == 0] = 1.0
    return span

def _active_core_cols(estimator, core_cols):
    dropper = estimator.named_steps.get("drop_channels")

    if dropper is None or not hasattr(dropper, "keep_channels_"):
        return core_cols

    return [core_cols[i] for i in dropper.keep_channels_]


def _plot_variance_breakdown(reducer, x_latent, path):
    noise_variance = _get_reducer_attr(reducer, "noise_variance_")
    components = _get_reducer_attr(reducer, "components_")
    explained_variance_ratio = _get_reducer_attr(reducer, "explained_variance_ratio_")

    if explained_variance_ratio is not None:
        values = list(explained_variance_ratio)
        labels = [f"dim {i + 1}" for i in range(len(values))]
        remaining = max(0.0, 1.0 - float(np.sum(explained_variance_ratio)))
        if remaining > 0:
            values.append(remaining)
            labels.append("not captured")
        title = "PCA explained variance share"
        y_label = "share of total variance"
    elif noise_variance is not None and components is not None:
        common_variance = np.sum(components * components, axis=1)
        noise_total = np.sum(noise_variance)
        total_variance = np.sum(common_variance) + noise_total
        values = list(common_variance / total_variance) + [noise_total / total_variance]
        labels = [f"factor {i + 1}" for i in range(len(common_variance))] + ["feature noise"]
        title = "FA modeled variance share"
        y_label = "share of modeled covariance"
    else:
        latent_variance = np.var(x_latent, axis=0, ddof=1)
        values = list(latent_variance / np.sum(latent_variance))
        labels = [f"dim {i + 1}" for i in range(len(values))]
        title = "Latent output variance share"
        y_label = "share of latent variance"

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(values))
    ax.bar(positions, values)
    ax.plot(positions, np.cumsum(values), color="black", marker="o", linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, min(1.05, max(values) * 1.25 if values else 1.0))
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_latent_variance_by_label(x_latent, y, path):
    n_dims = min(3, x_latent.shape[1])
    labels = np.unique(y)
    values = np.zeros((len(labels), n_dims))

    for i, label in enumerate(labels):
        mask = y == label
        values[i, :] = np.var(x_latent[mask, :n_dims], axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / n_dims
    positions = np.arange(len(labels))
    for dim in range(n_dims):
        ax.bar(
            positions + dim * width,
            values[:, dim],
            width=width,
            label=f"dim {dim + 1}",
        )

    ax.set_title("Latent variance within each label")
    ax.set_ylabel("variance")
    ax.set_xticks(positions + width * (n_dims - 1) / 2)
    ax.set_xticklabels([_label_text(label) for label in labels])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_loadings_heatmap(components, feature_names, path):
    feature_names = _fit_feature_names(feature_names, components.shape[1])
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(components, aspect="auto", cmap="coolwarm")
    ax.set_title("Reducer loadings by feature")
    ax.set_xlabel("wavelet feature")
    ax.set_ylabel("latent dimension")
    ax.set_yticks(np.arange(components.shape[0]))
    ax.set_yticklabels([f"dim {i + 1}" for i in range(components.shape[0])])
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=70, ha="right")
    fig.colorbar(image, ax=ax, label="loading")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_noise_variance(noise_variance, estimator, feature_names, path):
    feature_names = _fit_feature_names(feature_names, len(noise_variance))
    wave = estimator.named_steps.get("wave")
    active_cols = _active_core_cols(estimator, core_cols)

    detail_levels = wave.detail_levels() if wave is not None else []
    if wave is not None and len(noise_variance) == len(active_cols) * len(detail_levels):
        values = np.asarray(noise_variance).reshape(len(active_cols), len(detail_levels))
        detail_labels = [f"cD{level}" for level in detail_levels]

        fig, ax = plt.subplots(figsize=(8, 5))
        image = ax.imshow(values, aspect="auto", cmap="viridis")
        ax.set_title("FA feature noise variance")
        ax.set_xlabel("wavelet detail")
        ax.set_ylabel("channel")
        ax.set_xticks(np.arange(len(detail_levels)))
        ax.set_xticklabels(detail_labels)
        ax.set_yticks(np.arange(len(active_cols)))
        ax.set_yticklabels(active_cols)
        fig.colorbar(image, ax=ax, label="noise variance")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        positions = np.arange(len(noise_variance))
        ax.bar(positions, noise_variance)
        ax.set_title("FA feature noise variance")
        ax.set_ylabel("noise variance")
        ax.set_xticks(positions)
        #ax.set_xticklabels(feature_names, rotation=70, ha="right")
        ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_classifier_confidence(estimator, x, y, path):
    probabilities = estimator.predict_proba(x)
    confidence = np.max(probabilities, axis=1)
    labels = np.unique(y)
    groups = [confidence[y == label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(groups, tick_labels=[_label_text(label) for label in labels])
    ax.set_title("Classifier confidence by true label")
    ax.set_ylabel("max predicted probability")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _label_text(label):
    return LABEL_MAPPING.get(label, str(label))

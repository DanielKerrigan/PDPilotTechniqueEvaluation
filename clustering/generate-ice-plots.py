"Generate synthetic ICE plots."

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import shuffle


def fit_pca(real_ice_lines):
    """Fit PCA on ICE lines."""
    pca = PCA(n_components=8, svd_solver="full", whiten=True, random_state=1)
    pca.fit(real_ice_lines)
    return pca


def generate_synthetic_ice_plot(
    cluster_sizes, between_deviation, within_deviation, pca, rng, seed
):
    """Generate synthetic ICE plot."""

    start = rng.normal(loc=0, scale=1, size=pca.n_components_)

    clusters = []

    # repeat 0 cluster_sizes[0] times, 1 cluster_sizes[1] times, etc.
    labels = np.repeat(np.arange(len(cluster_sizes)), cluster_sizes)

    for cluster_size in cluster_sizes:
        center = start + rng.normal(
            loc=0, scale=between_deviation, size=pca.n_components_
        )
        cluster_lines = center + rng.normal(
            loc=0, scale=within_deviation, size=(cluster_size, pca.n_components_)
        )
        clusters.append(pca.inverse_transform(cluster_lines))

    lines = np.concatenate(clusters)

    shuffled_lines, shuffled_labels = shuffle(lines, labels, random_state=seed)

    return shuffled_lines.tolist(), shuffled_labels.tolist()


def main(input_path, output_path, num_plots):
    "Generate synthetic ICE plots."

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    real_ice_lines = np.array(json.loads(input_path.read_bytes()))
    pca = fit_pca(real_ice_lines)

    print(f"Explained variance: {pca.explained_variance_ratio_.sum()}")

    rng = np.random.default_rng(seed=1)

    synthetic_ice_plots = []

    for i in range(num_plots):
        num_clusters = rng.choice([2, 3, 4, 5])
        cluster_sizes = rng.integers(low=100, high=500, size=num_clusters)

        lines, labels = generate_synthetic_ice_plot(
            cluster_sizes=cluster_sizes,
            between_deviation=0.7,
            within_deviation=0.2,
            pca=pca,
            rng=rng,
            seed=i,
        )

        synthetic_ice_plots.append({"lines": lines, "labels": labels})

    output_path.write_text(json.dumps(synthetic_ice_plots), encoding="UTF-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic ICE plots.")
    parser.add_argument(
        "-i",
        "--input",
        default="./results/real-ice.json",
        help="path to ICE lines",
    )
    parser.add_argument(
        "-o", "--output", default="./results/synthetic-ice.json", help="output path"
    )
    parser.add_argument(
        "-n",
        "--num_plots",
        default=1000,
        type=int,
        help="number of synthetic ICE plots to create",
    )
    args = parser.parse_args()

    main(args.input, args.output, args.num_plots)

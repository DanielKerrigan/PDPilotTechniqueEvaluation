"Generate synthetic ICE plots."

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
from numpy.random import MT19937, RandomState
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances


def cluster_lines(
    ice_lines,
    cluster_preprocessing,
    random_state,
):
    """Cluster ICE lines with the given preprocessing."""

    if cluster_preprocessing == "diff":
        lines_to_cluster = np.diff(ice_lines)
    elif cluster_preprocessing == "cice":
        centered_ice_lines = ice_lines - ice_lines[:, 0].reshape(-1, 1)
        lines_to_cluster = centered_ice_lines
    elif cluster_preprocessing == "mean":
        lines_to_cluster = ice_lines - ice_lines.mean(axis=1, keepdims=True)

    distances = euclidean_distances(lines_to_cluster, lines_to_cluster)

    best_score = -math.inf
    best_labels = []

    for n_clusters in range(2, 6):
        cluster_model = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=5,
            max_iter=300,
            algorithm="lloyd",
            random_state=random_state,
        )

        with warnings.catch_warnings():
            # Supress ConvergenceWarning warning. Log it below.
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            cluster_model.fit(lines_to_cluster)

        labels = cluster_model.labels_

        if len(np.unique(labels)) < n_clusters:
            if not best_labels:
                best_labels = labels
            break

        score = silhouette_score(distances, cluster_model.labels_, metric="precomputed")

        if score > best_score:
            best_score = score
            best_labels = labels

    return best_labels


def main(input_path, output_dir, method_index):
    "Cluster ICE plots with the given preprocessing."

    methods = ["diff", "cice", "mean"]
    cluster_preprocessing = methods[method_index]

    input_path = Path(input_path).resolve()
    output_path = Path(output_dir).resolve() / f"{cluster_preprocessing}.json"

    ice_plots = json.loads(input_path.read_bytes())

    random_state = RandomState(MT19937(1))

    results = []

    for ice_plot in ice_plots:
        ice_lines = np.array(ice_plot["lines"])

        pred_labels = cluster_lines(
            ice_lines=ice_lines,
            cluster_preprocessing=cluster_preprocessing,
            random_state=random_state,
        )

        results.append(pred_labels.tolist())

    output_path.write_text(json.dumps(results), encoding="UTF-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster ICE plots.")
    parser.add_argument(
        "-i",
        "--input",
        default="./scratch/synthetic-ice.json",
        help="path to ICE lines",
    )
    parser.add_argument(
        "-o", "--output", default="./scratch/clusters", help="output directory"
    )
    parser.add_argument(
        "-m", "--method", choices=[0, 1, 2], type=int, help="preprocessing method"
    )
    args = parser.parse_args()

    main(args.input, args.output, args.method)

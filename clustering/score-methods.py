"Evaluate clusters from different methods."

import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def main(ice_path, cluster_dir, output_path):
    "Evaluate clusters from different methods."

    ice_path = Path(ice_path).resolve()
    cluster_dir = Path(cluster_dir).resolve()
    output_path = Path(output_path).resolve()

    all_true_labels = [plot["labels"] for plot in json.loads(ice_path.read_bytes())]

    cluster_paths = cluster_dir.glob("*.json")
    method_to_clusters = {
        path.stem: json.loads(path.read_bytes()) for path in cluster_paths
    }

    num_plots = len(all_true_labels)
    methods = list(method_to_clusters.keys())

    results = []

    for i in range(num_plots):
        true_labels = all_true_labels[i]

        for method in methods:
            pred_labels = method_to_clusters[method][i]

            result = {
                "plot": i,
                "method": method,
                "adjusted_rand": adjusted_rand_score(true_labels, pred_labels),
            }

            results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate clusters from different methods."
    )
    parser.add_argument(
        "-i",
        "--ice_path",
        default="./scratch/synthetic-ice.json",
        help="path to ICE lines",
    )
    parser.add_argument(
        "-c",
        "--cluster_dir",
        default="./scratch/clusters",
        help="clustering results directory",
    )
    parser.add_argument(
        "-o", "--output_path", default="./scratch/results.csv", help="output path"
    )
    args = parser.parse_args()

    main(args.ice_path, args.cluster_dir, args.output_path)

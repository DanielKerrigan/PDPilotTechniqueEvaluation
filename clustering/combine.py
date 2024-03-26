"Gather all clusters into one file."

import argparse
from itertools import permutations
from pathlib import Path
import json
import numpy as np


def check_equals(data, feature_data, key):
    """check that the values from the two different calls to PDPilot are equal"""
    for prop in [
        "x_values",
        "pdp",
        "centered_pdp",
        "ice",
        "centered_ice_min",
        "centered_ice_max",
    ]:
        assert np.array_equal(data[prop], feature_data[key][prop]), f"{key}, {prop}"


def get_data(datasets_group, input_dir):
    """combine the two clustering results from the different methods together"""
    feature_data = {}

    for ds in datasets_group:
        dataset = ds["name"]

        for cluster_preprocessing in ["diff", "center"]:
            input_path = input_dir / f"{dataset}-{cluster_preprocessing}.json"

            if not input_path.exists():
                print(f"No file for {dataset} {cluster_preprocessing}")
                continue

            pd_data = json.loads(Path(input_path).read_bytes())

            for owp in pd_data["one_way_pds"]:
                key = f"{dataset}_{owp['x_feature']}"
                num_clusters = owp["ice"]["num_clusters"]

                if num_clusters == 1:
                    print(
                        f"No clusters for {dataset} {owp['x_feature']} {cluster_preprocessing}"
                    )
                    continue

                clusters = owp["ice"]["clusterings"][str(num_clusters)]["clusters"]
                feature_info = pd_data["feature_info"][owp["x_feature"]]

                data = {
                    "id": len(feature_data),
                    "dataset": dataset,
                    "feature": owp["x_feature"],
                    "kind": feature_info["kind"],
                    "subkind": feature_info["subkind"],
                    "x_values": owp["x_values"],
                    "pdp": owp["mean_predictions"],
                    "centered_pdp": owp["ice"]["centered_pdp"],
                    "ice": pd_data["feature_to_ice_lines"][owp["x_feature"]],
                    "centered_ice_min": owp["ice"]["centered_ice_min"],
                    "centered_ice_max": owp["ice"]["centered_ice_max"],
                    "clusters": {cluster_preprocessing: clusters},
                }

                if key not in feature_data:
                    feature_data[key] = data
                else:
                    # if the key is already in feature_data, then the only thing
                    # that needs to be added is the clusters for this method.
                    # everything else is identical.
                    feature_data[key]["clusters"][cluster_preprocessing] = clusters
                    check_equals(data, feature_data, key)

    return list(feature_data.values())


def calculate_cluster_similarity(data):
    """For each feature, align the clusters from the different methods and calculate
    how many instances are classified into the same cluster."""

    for feature in data:
        if len(feature["clusters"]["diff"]) > len(feature["clusters"]["center"]):
            more_clusters = feature["clusters"]["diff"]
            less_clusters = feature["clusters"]["center"]
        else:
            more_clusters = feature["clusters"]["center"]
            less_clusters = feature["clusters"]["diff"]

        more_ids = [c["id"] for c in more_clusters]
        less_ids = [c["id"] for c in less_clusters]

        max_intersection = float("-inf")
        # best order of the clusters for the method with the larger number of clusters
        best_more_order = None

        # try all possible orders of the clusters
        for m_ids in permutations(more_ids):
            # get the number of instances that were put into the same clusters
            # by the different methods
            intersection = sum(
                len(
                    set(more_clusters[mi]["indices"])
                    & set(less_clusters[li]["indices"])
                )
                for mi, li in zip(m_ids, less_ids)
            )

            if intersection > max_intersection:
                max_intersection = intersection
                best_more_order = m_ids

        feature["intersection"] = max_intersection
        n_instances = len(feature["ice"])
        feature["percent_overlap"] = max_intersection / n_instances

        assert len(set(best_more_order)) == len(best_more_order) == len(more_clusters)

        for cluster in more_clusters:
            cluster["aligned_id"] = best_more_order.index(cluster["id"])

        for cluster in less_clusters:
            cluster["aligned_id"] = cluster["id"]


def main(dataset_group, datasets_file, input_dir, output_path):
    "Gather all clusters into one file."

    datasets = json.loads(Path(datasets_file).read_bytes())

    input_dir = Path(input_dir).resolve()
    output_path = Path(output_path).resolve()

    data = get_data(datasets[dataset_group], input_dir)

    calculate_cluster_similarity(data)

    output_path.write_text(json.dumps(data), encoding="UTF-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather all clusters into one file.")
    parser.add_argument(
        "-f",
        "--file",
        default="../data/datasets.json",
        help="path to datasets.json file",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="run on debug datasets"
    )
    parser.add_argument(
        "-p",
        "--path",
        help="directory containing clustering results",
    )
    parser.add_argument("-o", "--output", default="clusters.json", help="output path")
    args = parser.parse_args()

    DATASET_GROUP = "debug" if args.debug else "actual"

    main(DATASET_GROUP, args.file, args.path, args.output)

"""Combine all PDPs into one file."""

import argparse
import json


def read_json(path):
    """Read JSON file."""
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json(path, data):
    """Write to JSON file."""
    with open(path, "w", encoding="utf-8") as fp:
        return json.dump(data, fp)


def get_pdps(datasets_and_models):
    """Get all PDPs in one list."""
    pdps = []

    for info in datasets_and_models:
        pd_data = read_json(f'../data/{info["pd_data"]}')
        feature_info = pd_data["feature_info"]
        for owp in pd_data["one_way_pds"]:
            # don't include binary features or flat PDPs
            if len(owp["x_values"]) > 2 and len(set(owp["mean_predictions"])) > 1:
                pdp = {
                    "x": owp["x_values"],
                    "y": owp["mean_predictions"],
                    "kind": feature_info[owp["x_feature"]]["kind"],
                }
                pdps.append(pdp)

    return pdps


def main(trial):
    """Main method for script."""

    if trial:
        input_file = "../data/trial_datasets_and_models.json"
        output_file = "trial_pdps.json"
    else:
        input_file = "../data/datasets_and_models.json"
        output_file = "pdps.json"

    datasets_and_models = read_json(input_file)
    pdps = get_pdps(datasets_and_models)
    write_json(output_file, pdps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine all PDPs into one file.",
    )
    parser.add_argument(
        "-t", "--trial", action="store_true", help="trial run with different datasets"
    )
    args = parser.parse_args()

    main(args.trial)

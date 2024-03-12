"""Combine all PDPs into one file."""

import argparse
import json
from pathlib import Path


def get_pdps(pdpilot_paths):
    """Get all PDPs in one list."""
    pdps = []

    for pd_path in pdpilot_paths:
        if not pd_path.exists():
            print(f"{pd_path} does not exist")
            continue

        pd_data = json.loads(pd_path.read_bytes())
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


def main(dataset_group, input_dir, output):
    """Main method for script."""

    input_path = Path(input_dir).resolve()
    output_path = Path(output).resolve()

    datasets = json.loads(Path("../data/datasets.json").read_bytes())

    pdpilot_paths = [input_path / f"{x['name']}.json" for x in datasets[dataset_group]]

    pdps = get_pdps(pdpilot_paths)
    output_path.write_text(json.dumps(pdps), encoding="UTF-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine all PDPs into one file.",
    )

    parser.add_argument(
        "-d", "--debug", action="store_true", help="run on debug datasets"
    )
    parser.add_argument("-p", "--pdpilot", default=".", help="pdpilot data directory")
    parser.add_argument("-o", "--output", default=".", help="output path")

    args = parser.parse_args()

    DATASET_GROUP = "debug" if args.debug else "actual"

    main(DATASET_GROUP, args.pdpilot, args.output)

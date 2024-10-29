"Gather all ICE lines into one file."

import argparse
import json
from pathlib import Path

import numpy as np


def main(input_dir, output_path, include_flat=True):
    "Gather all ICE lines into one file."

    input_paths = Path(input_dir).resolve().glob("*/pdpilot/*.json")
    output_path = Path(output_path).resolve()

    data = []

    feature_count = 0

    for input_path in input_paths:
        # load PDPilot data for this dataset
        pd_data = json.loads(input_path.read_bytes())

        # min and max values in the ICE plots across all features
        lo, hi = pd_data["ice_line_extent"]

        for feature, ice_lines in pd_data["feature_to_ice_lines"].items():
            # only get quantiative features with 20 points in their lines
            if (
                len(ice_lines[0]) == 20
                and pd_data["feature_info"][feature]["kind"] == "quantitative"
            ):
                feature_count += 1

                lines = np.array(ice_lines)

                # optionally filter out ICE lines that are flat
                if not include_flat:
                    not_flat = np.sum(np.diff(lines), axis=1) != 0
                    lines = lines[not_flat]

                # globally normalize all ICE lines in this dataset to be between 0 and 1
                normalized = (lines - lo) / (hi - lo)

                data.extend(normalized.tolist())

    print(f"saving {len(data)} lines across {feature_count} features.")
    output_path.write_text(json.dumps(data), encoding="UTF-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather all ICE lines into one file.")
    parser.add_argument(
        "-i",
        "--input",
        default="../data/results",
        help="path to datasets, models, and PDPilot data.",
    )
    parser.add_argument(
        "-o", "--output", default="./scratch/real-ice.json", help="output path"
    )
    args = parser.parse_args()

    main(args.input, args.output)

"Gather all ICE lines into one file."

import argparse
import json
from pathlib import Path

import numpy as np


def main(input_dir, output_path, exclude_flat):
    "Gather all ICE lines into one file."

    input_paths = Path(input_dir).resolve().glob("*/pdpilot/*.json")
    output_path = Path(output_path).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []

    feature_count = 0

    for input_path in input_paths:
        # load PDPilot data for this dataset
        pd_data = json.loads(input_path.read_bytes())

        # min and max values in the ICE plots across all features
        lo, hi = pd_data["ice_line_extent"]

        for feature, ice_lines in pd_data["feature_to_ice_lines"].items():
            # only get features with at least 20 points in their lines
            if len(ice_lines[0]) >= 20:
                # take the first 20 values in each line
                lines = np.array(ice_lines)[:, :20]

                # optionally filter out ICE lines that are flat
                if exclude_flat:
                    not_flat = ~np.isclose(np.ptp(lines, axis=1), 0)
                    lines = lines[not_flat]

                # if there are still lines after filtering
                if lines.shape[0] > 0:
                    feature_count += 1

                    # globally normalize all ICE lines in this dataset to be between 0 and 1
                    normalized = (lines - lo) / (hi - lo)

                    data.extend(normalized.tolist())
                else:
                    print(f"Flat: dataset={input_path.stem} feature={feature}")

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
        "-o", "--output", default="./results/real-ice.json", help="output path"
    )
    parser.add_argument("--exclude_flat", action="store_true")
    args = parser.parse_args()

    main(args.input, args.output, args.exclude_flat)

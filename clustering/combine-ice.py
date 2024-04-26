"Gather all ICE lines into one file."

import argparse
from pathlib import Path
import json
import numpy as np


def main(input_dir, output_path):
    "Gather all ICE lines into one file."

    input_paths = Path(input_dir).resolve().glob("*/pdpilot/*.json")
    output_path = Path(output_path).resolve()

    data = []

    for input_path in input_paths:
        pd_data = json.loads(input_path.read_bytes())

        lo, hi = pd_data["ice_line_extent"]

        for _, ice_lines in pd_data["feature_to_ice_lines"].items():
            if len(ice_lines[0]) == 20:
                lines = np.array(ice_lines)

                not_flat = np.sum(np.diff(lines), axis=1) != 0

                normalized = (lines[not_flat] - lo) / (hi - lo)

                data.extend(normalized.tolist())

    output_path.write_text(json.dumps(data), encoding="UTF-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather all ICE lines into one file.")
    parser.add_argument(
        "-i",
        "--input",
        default="../data/results",
        help="path to datasets, models, and PDPilot data.",
    )
    parser.add_argument("-o", "--output", default="ice.json", help="output path")
    args = parser.parse_args()

    main(args.input, args.output)

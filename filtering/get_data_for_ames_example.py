"""Extract the PDPs and shapes from the Ames, Iowa Housing Dataset."""

import argparse
from pathlib import Path
import json
from evaluate_shapes import get_shape


def main(input_path, output_path):
    """Get the one-way PDPs and shapes."""

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    pd_data = json.loads(input_path.read_bytes())

    pdp_shapes = [
        {
            "feature": owp["x_feature"],
            "pdp": owp["mean_predictions"],
            "x_values": owp["x_values"],
            # The PDPilot data for the Ames Housing dataset used
            # the previous default tolerance, 0.15. Here we update
            # the shape to use the new tolerance, 0.29.
            "shape": get_shape({"y": owp["mean_predictions"]}, 0.29),
        }
        for owp in pd_data["one_way_pds"]
        if owp["ordered"]
    ]

    output_path.write_text(json.dumps(pdp_shapes), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract the PDPs and shapes from the Ames, Iowa Housing Dataset.",
    )
    parser.add_argument(
        "-i", "--input", default="scratch/ames_pd.json", help="path to PDPilot data"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="scratch/ames_pdp_shapes.json",
        help="path to output file",
    )
    args = parser.parse_args()

    main(args.input, args.output)

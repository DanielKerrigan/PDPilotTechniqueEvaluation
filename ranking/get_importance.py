"""Get importance scores."""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np


def get_scores(all_datasets, datasets_group, input_path):
    """Get importance scores."""

    datasets = all_datasets[datasets_group]

    results = []

    for ds in datasets:
        pd_path = input_path / f"pdpilot/{ds['name']}.json"

        if not pd_path.exists():
            print(f"{pd_path} does not exist")
            continue

        model_path = input_path / f"models/{ds['name']}.txt"
        booster = lgb.Booster(model_file=model_path)

        pd_data = json.loads(pd_path.read_bytes())

        lgb_scores = dict(
            zip(
                booster.feature_name(),
                booster.feature_importance(importance_type="gain"),
            )
        )

        pds = pd_data["one_way_pds"]
        feature = [p["x_feature"] for p in pds]

        score_ice = np.array([p["deviation"] for p in pds])
        score_pdp = np.array([np.std(p["mean_predictions"]) for p in pds])
        score_lgb = np.array(
            [lgb_scores[p["x_feature"].replace(" ", "_")] for p in pds]
        )

        rank_ice = np.argsort(-score_ice, kind="stable")
        rank_pdp = np.argsort(-score_pdp, kind="stable")
        rank_lgb = np.argsort(-score_lgb, kind="stable")

        results.append(
            {
                "dataset": ds["name"],
                "feature": feature,
                "score_ice": score_ice.tolist(),
                "score_pdp": score_pdp.tolist(),
                "score_lgb": score_lgb.tolist(),
                "rank_ice": rank_ice.tolist(),
                "rank_pdp": rank_pdp.tolist(),
                "rank_lgb": rank_lgb.tolist(),
            }
        )

    return results


def main(dataset_group, input_dir, output):
    """Main method for script."""

    input_path = Path(input_dir).resolve() / dataset_group
    output_path = Path(output).resolve()

    datasets = json.loads(Path("../data/datasets.json").read_bytes())

    scores = get_scores(datasets, dataset_group, input_path)

    output_path.write_text(json.dumps(scores), encoding="UTF-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get importance scores.",
    )

    parser.add_argument(
        "-d", "--debug", action="store_true", help="run on debug datasets"
    )
    parser.add_argument(
        "-i",
        "--input",
        default="../data/results",
        help="results directory containing datasets, models, and PDPilot data",
    )
    parser.add_argument("-o", "--output", default="scores.json", help="output path")

    args = parser.parse_args()

    DATASET_GROUP = "debug" if args.debug else "actual"

    main(DATASET_GROUP, args.input, args.output)

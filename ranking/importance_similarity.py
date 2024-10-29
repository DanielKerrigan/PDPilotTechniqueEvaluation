"""Compare feature importance scores."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy import stats


def get_dataset_to_type(dataset_path):
    """get dictionary from name of dataset to whether it is synthetic or real"""
    datasets = json.loads(Path(dataset_path).read_bytes())
    return {ds["name"]: ds["type"] for group in datasets.values() for ds in group}


def compare_scores(input_paths, dataset_to_type):
    """Compute how similar the rankings are between the
    difference feature importance methods for each dataset."""

    dfs = []

    clean_names = {
        "score_ice": "ICE",
        "score_pdp": "PDP",
        "score_lgb": "GBM",
        "score_shap": "SHAP",
        "score_perm": "PERM",
    }

    for path in input_paths:
        dataset = path.stem

        df_scores = pd.read_csv(path)

        score_cols = ["score_ice", "score_pdp", "score_lgb", "score_shap", "score_perm"]
        combos = list(combinations(score_cols, 2))

        similarities = []

        for a, b in combos:
            kendall_tau = stats.kendalltau(df_scores[a], df_scores[b])
            weighted_kendall_tau = stats.weightedtau(df_scores[a], df_scores[b])

            info = {
                "dataset": dataset,
                "type": dataset_to_type[dataset],
                "combo": f"{clean_names[a]} vs. {clean_names[b]}",
                "first_method": clean_names[a],
                "second_method": clean_names[b],
                "kendall_tau": kendall_tau.statistic,
                "weighted_kendall_tau": weighted_kendall_tau.statistic,
            }

            similarities.append(info)

        dfs.append(pd.DataFrame(similarities))

    return pd.concat(dfs)


def main(input_dir, output):
    """Main method for script."""

    input_paths = Path(input_dir).resolve().glob("*/importances/*.csv")
    output_path = Path(output).resolve()

    dataset_to_type = get_dataset_to_type("../data/datasets.json")

    results = compare_scores(input_paths, dataset_to_type)

    results.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare feature importance scores.",
    )

    parser.add_argument(
        "-i",
        "--input",
        default="../data/results",
        help="results directory containing datasets, models, and PDPilot data",
    )
    parser.add_argument("-o", "--output", default="similarity.csv", help="output path")

    args = parser.parse_args()

    main(args.input, args.output)

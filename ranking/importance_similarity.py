"""Compare feature importance scores."""

import argparse
from pathlib import Path
from itertools import combinations

import pandas as pd
from scipy import stats


def compare_scores(input_paths):
    """Compute how similar the rankings are between the
    difference feature importance methods for each dataset."""

    dfs = []

    clean_names = {
        "score_ice": "ICE",
        "score_pdp": "PDP",
        "score_lgb": "LGB",
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

            info = {
                "dataset": dataset,
                "combo": f"{clean_names[a]} vs. {clean_names[b]}",
                "first_method": clean_names[a],
                "second_method": clean_names[b],
                "kendall_tau": kendall_tau.statistic,
            }

            similarities.append(info)

        dfs.append(pd.DataFrame(similarities))

    return pd.concat(dfs)


def main(input_dir, output):
    """Main method for script."""

    input_paths = Path(input_dir).resolve().glob("*/importances/*.json")
    output_path = Path(output).resolve()

    results = compare_scores(input_paths)

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

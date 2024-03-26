"""Cluster ICE lines using different preprocessing strategies."""

import argparse
from datetime import datetime
from pathlib import Path
import json
import lightgbm as lgb
import optuna
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
import numpy as np
from pdpilot import partial_dependence


def sample(df, n, objective):
    """Stratified sample for binary datasets, random for regression."""
    if objective == "binary":
        return train_test_split(df, train_size=n, random_state=1, stratify=[0, 1])[0]
    else:
        return df.sample(n, random_state=1)


def get_time():
    """Return the current time as a string."""
    return datetime.now().strftime("%I:%M:%S")


def load_dataset(datasets_file, dataset_group, index, datasets_dir):
    "Download the dataset."

    datasets = json.loads(Path(datasets_file).read_bytes())
    dataset_info = datasets[dataset_group][index]
    dataset = dataset_info["name"]
    objective = dataset_info["objective"]
    exclude_features = dataset_info["exclude_features"]
    nominal_features = dataset_info["nominal_features"]

    df_all = fetch_data(dataset_info["name"], local_cache_dir=datasets_dir.as_posix())

    df_reduced = (
        df_all if df_all.shape[0] <= 200_000 else sample(df_all, 200_000, objective)
    )

    df_X = df_reduced.drop(columns=["target"] + exclude_features)

    # drop columns that only have one unique value
    nunique = df_X.nunique()
    df_X.drop(columns=nunique[nunique == 1].index, inplace=True)

    features = list(df_X.columns)

    # convert float columns that contain only integers to integers
    for feature in features:
        as_int = df_X[feature].astype(int)
        if np.array_equal(df_X[feature], as_int):
            df_X[feature] = as_int

    df_pd = df_X if df_X.shape[0] <= 2000 else sample(df_X, 2000, objective)

    return dataset, df_pd, features, nominal_features


def main(dataset_group, index, datasets_file, input_dir, output_dir, jobs):
    "Cluster ICE lines using different preprocessing strategies."

    print(
        f"{get_time()}",
        f"\t{datasets_file=}",
        f"\t{dataset_group=}",
        f"\t{index=}",
        f"\t{input_dir=}",
        f"\t{output_dir=}",
        f"\t{jobs=}",
        sep="\n",
    )

    # make output directories

    print(f"\n{get_time()} Making output directories")
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    # load the dataset

    print(f"\n{get_time()} Loading dataset")
    dataset, df_pd, features, nominal_features = load_dataset(
        datasets_file, dataset_group, index, input_dir / "datasets"
    )

    # Check that the trained model was better than the baseline.
    # If the model is not better than the baseline, then there is
    # no PDPilot data.
    if not (input_dir / f"pdpilot/{dataset}.json").exists():
        print(f"\nNo PDPilot data for {dataset}.")
        return

    # calculate PDP and ICE plots

    booster = lgb.Booster(model_file=input_dir / f"models/{dataset}.txt")

    output_paths = []

    for cluster_preprocessing in ["diff", "center"]:
        print(f"\n{get_time()} Calculating clusters with {cluster_preprocessing=}")

        output_path = output_dir / f"{dataset}-{cluster_preprocessing}.json"
        output_paths.append(output_path)

        partial_dependence(
            df=df_pd,
            predict=booster.predict,
            features=features,
            nominal_features=nominal_features,
            resolution=20,
            compute_two_way_pdps=False,
            cluster_preprocessing=cluster_preprocessing,
            n_jobs=jobs,
            seed=1,
            output_path=output_path.as_posix(),
            logging_level="WARNING",
        )

    print("Plots computed for the following instances:")
    print(list(df_pd.index))


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Cluster ICE lines using different preprocessing strategies.",
    )
    parser.add_argument(
        "-f",
        "--file",
        default="../data/datasets.json",
        help="path to datasets.json file",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="run on debug datasets"
    )
    parser.add_argument("-i", "--index", help="dataset index", type=int)
    parser.add_argument(
        "-p",
        "--path",
        default=".",
        help="directory containing datasets, models, and pdpilot directories",
    )
    parser.add_argument("-o", "--output", default=".", help="output directory")
    parser.add_argument(
        "-j", "--jobs", help="number of jobs to use", default=1, type=int
    )
    args = parser.parse_args()

    DATASET_GROUP = "debug" if args.debug else "actual"

    main(DATASET_GROUP, args.index, args.file, args.path, args.output, args.jobs)

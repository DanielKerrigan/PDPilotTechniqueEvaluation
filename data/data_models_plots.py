"""Download datasets, train models, and calculate PDP and ICE plots."""

import argparse
from datetime import datetime
from pathlib import Path
import json

from gbm import nested_cross_validation_and_train
from feature_importance import get_feature_importance

import optuna
from pmlb import fetch_data
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import train_test_split
import numpy as np
from pdpilot import partial_dependence


def sample(df, n, objective):
    """Stratified sample for binary datasets, random for regression."""
    if objective == "binary":
        return train_test_split(
            df, df["target"], train_size=n, random_state=1, stratify=[0, 1]
        )[0]
    else:
        return df.sample(n, random_state=1)


def get_time():
    """Return the current time as a string."""
    return datetime.now().strftime("%I:%M:%S")


def get_baseline_score(y_true, objective):
    """Get the score for a baseline model.
    For binary classification, this means assigning probabilities to the classes
    based on how often they occur in the dataset.
    For regression, this means predicting the mean value."""

    y_pred = np.array([y_true.mean()] * y_true.shape[0])

    if objective == "binary":
        return log_loss(y_true, y_pred)
    else:
        return mean_squared_error(y_true, y_pred)


def create_dirs(output, dataset):
    "Create directories to write output files to."

    output_dir = Path(output).resolve()

    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f"{dataset}.txt"

    pdpilot_dir = output_dir / "pdpilot"
    pdpilot_dir.mkdir(exist_ok=True)
    pdpilot_path = pdpilot_dir / f"{dataset}.json"

    importances_dir = output_dir / "importances"
    importances_dir.mkdir(exist_ok=True)
    importances_path = importances_dir / f"{dataset}.csv"

    stuff_dir = output_dir / "stuff"
    stuff_dir.mkdir(exist_ok=True)
    stuff_path = stuff_dir / f"{dataset}.json"

    return datasets_dir, model_path, pdpilot_path, importances_path, stuff_path


def load_dataset(dataset_info, datasets_dir):
    "Download the dataset."

    dataset = dataset_info["name"]
    objective = dataset_info["objective"]
    exclude_features = dataset_info["exclude_features"]
    nominal_features = dataset_info["nominal_features"]

    df_all = fetch_data(dataset_info["name"], local_cache_dir=datasets_dir.as_posix())

    df_reduced = (
        df_all if df_all.shape[0] <= 200_000 else sample(df_all, 200_000, objective)
    )

    df_X = df_reduced.drop(columns=["target"] + exclude_features)
    y = df_reduced["target"].to_numpy()

    # drop columns that only have one unique value
    nunique = df_X.nunique()
    df_X.drop(columns=nunique[nunique == 1].index, inplace=True)

    features = list(df_X.columns)

    # convert float columns that contain only integers to integers
    for feature in features:
        as_int = df_X[feature].astype(int)
        if np.array_equal(df_X[feature], as_int):
            df_X[feature] = as_int

    X = df_X.to_numpy()

    return dataset, objective, df_X, X, y, features, nominal_features


def main(dataset_group, index, output, jobs):
    """Download the dataset, train the model, and calculate the PDP and ICE plots."""

    datasets = json.loads(Path("datasets.json").read_bytes())
    dataset_info = datasets[dataset_group][index]
    dataset = dataset_info["name"]

    print(f"{get_time()} {dataset=} {output=} {jobs=}")

    # make output directories

    print(f"\n{get_time()} Making output paths and directories")
    datasets_dir, model_path, pdpilot_path, importances_path, stuff_path = create_dirs(
        output, dataset
    )

    # load the dataset

    print(f"\n{get_time()} Loading dataset")
    dataset, objective, df_X, X, y, features, nominal_features = load_dataset(
        dataset_info, datasets_dir
    )

    # train the model

    print(f"\n{get_time()} Training the model on {dataset}")

    results, booster = nested_cross_validation_and_train(
        X, y, features, nominal_features, objective, jobs=jobs
    )
    booster.save_model(model_path)

    baseline_score = get_baseline_score(y, objective)

    print(
        f"\t{dataset=}",
        f"\t{objective=}",
        f"\tstd_score={results['std_score']}",
        f"\tmean_score={results['mean_score']}",
        f"\tbaseline_score={baseline_score}",
        sep="\n",
    )

    if results["mean_score"] >= baseline_score:
        print("The model is not better than the baseline")
        return

    # calculate PDP and ICE plots

    print(f"\n{get_time()} Calculating PDP and ICE plots")

    df_Xy = df_X.copy()
    df_Xy["target"] = y

    df_Xy_sample = df_Xy if df_Xy.shape[0] <= 2000 else sample(df_Xy, 2000, objective)

    df_pd = df_Xy_sample.drop(columns=["target"])
    y_pd = df_Xy_sample["target"].to_numpy()

    partial_dependence(
        df=df_pd,
        predict=booster.predict,
        features=features,
        nominal_features=nominal_features,
        resolution=20,
        n_jobs=jobs,
        seed=1,
        output_path=pdpilot_path.as_posix(),
        logging_level="WARNING",
    )

    # calculate feature importances

    print(f"\n{get_time()} Calculating feature importances")

    df_importance = get_feature_importance(
        booster, df_pd, y_pd, pdpilot_path, objective
    )
    df_importance.to_csv(importances_path, index=False)

    # save indices used to compute PDP and ICE plots

    print(f"\n{get_time()} Saving additional info")

    results["scores"] = results["scores"].to_json(orient="records")
    stuff = {"pdpilot_indices": df_pd.index.to_list(), "cv_results": results}
    stuff_path.write_text(json.dumps(stuff), encoding="UTF-8")


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Download datasets, train models, and calculate PDP and ICE plots.",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="run on debug datasets"
    )
    parser.add_argument("-i", "--index", help="dataset index", type=int)
    parser.add_argument("-o", "--output", default=".", help="output directory")
    parser.add_argument(
        "-j", "--jobs", help="number of jobs to use", default=1, type=int
    )
    args = parser.parse_args()

    DATASET_GROUP = "debug" if args.debug else "actual"

    main(DATASET_GROUP, args.index, args.output, args.jobs)

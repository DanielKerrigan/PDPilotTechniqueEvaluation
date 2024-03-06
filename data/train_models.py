"""Train LightGBM models on PMLB datasets."""

from datetime import datetime
from pmlb import fetch_data
from gbm import nested_cross_validation_and_train
from sklearn.metrics import mean_squared_error, log_loss
import pandas as pd
import numpy as np
from pdpilot import partial_dependence


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


def get_time():
    """Return the current time as a string."""
    return datetime.now().strftime("%I:%M:%S")


def train_models(datasets, objective):
    """Train a model on each dataset and calculate the PDP and ICE plots."""

    mean_scores = []
    std_scores = []
    baseline_scores = []
    model_paths = []
    pd_paths = []

    for i, dataset in enumerate(datasets):
        print(f"\n{i + 1}/{len(datasets)}: {dataset}")

        print(f"{get_time()} Downloading dataset")

        df_all = fetch_data(dataset, local_cache_dir="./datasets")
        df_X = df_all.drop(columns=["target"])
        features = list(df_X.columns)

        # convert float columns that contain only integers to integers
        for feature in features:
            as_int = df_X[feature].astype(int)
            if np.array_equal(df_X[feature], as_int):
                df_X[feature] = as_int

        X = df_X.to_numpy()
        y = df_all["target"].to_numpy()

        print(f"{get_time()} Training model")

        results, booster = nested_cross_validation_and_train(X, y, features, objective)

        model_path = f"models/{dataset}.txt"
        booster.save_model(model_path)

        baseline_score = get_baseline_score(y, objective)

        print(
            f"{objective}: score={results['mean_score']:.4f} baseline={baseline_score:.4f} std={results['std_score']:.4f}"
        )

        print(f"{get_time()} Calculating PDP and ICE plots")

        pd_path = f"pdpilot/{dataset}.json"

        df_pd = df_X if df_X.shape[0] <= 2000 else df_X.sample(2000, random_state=1)

        partial_dependence(
            df=df_pd,
            predict=booster.predict,
            features=features,
            resolution=20,
            n_jobs=8,
            seed=1,
            output_path=pd_path,
        )

        mean_scores.append(results["mean_score"])
        std_scores.append(results["std_score"])
        model_paths.append(model_path)
        pd_paths.append(pd_path)
        baseline_scores.append(baseline_score)

    return pd.DataFrame(
        {
            "dataset": datasets,
            "objective": objective,
            "model": model_paths,
            "pd_data": pd_paths,
            "mean_score": mean_scores,
            "std_score": std_scores,
            "baseline": baseline_scores,
        }
    )

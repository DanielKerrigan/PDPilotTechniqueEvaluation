"""Training and evaluating LightGBM models."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from optuna.integration.lightgbm import LightGBMTunerCV
from sklearn.model_selection import KFold, StratifiedKFold


def train(
    X,
    y,
    features,
    nominal_features,
    objective,
    splitter,
    early_stopping_rounds=10,
    seed=1,
    jobs=1,
):
    """Train a LightGBM model with parameters chosen by cross-validation."""

    train_set = lgb.Dataset(X, label=y, params={"verbose": -1})

    params = {
        "objective": objective,
        "metric": objective,
        "verbose": -1,
        "deterministic": True,
        "force_col_wise": True,
        "seed": seed,
        "num_threads": jobs,
    }

    tuner = LightGBMTunerCV(
        params,
        train_set,
        feature_name=features,
        categorical_feature=nominal_features,
        folds=splitter,
        num_boost_round=10000,
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        return_cvbooster=True,
        optuna_seed=seed + 1,
        show_progress_bar=False,
    )

    tuner.run()

    best_params = tuner.best_params

    cv_booster = tuner.get_best_booster()
    best_iteration = cv_booster.best_iteration

    booster = lgb.train(
        params=best_params,
        train_set=train_set,
        feature_name=features,
        categorical_feature=nominal_features,
        num_boost_round=best_iteration,
    )

    return booster


def nested_cross_validation(
    X,
    y,
    features,
    nominal_features,
    objective,
    outer_splitter,
    inner_splitter,
    seed=1,
    jobs=1,
):
    """Performs nested cross validation."""

    # tracking outer cross-validation scores and best parameters
    scores = []
    params = []
    folds = []

    for i, (train_index, eval_index) in enumerate(outer_splitter.split(X, y)):
        X_train, y_train = X[train_index], y[train_index]
        X_eval, y_eval = X[eval_index], y[eval_index]

        booster = train(
            X=X_train,
            y=y_train,
            features=features,
            nominal_features=nominal_features,
            objective=objective,
            splitter=inner_splitter,
            seed=seed + i,
            jobs=jobs,
        )

        y_pred = booster.predict(X_eval)

        score = np.sqrt(np.mean((y_pred - y_eval) ** 2))

        scores.append(score)
        params.append(booster.params)
        folds.append(i + 1)

    df_scores = pd.DataFrame({"fold": scores, "score": scores})

    return {
        "scores": df_scores,
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "params": params,
    }


def nested_cross_validation_and_train(
    X,
    y,
    features,
    nominal_features,
    objective,
    n_outer_splits=5,
    n_inner_splits=5,
    seed=1,
    jobs=1,
):
    """Perform nested cross-validation and then train the model on the whole dataset."""

    if objective == "binary":
        outer_splitter = StratifiedKFold(
            n_splits=n_outer_splits, random_state=seed, shuffle=True
        )
        inner_splitter = StratifiedKFold(
            n_splits=n_inner_splits, random_state=seed, shuffle=True
        )
    elif objective == "regression":
        outer_splitter = KFold(n_splits=n_outer_splits, random_state=seed, shuffle=True)
        inner_splitter = KFold(n_splits=n_inner_splits, random_state=seed, shuffle=True)

    results = nested_cross_validation(
        X=X,
        y=y,
        features=features,
        nominal_features=nominal_features,
        objective=objective,
        outer_splitter=outer_splitter,
        inner_splitter=inner_splitter,
        seed=seed,
        jobs=jobs,
    )

    booster = train(
        X=X,
        y=y,
        features=features,
        nominal_features=nominal_features,
        objective=objective,
        splitter=inner_splitter,
        seed=seed,
        jobs=jobs,
    )

    return results, booster

from gbm import score
import json
import shap
import numpy as np
import pandas as pd


def get_shap_importance(booster, df):
    """SHAP feature importance"""
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer(df)
    return dict(zip(df.columns, np.abs(shap_values.values).mean(axis=0)))


def get_permutation_importance(booster, df_original, y, objective, trials=10):
    """Calculate permutation feature importance."""

    importances = {}

    rng = np.random.default_rng(seed=1)

    # don't mutate the dataframe
    df_mod = df_original.copy()

    baseline_score = score(y, booster.predict(df_original), objective)

    for feature in df_original.columns:
        differences = []

        for _ in range(trials):
            df_mod[feature] = rng.permutation(df_original[feature].to_numpy())
            permuted_score = score(y, booster.predict(df_mod), objective)
            # lower score is better, so we put the permuted score first
            # so that the importances are positive values
            score_diff = permuted_score - baseline_score
            differences.append(score_diff)

        importances[feature] = np.mean(differences)

        df_mod[feature] = df_original[feature]

    return importances


def get_feature_importance(booster, df, y, pd_path, objective):
    """Calculate feature importance"""
    pd_data = json.loads(pd_path.read_bytes())

    lgb_scores = dict(
        zip(
            booster.feature_name(),
            booster.feature_importance(importance_type="gain"),
        )
    )

    permutation_scores = get_permutation_importance(booster, df, y, objective)

    shap_importances = get_shap_importance(booster, df)

    pds = pd_data["one_way_pds"]
    feature = [p["x_feature"] for p in pds]

    score_ice = np.array([p["deviation"] for p in pds])
    score_pdp = np.array([np.std(p["mean_predictions"]) for p in pds])
    score_lgb = np.array([lgb_scores[f.replace(" ", "_")] for f in feature])
    score_perm = np.array([permutation_scores[f] for f in feature])
    score_shap = np.array([shap_importances[f] for f in feature])

    results = pd.DataFrame(
        {
            "feature": feature,
            "score_ice": score_ice.tolist(),
            "score_pdp": score_pdp.tolist(),
            "score_lgb": score_lgb.tolist(),
            "score_perm": score_perm.tolist(),
            "score_shap": score_shap.tolist(),
        }
    )

    rank_cols = [
        "rank_ice",
        "rank_pdp",
        "rank_lgb",
        "rank_perm",
        "rank_shap",
    ]
    score_cols = [
        "score_ice",
        "score_pdp",
        "score_lgb",
        "score_perm",
        "score_shap",
    ]

    results[rank_cols] = results[score_cols].rank(method="first", ascending=False)

    return results

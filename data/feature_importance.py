import json
import shap
import numpy as np
import pandas as pd


def get_shap_importance(booster, df):
    """SHAP feature importance"""
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer(df)
    return dict(zip(df.columns, np.abs(shap_values.values).mean(axis=0)))


def get_feature_importance(booster, df_all, df_pdp, pd_path):
    """Calculate feature importance"""
    pd_data = json.loads(pd_path.read_bytes())

    lgb_scores = dict(
        zip(
            booster.feature_name(),
            booster.feature_importance(importance_type="gain"),
        )
    )

    shap_all = get_shap_importance(booster, df_all)
    shap_subset = get_shap_importance(booster, df_pdp)

    pds = pd_data["one_way_pds"]
    feature = [p["x_feature"] for p in pds]

    score_ice = np.array([p["deviation"] for p in pds])
    score_pdp = np.array([np.std(p["mean_predictions"]) for p in pds])
    score_lgb = np.array([lgb_scores[f.replace(" ", "_")] for f in feature])
    score_shap_all = np.array([shap_all[f] for f in feature])
    score_shap_subset = np.array([shap_subset[f] for f in feature])

    results = pd.DataFrame(
        {
            "feature": feature,
            "score_ice": score_ice.tolist(),
            "score_pdp": score_pdp.tolist(),
            "score_lgb": score_lgb.tolist(),
            "score_shap_all": score_shap_all.tolist(),
            "score_shap_subset": score_shap_subset.tolist(),
        }
    )

    rank_cols = [
        "rank_ice",
        "rank_pdp",
        "rank_lgb",
        "rank_shap_all",
        "rank_shap_subset",
    ]
    score_cols = [
        "score_ice",
        "score_pdp",
        "score_lgb",
        "score_shap_all",
        "score_shap_subset",
    ]

    results[rank_cols] = results[score_cols].rank(method="first", ascending=False)

    return results

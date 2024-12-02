from collections import defaultdict
from math import isclose, sqrt

import altair as alt
import numpy as np
import pandas as pd

# DISTANCE METRICS


def get_pdpilot_cluster_dist(feature_val):
    """
    This code calculates PDPilot's cluster difference metric
    when the input is in the format that VINE uses.
    """
    cluster_distance = np.float64(0)

    centered_pdp = feature_val["pdp_line"]

    for cluster in feature_val["clusters"]:
        centered_cluster_mean = cluster["line"]
        distance = np.mean(np.absolute(centered_cluster_mean - centered_pdp))
        cluster_distance += distance

    return cluster_distance


def _default_factory():
    """From VINE"""
    return float("inf")


def _get_dtw_distance(s1, s2, w=4):
    """From VINE"""
    w = max(w, abs(len(s1) - len(s2)))
    DTW = defaultdict(_default_factory)
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w)):
            DTW[(i, j)] = (s1[i] - s2[j]) ** 2 + min(
                DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)]
            )

    return sqrt(DTW[len(s1) - 1, len(s2) - 1])


def get_vine_cluster_dist(feature_val):
    """From VINE"""
    return np.mean(
        np.abs(
            [
                _get_dtw_distance(
                    np.array(feature_val["pdp_line"]), np.array(x["line"])
                )
                for x in feature_val["clusters"]
            ]
        )
    ) / np.max(np.abs(np.array(feature_val["pdp_line"])))


# DATA PROCESSING


def create_clusters(x_values, cluster_center_functions, centering):
    """
    Create synthetic ICE plot clusters in the format used by VINE.
    Each cluster center is defined as a function of the x values.
    """
    clusters = [{"line": func(x_values)} for func in cluster_center_functions]

    if centering == "mean":
        for c in clusters:
            c["line"] = c["line"] - c["line"].mean()
    elif centering == "0":
        for c in clusters:
            c["line"] = c["line"] - c["line"][0]
    elif centering != "none":
        raise ValueError(f"Unknown centering {centering}")

    pdp = np.array([c["line"] for c in clusters]).mean(axis=0)

    return {
        "x_values": x_values,
        "pdp_line": pdp,
        "clusters": clusters,
    }


def get_scores_for_method(
    method, min_n, max_n, min_x, max_x, cluster_center_functions, relative_n
):
    n_points = []
    scores = []

    base_score_check = -1.0

    for n in range(min_n, max_n + 1):
        x_values = np.linspace(min_x, max_x, n)

        if method == "PDPilot":
            feature_val = create_clusters(
                x_values,
                cluster_center_functions=cluster_center_functions,
                centering="0",
            )
            score = get_pdpilot_cluster_dist(feature_val)
        elif method == "VINE":
            feature_val = create_clusters(
                x_values,
                cluster_center_functions=cluster_center_functions,
                centering="mean",
            )
            score = get_vine_cluster_dist(feature_val)
        else:
            raise ValueError(f"Unknown method {method}")

        n_points.append(n)
        scores.append(score)

        if n == relative_n:
            base_score_check = score

    df = pd.DataFrame({"n_points": n_points, "score": scores, "method": method})

    base_score = df[df["n_points"] == relative_n]["score"].to_numpy()[0]

    assert isclose(base_score, base_score_check), f"{base_score} != {base_score_check}"

    df["relative_score"] = df["score"] / base_score

    return df


def get_method_comparison_data(
    methods, min_n, max_n, min_x, max_x, cluster_center_functions, relative_n
):
    dfs = []

    for method in methods:
        df = get_scores_for_method(
            method=method,
            min_n=min_n,
            max_n=max_n,
            min_x=min_x,
            max_x=max_x,
            cluster_center_functions=cluster_center_functions,
            relative_n=relative_n,
        )

        dfs.append(df)

    return pd.concat(dfs)


# VISUALIZATION


def plot_clustered_ice(feature_val, y_label):
    """Plot a clustered ICE plot, like in PDPilot."""
    df_pdp = pd.DataFrame(
        {
            "x": feature_val["x_values"],
            "y": feature_val["pdp_line"],
        }
    )

    pdp = (
        alt.Chart(df_pdp)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("x").axis(grid=False),
            y=alt.Y("y").axis(grid=False).title(y_label),
            color=alt.value("black"),
        )
    )

    clusters = []

    for cluster in feature_val["clusters"]:
        df_cluster = pd.DataFrame({"x": feature_val["x_values"], "y": cluster["line"]})
        plot = (
            alt.Chart(df_cluster)
            .mark_line(point=True, strokeWidth=2)
            .encode(x="x", y="y", color=alt.value("rgb(198, 198, 198)"))
        )
        clusters.append(plot)

    return alt.layer(*clusters, pdp).properties(height=150)


def plot_example(x_values, cluster_center_functions, centerings):
    if not centerings:
        raise ValueError("centerings is empty")

    plots = []

    for c in centerings:
        feature = create_clusters(
            x_values=x_values,
            cluster_center_functions=cluster_center_functions,
            centering=c,
        )

        if c == "mean":
            y_label = "y (mean centered)"
        elif c == "0":
            y_label = "y (centered at 0)"
        elif c == "none":
            y_label = "y"
        else:
            raise ValueError(f"Unknown centering {c}")

        plots.append(plot_clustered_ice(feature, y_label=y_label))

    if len(centerings) == 1:
        return plots[0]
    else:
        return alt.hconcat(*plots)


def plot_relative_scores(df, relative_n):
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("n_points").title("Number of points in the PDP"),
            y=alt.Y("relative_score").title(
                f"Cluster difference score relative to {relative_n} points"
            ),
            color=alt.Color("method").title(None).legend(orient="top", symbolOpacity=1),
            opacity=alt.value(0.7),
        )
        .properties(height=200)
    )

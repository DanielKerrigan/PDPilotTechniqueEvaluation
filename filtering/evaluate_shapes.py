"""Determine the threshold that best aligns with the labels."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt


def read_json(path):
    """Read JSON file."""
    return json.loads(Path(path).read_bytes())


def get_shape(curve, t):
    """Get the shape of the curve for the given threshold."""

    y = np.array(curve["y"])
    diff = np.diff(y)
    pos = diff[diff > 0].sum()
    neg = np.abs(diff[diff < 0].sum())
    percent_pos = pos / (pos + neg) if pos + neg != 0 else 0.5

    if percent_pos >= (0.5 + t):
        return "increasing"
    elif percent_pos <= (0.5 - t):
        return "decreasing"
    else:
        return "mixed"


def calculate_accuracy(labels_a, labels_b):
    """Get accuracy between two sets of labels"""

    num_curves = len(labels_a)

    assert num_curves == len(labels_b)

    correct = 0

    for a, b in zip(labels_a, labels_b):
        if a == b:
            correct += 1

    return correct / num_curves


def get_scores(curves):
    """Find the threshold in the range [0, 0.5) that gives the best accuracy."""

    user_labels = [curve["shape"] for curve in curves]

    thresholds = np.linspace(0, 0.5, 101)
    accuracies = []
    labels = []

    for t in thresholds:
        heuristic_labels = [get_shape(curve, t) for curve in curves]
        accuracy = calculate_accuracy(user_labels, heuristic_labels)
        accuracies.append(accuracy)
        labels.append(heuristic_labels)

    return pd.DataFrame(
        {"threshold": thresholds, "accuracy": accuracies, "labels": labels}
    )


def plot_accuracy_vs_threshold(df):
    """Plot line chart."""

    best = df.iloc[df["accuracy"].idxmax()]
    title = f"Best t = {best['threshold']} ({best['accuracy']:.2%})"

    plot = (
        alt.Chart(df, title=title)
        .mark_line()
        .encode(
            x=alt.X("threshold").title("t"),
            y=alt.Y("accuracy").title("Accuracy").axis(format=".2~%"),
        )
        .properties(width=400)
    )

    return plot, best


def check_labels(curves, heuristic_labels):
    """if there are cases where the user labeled it increasing and the
    heurisitc labels it decreasing, then it is likely
    due to a mistake when labeling."""

    user_labels = [curve["shape"] for curve in curves]

    bad = []

    for i, (heuristic_label, user_label) in enumerate(
        zip(heuristic_labels, user_labels)
    ):
        if (heuristic_label == "increasing" and user_label == "decreasing") or (
            heuristic_label == "decreasing" and user_label == "increasing"
        ):
            # if this is true, it's likely because of a mistake when labeling
            bad.append(
                {
                    "index": i,
                    "user_label": user_label,
                    "heuristic_label": heuristic_label,
                }
            )

    return bad


def fix_labels(curves, bad_labels):
    """Correct labeling mistakes."""
    for bl in bad_labels:
        curves[bl["index"]]["shape"] = bl["heuristic_label"]


def plot_disagreements(curves, shapes_a, shapes_b, title_a, title_b):
    """Show plots where there are disagreements between labels."""

    disagree_plots = []

    height = 200
    width = 1.95 * height

    for curve, a, b in zip(curves, shapes_a, shapes_b):
        if a != b:
            df_plot = pd.DataFrame(
                {
                    "x": curve["x"],
                    "y": curve["y"],
                }
            )

            plot = (
                alt.Chart(df_plot, title=f"{title_a} = {a}, {title_b} = {b}")
                .mark_line()
                .encode(x="x", y=alt.Y("y").scale(zero=False))
                .properties(width=width, height=height)
            )

            disagree_plots.append(plot)

    return alt.hconcat(*disagree_plots)


def plot_label_counts(labels_a, labels_b):
    """Plot number of times labels used for each user."""

    df_labels = pd.DataFrame(
        {
            "user": ["a"] * len(labels_a) + ["b"] * len(labels_b),
            "label": labels_a + labels_b,
        }
    )

    return (
        alt.Chart(df_labels)
        .mark_bar()
        .encode(
            x="user",
            y="count()",
            column=alt.Column("label").sort(["decreasing", "mixed", "increasing"]),
            color="user",
        )
    )

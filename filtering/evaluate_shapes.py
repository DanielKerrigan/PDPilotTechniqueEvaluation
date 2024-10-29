"""Determine the threshold that best aligns with the labels."""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd


def read_json(path):
    """Read JSON file."""
    return json.loads(Path(path).read_bytes())


def add_indices(curves):
    """add index key to dicts in the list"""
    for i, curve in enumerate(curves):
        curve["index"] = i

    return curves


def get_shape(curve, t):
    """Get the shape of the curve for the given threshold. Copied from PDPilot."""

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


def calculate_num_correct(labels_a, labels_b):
    """Get number of agreements between two lists of labels"""

    assert len(labels_a) == len(labels_b)

    correct = 0

    for a, b in zip(labels_a, labels_b):
        if a == b:
            correct += 1

    return correct


def get_scores(curves):
    """For each threshold in the range [0, 0.5] in increments of 0.005,
    calculate the heuristic's labels for the curves and the accuracy between
    those labels and the user's labels."""

    user_labels = [curve["shape"] for curve in curves]

    num_curves = len(curves)

    thresholds = np.linspace(0, 0.5, 101)
    accuracies = []
    correct = []
    labels = []

    for t in thresholds:
        heuristic_labels = [get_shape(curve, t) for curve in curves]
        num_correct = calculate_num_correct(user_labels, heuristic_labels)
        correct.append(num_correct)
        accuracies.append(num_correct / num_curves)
        labels.append(heuristic_labels)

    return pd.DataFrame(
        {
            "threshold": thresholds,
            "accuracy": accuracies,
            "correct": correct,
            "labels": labels,
        }
    )


def plot_accuracy_vs_threshold(df):
    """Plot line chart that compares accuracy of users's labels and the threshold."""

    # get max accuracy

    plot = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("threshold").title("t"),
            y=alt.Y("accuracy").title("Accuracy").axis(format=".2~%"),
        )
        .properties(width=400)
    )

    return plot


def get_best_thresholds(df):
    """Get the rows of df that correspond to the highest scores."""
    return df[df["correct"] == df["correct"].max()]


def check_labels(curves, heuristic_labels):
    """Check if there are cases where the user labeled it increasing and the
    heurisitc labels it decreasing (or vice versa). If there are, then they are
    likely due to a mistake when labeling that should be corrected."""

    user_labels = [curve["shape"] for curve in curves]

    bad = []

    for i, (heuristic_label, user_label) in enumerate(
        zip(heuristic_labels, user_labels)
    ):
        if (heuristic_label == "increasing" and user_label == "decreasing") or (
            heuristic_label == "decreasing" and user_label == "increasing"
        ):
            bad.append(
                {
                    "index": i,
                    "user_label": user_label,
                    "heuristic_label": heuristic_label,
                }
            )

    return bad


def fix_labels(curves, bad_labels):
    """Correct labeling mistakes that were identified by check_labels."""
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
                alt.Chart(
                    df_plot,
                    title=f"{curve['index']}: {title_a} = {a}, {title_b} = {b}",
                )
                .mark_line()
                .encode(x="x", y=alt.Y("y").scale(zero=False))
                .properties(width=width, height=height)
            )

            disagree_plots.append(plot)

    return alt.hconcat(*disagree_plots)


def set_consensus_labels(curves_consensus, labels_a, labels_b, corrections):
    """modify curves_consensus based on the provided list of corrections."""

    correction_index = 0

    for curve, a, b in zip(curves_consensus, labels_a, labels_b):
        if a != b:
            i, new_label = corrections[correction_index]
            assert i == curve["index"]
            curve["shape"] = new_label
            correction_index += 1
        else:
            assert a == b == curve["shape"]


def plot_label_counts(labels_a, labels_b):
    """Plot number of times each shape was picked for both sets of labels."""

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


def check_same_curves(curves_a, curves_b):
    """Check that the curves are the same."""
    for a, b in zip(curves_a, curves_b):
        assert a["x"] == b["x"]
        assert a["y"] == b["y"]

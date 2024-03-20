"""Determine the threshold that best aligns with the labels."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt


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


def calculate_accuracy(my_labels, heuristic_labels):
    """Get the heuristic's accuracy for the given threshold"""

    correct = 0

    for i, (a, b) in enumerate(zip(my_labels, heuristic_labels)):
        if a == b:
            correct += 1

    return correct / len(my_labels)


def get_scores(curves):
    """Find the threshold in the range [0, 0.5) that gives the best accuracy."""

    my_labels = [curve["shape"] for curve in curves]

    thresholds = np.linspace(0, 0.5, 101)
    accuracies = []
    labels = []

    for t in thresholds:
        heuristic_labels = [get_shape(curve, t) for curve in curves]
        accuracy = calculate_accuracy(my_labels, heuristic_labels)
        accuracies.append(accuracy)
        labels.append(heuristic_labels)

    return pd.DataFrame(
        {"threshold": thresholds, "accuracy": accuracies, "labels": labels}
    )


def read_json(path):
    """Read JSON file."""
    return json.loads(Path(path).read_bytes())


def plot_accuracy_vs_threshold(df):
    """Plot line chart."""
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("threshold").title("t"),
            y=alt.Y("accuracy").title("Accuracy"),
        )
        .properties(width=400)
    )


def check_labels(curves, heuristic_labels):
    """if there are cases where I labeled it increasing and the
    heurisitc labels it decreasing, then I likely
    made a mistake when labeling."""

    my_labels = [curve["shape"] for curve in curves]

    bad = []

    for i, (a, b) in enumerate(zip(heuristic_labels, my_labels)):
        if (a == "increasing" and b == "decreasing") or (
            a == "decreasing" and b == "increasing"
        ):
            # if this is true, it's likely because I made a mistake when labeling
            bad.append(i)
            print(f"curve {i + 1}: my label is {a}, heuristic label is {b}")

    return bad

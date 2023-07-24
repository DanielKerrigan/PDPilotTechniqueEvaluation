#!/usr/bin/env python
# coding: utf-8

"""
Compute partial dependence plots
"""

import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.random import MT19937, RandomState, SeedSequence
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

from pte.metadata import Metadata
from pte.tqdm_joblib import tqdm_joblib


def calculate_data(
    *,
    predict: Callable[[pd.DataFrame], List[float]],
    model_name: str,
    df: pd.DataFrame,
    dataset_name: str,
    y_label: str,
    features: List[str],
    resolution: int = 20,
    one_hot_features: Union[Dict[str, List[Tuple[str, str]]], None] = None,
    nominal_features: Union[List[str], None] = None,
    ordinal_features: Union[List[str], None] = None,
    feature_value_mappings: Union[Dict[str, Dict[str, str]], None] = None,
    num_clusters_extent: Tuple[int, int] = (2, 5),
    n_jobs: int = 1,
    seed: Union[int, None] = None,
    output_path: Union[str, None] = None,
) -> Union[dict, None]:
    """Calculates the data needed to evaluate the filtering, ranking, and clustering techniques.

    :param predict: A function whose input is a DataFrame of instances and
        returns the model's predictions on those instances.
    :type predict: Callable[[pd.DataFrame], list[float]]
    :param model_name: Name of the model.
    :type model_name: string
    :param df: Instances to use to compute the PDPs and ICE plots.
    :type df: pd.DataFrame
    :param dataset_name: Name of the dataset.
    :type dataset_name: string
    :param y_label: Name of the dataset.
    :type y_label: string
    :param features: List of feature names to compute the plots for.
    :type features: list[str]
    :param resolution: For quantitative features, the number of evenly
        spaced to use to compute the plots, defaults to 20.
    :type resolution: int, optional
    :param one_hot_features: A dictionary that maps from the name of a feature
        to a list tuples containg the corresponding one-hot encoded column
        names and feature values, defaults to None.
    :type one_hot_features: dict[str, list[tuple[str, str]]] | None, optional
    :param nominal_features: List of nominal and binary features in the
        dataset that are not one-hot encoded. If None, defaults to binary
        features in the dataset.
    :type nominal_features: list[str] | None, optional
    :param ordinal_features: List of ordinal features in the dataset.
        If None, defaults to integer features with 3-12 unique values.
    :type ordinal_features: list[str] | None, optional
    :param feature_value_mappings: Nested dictionary that maps from the name
        of a nominal or ordinal feature, to a value for that feature in
        the dataset, to the desired label for that value in the UI,
        defaults to None.
    :type feature_value_mappings: dict[str, dict[str, str]] | None, optional
    :param num_clusters_extent: The minimum and maximum number of clusters to
        try when clustering the lines of ICE plots. Defaults to (2, 5).
    :type num_clusters_extent: tuple[int, int]
    :param n_jobs: Number of jobs to use to parallelize computation,
        defaults to 1.
    :type n_jobs: int, optional
    :param seed:  Random state for clustering. Defaults to None.
    :type seed: int | None, optional
    :param output_path: A file path to write the results to.
        If None, then the results are instead returned.
    :type output_path: str | None, optional
    :raises OSError: Raised when the ``output_path``, if provided, cannot be written to.
    :return: Calculated data, or None if an ``output_path`` is provided.
    :rtype: dict | None
    """

    # first check that the output path exists if provided so that the function
    # can fail quickly, rather than waiting until all the work is done
    if output_path:
        path = Path(output_path).resolve()

        if not path.parent.is_dir():
            raise OSError(f"Cannot write to {path.parent}")

    md = Metadata(
        df,
        resolution,
        one_hot_features,
        nominal_features,
        ordinal_features,
        feature_value_mappings,
    )

    subset = df.copy()
    subset_copy = df.copy()

    seed_sequence = SeedSequence(seed)
    seeds = seed_sequence.spawn(len(features))

    one_way_work = [
        {
            "predict": predict,
            "data": subset,
            "data_copy": subset_copy,
            "feature": feature,
            "md": md,
            "num_clusters_extent": num_clusters_extent,
            "seed_sequence": seeds[i],
        }
        for i, feature in enumerate(features)
    ]

    num_one_way = len(features)
    print(f"Calculating {num_one_way} one-way PDPs")

    if n_jobs == 1:
        one_way_results = [
            _calc_one_way_pd(**args) for args in tqdm(one_way_work, ncols=80)
        ]
    else:
        with tqdm_joblib(tqdm(total=num_one_way, unit="PDP", ncols=80)) as _:
            one_way_results = Parallel(n_jobs=n_jobs)(
                delayed(_calc_one_way_pd)(**args) for args in one_way_work
            )

    # output

    results = {
        "dataset": dataset_name,
        "model": model_name,
        "y_label": y_label,
        "features": one_way_results,
    }

    if output_path:
        path.write_text(json.dumps(results), encoding="utf-8")
    else:
        return results


def _calc_one_way_pd(
    predict,
    data,
    data_copy,
    feature,
    md,
    num_clusters_extent,
    seed_sequence,
):
    random_state = RandomState(MT19937(seed_sequence))

    feat_info = md.feature_info[feature]

    ice_lines = []

    for value in feat_info["values"]:
        _set_feature(feature, value, data, feat_info)
        predictions = predict(data)
        ice_lines.append(predictions.tolist())

    _reset_feature(feature, data, data_copy, feat_info)

    ice_lines = np.array(ice_lines).T
    ice_deviation = np.std(ice_lines, axis=1).mean().item()
    mean_predictions = np.mean(ice_lines, axis=0)

    x_values = (
        feat_info["values"]
        if "value_map" not in feat_info
        else [feat_info["value_map"].get(v, v) for v in feat_info["values"]]
    )

    clustering = _calculate_clusterings(
        ice_lines=ice_lines,
        num_clusters_extent=num_clusters_extent,
        random_state=random_state,
    )

    subkind = "nominal" if feat_info["subkind"] == "one_hot" else feat_info["subkind"]

    feature_data = {
        "name": feature,
        "kind": feat_info["kind"],
        "subkind": subkind,
        "ordered": feat_info["ordered"],
        "x_values": x_values,
        "ice_lines": ice_lines.tolist(),
        "pdp": mean_predictions.tolist(),
        "deviation": ice_deviation,
        "clustering": clustering,
    }

    if feat_info["ordered"]:
        # shape
        y = np.array(mean_predictions)
        diff = np.diff(y)
        pos = diff[diff > 0].sum()
        neg = np.abs(diff[diff < 0].sum())
        percent_pos = pos / (pos + neg) if pos + neg != 0 else 0.5

        threshold_increasing = percent_pos - 0.5

        feature_data["shape_tolerance_0"] = (
            "increasing" if threshold_increasing >= 0 else "decreasing"
        )
        feature_data["tolerance_threshold"] = abs(threshold_increasing)

    return feature_data


def _set_feature(feature, value, data, feature_info):
    if feature_info["subkind"] == "one_hot":
        col = feature_info["value_to_column"][feature_info["value_map"][value]]
        all_features = [feat for feat, _ in feature_info["columns_and_values"]]
        data[all_features] = 0
        data[col] = 1
    else:
        data[feature] = value


def _reset_feature(
    feature,
    data,
    data_copy,
    feature_info,
):
    if feature_info["subkind"] == "one_hot":
        all_features = [col for col, _ in feature_info["columns_and_values"]]
        data[all_features] = data_copy[all_features]
    else:
        data[feature] = data_copy[feature]


def _calculate_clusterings(ice_lines, num_clusters_extent, random_state):
    centered_ice_lines = ice_lines - ice_lines[:, 0].reshape(-1, 1)

    diffs = np.diff(ice_lines)

    centered_distances = euclidean_distances(centered_ice_lines, centered_ice_lines)
    diffs_distances = euclidean_distances(diffs, diffs)

    methods = [
        {
            "name": "centered_kmeans",
            "lines": centered_ice_lines,
            "model": kmeans,
            "distances": centered_distances,
        },
        {
            "name": "diff_kmeans",
            "lines": diffs,
            "model": kmeans,
            "distances": diffs_distances,
        },
    ]

    clusterings = []

    for method in methods:
        best_score = -math.inf
        best_n_clusters = -1
        best_labels = None

        model = method["model"]
        lines = method["lines"]
        distances = method["distances"]

        for n_clusters in range(num_clusters_extent[0], num_clusters_extent[1] + 1):
            labels = model(lines, n_clusters, random_state)

            if len(np.unique(labels)) < n_clusters:
                # Fewer than n_clusters clusters were found.
                # This could happen if the feature is not used by the model, causing
                # all of the centered ice lines to be the same.
                if best_n_clusters == -1:
                    best_n_clusters = 1
                    best_labels = labels

                break

            score = silhouette_score(distances, labels, metric="precomputed")

            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_labels = labels

        clusterings.append(
            {
                "method": method["name"],
                "num_clusters": best_n_clusters,
                "labels": best_labels.tolist(),
            }
        )

    return clusterings


def kmeans(lines, n_clusters, random_state):
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=5,
        max_iter=300,
        algorithm="lloyd",
        random_state=random_state,
    ).fit(lines)
    return model.labels_

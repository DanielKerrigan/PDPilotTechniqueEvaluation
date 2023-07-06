#!/usr/bin/env python
# coding: utf-8

"""
Compute partial dependence plots
"""

import json
from pathlib import Path
from typing import Callable, Union, Dict, Tuple, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .metadata import Metadata
from .tqdm_joblib import tqdm_joblib


def calculate_ice(
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
    n_jobs: int = 1,
    output_path: Union[str, None] = None,
) -> Union[dict, None]:
    """Calculates the ICE plots.

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
    :param n_jobs: Number of jobs to use to parallelize computation,
        defaults to 1.
    :type n_jobs: int, optional
    :param output_path: A file path to write the results to.
        If None, then the results are instead returned.
    :type output_path: str | None, optional
    :raises OSError: Raised when the ``output_path``, if provided, cannot be written to.
    :return: Wigdet data, or None if an ``output_path`` is provided.
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

    # TODO: reset index?
    subset = df.copy()
    subset_copy = df.copy()

    work = [
        {
            "predict": predict,
            "data": subset,
            "data_copy": subset_copy,
            "feature": feature,
            "md": md,
        }
        for feature in features
    ]

    num_one_way = len(features)
    print(f"Calculating {num_one_way} ICE plots")

    if n_jobs == 1:
        one_way_results = [_calc_ice_lines(**args) for args in tqdm(work, ncols=80)]
    else:
        with tqdm_joblib(tqdm(total=num_one_way, unit="plot", ncols=80)) as _:
            one_way_results = Parallel(n_jobs=n_jobs)(
                delayed(_calc_ice_lines)(**args) for args in work
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


def _calc_ice_lines(
    predict,
    data,
    data_copy,
    feature,
    md,
):
    feat_info = md.feature_info[feature]

    ice_lines = []

    for value in feat_info["values"]:
        _set_feature(feature, value, data, feat_info)
        predictions = predict(data)
        ice_lines.append(predictions.tolist())

    _reset_feature(feature, data, data_copy, feat_info)

    ice_lines = np.array(ice_lines).T

    x_values = (
        feat_info["values"]
        if "value_map" not in feat_info
        else [feat_info["value_map"].get(v, v) for v in feat_info["values"]]
    )

    feature = {
        "name": feature,
        "kind": feat_info["kind"],
        "subkind": feat_info["subkind"],
        "x_values": x_values,
        "ice_values": ice_lines.tolist(),
    }

    return feature


def _set_feature(feature, value, data, feature_info):
    if feature_info["one_hot"]:
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
    if feature_info["one_hot"]:
        all_features = [col for col, _ in feature_info["columns_and_values"]]
        data[all_features] = data_copy[all_features]
    else:
        data[feature] = data_copy[feature]

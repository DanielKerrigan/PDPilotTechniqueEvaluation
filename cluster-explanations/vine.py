"""
This file is a modified version of `main.py` from VINE.
https://github.com/MattJBritton/VINE.

We have updated it to work with Python 3 and more recent
version of packages like pandas, numpy, and scikit-learn.

All non-formatting changes are marked with a comment
that begins with CHANGE.
"""

########IMPORTS#############

# CHANGE: Removed `from __future__ import division`

import datetime
import json
import sys
import time
from collections import defaultdict
from math import sqrt

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import mode

# scikit-learn
from sklearn import datasets, metrics
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier

# for reproducibility
# CHANGE: we pass the random seed as an argument and set it
# inside the calculate function rather than setting it globally
# np.random.seed(2019)

########LOAD DATASETS#############


# functions to parse datasets
def load_bike_dataset():
    def _datestr_to_timestamp(s):
        return time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d").timetuple())

    data = pd.read_csv("data/bike.csv")
    data["dteday"] = data["dteday"].apply(_datestr_to_timestamp)
    # CHANGE: Added `dtype=np.uint8` so that binary values are represented with
    # 1 and 0 rather than True and False.
    data = pd.get_dummies(
        data,
        prefix=["weathersit"],
        columns=["weathersit"],
        drop_first=False,
        dtype=np.uint8,
    )

    # de-normalize data to produce human-readable features.
    # Original range info from http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
    data["hum"] = data["hum"].apply(lambda x: x * 100.0)
    data["windspeed"] = data["windspeed"].apply(lambda x: x * 67.0)
    # convert Celsius to Fahrenheit
    data["temp"] = data["temp"].apply(lambda x: (x * 47.0 - 8) * 9 / 5 + 32)
    data["atemp"] = data["atemp"].apply(lambda x: (x * 66.0 - 16) * 9 / 5 + 32)

    # rename features to make them interpretable for novice users
    feature_names_dict = {
        "yr": "First or Second Year",
        "season": "Season",
        "hr": "Hour of Day",
        "workingday": "Work Day",
        "weathersit_2": "Misty Weather",
        "weathersit_3": "Light Precipitation",
        "weathersit_4": "Heavy Precipitation",
        "temp": "Temperature",
        "atemp": "Feels Temperature",
        "hum": "Humidity",
        "windspeed": "Wind Speed",
    }
    data = data.rename(mapper=feature_names_dict, axis=1)

    features = feature_names_dict.values()

    X = data[features]
    y = data["cnt"]

    return X, y


def load_diabetes_dataset():
    diabetes_dataset = datasets.load_diabetes()

    return pd.DataFrame(
        diabetes_dataset.data, columns=diabetes_dataset.feature_names
    ), diabetes_dataset.target


def load_boston_dataset():
    boston_dataset = datasets.load_boston()

    return pd.DataFrame(
        boston_dataset.data, columns=boston_dataset.feature_names
    ), boston_dataset.target


def load_dataset(name="bike"):
    if name == "bike":
        X, y = load_bike_dataset()
    elif name == "boston":
        X, y = load_boston_dataset()
    elif name == "diabetes":
        X, y = load_diabetes_dataset()
    gbm = GradientBoostingRegressor(min_samples_leaf=10, n_estimators=300)
    gbm.fit(X, y)
    return X, y, gbm


########INTERNAL METHODS#############


# from original PyCEBox library
# get the x_values for a given granularity of curve
def _get_grid_points(x, num_grid_points):
    if sorted(list(x.unique())) == [0, 1]:
        return [0.0, 1.0], "categorical"
    if num_grid_points is None:
        return x.unique(), "numeric"
    else:
        # unique is necessary, because if num_grid_points is too much larger
        # than x.shape[0], there will be duplicate quantiles (even with
        # interpolation)
        return x.quantile(np.linspace(0, 1, num_grid_points)).unique(), "numeric"


# from original PyCEBox library
# average the PDP lines (naive method seems to work fine)
def _pdp(ice_data):
    return np.array(ice_data.mean(axis=0))


# from http://nbviewer.jupyter.org/github/alexminnaar/time-series-classification-and-clustering/blob/master/Time%20Series%20Classification%20and%20Clustering.ipynb
def _default_factory():
    return float("inf")


def _get_dtw_distance(s1, s2, w=4):
    w = max(w, abs(len(s1) - len(s2)))
    DTW = defaultdict(_default_factory)
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w)):
            DTW[(i, j)] = (s1[i] - s2[j]) ** 2 + min(
                DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)]
            )

    return sqrt(DTW[len(s1) - 1, len(s2) - 1])


# transform curves before distance measurement
def _differentiate(series):
    dS = np.diff(series)
    return dS


# interpolate lines to num_grid_points when comparing features for feature-space statistics
def _interpolate_line(x, y, length):
    if len(y) == length:
        return y
    else:
        f = interp1d(x, y, kind="cubic")
        return list(f(np.linspace(x[0], x[-1], num=length, endpoint=True)))


def _get_model_split(columns, model):
    split_feature = (
        columns[model.tree_.feature[0]] if model.tree_.value.shape[0] > 1 else "none"
    )
    split_val = round(model.tree_.threshold[0], 2)
    split_direction = (
        "<="
        if model.tree_.value.shape[0] == 1
        or model.classes_[np.argmax(model.tree_.value[1])] == 1
        else ">"
    )
    return split_feature, split_val, split_direction


########MAIN ALGORITHM#############


# main function - run this to export JSON file for vis
# CHANGE: Separating the calculation of the data from
# exporting it to a JSON file.
# CHANGE: add `merge_clusters` and `seed` parameters
def calculate(
    data,
    y,
    predict_func,
    num_clusters=5,
    num_grid_points=40,
    ice_curves_to_export=100,
    cluster_method="good",
    prune_clusters=True,
    merge_clusters=True,
    seed=1,
):
    # CHANGE: create random number generator
    rng = np.random.default_rng(seed=seed)

    export_dict = {
        "features": {},
        "distributions": {},
        "average_prediction": np.mean(y),
    }

    # generate data for one ICE plot per column
    for column_of_interest in data.columns:
        ice_data = pd.DataFrame(np.ones(data.shape[0]), columns=["tempPlaceholderCol"])

        x_s, column_type = _get_grid_points(data[column_of_interest], num_grid_points)

        # create dataframe with synthetic points (one for each x value returned by _get_grid_points)
        for x_val in x_s:
            kwargs = {column_of_interest: x_val}
            ice_data[x_val] = predict_func(data.assign(**kwargs))

        ice_data.drop("tempPlaceholderCol", axis=1, inplace=True)

        # center all curves at the mean point of the feature's range
        if column_type == "numeric":
            ice_data = ice_data.sub(ice_data.mean(axis=1), axis="index")
        else:  # categorical
            # CHANGE: mode returns the mode and the count
            # However, this looks like it was the case even in old version
            # of scipy, so it may be a bug in the original code:
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.mode.html
            if mode(data.loc[:, column_of_interest])[0] == 1:
                ice_data = ice_data.sub(ice_data.iloc[:, 1], axis="index")
            else:
                ice_data = ice_data.sub(ice_data.iloc[:, 0], axis="index")

        pdp_data = _pdp(ice_data)
        hist_counts, hist_bins = np.histogram(
            a=np.array(data.loc[:, column_of_interest]), bins="auto"
        )
        hist_zip = [{"x": x[0], "y": x[1]} for x in zip(hist_bins, hist_counts)]
        export_dict["distributions"][column_of_interest] = hist_zip
        export_dict["features"][column_of_interest] = {
            "feature_name": column_of_interest,
            "x_values": list(x_s),
            "pdp_line": list(pdp_data),
            "importance": np.std(pdp_data),
            "clusters": [],
            "data_type": column_type,
        }

        # perform clustering
        if cluster_method == "good":
            ice_data["cluster_label"] = (
                AgglomerativeClustering(n_clusters=num_clusters)
                .fit(_differentiate(ice_data.values))
                .labels_
            )
        elif cluster_method == "fast":
            ice_data["cluster_label"] = (
                Birch(n_clusters=num_clusters, threshold=0.1)
                .fit(_differentiate(ice_data.values))
                .labels_
            )

        ice_data["points"] = ice_data[x_s].values.tolist()

        # generate all the ICE curves per cluster
        all_curves_by_cluster = ice_data.groupby("cluster_label")["points"].apply(
            lambda x: np.array(x)
        )

        splits_first_pass = []
        for cluster_num in range(len(all_curves_by_cluster)):
            num_curves_in_cluster = len(all_curves_by_cluster[cluster_num])

            # build model to predict cluster membership
            rdwcY = ice_data["cluster_label"].apply(
                lambda x: 1 if x == cluster_num else 0
            )
            # 1-node decision tree to get best split for each cluster
            # CHANGE: DecisionTreeClassifer no longer has a `presort` parameter.
            # Previously it was set to False, which was the default value.
            # CHANGE: set random_state
            model = DecisionTreeClassifier(
                criterion="entropy",
                max_depth=1,
                class_weight="balanced",
                random_state=seed,
            )
            model.fit(data, rdwcY)
            split_feature, split_val, split_direction = _get_model_split(
                data.columns, model
            )
            splits_first_pass.append(
                {
                    "feature": split_feature,
                    "val": split_val,
                    "direction": split_direction,
                    "model": model,
                }
            )

        # loop through splits to find duplicates
        duplicate_splits = {}
        # CHANGE: add option to not merge clusters
        if merge_clusters:
            for i, split_def in enumerate(splits_first_pass[:-1]):
                for j, split_def_2 in enumerate(splits_first_pass):
                    if j <= i or i in duplicate_splits or j in duplicate_splits:
                        continue
                    elif (
                        split_def["feature"] == split_def_2["feature"]
                        and split_def["direction"] == split_def_2["direction"]
                        and (split_def["val"] - split_def_2["val"])
                        / (np.ptp(data.loc[:, split_def["feature"]]))
                        <= 0.1
                    ):
                        duplicate_splits[j] = i

        # CHANGE: When using a nested dict for `to_replace`, the `value`
        # parameter cannot be set. Previously, it was set to `None`.
        ice_data = ice_data.replace(to_replace={"cluster_label": duplicate_splits})
        # generate all the ICE curves per cluster
        all_curves_by_cluster = ice_data.groupby("cluster_label")["points"].apply(
            lambda x: np.array(x)
        )

        # average the above to get the mean cluster line
        # CHANGE: Replaced `iteritems()` with `items()`
        cluster_average_curves = {
            key: np.mean(np.array(list(value)), axis=0)
            for key, value in all_curves_by_cluster.items()
        }

        for cluster_num in all_curves_by_cluster.keys():
            num_curves_in_cluster = len(all_curves_by_cluster[cluster_num])

            # build model to predict cluster membership
            rdwcY = ice_data["cluster_label"].apply(
                lambda x: 1 if x == cluster_num else 0
            )
            model = splits_first_pass[cluster_num]["model"]
            predY = model.predict(data)
            split_feature, split_val, split_direction = _get_model_split(
                data.columns, model
            )

            if prune_clusters:
                # do not use cluster if it has low accuracy or is highly similar to the PDP
                if int(round(100.0 * metrics.accuracy_score(rdwcY, predY))) <= 50:
                    continue

                pdp_distance = np.mean(
                    np.absolute(
                        (cluster_average_curves[cluster_num] - pdp_data)
                        / np.max(np.absolute(pdp_data))
                    )
                )
                if pdp_distance < 0.2 or np.isnan(pdp_distance):
                    continue

            # get random curves if there are more than 100
            # no reason to make the visualization display 1000+ ICE curves for this tool
            # CHANGE: use rng.choice rather than np.random.choice
            if num_curves_in_cluster > ice_curves_to_export:
                individual_ice_samples = [
                    list(x)
                    for x in list(
                        all_curves_by_cluster[cluster_num][
                            rng.choice(
                                num_curves_in_cluster,
                                size=ice_curves_to_export,
                                replace=False,
                            )
                        ]
                    )
                ]
            else:
                individual_ice_samples = [
                    list(x) for x in list(all_curves_by_cluster[cluster_num])
                ]

            # add cluster-level metrics to the JSON file
            export_dict["features"][column_of_interest]["clusters"].append(
                {
                    "accuracy": int(
                        round(100.0 * metrics.accuracy_score(rdwcY, predY))
                    ),
                    "precision": int(
                        round(100.0 * metrics.precision_score(rdwcY, predY))
                    ),
                    "recall": int(round(100.0 * metrics.recall_score(rdwcY, predY))),
                    "split_feature": split_feature,
                    "split_val": split_val,
                    "split_direction": split_direction,
                    "cluster_size": num_curves_in_cluster,
                    "line": list(cluster_average_curves[cluster_num]),
                    "individual_ice_curves": individual_ice_samples,
                }
            )

        # feature-level calculation for cluster distance to pdp
        feature_val = export_dict["features"][column_of_interest]
        # CHANGE: if there are no clusters, set cluster_deviation to 0
        # and skip the calculation. This avoids runtime warnings.
        if len(feature_val["clusters"]) == 0:
            feature_val["cluster_deviation"] = 0
        else:
            feature_val["cluster_deviation"] = np.mean(
                np.abs(
                    [
                        _get_dtw_distance(
                            np.array(feature_val["pdp_line"]), np.array(x["line"])
                        )
                        for x in feature_val["clusters"]
                    ]
                )
            ) / np.max(np.abs(np.array(feature_val["pdp_line"])))
            if np.isnan(feature_val["cluster_deviation"]) or feature_val[
                "cluster_deviation"
            ] == float("inf"):
                feature_val["cluster_deviation"] = 0

        # EOF feature loop

    return export_dict


# CHANGE: Separating the calculation of the data from
# exporting it to a JSON file.
def export(export_dict, output_path):
    with open(output_path, "w") as outfile:
        # CHANGE: Convert numpy values to Python values so that
        # they can be encoded in JSON.
        # https://stackoverflow.com/a/57915246/5016634
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        json.dump(export_dict, outfile, cls=NumpyEncoder)


# CHANGE: Separating the calculation of the data from
# exporting it to a JSON file.
# CHANGE: add `merge_clusters` and `seed` parameters
def calculate_and_export(
    data,
    y,
    predict_func,
    num_clusters=5,
    num_grid_points=40,
    ice_curves_to_export=100,
    cluster_method="good",
    prune_clusters=True,
    merge_clusters=True,
    seed=1,
    output_path="static/data.json",
):
    export_dict = calculate(
        data=data,
        y=y,
        predict_func=predict_func,
        num_clusters=num_clusters,
        num_grid_points=num_grid_points,
        ice_curves_to_export=ice_curves_to_export,
        cluster_method=cluster_method,
        prune_clusters=prune_clusters,
        merge_clusters=merge_clusters,
        seed=seed,
    )

    export(export_dict=export_dict, output_path=output_path)


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    num_clusters = int(sys.argv[2])
    num_grid_points = int(sys.argv[3])
    cluster_method = sys.argv[4]
    prune_clusters = sys.argv[5]
    dfX, y, predictor = load_dataset(name=dataset_name)

    calculate_and_export(
        data=dfX,
        y=y,
        predict_func=predictor.predict,
        num_clusters=num_clusters,
        num_grid_points=num_grid_points,
        ice_curves_to_export=50,
        cluster_method=cluster_method,
        prune_clusters=prune_clusters,
    )

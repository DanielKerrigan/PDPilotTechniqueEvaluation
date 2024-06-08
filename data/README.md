# Data

This directory contains the code for downloading datasets, training models, computing PDP and ICE plots, and calculating feature importance scores.

We train LightGBM models on binary classification and regression datasets from the [Penn Machine Learning Benchmark](https://epistasislab.github.io/pmlb/). We use [Optuna](https://optuna.org) for hyperpameter tuning and perform nested cross-validaton to evaluate the modeling processes. For regression datasets, we evaluate the models using root mean squared error. For classification datasets, we use binary log loss.

For a given dataset, if the average score of the outer cross-validation folds is not better than a simple baseline model, then we do not use that dataset. For regression datasets, the baseline model predicts the mean of the targets. For binary classification datasets, the baseline model assigns probabilities to the classes equal to their proportions in the dataset. Based on this, of the 55 datasets we used, only "Hill_Valley_with_noise" was excluded.

We ran this code on Northeastern's HPC cluster and then copied the results to my local computer, which are available in the `results` directory of the zip file released with this repository.

## Contents

- [pmlb_datasets.csv](pmlb_datasets.csv): A list of [all of the datasets](https://github.com/EpistasisLab/pmlb/blob/master/pmlb/all_summary_stats.tsv) in PMLB. We selected binary classification and regression datasets with at least 500 instances. For datasets that were excluded, we list a reason for their exclusion.
- [datasets.json](datasets.json): The list of datasets that we use from PMLB. The datasets are split into two groups: small and big. The small group contains 18 datasets that mostly have fewer than 1000 instances. The big group contains 37 datasets that have 1000+ instances. The datasets are split up so that code can be debugged on the small group.
- [data_models_plots.py](data_models_plots.py): Python script to download the specified dataset, train a LightGBM model on it, compute PDP and ICE plots for it, and calculate feature importance scores.
- [gbm.py](gbm.py): Code for training and evaluating LightGBM models using nested cross-validation.
- [feature_importance.py](feature_importance.py): Code for calculating feature importance scores with the following 5 methods: deviation of PDP, deviation of ICE plot, permutation feature importance, mean absolute SHAP value, and LightGBM's built-in importance.
- [job-big.sh](job-big.sh): Bash script for submitting a Slurm job to run `data_models_plots.py` on all big datasets.
- [job-small.sh](job-small.sh): Bash script for submitting a Slurm job to run `data_models_plots.py` on all small datasets.
- [copy.sh](copy.sh): Bash script to copy results from Northeastern's HPC cluster to my local computer.

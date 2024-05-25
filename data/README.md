# Data

This directory contains the code for downloading datasets, training models, computing PDP and ICE plots, and calculating feature importance scores.

- [pmlb_datasets.csv](pmlb_datasets.csv): A list of [all of the datasets](https://github.com/EpistasisLab/pmlb/blob/master/pmlb/all_summary_stats.tsv) in the [Penn Machine Learning Benchmark](https://epistasislab.github.io/pmlb/). We selected binary classification and regression datasets with at least 500 instances. For datasets that were excluded, we list a reason for their exclusion.
- [datasets.json](datasets.json): The list of datasets that we use from PMLB. The datasets are split into two groups: small and big. The small group contains 18 datasets that mostly have fewer than 1000 instances. The big group contains 37 datasets that have 1000+ instances. The datasets are split up so that code can be debugged on the small group.
- [data_models_plots.py](data_models_plots.py): Python script to download the specified dataset, train a LightGBM model on it, compute PDP and ICE plots for it, and calculate feature importance scores.
- [gbm.py](gbm.py): Code for training and evaluating LightGBM models using nested cross-validation.
- [feature_importance.py](feature_importance.py): Code for calculating feature importance scores with 5 different methods.
- [job-big.sh](job-big.sh): Bash script for submitting a Slurm job to run `data_models_plots.py` on all big datasets.
- [job-small.sh](job-small.sh): Bash script for submitting a Slurm job to run `data_models_plots.py` on all small datasets.
- [copy.sh](copy.sh): Bash script to copy results from Northeastern's HPC Cluster to my local computer.

# Clustering

This directory contains code for evaluating different preprocessing strategies for clustering ICE plots.

We compare clustering mean-centered ICE plots, centered ICE plots, and the estimated piecewise slopes of ICE lines. We generate synthetic ICE plots where we control the number of clusters to serve as ground truth to evaluate the results against. Each generated plot has between 2-5 clusters with 100-500 lines per cluster.

We use PCA as a simple generative model for creating the synthetic plots. We fit PCA with 8 components to the ICE lines calculated in the [data](../data/) directory. So that all lines are the same size, we only use ICE lines for quantative features with 20 points in their PDP. We set [`whiten=True`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) so that the features in the lower dimensional space have a mean of 0 and a standard deviation of 1. To generate a synthetic ICE line, we can them sample an array of size 8 from the normal distribution and use [`inverse_transform`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA.inverse_transform) to project it back to being a line with 20 points. To create clusters of ICE lines, we can sample from neighborhoods of the normal distribution.

In more detail, to create a synthetic ICE plot, we get a random starting point by sampling an array of size 8 from the normal distribution with a mean of 0 and a standard deviation of 1. For one cluster, we then sample another array from the normal distribution with a mean of 0 and a standard deviation of `between_deviation`. We add this to the starting point to get the "center" of the cluster. To generate lines in the cluster, we again sample from the normal distribution, this time with a mean of 0 and a standard deviation of `within_deviation` and add it to the center of the cluster. `between_deviation` controls how different the clusters in a plot are. `within_deviation` controls how different the lines in a cluster are. We visually determined `between_deviation=0.7` and `within_deviation=0.2` to be reasonable values to use.

## Contents

- [combine-ice.py](combine-ice.py): Combine all ICE lines for quantitative features with 20 points into one file.
- [generate-ice-plots.py](generate-ice-plots.py): Use PCA to generate synthetic ICE plots.
- [cluster-ice-plots.py](cluster-ice-plots.py): Cluster all ICE plots using the given preprocessing method.
- [score-methods.py](score-methods.py): Calculate the adjusted Rand index for each preprocessing method and synthetic ICE plot.
- [run-all.sh](run-all.sh): Run the above four scripts.
- [analyze-results.ipynb](analyze-results.ipynb): Calculate the mean and standard deviation of the adjusted Rand index for each preprocessing step.
- [view-synthetic-ice-plots.ipynb](view-synthetic-ice-plots.ipynb): Visualize the synthetic ICE plots and their clusters.
- [explore-pca.ipynb](explore-pca.ipynb): Explore the generated plots when modifying `between_deviation` and `within_deviation`.

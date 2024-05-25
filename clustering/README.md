# Clustering

This directory contains code for evaluating different preprocessing strategies for clustering ICE plots.

- [combine-ice.py](combine-ice.py): Combine all ICE lines for quantitative features with 20 points into one file.
- [generate-ice-plots.py](generate-ice-plots.py): Use PCA to generate synthetic ICE plots.
- [cluster-ice-plots.py](cluster-ice-plots.py): Cluster all ICE plots using the given preprocessing method.
- [score-methods.py](score-methods.py): Calculate the adjusted Rand index for each preprocessing method and synthetic ICE plot.
- [analyze-results.ipynb](analyze-results.ipynb): Calculate the mean and standard deviation of the adjusted Rand index for each preprocessing step.

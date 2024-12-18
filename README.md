# PDPilot Techniques Evaluation

This repository contains code and data for evaluating techniques in [PDPilot](https://github.com/DanielKerrigan/PDPilot).

## Contents

- [data](data): Downloading datasets, training models, computing PDP and ICE plots, and calculating feature importance scores.
- [clustering](clustering): Evaluating the effect of preprocessing on the clustering results.
- [ranking](ranking): Comparing feature importance rankings.
- [filtering](filtering): Analyzing the effect of the tolerance parameter for filtering PDPs by shape.
- [tool](tool): Web app for labeling the shape of PDPs.
- [cluster-difference](cluster-difference): Analyzing PDPilot's cluster difference metric.
- [cluster-explanations](cluster-explanations): Analyzing the effect of decision tree depth on ICE plot cluster explanations.
- [local-environment.yml](local-environment.yml) and [local-requirements.txt](local-requirements.txt): Conda environment file and exact versions of all packages that were used locally on my Mac.
- [hpc-environment.yml](hpc-environment.yml) and [hpc-requirements.txt](hpc-requirements.txt): Conda environment file and exact versions of all packages that were used on the Linux cluster.

## Installation

```
conda env create -f local-environment.yml
conda activate pdpilot-eval
```

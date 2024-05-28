# PDPilot Techniques Evaluation

This repository contains code and data for evaluating techniques in [PDPilot](https://github.com/DanielKerrigan/PDPilot).

## Contents

- [data](data): Downloading datasets, training models, computing PDP and ICE plots, and calculating feature importance scores.
- [clustering](clustering): Evaluating the effect of preprocessing on the clustering results.
- [ranking](ranking): Comparing feature importance rankings.
- [filtering](filtering): Analyzing the effect of the tolerance parameter for filtering PDPs by shape.
- [tool](tool): Web app for labeling the shape of PDPs.
- [environment.yml](environment.yml): conda environment file.

## Installation

```
conda env create -f environment.yml
conda activate pdpilot-eval
```

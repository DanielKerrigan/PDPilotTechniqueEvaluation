# Filtering

This directory contains the code for the analysis of filtering PDPs by their shape. We labeled the PDPs using the web app in the [tool](../tool/) directory.

Note that the PDPs that we labeled in [small-pdps.json](small-pdps.json) are not from the exact same models that we used for the ranking and clustering evaluations. Labeling the PDPs was the first evaluation that we did and we later made changes to the data preprocessing and modeling code for the ranking and clustering evaluations.

## Contents

- [get_pdps.py](get_pdps.py): Combine ordered, non-flat PDPs into one file.
- [small-pdps.json](small-pdps.json): 171 PDPs that we labeled.
- [dan-small-labeled-pdps.json](dan-small-labeled-pdps.json): Dan's labels.
- [enrico-small-labeled-pdps.json](enrico-small-labeled-pdps.json): Enrico's labels.
- [evaluate_shapes.py](evaluate_shapes.py): Functions for comparing labels and finding the best value for the tolerance parameter.
- [comparison.ipynb](comparison.ipynb): Comparing our labels with the labeling function's for different values for the tolerance parameter.

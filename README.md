# KNN+X
`KNN+X` is a proposed improvement to the well-known `KNN` algorithm.  
To label a new sample, the `k` nearest samples from the training data are found  
and then another classification algorithm `X` is used on the `k` neighbors.  
This is a replacement for the original majority choice in `KNN`, this is meant to make a more informed choice in cases where the majority is slight.

We enhance the proposed method by adding _thresholds_. Using thresholds, we can decide that if the majority label is 80% of the k-neighborhood, we choose it (`t=0.8` in code).  
So if we have a sufficient majority we don't train a new classifier - this saves runtime.

# Usage
## Dependencies
The experiment python code requires:
- numpy
- pandas
- h5py
- sklearn
- joblib

The plotting notebook also requires Matplotlib.
## Experiment code
When running experiments you can control various variables:
- Runs amount - how many experiment rounds to run.
- K values - which values of `k` to experiment with. e.g. 5, 13, 20, 100. These should be positive integers
- t values  - which values of `t` to experiment with. e.g. 0.2,0.6,0.9,1. We require `0 <= t <= 1`.
- Datasets - which `datasets` to experiment with (Currently implemented: `covertype`, `glass`, `mnist`, `skin`, `shuttle`, `usps`, `wine` and `yeast`).
- KNN algo - which KNN algorithm to use. (Choose between `brute`, `kd_tree` and `ball_tree`).
    
To run the experiment code, optionally choose the run amount, datasets, threshold values, k values, and knn algorithm to try in the __main__ function in `knn_plus_x.py` and run it.  
`python knn_plus_x.py`

## Result notebooks
### Plotting notebook
You can plot the results using `plot_results.ipynb`

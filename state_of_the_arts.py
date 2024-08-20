import os, sys
import struct
from array import array
from typing import List, Literal
import logging
import pickle

from timeit import default_timer as timer
import pprint
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from knn_plus_x import load_dataset


logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

delimiter = '#'*25

def generate_results(dataset: str):
    logger.info(f'Generating results for dataset: {dataset}')

    all_results = []
    
    X, y = load_dataset(dataset)
    from collections import Counter

    class_counts = Counter(y)

    print("Class distribution:")
    class_counts = Counter(y)
    print(delimiter)
    for cls, count in class_counts.items():
        print(f"Class {cls}: {count} members")
    print(delimiter)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
            
    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index]
            
        def pipeline_name(clf):
            if clf.__class__.__name__ == "Pipeline":
                return clf[-1].__class__.__name__
            else:
                return clf.__class__.__name__
        

        clfs = [
            RandomForestClassifier(random_state=42),
            GradientBoostingClassifier(random_state=42),
            HistGradientBoostingClassifier(random_state=42)
        ]

        clfs = tuple(sorted(clfs, key=lambda clf: pipeline_name(clf)))
        
        baseline_acc = np.empty((len(clfs)))
        baseline_time = np.empty((len(clfs)))

        for iclf, clf in enumerate(clfs):
            # logger.info(f"calcing (baseline) for clf: {pipeline_name(clf)}")
            start = timer()
            clf.fit(X_train, y_train)
            y_pred_test_rf = clf.predict(X_test)
            end = timer()
            
            baseline_time[iclf]=end-start
            baseline_acc[iclf] = accuracy_score(y_test, y_pred_test_rf)
        
        
        # logger.info(f'~~Finished~~ Generating results for dataset: {dataset}')
        results = {
            "dataset": dataset,
            "RandomForest_acc": baseline_acc[0],
            "GradientBoosting_acc": baseline_acc[1],
            "HistGradientBoosting_acc": baseline_acc[2]
        }
        all_results.append(results)
    
    return create_sota_results(all_results)
    

from collections import OrderedDict
def create_sota_results(l: List):
    df = pd.DataFrame(l)
    to_average = df.drop(columns=["dataset"], errors='ignore')
    to_keep = df["dataset"].iloc[0] if "dataset" in df.columns else None

    averaged_results = to_average.mean().add_suffix('_avg').to_dict()
    
    result = OrderedDict([("dataset", to_keep)])
    result.update(averaged_results)

    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(result)
    return result

import json
import os
if __name__ == "__main__":
    datasets = ['covertype', 'glass', 'mnist', 'skin', 'shuttle', 'usps', 'wine', 'yeast']
    if os.path.exists('results_file'):
        os.remove('results_file')

with open('results.txt', 'w') as f:
    for dataset in datasets:
        result = generate_results(dataset)
        f.write(f"Dataset: {dataset}\n")
        f.write(json.dumps(result, indent=4))
        f.write("\n\n") 

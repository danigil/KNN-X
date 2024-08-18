#try
import os, numpy as np
from typing import Literal
from glob import glob
import pickle

def ret_avg_results(datasets=['wine', 'mnist', 'usps', 'glass', 'yeast'], knn_algo:Literal['brute', 'kd_tree', 'ball_tree']='brute'):
    ret = {}
    for dataset in datasets:
        baseline_knn_accs = []
        baseline_accs = []
        smart_accs = []

        baseline_knn_times = []
        baseline_times = []
        smart_times = []

        for result_file in glob(f'{os.path.join("results", dataset, "*.pickle")}'):
            with open(result_file, 'rb') as f:
                result_dict = pickle.load(f)

            if result_dict.get("knn_algo",'brute') != knn_algo:
                continue

            baseline_knn_acc = result_dict["baseline_knn_acc"]
            baseline_acc = result_dict["baseline_acc"]
            smart_acc = result_dict["smart_acc"]

            baseline_knn_accs.append(baseline_knn_acc)
            baseline_accs.append(baseline_acc)
            smart_accs.append(smart_acc)


            baseline_knn_time = result_dict["baseline_knn_time"]
            baseline_time = result_dict["baseline_time"]
            smart_time = result_dict["smart_time"]

            baseline_knn_times.append(baseline_knn_time)
            baseline_times.append(baseline_time)
            smart_times.append(smart_time)

        if len(baseline_knn_accs) == 0:
            ret[dataset] = None
        else:
            ret[dataset] = {
                'dataset': dataset,
                'knn_algo': knn_algo,

                "baseline_knn_acc": np.average(np.array(baseline_knn_accs), axis=0),
                "baseline_acc": np.average(np.array(baseline_accs), axis=0),
                "smart_acc": np.average(np.array(smart_accs), axis=0),

                "baseline_knn_time": np.average(np.array(baseline_knn_times), axis=0),
                "baseline_time": np.average(np.array(baseline_times), axis=0),
                "smart_time": np.average(np.array(smart_times), axis=0),

                'clfs': result_dict['clfs'],
                'ks': result_dict['ks'],
                'thresholds': result_dict['tresholds']
            }

    ret = {k: v for k, v in ret.items() if v is not None}
    if len(ret) == 0:
        return None
    else:
        return ret
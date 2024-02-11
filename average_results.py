import os, json, re, numpy as np
from typing import Literal

def parse_results_dict(results, clfs, ks):
    
    baseline_knn_acc=np.array(results["baseline_acc"]["KNN"]) #shape (n_k,)
    baseline_acc = np.array([results["baseline_acc"][clf] for clf in clfs]) #shape (n_clfs,)
    smart_acc = np.array([results["smart_acc"][clf] for clf in clfs]) #shape (n_clfs,n_k)
    
    baseline_knn_time= np.array(results["baseline_time"]["KNN"]) #shape (n_k,)
    baseline_time = np.array([results["baseline_time"][clf] for clf in clfs]) #shape (n_clfs,)
    smart_time = np.array([results["smart_time"][clf] for clf in clfs]) #shape (n_clfs,n_k)
    
    #print(f'n_k: {len(ks)}')
    #print(f'n_clfs: {len(clfs)}\n')
    
    accs = [baseline_knn_acc, smart_acc]
    times = [baseline_knn_time, smart_time]
    
    k_len_arrs = np.vstack(accs+times) # shape (2(n_clfs+1), n_k)
    #print(k_len_arrs.shape)
    clf_len_arrs = np.vstack([baseline_acc, baseline_time])
    #print(clf_len_arrs.shape) # shape (2, n_clfs)
    return k_len_arrs, clf_len_arrs
    
    
    
    #return baseline_knn_acc, baseline_acc, smart_acc, baseline_knn_time, baseline_time, smart_time

def ret_clfs_ks(results):
    clfs = list(results["smart_acc"].keys())
    ks = results["ks"]
    return clfs, ks 

def ret_avg_results(datasets=['wine', 'mnist', 'usps', 'glass', 'yeast'], knn_algo:Literal['brute', 'kd_tree', 'ball_tree']='brute'):
    from glob import glob
    import pickle

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


if __name__=='__main__':
    #dataset_dirs = ['glass', 'wine', 'yeast']
    dataset_dirs = ['mnist', 'usps']
    results_dict = {dataset_dir:[] for dataset_dir in dataset_dirs}
    p=True
    for dataset_dir in dataset_dirs:
        dataset_results_path = os.path.join('./results',dataset_dir)
        
        for result in os.listdir(dataset_results_path):
            if not re.fullmatch(r"results_(\d)+.json" ,result):
                continue
            
            result_path = os.path.join(dataset_results_path, result)
            with open(result_path, 'r') as f:
                result_json = f.read()
                
            results_dict[dataset_dir].append(json.loads(result_json))
    clfs_ks = {dataset_dir:ret_clfs_ks(results_dict[dataset_dir][0]) for dataset_dir in dataset_dirs}
    #parse_results_dict(results_dict['glass'][0], *clfs_ks['glass'])
    parsed_results = {dataset_dir:[parse_results_dict(result, *clfs_ks[dataset_dir]) for result in results_dict[dataset_dir]] for dataset_dir in dataset_dirs}
    parsed_results_list_split = {dataset_dir:[*zip(*parsed_results[dataset_dir])] for dataset_dir in dataset_dirs}
    parsed_results_list_stacked = {dataset_dir:(np.array(parsed_results_list_split[dataset_dir][0]), np.array(parsed_results_list_split[dataset_dir][1])) for dataset_dir in dataset_dirs}
    averages = {dataset_dir:(np.average(parsed_results_list_stacked[dataset_dir][0], axis=0), np.average(parsed_results_list_stacked[dataset_dir][1], axis=0)) for dataset_dir in dataset_dirs}
    for dataset_dir in averages:
        clfs, ks = clfs_ks[dataset_dir]
        n_clfs = len(clfs)
        n_k = len(ks)
        
        arr1, arr2 = averages[dataset_dir]
        baseline_knn_acc, smart_acc, baseline_knn_time, smart_time = np.split(arr1, [1, n_clfs+1, n_clfs+2])
        baseline_acc, baseline_time = np.split(arr2, [1])
        
        baselines_dict_acc = {"KNN": list(baseline_knn_acc[0])}
        baselines_dict_acc.update({clf:baseline_acc[0][iclf] for iclf, clf in enumerate(clfs)})
        
        baselines_dict_time = {"KNN": list(baseline_knn_time[0])}
        baselines_dict_time.update({clf:baseline_time[0][iclf] for iclf, clf in enumerate(clfs)})
        x={
            "baseline_acc": baselines_dict_acc,
            "smart_acc": {clf:list(smart_acc[iclf]) for iclf, clf in enumerate(clfs)},
            "baseline_time": baselines_dict_time,
            "smart_time": {clf:list(smart_time[iclf]) for iclf, clf in enumerate(clfs)},
            "ks": ks
        }
        save_folder = os.path.join('./results',dataset_dir)
        with open(os.path.join(save_folder, f'results_avg.json'),'w') as f:
            f.write(json.dumps(x, indent=4))
        # if dataset_dir=='glass':
        #     print(baseline_acc)
        #     print(baseline_acc.shape)
        #     print(baseline_time.shape)
        #     print(baseline_knn_acc.shape)
        #     print(smart_acc.shape)
        #     print(baseline_knn_time.shape)
        #     print(smart_time.shape)
        

        
        
    #print(len(parsed_results['glass']))
    #print(*zip(*parsed_results['glass']))

import os, sys
import struct
from array import array
from typing import List, Literal
import logging
import pickle

from timeit import default_timer as timer

import numpy as np
import pandas as pd
import h5py

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

from joblib import Parallel, delayed

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_dataset(dataset: Literal['covertype', 'glass', 'mnist', 'skin', 'shuttle', 'usps', 'wine', 'yeast']):
    if dataset == 'mnist':
        def read_images_labels(images_filepath, labels_filepath):        
            labels = []
            with open(os.path.join(os.path.dirname(__file__), 'datasets', 'mnist', labels_filepath), 'rb') as file:
                magic, size = struct.unpack(">II", file.read(8))
                if magic != 2049:
                    raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
                labels = array("B", file.read())        
            
            with open(os.path.join(os.path.dirname(__file__), 'datasets', 'mnist', images_filepath), 'rb') as file:
                magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
                if magic != 2051:
                    raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
                image_data = array("B", file.read())        
            images = []
            for i in range(size):
                images.append([0] * rows * cols)
            for i in range(size):
                img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
                img = img.reshape(28, 28)
                images[i][:] = img            
            
            return np.array(images), np.array(labels)
                
        def load_data():
            x_train, y_train = read_images_labels('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
            x_test, y_test = read_images_labels('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
            return (x_train, y_train),(x_test, y_test)   


        (X_train, y_train),(X_test, y_test) = load_data()
        X_train = X_train.reshape(-1, 28 * 28)
        X_test = X_test.reshape(-1, 28 * 28)
        
        X = np.vstack((X_train, X_test))
        y = np.hstack((y_train, y_test))
        
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)
    elif dataset == 'usps': 
        with h5py.File(os.path.join('datasets', 'usps', 'usps.h5'), 'r') as hf:
                train = hf.get('train')
                X_train = train.get('data')[:]
                y_train = train.get('target')[:]
                test = hf.get('test')
                X_test = test.get('data')[:]
                y_test = test.get('target')[:]
        
        X = np.vstack((X_train, X_test))
        y = np.hstack((y_train, y_test))
        
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)
    elif dataset == 'wine':
        df = pd.read_csv(os.path.join('datasets', 'wine', 'winequality-white.csv'), delimiter=';')
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        y = LabelEncoder().fit_transform(y)
        X = X.to_numpy(dtype=float)

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)

        # X_train = X_train.to_numpy(dtype=float)
        # X_test = X_test.to_numpy(dtype=float)
    elif dataset == 'yeast':
        df = pd.read_csv(os.path.join('datasets', 'yeast', 'yeast.data'), sep='\\s+', header=None)
        X = df.iloc[:, 1:-1]
        y = df.iloc[:, -1]

        y = LabelEncoder().fit_transform(y)
        X = X.to_numpy(dtype=float)

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)

        # X_train = X_train.to_numpy(dtype=float)
        # X_test = X_test.to_numpy(dtype=float)
    elif dataset == 'glass':
        df = pd.read_csv(os.path.join('datasets', 'glass', 'glass.data'), sep=',', header=None)
        X = df.iloc[:, 1:-1]
        y = df.iloc[:, -1]
        y = LabelEncoder().fit_transform(y)

        X = X.to_numpy(dtype=float)

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)

        # X_train = X_train.to_numpy(dtype=float)
        # X_test = X_test.to_numpy(dtype=float)
    elif dataset == 'covertype':
        df = pd.read_csv(os.path.join('datasets', 'covertype', 'covtype.data'), header=None)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        y = LabelEncoder().fit_transform(y)
        X= X.to_numpy(dtype=int)

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)


        # X_train = X_train.to_numpy(dtype=int)
        # X_test = X_test.to_numpy(dtype=int)
    elif dataset == 'skin':
        df = pd.read_csv(os.path.join('datasets', 'skin_nonskin', 'Skin_NonSkin.txt'), sep='\\s+', header=None)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        y = LabelEncoder().fit_transform(y)

        X= X.to_numpy(dtype=int)

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)

        # X_train = X_train.to_numpy(dtype=int)
        # X_test = X_test.to_numpy(dtype=int)
    elif dataset == 'statlog':
        df_train = pd.read_csv(os.path.join('datasets', 'statlog', 'shuttle.trn'), sep='\\s+', header=None)
        df_test = pd.read_csv(os.path.join('datasets', 'statlog', 'shuttle.tst'), sep='\\s+', header=None)
        X_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1]
        X_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1]

        y_train = LabelEncoder().fit_transform(y_train)
        y_test = LabelEncoder().fit_transform(y_test)

        X_train = X_train.to_numpy(dtype=int)
        X_test = X_test.to_numpy(dtype=int)

        X = np.vstack((X_train, X_test))
        y = np.hstack((y_train, y_test))
        
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)
    else:
        raise Exception(f'unknown dataset {dataset}')

    return X, y

def generate_results(dataset: str, ks: List[int], thresholds: List[float], knn_algo:Literal['brute', 'kd_tree', 'ball_tree']='brute'):
    logger.info(f'Generating results for dataset: {dataset}')
    save_folder = os.path.join('results', dataset)
    os.makedirs(save_folder, exist_ok=True)

    
    X, y = load_dataset(dataset)
    from collections import Counter

    # class_counts = Counter(y)

    # # # Find the class with the least members
    # # least_common_class, least_common_count = min(class_counts.items(), key=lambda item: item[1])
    # # print(f"Class with the least members: {least_common_class}, Number of members: {least_common_count}")
    # print("Class distribution before removal:")
    # class_counts = Counter(y)
    # for cls, count in class_counts.items():
    #     print(f"Class {cls}: {count} members")


    rskf = RepeatedKFold(n_splits=10, n_repeats=5)
    # for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    # if not dataset in ('statlog', 'mnist', 'usps'):
    #     X = X.to_numpy(dtype=int)

    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index]
        

        def smart_decision(clf, sample, neighbor_idxs):
            X_neighbors, y_neighbors = X_train[neighbor_idxs], y_train[neighbor_idxs]

            unique, counts = np.unique(y_neighbors, return_counts=True)
            dominant_class = unique[np.argmax(counts)]
            if counts[np.argmax(counts)] >= treshold*k:
                return dominant_class
            
            else:
                clf.fit(X_neighbors, y_neighbors)

                return clf.predict(sample.reshape(1, -1))[0]
            
        def pipeline_name(clf):
            if clf.__class__.__name__ == "Pipeline":
                return clf[-1].__class__.__name__
            else:
                return clf.__class__.__name__
        
        logistic_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        svm_clf = make_pipeline(StandardScaler(), SVC())

        clfs = [svm_clf, GaussianNB(), logistic_clf, DecisionTreeClassifier()]
        clfs = tuple(sorted(clfs, key=lambda clf: pipeline_name(clf)))
        
        baseline_knn_acc = np.empty((len(ks)))
        baseline_knn_time = np.empty((len(ks)))
        
        baseline_acc = np.empty((len(clfs)))
        baseline_time = np.empty((len(clfs)))
        
        smart_acc = np.empty((len(clfs), len(ks), len(thresholds)))
        smart_time = np.empty((len(clfs), len(ks), len(thresholds)))

        for iclf, clf in enumerate(clfs):
            logger.info(f"calcing (baseline) for clf: {pipeline_name(clf)}")
            start = timer()
            clf.fit(X_train, y_train)
            y_pred_test_rf = clf.predict(X_test)
            end = timer()
            
            baseline_time[iclf]=end-start
            baseline_acc[iclf] = accuracy_score(y_test, y_pred_test_rf)
        
        for ik, k in enumerate(ks):
            logger.info(f"\tcalcing for k: {k}")

            knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', metric='minkowski', p=2, n_jobs=-1)
            
            start = timer()
            
            knn.fit(X_train, y_train)
            y_pred_test_knn = knn.predict(X_test)
            
            end = timer()
            
            baseline_knn_time[ik] = end - start
            baseline_knn_acc[ik] = accuracy_score(y_test, y_pred_test_knn)

            neighbors_test = knn.kneighbors(X_test, return_distance=False)   

            for iclf, clf in enumerate(clfs):
                logger.info(f"\t\tcalcing (smart) for clf: {pipeline_name(clf)}")
                for itreshold, treshold in enumerate(thresholds):
                    logger.info(f"\t\t\tcalcing (smart) for threshold: {treshold}")
                    start = timer()
                    y_pred_test_smart=Parallel(n_jobs=-1)(delayed(smart_decision)(clone(clf), X_test[i], idxs) for i, idxs in enumerate(neighbors_test))
                    end = timer()

                    smart_time[iclf, ik, itreshold]=end-start
                    smart_acc[iclf, ik, itreshold]=accuracy_score(y_test, y_pred_test_smart)

            
        logger.info(f'~~Finished~~ Generating results for dataset: {dataset}')
        results = {
            "baseline_knn_time": baseline_knn_time,
            "baseline_knn_acc": baseline_knn_acc,
            "baseline_time": baseline_time,
            "baseline_acc": baseline_acc,
            "smart_time": smart_time,
            "smart_acc": smart_acc,
            "clfs": [pipeline_name(clf) for clf in clfs],
            "ks": ks,
            "tresholds": thresholds,
            "knn_algo":knn_algo
        }

        with open(os.path.join(save_folder, f'results_{i}_{knn_algo}.pickle'), 'wb') as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 80, 100]
    thresholds = [1.0,]
    datasets = ['covertype', 'glass', 'mnist', 'skin', 'shuttle', 'usps', 'wine', 'yeast']
    # datasets = ['glass']
    knn_algo: Literal['brute', 'kd_tree', 'ball_tree'] = 'brute'
    
    for dataset in datasets:
        generate_results(dataset, ks=ks, thresholds=thresholds, knn_algo=knn_algo)
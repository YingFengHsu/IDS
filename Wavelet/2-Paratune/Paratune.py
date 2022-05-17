#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial

import csv
import optuna
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
import json

randomseed=42

def objective(X, y, trial):
    max_features = trial.suggest_categorical('max_features', ["log2", "sqrt", None])
    n_estimators = trial.suggest_categorical('n_estimators', [2, 4, 8, 16, 32, 64, 128, 256])
    min_samples_leaf = int(trial.suggest_discrete_uniform('min_samples_leaf', 1, 241, 20))
    
    min_samples_split = int(trial.suggest_discrete_uniform('min_samples_split', 2, 30, 2))
    max_leaf_nodes = trial.suggest_categorical('max_leaf_nodes', [2, 4, 8, 16, 32 ,64, 128, 256, None])
    criterion = trial.suggest_categorical('criterion', ["gini", "entropy"])
    max_depth = trial.suggest_categorical('max_depth', [16, 32, 64, 128, None])
    
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, min_samples_split=min_samples_split, max_leaf_nodes=max_leaf_nodes, criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features = max_features, n_jobs=2)
    
    skf = KFold(n_splits=5, shuffle=False)
                          
    y_pred = cross_val_score(clf, X, y, cv=skf)
    average=sum(y_pred)/len(y_pred)
    return average

def main():
    #This opens the target dataset and labels generated in step 1
    with open("/data/lee/wavelet/FullTest-DoS-TrainVali-haar-5.txt", "r") as fp:
        features=json.load(fp)
    with open("/data/lee/wavelet/DoS-TrainVali-Labels.txt", "r") as fp:
        labels=json.load(fp)
        
    f = partial(objective, features, labels)
    study = optuna.create_study(direction='maximize')
    study.optimize(f,timeout=20000000, n_jobs=20)

    print('params:', study.best_params)


if __name__ == '__main__':
    main()

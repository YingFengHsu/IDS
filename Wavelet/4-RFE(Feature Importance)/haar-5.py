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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
import codecs
import json

import random
#randomseed=random.randint(1,100)

fp=open('/data/lee/newdataset/FullTest-DoS-TrainVali-haar-5.txt')
X=json.load(fp)
fp=open('/data/lee/wavelet/DoS-TrainVali-Labels.txt')
y=json.load(fp)

clf = RandomForestClassifier(max_features='sqrt', n_estimators=256, min_samples_leaf=21, min_samples_split=18, max_leaf_nodes=None, criterion='entropy', max_depth=128, random_state=42, n_jobs=-1)

rfe = RFE(estimator=clf, n_features_to_select=1, step=6)

rfe.fit(X, y)

print(str(rfe.ranking_), file=codecs.open('RFE-Ranking-haar5.txt', 'a', 'utf-8'))

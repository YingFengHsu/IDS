#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
import json

import csv
import optuna
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import numpy as np

from sklearn.metrics import confusion_matrix
with open('/data/lee/newdataset/FullTest-DoS-TrainVali-coif3-2.txt') as f:
    features = json.load(f)
with open('/data/lee/wavelet/DoS-TrainVali-Labels.txt') as f:
    labels = json.load(f)
    
clf = RandomForestClassifier(max_features='sqrt', n_estimators=128, min_samples_leaf=1, min_samples_split=4, max_leaf_nodes=None, criterion='gini', max_depth=None, random_state=42, n_jobs=-1)
clf.fit(features,labels)

with open('/data/lee/newdataset/FullTest-DoS-Test-coif3-2.txt') as f:
    features = json.load(f)
with open('/data/lee/wavelet/DoS-Test-Labels.txt') as f:
    labels = json.load(f)
    
print('coif3-2')
predictedlabels=clf.predict(features)
confu=confusion_matrix(labels,predictedlabels)
confulist=confu.tolist()
print(confu)
print(confulist)
print(classification_report(labels,predictedlabels))

print(accuracy_score(labels,predictedlabels))
precision=confulist[0][0]/(confulist[0][0]+confulist[1][0])
recall=confulist[0][0]/(confulist[0][0]+confulist[0][1])
print(precision)
print(recall)
f2=(5*precision*recall)/(4*precision+recall)
print(f2)
#print(clf.feature_importances_)

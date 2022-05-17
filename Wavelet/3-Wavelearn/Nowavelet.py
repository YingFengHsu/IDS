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
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.metrics import confusion_matrix

inputfile=open('/data/lee/newdataset/DoS-TrainVali.csv')
reader=csv.reader(inputfile)
inputdata=[row for row in reader]
features=[row[:-1] for row in inputdata]
labels=[row[-1] for row in inputdata]
clf = RandomForestClassifier(n_estimators=4, random_state=42, min_samples_split=6, max_leaf_nodes=256, criterion='entropy', max_depth=16, min_samples_leaf=31, max_features = None, n_jobs=-1)
clf.fit(features,labels)

inputfile=open('/data/lee/newdataset/DoS-Test.csv')
reader=csv.reader(inputfile)
inputdata=[row for row in reader]
features=[row[:-1] for row in inputdata]
labels=[row[-1] for row in inputdata]
predictedlabels=clf.predict(features)
print('nowavelet')
confu=confusion_matrix(labels,predictedlabels)
confulist=confu.tolist()
print(confu)
print(classification_report(labels,predictedlabels))

print(accuracy_score(labels,predictedlabels))
precision=confulist[0][0]/(confulist[0][0]+confulist[1][0])
recall=confulist[0][0]/(confulist[0][0]+confulist[0][1])
print(precision)
print(recall)
f2=(5*precision*recall)/(4*precision+recall)
print(f2)
#print(clf.feature_importances_)

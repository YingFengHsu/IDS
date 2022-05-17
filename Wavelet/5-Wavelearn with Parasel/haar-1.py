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
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

support=[1,24,47,16,52,9,57,14,16,6,23,7,28,27,56,12,33,18,21,3,30,4,57,3,19,2,12,13,30,26,35,22,53,25,39,13,35,27,45,19,41,29,54,18,44,17,31,20,36,43,59,25,51,39,49,33,48,32,55,54,63,69,68,66,66,73,72,8,47,17,48,26,43,10,49,11,41,15,46,9,42,10,40,4,34,44,62,53,64,77,75,24,64,37,63,29,65,71,67,75,73,45,59,5,22,7,36,2,20,76,68,70,77,69,74,76,71,70,74,72,67,15,62,14,23,11,51,6,28,8,34,5,38,38,60,21,50,50,58,55,61,42,58,56,61,37,46,60,65,32,31,52,40]

for choosecount in range(1,78):

    with open('/data/lee/newdataset/FullTest-DoS-TrainVali-haar-1.txt') as f:
        features = json.load(f)
    with open('/data/lee/wavelet/DoS-TrainVali-Labels.txt') as f:
        labels = json.load(f)
        
    features_array=np.array(features)
    deletecol=[]
    for i in range(len(support)):
        if support[i]>choosecount:
            deletecol.append(i)
            
    features_array=np.delete(features_array,deletecol,1)
    features=features_array.tolist()
        
    clf = RandomForestClassifier(max_features='sqrt', n_estimators=64, min_samples_leaf=41, min_samples_split=2, max_leaf_nodes=None, criterion='entropy', max_depth=16, random_state=42, n_jobs=-1)
    clf.fit(features,labels)

    with open('/data/lee/newdataset/FullTest-DoS-Test-haar-1.txt') as f:
        features = json.load(f)
    with open('/data/lee/wavelet/DoS-Test-Labels.txt') as f:
        labels = json.load(f)
        
    features_array=np.array(features)
    deletecol=[]
    for i in range(len(support)):
        if support[i]>choosecount:
            deletecol.append(i)
            
    features_array=np.delete(features_array,deletecol,1)
    features=features_array.tolist()

    print("choosecount="+str(choosecount))
    predictedlabels=clf.predict(features)
    confu=confusion_matrix(labels,predictedlabels)
    confulist=confu.tolist()
    print(confu)

    print(accuracy_score(labels,predictedlabels))
    precision=confulist[0][0]/(confulist[0][0]+confulist[1][0])
    recall=confulist[0][0]/(confulist[0][0]+confulist[0][1])
    print(precision)
    print(recall)
    f2=(5*precision*recall)/(4*precision+recall)
    print(f2)
    #print(clf.feature_importances_)

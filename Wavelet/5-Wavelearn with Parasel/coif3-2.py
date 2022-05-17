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

support=[2,37,62,31,10,57,50,4,41,40,16,45,18,7,32,13,11,39,21,4,29,33,5,47,24,21,41,26,4,54,40,1,9,6,3,48,34,2,39,25,7,22,18,20,30,34,23,64,59,21,63,59,20,51,38,11,31,25,32,58,44,30,65,63,22,52,55,15,57,38,24,43,41,43,65,61,34,61,53,25,42,33,36,63,53,37,55,46,15,28,27,77,77,77,76,72,70,75,75,72,10,62,60,8,51,49,10,40,36,39,54,46,3,20,16,13,64,56,11,65,52,7,58,48,6,29,29,13,23,26,12,28,37,66,66,67,12,35,27,14,61,55,12,36,45,74,74,72,66,67,67,14,62,60,6,64,49,5,42,23,2,50,27,75,76,76,71,68,74,68,73,69,73,71,71,69,70,73,70,68,69,17,56,53,18,47,17,5,38,43,9,30,15,8,44,35,3,9,14,17,57,54,8,59,56,16,49,48,19,28,26,19,42,44,24,50,52,22,58,33,32,51,35,19,45,47,31,60,46]

for choosecount in range(1,78):

    with open('/data/lee/newdataset/FullTest-DoS-TrainVali-coif3-2.txt') as f:
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
        
    clf = RandomForestClassifier(max_features='sqrt', n_estimators=128, min_samples_leaf=1, min_samples_split=4, max_leaf_nodes=None, criterion='gini', max_depth=None, random_state=42, n_jobs=-1)
    clf.fit(features,labels)

    with open('/data/lee/newdataset/FullTest-DoS-Test-coif3-2.txt') as f:
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

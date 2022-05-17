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

support=[2,34,50,50,14,51,38,5,47,34,13,35,12,9,15,15,8,50,19,4,48,35,5,56,24,19,43,20,3,53,32,1,20,16,3,37,25,2,49,28,8,28,27,24,58,47,27,56,54,26,42,41,24,37,26,13,36,32,40,48,46,35,64,55,21,33,31,18,19,21,23,46,42,45,65,61,36,59,49,30,57,36,39,65,46,45,63,51,16,39,51,72,69,71,70,73,77,73,76,75,10,60,38,9,48,33,20,57,44,43,64,54,4,17,18,11,65,60,7,64,40,9,63,52,6,49,39,17,43,37,14,47,44,66,66,67,10,33,30,10,62,59,13,54,63,75,75,73,67,66,67,12,62,53,6,61,41,5,45,15,2,28,22,71,69,74,72,69,68,72,68,68,76,76,71,74,74,77,77,70,70,16,57,29,14,41,12,4,42,32,7,11,8,6,40,31,3,26,18,25,62,56,7,53,38,23,58,52,21,52,44,17,61,55,25,55,58,23,29,29,34,60,59,22,22,11,30,31,27]

for choosecount in range(1,78):

    with open('/data/lee/newdataset/FullTest-DoS-TrainVali-sym5-2.txt') as f:
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
        
    clf = RandomForestClassifier(max_features='sqrt', n_estimators=128, min_samples_leaf=1, min_samples_split=8, max_leaf_nodes=None, criterion='entropy', max_depth=16, random_state=42, n_jobs=-1)
    clf.fit(features,labels)

    with open('/data/lee/newdataset/FullTest-DoS-Test-sym5-2.txt') as f:
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

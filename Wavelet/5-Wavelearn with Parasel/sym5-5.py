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

support=[2,25,62,60,63,31,32,19,41,46,47,41,17,3,27,27,34,16,10,11,52,64,54,37,6,20,46,51,43,12,7,8,49,59,59,38,4,1,23,31,40,14,16,8,50,61,63,25,12,9,47,62,58,42,12,3,29,49,53,47,15,2,14,13,19,10,4,3,27,48,47,34,15,2,13,16,37,25,12,4,31,39,28,21,12,21,38,32,43,46,24,8,43,61,65,39,34,8,29,46,69,40,31,10,26,55,64,21,21,6,44,40,48,19,17,18,54,69,64,40,29,27,48,49,60,59,38,8,39,57,62,28,15,14,35,57,61,24,23,7,36,35,47,32,20,35,58,59,68,59,37,22,23,40,56,44,25,25,26,33,51,18,14,28,42,53,64,51,26,30,41,39,56,35,24,6,50,43,33,27,17,75,75,75,69,77,68,68,70,71,71,73,73,73,71,71,76,71,70,10,53,60,57,42,21,17,44,41,39,23,15,3,36,49,50,29,19,22,58,62,61,52,32,2,20,36,21,5,5,6,45,57,57,52,33,11,54,59,64,48,33,11,38,54,61,54,36,7,22,29,36,19,18,5,36,30,26,18,11,3,41,43,42,22,17,69,63,70,72,76,76,7,55,60,50,28,18,4,63,64,70,55,37,7,25,37,40,34,42,76,76,74,76,74,74,65,66,71,70,66,75,3,50,63,65,44,33,5,55,60,60,49,33,7,53,63,52,31,12,4,29,45,50,23,11,69,66,66,77,77,77,68,73,68,65,72,66,65,65,67,73,73,67,70,67,72,77,77,74,72,72,72,67,67,67,66,75,75,74,74,68,14,30,48,56,30,16,19,49,69,58,45,4,2,22,31,32,20,9,18,45,51,45,15,8,10,53,51,53,32,24,2,23,17,9,5,9,11,42,56,61,35,26,5,58,62,62,56,41,6,38,48,44,27,26,13,24,35,34,14,9,13,45,56,47,28,20,9,34,51,52,43,30,13,38,52,55,15,13,22,37,55,54,28,16,6,46,46,58,30,10,20,44,39,57,16,24]

for choosecount in range(1,78):

    with open('/data/lee/newdataset/FullTest-DoS-TrainVali-sym5-5.txt') as f:
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
        
    clf = RandomForestClassifier(max_features='sqrt', n_estimators=32, min_samples_leaf=1, min_samples_split=24, max_leaf_nodes=256, criterion='gini', max_depth=64, random_state=42, n_jobs=-1)
    clf.fit(features,labels)

    with open('/data/lee/newdataset/FullTest-DoS-Test-sym5-5.txt') as f:
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

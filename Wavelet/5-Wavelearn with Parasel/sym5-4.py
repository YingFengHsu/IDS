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

support=[1,24,64,62,31,21,19,53,51,30,29,3,41,47,25,18,17,61,64,36,10,12,51,43,7,4,15,56,63,23,10,2,41,47,22,22,8,62,56,37,12,19,60,55,49,12,3,44,50,37,20,2,14,21,7,9,3,46,55,18,11,2,30,42,32,19,9,33,30,22,15,20,39,44,40,27,15,59,63,47,31,13,55,62,31,20,21,50,50,14,15,9,37,39,28,17,32,62,64,31,33,27,61,65,60,46,11,50,52,40,24,16,54,54,15,14,12,32,41,35,26,38,65,65,64,57,26,56,55,39,30,21,48,45,29,23,25,53,58,52,37,32,51,58,43,34,4,33,43,38,28,71,71,71,76,71,77,69,77,74,70,69,69,75,75,75,14,57,62,47,34,5,49,48,27,17,10,51,48,36,25,35,57,54,41,28,2,33,23,11,9,9,55,63,53,35,6,59,63,48,33,4,47,58,49,34,5,28,37,27,23,7,38,34,29,17,8,43,46,40,26,66,66,67,66,67,5,52,45,24,18,4,61,65,53,38,8,28,43,49,54,73,76,76,73,73,66,66,67,67,67,6,65,60,42,31,11,57,64,50,32,6,61,59,22,6,2,52,56,26,16,70,73,70,70,68,72,68,73,77,77,77,70,69,68,68,72,72,72,72,68,76,76,74,74,75,75,74,74,71,69,13,58,59,25,27,18,63,60,39,6,3,35,46,22,11,13,51,44,12,8,5,48,54,29,23,3,19,20,13,4,19,56,60,42,25,5,59,58,38,30,7,41,53,35,34,13,36,40,16,20,16,42,45,40,29,8,42,45,45,36,14,44,57,17,18,24,52,46,36,26,10,39,44,10,7,21,49,61,24,16]

for choosecount in range(1,78):

    with open('/data/lee/newdataset/FullTest-DoS-TrainVali-sym5-4.txt') as f:
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
        
    clf = RandomForestClassifier(max_features='sqrt', n_estimators=64, min_samples_leaf=1, min_samples_split=24, max_leaf_nodes=None, criterion='entropy', max_depth=32, random_state=42, n_jobs=-1)
    clf.fit(features,labels)

    with open('/data/lee/newdataset/FullTest-DoS-Test-sym5-4.txt') as f:
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

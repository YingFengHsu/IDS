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

support=[2,35,65,51,31,17,58,36,25,3,55,41,24,14,61,29,19,8,45,7,7,11,62,39,5,4,45,32,27,4,65,35,15,17,60,38,11,2,56,49,23,1,21,13,12,2,53,27,17,2,45,43,22,6,32,21,21,17,45,49,34,23,63,57,42,19,60,30,25,10,52,27,20,18,40,33,22,27,64,44,34,29,65,62,48,23,58,33,25,16,57,22,8,13,39,44,31,35,65,63,55,29,61,54,33,25,58,36,24,40,63,56,34,37,59,51,42,12,49,37,30,74,72,70,76,70,74,70,68,68,75,75,75,9,59,46,26,6,47,44,22,9,54,44,28,36,61,55,40,4,28,18,14,9,64,60,39,8,63,55,46,7,61,57,47,5,51,43,34,13,38,28,26,5,39,43,40,66,66,66,67,6,52,30,30,11,62,53,47,10,31,48,56,75,73,74,74,66,67,67,67,5,64,51,41,4,64,53,37,8,57,48,12,3,50,31,15,69,72,76,71,72,71,76,77,77,77,73,69,76,69,71,73,69,72,77,68,71,70,68,73,15,49,41,24,10,60,29,7,3,50,35,19,14,41,18,13,6,50,37,28,3,24,16,11,21,59,54,38,9,59,42,33,15,56,48,43,23,46,32,26,10,50,47,36,12,52,53,42,20,62,19,16,32,54,46,38,18,52,14,16,26,58,20,20]

for choosecount in range(1,78):

    with open('/data/lee/newdataset/FullTest-DoS-TrainVali-sym5-3.txt') as f:
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
        
    clf = RandomForestClassifier(max_features='sqrt', n_estimators=64, min_samples_leaf=1, min_samples_split=20, max_leaf_nodes=None, criterion='entropy', max_depth=None, random_state=42, n_jobs=-1)
    clf.fit(features,labels)

    with open('/data/lee/newdataset/FullTest-DoS-Test-sym5-3.txt') as f:
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

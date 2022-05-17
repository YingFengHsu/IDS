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

support=[2,39,47,34,50,61,57,3,56,35,23,38,59,22,17,7,11,43,20,21,33,38,28,62,54,24,55,42,16,57,60,1,58,65,26,45,49,2,39,24,4,35,13,7,29,15,3,40,38,6,31,32,30,33,25,6,25,9,27,37,44,30,52,11,22,37,17,15,14,15,3,34,26,45,59,44,12,54,32,8,28,30,22,19,47,32,46,43,35,55,41,73,71,68,77,72,73,67,69,69,12,48,34,26,42,28,12,18,9,10,23,10,5,40,27,48,53,54,18,49,50,21,49,50,4,19,39,20,40,31,47,45,27,67,66,74,14,44,52,25,63,41,18,64,52,73,68,75,75,72,72,37,56,64,6,59,36,36,53,61,16,65,56,74,74,76,68,70,75,71,71,70,66,66,77,67,76,77,76,70,69,55,64,46,23,60,51,5,63,20,8,29,7,4,21,17,2,24,13,11,63,51,8,16,13,29,51,61,48,43,33,62,58,65,42,58,62,19,9,31,53,46,57,10,14,5,60,36,41]

for choosecount in range(1,78):

    with open('/data/lee/newdataset/FullTest-DoS-TrainVali-haar-2.txt') as f:
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
        
    clf = RandomForestClassifier(max_features=None, n_estimators=64, min_samples_leaf=1, min_samples_split=6, max_leaf_nodes=None, criterion='entropy', max_depth=64, random_state=42, n_jobs=-1)
    clf.fit(features,labels)

    with open('/data/lee/newdataset/FullTest-DoS-Test-haar-2.txt') as f:
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

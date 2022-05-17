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

support=[2,30,30,30,30,30,29,21,21,29,17,31,31,31,6,12,4,16,32,32,34,34,35,35,35,35,25,35,34,37,18,34,23,38,39,37,39,12,25,40,40,39,22,27,38,33,36,36,43,36,25,24,41,41,42,5,41,42,42,43,43,1,4,14,5,39,19,11,36,44,32,45,46,2,7,47,47,53,8,28,47,28,55,54,47,48,15,25,22,48,44,3,58,18,49,49,50,8,11,50,52,53,57,53,53,55,55,55,56,56,23,22,56,54,12,58,60,59,59,59,28,59,63,59,68,58,58,27,24,61,24,61,62,61,13,18,62,34,64,23,13,64,64,65,26,65,65,65,66,66,66,12,10,26,16,27,20,16,8,13,6,26,17,21,6,71,70,60,72,27,72,72,21,73,9,6,19,74,74,75,75,75,76,76,76,76,76,75,75,74,74,73,31,40,67,29,77,77,77,17,24,38,24,34,31,20,15,12,8,21,41,2,20,48,49,50,46,18,59,61,62,66,65,11,15,23,67,23,2,67,75,70,67,74,72,73,15,47,42,39,40,40,20,43,43,39,38,19,13,9,11,14,5,5,20,19,7,9,12,10,31,41,37,35,33,32,32,32,45,45,46,3,17,7,16,14,3,8,64,53,55,55,28,46,46,15,23,28,49,56,56,49,50,50,50,64,49,48,48,48,46,51,26,51,51,52,52,52,25,20,54,54,54,54,71,71,52,63,63,7,25,51,58,58,45,64,61,61,65,62,62,66,66,69,69,70,70,70,74,77,77,72,72,70,73,73,73,69,69,69,68,68,68,68,68,62,37,60,71,71,60,10,30,14,57,57,57,57,57,51,56,53,28,63,11,17,5,63,22,29,52,44,22,44,37,3,44,27,47,44,26,3,14,2,2,7,4,5,6,21,13,33,71,3,19,40,33,26,37,9,18,18,38,33,63,22,10,15,8,7,6,14,29,9,13,17,41,19,67,16,60,60,36,51,10,33,29,42,36,16,10,9,4,11,4,43,4,24,69,76,77,45,45,38,42,67,27]

for choosecount in range(1,78):

    with open('/data/lee/newdataset/FullTest-DoS-TrainVali-coif3-5.txt') as f:
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
        
    clf = RandomForestClassifier(max_features=None, n_estimators=32, min_samples_leaf=41, min_samples_split=8, max_leaf_nodes=64, criterion='gini', max_depth=128, random_state=42, n_jobs=-1)
    clf.fit(features,labels)

    with open('/data/lee/newdataset/FullTest-DoS-Test-coif3-5.txt') as f:
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

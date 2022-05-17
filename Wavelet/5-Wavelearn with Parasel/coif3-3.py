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

support=[1,37,37,54,28,17,42,59,54,4,26,43,35,18,18,46,12,10,13,20,7,12,27,40,15,3,18,24,25,5,24,48,17,16,35,47,23,4,53,56,26,2,6,12,5,2,51,43,20,2,44,50,30,9,23,20,14,22,44,42,39,24,61,65,58,15,53,62,52,16,33,48,22,11,21,33,25,26,41,55,26,34,64,65,60,15,57,59,52,19,25,45,19,16,30,48,41,44,63,65,56,36,53,61,49,28,35,49,43,38,56,62,55,40,47,55,41,11,29,39,45,71,70,73,71,74,74,74,74,76,71,71,75,11,54,64,61,9,42,55,43,10,51,50,34,36,50,46,31,3,13,14,9,8,62,64,46,7,62,65,42,6,60,63,48,4,32,37,32,14,35,25,30,10,27,39,41,66,66,67,67,6,29,28,18,8,63,64,58,10,27,31,53,72,77,69,72,66,66,67,67,8,57,60,52,4,61,63,40,5,31,44,19,2,45,49,23,68,68,76,68,69,77,77,70,77,75,75,75,70,70,69,72,72,73,73,73,69,68,76,76,13,38,59,51,17,23,38,12,3,29,46,36,9,16,22,7,6,34,40,31,3,5,13,8,19,50,59,58,7,58,60,57,15,36,51,45,17,21,39,24,11,29,47,38,14,33,47,52,22,32,57,33,37,34,56,30,21,21,49,27,20,28,54,32]

for choosecount in range(1,78):

    with open('/data/lee/newdataset/FullTest-DoS-TrainVali-coif3-3.txt') as f:
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
        
    clf = RandomForestClassifier(max_features='sqrt', n_estimators=256, min_samples_leaf=1, min_samples_split=4, max_leaf_nodes=None, criterion='entropy', max_depth=None, random_state=42, n_jobs=-1)
    clf.fit(features,labels)

    with open('/data/lee/newdataset/FullTest-DoS-Test-coif3-3.txt') as f:
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

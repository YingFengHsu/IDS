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

support=[1,23,61,35,33,13,59,56,38,5,59,52,38,8,56,31,11,4,27,9,9,7,33,30,5,3,46,49,37,5,47,44,15,14,48,34,9,3,45,52,24,2,17,28,20,2,42,28,15,2,40,36,27,6,25,21,19,9,46,39,30,21,55,41,27,22,46,32,22,14,37,23,12,11,36,29,20,18,55,40,35,25,58,56,35,16,44,42,22,10,38,15,12,16,34,35,29,24,62,63,49,26,51,51,31,18,48,45,36,21,57,54,26,26,48,46,37,30,62,64,47,67,72,72,73,73,73,72,70,70,67,75,77,8,58,52,32,8,57,42,29,11,50,39,28,6,49,37,20,3,31,33,24,12,53,55,41,7,50,45,29,7,47,59,38,4,27,39,32,14,56,53,53,23,60,63,43,66,66,66,72,16,61,62,63,10,65,64,60,13,59,65,65,76,76,76,68,67,66,67,75,17,65,64,57,4,42,45,28,6,48,34,17,2,41,30,15,75,70,71,76,71,71,71,74,75,74,77,74,68,68,69,77,73,69,69,68,69,74,77,70,12,58,58,40,10,50,44,11,5,54,49,36,4,26,8,6,7,33,39,32,3,22,20,17,18,64,63,50,10,54,51,43,14,62,57,43,19,44,41,31,13,55,53,40,18,60,61,51,16,47,24,19,34,60,61,54,13,43,25,19,23,52,21,25]

for choosecount in range(1,78):

    with open('/data/lee/newdataset/FullTest-DoS-TrainVali-haar-3.txt') as f:
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
        
    clf = RandomForestClassifier(max_features='sqrt', n_estimators=256, min_samples_leaf=1, min_samples_split=4, max_leaf_nodes=None, criterion='entropy', max_depth=64, random_state=42, n_jobs=-1)
    clf.fit(features,labels)

    with open('/data/lee/newdataset/FullTest-DoS-Test-haar-3.txt') as f:
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

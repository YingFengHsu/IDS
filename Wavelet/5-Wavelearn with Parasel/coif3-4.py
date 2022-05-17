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

support=[1,31,55,36,56,22,15,37,40,59,53,2,22,24,45,31,15,33,19,38,10,13,33,12,21,6,11,45,25,35,12,2,17,14,23,17,8,43,26,42,16,12,40,33,44,16,3,47,50,55,21,2,7,7,8,4,3,51,49,47,19,2,36,43,48,25,11,27,19,22,11,25,42,49,45,32,18,57,61,65,56,15,49,58,64,48,18,39,38,55,19,12,20,20,32,23,30,57,34,58,26,31,60,65,65,61,14,52,56,60,54,13,36,34,52,21,10,27,27,48,40,43,63,63,65,57,35,42,47,61,47,33,29,34,46,42,40,54,53,62,53,31,35,41,55,37,8,17,28,38,39,71,68,72,72,73,73,73,72,72,70,74,76,74,74,74,11,55,53,64,58,5,37,35,54,41,9,50,50,48,36,36,52,45,46,24,3,13,10,9,5,7,60,62,64,49,5,62,60,65,41,6,57,58,61,49,4,27,32,39,27,11,34,32,26,20,5,19,25,28,46,66,66,67,67,67,6,32,29,25,15,4,63,64,63,60,7,22,26,39,50,71,69,69,75,77,66,66,66,67,67,3,59,59,61,52,5,62,62,64,38,6,40,21,44,14,3,53,52,43,18,75,72,71,75,77,77,77,77,71,76,76,76,76,75,68,70,70,73,70,73,70,68,74,68,69,69,68,71,69,75,13,41,43,59,51,16,33,16,38,9,4,24,21,45,29,10,24,14,17,4,7,37,37,42,30,2,10,9,12,6,16,44,51,63,54,8,56,56,59,58,8,23,29,51,44,14,15,18,31,18,9,24,30,47,35,13,23,30,46,50,23,46,28,57,30,41,39,34,51,29,20,28,17,48,28,20,44,22,54,26]

for choosecount in range(1,78):

    with open('/data/lee/newdataset/FullTest-DoS-TrainVali-coif3-4.txt') as f:
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
        
    clf = RandomForestClassifier(max_features='sqrt', n_estimators=256, min_samples_leaf=1, min_samples_split=8, max_leaf_nodes=None, criterion='entropy', max_depth=32, random_state=42, n_jobs=-1)
    clf.fit(features,labels)

    with open('/data/lee/newdataset/FullTest-DoS-Test-coif3-4.txt') as f:
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

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

support=[1,22,58,62,59,25,23,14,56,60,60,43,37,3,49,49,51,33,24,10,57,58,46,24,10,5,44,31,16,6,4,7,44,56,32,16,4,2,32,40,51,42,26,7,50,51,44,23,10,11,48,57,43,17,7,2,25,35,45,33,16,2,18,15,15,14,10,2,33,30,35,21,9,2,17,24,39,35,16,3,28,21,21,15,10,8,38,46,40,28,22,13,50,58,61,34,24,9,29,41,49,30,27,9,42,26,43,18,13,4,30,29,33,25,19,19,57,61,57,25,23,19,52,62,61,52,29,13,39,49,50,28,17,11,32,31,36,15,12,4,37,46,41,31,23,21,63,64,64,63,37,15,22,47,55,42,27,18,28,48,56,40,27,23,43,57,59,46,22,13,42,50,53,41,28,8,40,63,64,59,37,69,74,74,69,71,75,75,72,71,71,71,76,73,74,73,69,69,68,12,58,60,64,54,29,5,50,52,48,33,20,5,38,43,40,30,20,12,52,47,36,28,12,2,25,26,17,20,14,12,39,53,60,49,27,6,45,51,54,32,20,8,36,48,55,44,26,5,20,26,34,32,24,9,33,38,46,51,48,9,45,58,64,61,42,73,73,70,66,66,66,11,52,62,63,59,60,7,63,65,65,65,57,7,39,50,60,65,65,74,66,68,72,69,67,76,76,76,76,76,66,3,52,63,65,62,41,4,38,55,55,34,18,5,56,55,53,24,11,3,26,29,43,18,11,77,77,77,70,77,77,67,66,67,68,70,72,75,70,69,73,68,68,71,71,74,67,70,77,73,72,75,75,75,70,72,72,74,67,68,67,14,56,58,62,46,30,10,39,59,37,23,8,3,47,49,54,36,21,6,34,30,19,6,4,6,38,38,47,29,31,3,19,20,16,12,11,14,53,59,64,61,42,7,35,54,62,54,40,6,35,48,55,44,34,8,14,31,32,22,22,5,34,47,53,37,27,9,41,51,61,56,45,13,36,41,54,15,17,17,25,47,53,44,36,8,27,31,45,18,16,13,35,45,39,21,19]

for choosecount in range(1,78):

    with open('/data/lee/newdataset/FullTest-DoS-TrainVali-haar-5.txt') as f:
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
        
    clf = RandomForestClassifier(max_features='sqrt', n_estimators=256, min_samples_leaf=21, min_samples_split=18, max_leaf_nodes=None, criterion='entropy', max_depth=128, random_state=42, n_jobs=-1)
    clf.fit(features,labels)

    with open('/data/lee/newdataset/FullTest-DoS-Test-haar-5.txt') as f:
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

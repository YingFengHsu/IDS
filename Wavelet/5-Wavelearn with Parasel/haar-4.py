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

support=[1,22,58,54,34,31,10,55,54,52,33,3,46,53,44,28,9,58,61,36,9,5,34,37,6,6,4,50,46,17,3,4,48,48,46,29,5,56,51,37,13,9,56,59,31,12,2,32,50,39,19,2,34,25,20,12,3,35,39,25,9,2,28,29,38,17,4,21,24,19,10,7,47,40,30,20,15,56,60,47,26,16,41,51,33,19,10,35,35,18,15,11,31,23,23,16,22,59,57,27,29,20,60,59,50,35,12,39,44,28,21,21,30,29,17,8,14,37,37,32,24,25,63,65,64,43,16,32,55,41,27,20,38,49,45,21,21,53,56,45,26,26,41,43,44,27,18,52,64,55,33,67,66,66,75,73,69,66,67,73,67,67,76,73,68,70,7,59,63,50,28,5,49,53,39,19,11,44,42,30,24,7,47,40,26,13,2,33,30,27,16,6,49,62,57,27,7,52,60,43,29,4,48,56,49,26,5,22,33,34,23,8,48,42,40,51,10,48,62,52,38,68,68,71,71,71,14,51,61,58,53,8,63,64,65,63,13,46,63,65,65,75,75,72,72,76,72,72,69,69,69,18,60,65,62,42,6,42,57,42,17,8,54,57,35,15,2,31,39,23,12,77,74,74,71,69,66,76,68,68,75,72,66,67,70,71,76,76,75,73,70,74,74,74,73,77,77,77,77,70,70,13,53,62,49,32,11,58,61,38,14,4,61,54,45,24,3,38,25,8,5,7,36,40,28,23,3,22,18,16,11,18,64,64,60,51,6,40,55,43,37,9,47,62,41,34,15,31,30,36,19,10,50,54,41,32,13,55,61,52,45,20,46,57,22,14,25,43,59,58,36,15,36,45,11,14,12,47,44,17,24]

for choosecount in range(1,78):

    with open('/data/lee/newdataset/FullTest-DoS-TrainVali-haar-4.txt') as f:
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
        
    clf = RandomForestClassifier(max_features='sqrt', n_estimators=64, min_samples_leaf=1, min_samples_split=26, max_leaf_nodes=None, criterion='entropy', max_depth=32, random_state=42, n_jobs=-1)
    clf.fit(features,labels)

    with open('/data/lee/newdataset/FullTest-DoS-Test-haar-4.txt') as f:
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

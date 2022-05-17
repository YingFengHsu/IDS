import pywt
import numpy as np
import json
import csv

for filename in ['DoS-Test','DoS-TrainVali']:
    inputfile=open('/data/lee/'+filename+'.csv')
    reader=csv.reader(inputfile)
    labels=[row[77] for row in reader]
    print(labels)
    with open(filename+"-Labels.txt", "w") as fp:
        json.dump(labels, fp)

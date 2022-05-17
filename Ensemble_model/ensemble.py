import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras import optimizers
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import  IsolationForest as IF
from sklearn.svm import OneClassSVM, SVC
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score, f1_score
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

train = np.loadtxt("train2.csv", delimiter=',', dtype=np.float32)
test = np.loadtxt("test2.csv", delimiter=',', dtype=np.float32)

x_train = np.vstack((train[train[:,43]==0][:,:42], test[test[:,43]==0][:0,:42]))
x_test = train[train[:, 43]!=0][:, :42]

model = Sequential()
model.add(Dense(20, input_dim=42, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(10, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(20, activation='relu'))
model.add(Dense(42, activation='sigmoid'))

model.compile(optimizer='sgd',
              loss='mean_absolute_error')
loss = model.fit(x_train, x_train,
          epochs=50,
          batch_size=256,
          shuffle=True,
         # validation_split=0.1,
          validation_data=(x_test, x_test))

predtrain = model.predict(train[:,:42],batch_size=256)
predtest = model.predict(test[:,:42],batch_size=256)
losstrain = train[:,:42]-predtrain
losstest = test[:,:42]-predtest

y_train = train[:, 43]
y_test = test[:, 43]

rfc = RFC(n_estimators=100, max_features='auto', n_jobs=-1)
rfc.fit(x_train, y_train)
p=rfc.predict(x_train)
p2=rfc.predict(x_test)

svc=SVC(gamma=10, probability=True)
svc.fit(losstrain, y_train)
p=svc.predict(x_train)
p2=svc.predict(x_test)

cfr = OneClassSVM(kernel='rbf', gamma=10, nu=0.05)
# cfr = IF(n_estimators=500, max_samples=1000, contamination=0.1, max_features=1.0, n_jobs=-1, behaviour='new')
cfr.fit(losstrain[y_train==0])#, y_train)
pred = cfr.predict(losstrain)==-1
pred2 = cfr.predict(losstest)==-1
#print(pred)

print(accuracy_score(y_train, pred))
print(recall_score(y_train, pred))
print(confusion_matrix(y_train, pred))
print(accuracy_score(y_test, pred2))
print(recall_score(y_test, pred2))
print(confusion_matrix(y_test, pred2))

score=cfr.decision_function(losstest)
proba=rfc.predict_proba(x_test)
en2=proba[:,1]

x=range(1,51)
plt.plot(x, loss.history['loss'], label='normal loss')
plt.plot(x, loss.history['val_loss'], label='attack loss')
plt.title('loss')
plt.xlabel('epochs')
plt.ylim(0,0.4)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
plt.savefig('loss.png')

x_train=train[:,:42]
x_test=test[:,:42]

y_train = train[:, 43]
y_test = test[:, 43]

model = Sequential()
model.add(Dense(200, input_dim=42, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dense(1000, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
loss = model.fit(x_train, y_train,
          epochs=50,
          batch_size=256,
          shuffle=True,
         # validation_split=0.1,
          validation_data=(x_test, y_test))

pred=model.predict_classes(x_train, batch_size=256)
pred2=model.predict_classes(x_test, batch_size=256)
print(accuracy_score(y_train, pred))
print(recall_score(y_train, pred))
print(confusion_matrix(y_train, pred))
print(accuracy_score(y_test, pred2))
print(recall_score(y_test, pred2))
print(confusion_matrix(y_test, pred2))

acc={}
recall={}

en1=1.0/(1.0+np.exp(2*score))

for i in range(50,100):
    en = i*en1+(100-i)*en2>50
    acc[i]=accuracy_score(y_test,en)
    recall[i]=recall_score(y_test,en)

for k, v in acc.items():
        print(str(k) + ':' + str(v))


print('-----')

for k, v in recall.items():
    print(str(k) + ':' + str(v))



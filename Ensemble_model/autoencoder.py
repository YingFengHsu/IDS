import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras import optimizers
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

train = np.loadtxt("nsltrain.csv", dtype=np.float32)
testlabel = np.loadtxt("test-5class.csv", delimiter=",", dtype=np.float32)
test = np.loadtxt("nsltest.csv", dtype=np.float32)

x_train = np.vstack((train[train[:,41]==0][:,:41], test[test[:,41]==0][:,:41]))
x_test = test[test[:, 41]!=0][:, :41]

model = Sequential()
model.add(Dense(20, input_dim=41, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(10, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(20, activation='relu'))
model.add(Dense(41, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='mean_absolute_error')
loss = model.fit(x_train, x_train,
          epochs=50,
          batch_size=256,
          shuffle=True,
         # validation_split=0.1,
          validation_data=(x_test, x_test))

predtrain = model.predict(train[:,:41],batch_size=256)
predtest = model.predict(test[:,:41],batch_size=256)
losstrain = train[:,:41]-predtrain
losstest = test[:,:41]-predtest

y_train = train[:, 41]
y_test = test[:, 41]

cfr = OneClassSVM(kernel='rbf', gamma=100, nu=0.05)
cfr.fit(losstrain[y_train==0])#, y_train)
pred = cfr.predict(losstrain)==-1
t1 = time.time()
pred2 = cfr.predict(losstest)==-1
t = time.time() - t1

#print(pred)

print(accuracy_score(y_train, pred))
print(recall_score(y_train, pred))
print(confusion_matrix(y_train, pred))
print(accuracy_score(y_test, pred2))
print(recall_score(y_test, pred2))
print(confusion_matrix(y_test, pred2))

# x=range(1,101)
# plt.plot(x, loss.history['loss'], label='train loss')
# plt.plot(x, loss.history['val_loss'], label='val loss')
# plt.title('loss')
# plt.xlabel('epochs')
# plt.ylim(0,0.2)
# plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
# plt.savefig('loss2.png')

train = np.loadtxt("nsltrain.csv", dtype=np.float32)
test = np.loadtxt("nsltest.csv", dtype=np.float32)
label = np.genfromtxt("test-5class.txt", delimiter=",", dtype=np.float32)

y_train = train[:, 41]
y_test = test[:, 41]
label = label[:, 41]

x_train = np.vstack((train[train[:,41]==0][:,:41], test[test[:,41]==0][:0,:41]))

cfr = OneClassSVM(kernel='rbf', gamma=100, nu=0.05)
t1 = time.time()
# cfr.fit(x_train)
pred=cfr.predict(test[:, :41])==-1
t = time.time() - t1
pred=cfr.predict(test[:, :41])==-1
accuracy_score(y_test, pred)
recall_score(y_test, pred)
confusion_matrix()
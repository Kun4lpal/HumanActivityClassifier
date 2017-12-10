##############################################################################################################################################
# AUTHOR: KUNAL PALIWAL
# EMAIL ID: kupaliwa@syr.edu
# COURSE: ARTIFICAL NEURAL NETWORKS 
# Assignment 2
#
# This implementation uses Keras. It allowed me to design my own network and train a model for achieving high accuracy of 98 %.
# I believe testing this dataset using Keras helped me in understanding how to design my own network from scratch using only numpy and pandas
# as dependencies
##############################################################################################################################################
import keras
from keras.models import Sequential
from numpy import genfromtxt
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np

dataframe  = genfromtxt('X_train.csv', delimiter=',')
roomframe  = genfromtxt('X_train.csv', delimiter=',')
dataframe = np.delete(dataframe, [561], axis=1)
X = dataframe
# print('Testing the shape of X: ',X.shape)
# print("v2",X)
X = X.astype('float64')
y = roomframe[:,561].reshape(7352,1)
y=y-1
print(y.shape)
print(np.random.randint(10, size=(1000, 1)).shape)
y = keras.utils.to_categorical(y.astype('int64'), num_classes=6)

x_train = X
y_train = y

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=561))
# model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))
# stochastic gradient descent
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)

# score = model.evaluate(x_test, y_test, batch_size=128)
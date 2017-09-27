import unittest
import sys
import os
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path = [os.path.dirname(os.path.dirname(os.path.dirname(curr_path))), curr_path] + sys.path
os.environ['is_test_suite'] = 'True'
curr_path = None
import numpy as np
np.random.seed(1337) # for reproducibility
import logging
from auto_deepnet.models import BasicClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical

logger = logging.getLogger("auto_deepnet")
logger.setLevel(logging.CRITICAL)

nn = BasicClassifier(dropout=1.0, epochs=10, verbose=1, layers=[(2., 'tanh'), (2., 'tanh'), (1.5, 'relu')])
X = np.random.randn(100000, 5)
Y = np.argmax(X, axis=1)
nn.fit(X, Y)
nn.save_adn_model()

model = BasicClassifier()
model = Sequential()
model.add(Dense(5, input_dim=5))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))
X = np.random.randn(60000, 5)
Y = to_categorical(np.argmax(X, axis=1), 5)
X_dev = np.random.randn(20000, 5)
Y_dev = to_categorical(np.argmax(X_dev, axis=1), 5)
X_test = np.random.randn(20000, 5)
Y_test = to_categorical(np.argmax(X_test, axis=1))
X_predict = np.random.randn(2000, 5)
Y_predict = np.argmax(X_predict, axis=1)
model.compile(loss='categorical_crossentropy', optimizer='adagrad')
model.fit(X, Y, batch_size=256, validation_data=(X_dev, Y_dev), epochs=20, verbose=1)
print(model.test_on_batch(X_test, Y_test))
print(np.sum(np.argmax(model.predict(X_predict), axis=1) - Y_predict))
correct = 0.453789

from models.basic_classifier import *
import numpy as np

nn = BasicClassifier(dropout=1.0, epochs=10, verbose=1, layers=[(2., 'tanh'), (2., 'tanh'), (1.5, 'relu')])
X = np.random.randn(100000, 5)
Y = np.argmax(X, axis=1)
nn.fit(X, Y)



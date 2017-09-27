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

class TestBase(unittest.TestCase):

    def setUp(self):
        self.config = {
                'batch_size': 32,
                'epochs': 10,
                'dropout': 1.,
                'optimizer': 'adagrad',
                'loss': 'categorical_crossentropy',
                'metrics': ['accuracy'],
                'verbose': 0,
                'model_checkpoint_path': './model_checkpoint.hdf5',
                'model_path': './model.tgz'
                }
        self.nn = BasicClassifier(**self.config)

    def tearDown(self):
        self.nn = None
        if os.path.isfile(self.config['model_path']):
            os.remove(self.config['model_path'])
        if os.path.isfile(self.config['model_checkpoint_path']):
            os.remove(self.config['model_checkpoint_path'])

    def test_sample_dataset_training(self):
        X_train = np.random.randn(10000, 5)
        Y_train = np.argmax(X_train, axis=1)
        self.nn.fit(X_train, Y_train)
        X_test = np.random.randn(1000, 5)
        Y_test = np.argmax(X_test, axis=1)
        np.testing.assert_allclose([0.46882305932044982, 0.87], self.nn.model.evaluate(X_test, Y_test))


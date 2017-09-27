import unittest
import sys
import os
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path = [os.path.dirname(os.path.dirname(os.path.dirname(curr_path))), curr_path] + sys.path
os.environ['is_test_suite'] = 'True'
curr_path = None
import numpy as np
import logging
from auto_deepnet.models import Base
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

logger = logging.getLogger("auto_deepnet")
logger.setLevel(logging.DEBUG)

class TestBase(unittest.TestCase):

    def setUp(self):
        self.correct_config = {
                'input_shape': (None, 256),
                'output_shape': (None, 5),
                'batch_size': 512,
                'epochs': 50,
                'prediction_batch_size': 16,
                'dropout': 0.75,
                'optimizer': 'adagrad',
                'loss': 'categorical_crossentropy',
                'metrics': ['accuracy'],
                'verbose': True,
                'model_checkpoint_path': './model_checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5',
                'model_path': './model.tar'
                }
        kwargs = self.correct_config.copy()
        self.base = Base(**kwargs)

    def tearDown(self):
        self.base = None

    def test_config(self):
        self.assertDictEqual(self.base.get_config(), self.correct_config)
    
    def test_custom_options(self):
        self.correct_config['batch_size'] = 87
        self.correct_config['foo'] = 'bar'
        self.base = Base(**self.correct_config)
        self.assertDictEqual(self.base.get_config(), self.correct_config)

    def test_update_options(self):
        self.base.update_config(**{'input_shape': 65, 'foo': 'bar'})
        self.correct_config['input_shape'] = 65
        self.correct_config['foo'] = 'bar'
        self.assertDictEqual(self.base.get_config(), self.correct_config)

    def test_generate_kwargs(self):
        kwargs = self.base._generate_kwargs('input_shape', 'batch_size', **{'input_shape': 5})
        self.assertDictEqual(kwargs, {'input_shape': 5, 'batch_size': self.correct_config['batch_size']})

    def test_save_load_adn_model(self):
        model = Sequential()
        model.add(Dense(5, input_dim=5))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adagrad')
        self.base.model = model
        self.base.save_adn_model(self.correct_config['model_path'])
        self.base = None
        self.base = Base()
        self.base.load_adn_model(self.correct_config['model_path'])
        os.remove(self.correct_config['model_path'])
        self.assertDictEqual(self.base.get_config(), self.correct_config)

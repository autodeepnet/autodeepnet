from __future__ import absolute_import, division, print_function
import sys
import os
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path = [os.path.dirname(os.path.dirname(curr_path)), curr_path] + sys.path
curr_path = None
from auto_deepnet.utils import data_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import logging

logger = logging.getLogger("auto_deepnet")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def generate_config(**kwargs):
    return kwargs


class Base(object):
    def __init__(self,
                 input_dim,
                 layers=[(1., 'relu'), (1., 'relu')],
                 batch_size=512,
                 epochs=50,
                 prediction_batch_size=16,
                 dropout=0.75,
                 verbose=True,
                 **kwargs):
        self.config = {
                generate_config(
                    input_dim=input_dim,
                    layers=layers,
                    batch_size=batch_size,
                    epochs=epochs,
                    prediction_batch_size=prediction_batch_size,
                    verbose=verbose
                )}
    def build_model(self):
        self.model = Sequential()
        input_dim = self.config['input_dim']
        for i, layer in enumerate(self.config['layers']):
            if i == 0:
                model.add(Dense(layer[0]*input_dim, input_dim=input_dim))
            else:
                model.add(Dense(layer[0]*input_dim))
            model.add(Activation(layer[1]))
            model.add(Dropout(self.config['dropout']))


        


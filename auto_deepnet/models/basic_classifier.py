from __future__ import absolute_import, division, print_function
import sys
import os
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path = [os.path.dirname(os.path.dirname(curr_path)), curr_path] + sys.path
curr_path = None
from auto_deepnet.utils import data_transform_utils, data_io_utils
import auto_deepnet.utils.exceptions as exceptions
from auto_deepnet.models import Base
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
import numpy as np
import logging

logger = logging.getLogger("auto_deepnet")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class BasicClassifier(Base):
    def __init__(self, layers=[(1., 'relu'), (1., 'relu')], data_input_pipeline='basic_classifier_input_pipeline', data_output_pipeline='basic_classifier_output_pipeline', **kwargs):
        super(BasicClassifier, self).__init__(layers=layers, data_input_pipeline=data_input_pipeline, data_output_pipeline=data_output_pipeline, **kwargs)

    def build_model(self, **kwargs):
        if self.config['data_info']['data_type'] == 'sparse':
            self.config['data_info']['d_y'] = kwargs.pop('num_classes', self.config['data_info']['d_y'])
        if self.config['data_info']['data_type'] == 'sparse' and self.config['loss'] == 'categorical_crossentropy':
            logger.info("Dataset is sparse, changing loss to sparse categorical crossentropy")
            self.config['loss'] = 'sparse_categorical_crossentropy'
        d_x = self.config['data_info'].get('d_x', None)
        d_y = self.config['data_info'].get('d_y', None)
        if not d_x:
            logger.error("Need an input dimension!")
            raise Exception
        if not d_y:
            logger.error("Need the num_classes!")
            raise Exception
        self.model = Sequential()
        for i, layer in enumerate(self.config['layers']):
            if i == 0:
                self.model.add(Dense(int(layer[0]*d_x), input_dim=d_x))
            else:
                self.model.add(Dense(int(layer[0]*d_x)))
            self.model.add(Activation(layer[1]))
            self.model.add(Dropout(self.config['dropout']))
        self.model.add(Dense(d_y))
        self.model.add(Activation('softmax'))
        self.model.compile(optimizer=self.config['optimizer'],
                      loss=self.config['loss'],
                      metrics=self.config['metrics'])
        plot_model(self.model, to_file='model.png')


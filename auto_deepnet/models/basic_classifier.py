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

    def _build_model(self, hyperparameters=self.__hyperparameters, **kwargs):
        if self.__config['data_info']['data_type'] == 'sparse':
            self.__config['data_info']['d_y'] = kwargs.pop('num_classes', self.__config['data_info']['d_y'])
        if self.__config['data_info']['data_type'] == 'sparse' and self.__config['loss'] == 'categorical_crossentropy':
            logger.info("Dataset is sparse, changing loss to sparse categorical crossentropy")
            self.__config['loss'] = 'sparse_categorical_crossentropy'
        d_x = self.__config['data_info'].get('d_x', None)
        d_y = self.__config['data_info'].get('d_y', None)
        if not d_x:
            logger.error("Need an input dimension!")
            raise Exception
        if not d_y:
            logger.error("Need the num_classes!")
            raise Exception
        self.__model = Sequential()
        for i, layer in enumerate(self.__config['layers']):
            if i == 0:
                self.__model.add(Dense(int(layer[0]*d_x), input_dim=d_x))
            else:
                self.__model.add(Dense(int(layer[0]*d_x)))
            self.__model.add(Activation(layer[1]))
            self.__model.add(Dropout(self.__config['dropout']))
        self.__model.add(Dense(d_y))
        self.__model.add(Activation('softmax'))
        self.__model.compile(optimizer=self.__config['optimizer'],
                      loss=self.__config['loss'],
                      metrics=self.__config['metrics'])


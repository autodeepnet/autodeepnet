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
    def __init__(self, layers=[(1., 'relu'), (1., 'relu')], **kwargs):
        super(BasicClassifier, self).__init__(layers=layers, **kwargs)

    def build_model(self, input_shape=None, num_classes=None, **kwargs):
        if not input_shape:
            logger.error("Need an input shape!")
            raise Exception
        if not num_classes:
            logger.error("Need the num_classes!")
            raise Exception
        self.model = Sequential()
        input_dim = input_shape[1]
        for i, layer in enumerate(self.config['layers']):
            if i == 0:
                self.model.add(Dense(int(layer[0]*input_dim), input_dim=input_dim))
            else:
                self.model.add(Dense(int(layer[0]*input_dim)))
            self.model.add(Activation(layer[1]))
            self.model.add(Dropout(self.config['dropout']))
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))
        self.model.compile(optimizer=self.config['optimizer'],
                      loss=self.config['loss'],
                      metrics=self.config['metrics'])
        plot_model(self.model, to_file='model.png')


    def fit(self, X, Y, **kwargs):
        self.data_pipeline = kwargs.pop('data_pipeline', data_transform_utils.basic_classifier_input_pipeline)
        try:
            X, Y, self.data_info = self.data_pipeline(X, Y)
        except Exception as e:
            logger.exception("Error with data pipeline")
            raise exceptions.DataTransformError("Error with data pipeline")
        input_dims = X.shape[1]
        if self.data_info['data_type'] == 'sparse':
            num_classes = kwargs.pop('num_classes', np.max(Y) + 1)
        else:
            num_classes = Y.shape[1]
        if self.data_info['data_type'] == 'sparse' and self.config['loss'] == 'categorical_crossentropy':
            logger.info("Dataset is sparse, changing loss to sparse categorical crossentropy")
            self.config['loss'] = 'sparse_categorical_crossentropy'
        self.build_model(input_shape=X.shape, output_shape=Y.shape, num_classes=num_classes)
        if 'callbacks' not in kwargs:
            kwargs['callbacks'] = [
                ModelCheckpoint(
                    self.config['model_path'],
                    monitor='val_loss',
                    verbose=self.config['verbose'],
                    save_best_only=True,
                    save_weights_only=False)]
        kwargs = self._generate_kwargs('batch_size', 'epochs', 'verbose', **kwargs)
        kwargs['validation_split'] = kwargs.get('validation_split', 0.2)
        return self.model.fit(X, Y, **kwargs)

    def predict(self, X, **kwargs):
        kwargs = self._generate_kwargs('verbose', **kwargs)
        kwargs['batch_size'] = kwargs.get('batch_size', self.config['prediction_batch_size'])
        return self.model.predict(X, **kwargs)



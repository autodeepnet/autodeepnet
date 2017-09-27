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

    def build_model(self, **kwargs):
        d_x = self.config.get('d_x', None)
        d_y = self.config.get('d_y', None)
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


    def fit(self, X, Y, **kwargs):

        self.config['data_pipeline'] = kwargs.pop('data_pipeline', 'basic_classifier_input_pipeline')
        try:
            X, Y, self.config['data_info'] = getattr(data_transform_utils, self.config['data_pipeline'])(X, Y)
        except Exception as e:
            logger.exception("Error with data pipeline")
            raise exceptions.DataTransformError("Error with data pipeline")
        self.config['d_x'] = X.shape[1]
        if self.config['data_info']['data_type'] == 'sparse':
            self.config['d_y'] = kwargs.pop('num_classes', np.max(Y) + 1)
        else:
            self.config['d_y'] = Y.shape[1]
        if self.config['data_info']['data_type'] == 'sparse' and self.config['loss'] == 'categorical_crossentropy':
            logger.info("Dataset is sparse, changing loss to sparse categorical crossentropy")
            self.config['loss'] = 'sparse_categorical_crossentropy'
        self.build_model()
        if 'callbacks' not in kwargs:
            kwargs['callbacks'] = [
                ModelCheckpoint(
                    self.config['model_checkpoint_path'],
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



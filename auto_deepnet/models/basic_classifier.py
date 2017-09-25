from __future__ import absolute_import, division, print_function
import sys
import os
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path = [os.path.dirname(os.path.dirname(curr_path)), curr_path] + sys.path
curr_path = None
from auto_deepnet.utils import data_utils
from auto_deepnet.models import Base
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
import logging

logger = logging.getLogger("auto_deepnet")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class BasicClassifier(Base):
    def __init__(self, layers=[(1., 'relu'), (1., 'relu')], **kwargs):
        super(BasicClassifier, self).__init__(layers=layers, num_classes=num_classes, **kwargs)

    def build_model(self):
        self.model = Sequential()
        input_shape = self.config['input_shape']
        input_dim = input_shape[1]
        if len(input_shape) > 2:
            for dim in input_shape[2:]:
                input_dim *= dim
            model.add(Flatten())
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

    def fit(X, Y, **kwargs):
        if Y.shape[1] == 1 and np.max(Y) > 1:
            Y = to_categorical(Y, num_classes=self.config['num_classes'])

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

    def predict(X, **kwargs):
        kwargs = self._generate_kwargs('verbose', **kwargs)
        kwargs['batch_size'] = kwargs.get('batch_size', self.config['prediction_batch_size'])
        return self.model.predict(X, **kwargs)



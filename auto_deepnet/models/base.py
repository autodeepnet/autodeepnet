from __future__ import absolute_import, division, print_function
import sys
import os
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path = [os.path.dirname(os.path.dirname(curr_path)), curr_path] + sys.path
curr_path = None
from auto_deepnet.utils import data_io_utils, exceptions
import logging
import tarfile
import pandas as pd
from keras.models import load_model

logger = logging.getLogger("auto_deepnet")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class Base(object):
    def __init__(self,
                 batch_size=512,
                 prediction_batch_size=16,
                 epochs=50,
                 dropout=0.75,
                 optimizer='adagrad',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'],
                 verbose=True,
                 model_checkpoint_path='./model/model_checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5',
                 model_path='./model/model.tar',
                 **kwargs):
        self.config = self._generate_config(
            batch_size=batch_size,
            prediction_batch_size=prediction_batch_size,
            epochs=epochs,
            dropout=dropout,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            verbose=verbose,
            model_path=model_path,
            model_checkpoint_path=model_checkpoint_path,
            **kwargs)
        data_io_utils.verify_dir(model_checkpoint_path)
        data_io_utils.verify_dir(model_path)

    def update_config(self, **kwargs):
        self.config.update(kwargs)

    def _generate_config(self, **kwargs):
        return kwargs

    def _generate_kwargs(self, *args, **kwargs):
        for arg in args:
            kwargs[arg] = kwargs.get(arg, self.config[arg])
        return kwargs

    def get_config(self):
        return self.config

    def build_model(self, **kwargs):
        pass

    def fit(self, X, Y, **kwargs):
        pass

    def predict(self, X, **kwargs):
        pass

    def get_adn_model(self):
        return self.config, self.model

    def save_adn_model(self, model_path=None, **kwargs):
        if not model_path:
            model_path = self.config['model_path']
        data_io_utils.save_adn_model(model_path, self.get_adn_model())

    def load_adn_model(self, model_path=None, **kwargs):
        if not model_path:
            model_path = self.config['model_path']
        adn_model = data_io_utils.load_adn_model(model_path)
        if not adn_model:
            return
        try:
            self.config, self.model = adn_model
        except Exception as e:
            logger.exception("Error loading data from adn model: {}".format(e))

from __future__ import absolute_import, division, print_function
import sys
import os
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path = [os.path.dirname(os.path.dirname(curr_path)), curr_path] + sys.path
curr_path = None
from auto_deepnet.utils import data_utils
import logging

logger = logging.getLogger("auto_deepnet")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class Base(object):
    def __init__(self,
                 input_dim,
                 batch_size=512,
                 epochs=50,
                 prediction_batch_size=16,
                 dropout=0.75,
                 optimizer='adagrad',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'],
                 verbose=True,
                 model_path='./model_checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5'
                 **kwargs):
        self.config = {
            self.generate_config(
                input_dim=input_dim,
                batch_size=batch_size,
                epochs=epochs,
                prediction_batch_size=prediction_batch_size,
                verbose=verbose,
                **kwargs)}
        self.build_model()

    def _generate_config(self, **kwargs):
        return kwargs

    def _generate_kwargs(self, *args, **kwargs):
        for arg in args:
            kwargs[arg] = kwargs.get(arg, self.config[arg])
        return kwargs

    def build_model(self):
        pass

    def fit(X, Y, **kwargs):
        pass

    def predict(X, **kwargs):
        pass

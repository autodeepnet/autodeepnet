from __future__ import absolute_import, division, print_function
import sys
import os
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path = [os.path.dirname(os.path.dirname(curr_path)), curr_path] + sys.path
curr_path = None

import logging
import numpy as np
import pandas as pd
import re
import auto_deepnet.utils.exceptions as exceptions
from keras.utils.np_utils import to_categorical

logger = logging.getLogger("auto_deepnet")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def get_2D_tensor(data):
    shape = data.shape
    if len(shape) == 0:
        logger.error("Data is a 0D Tensor!!")
        raise exceptions.DataTransformError("Data has no batch-size")
    if len(shape) == 1:
        return np.expand_dims(data, axis=1)
    elif len(data.shape) == 2:
        return data
    else:
        return np.squeeze(data, axis=[i for i in range(2, len(shape))])


def get_numpy_array(data):
    if isinstance(data, pd.DataFrame):
        return data.values
    elif isinstance(data, np.ndarray):
        return data
    else:
        logger.info("Invalid data!")
        return data


def is_flat(data):
    return not len(data.shape) > 2


def get_flattened_shape(data):
    input_dims = data.shape[1]
    if not is_flat(data):
        for dims in data.shape[2:]:
            input_dims *= dims
    return (data.shape[0], input_dims)


def is_sparse(data):
    return data.shape[1] == 1


def onehot_to_sparse(data):
    try:
        return np.argmax(get_2D_tensor(data), axis=1)
    except Exception as e:
        logger.exception("Error transforming to sparse vectors: {}".format(e))
        raise exceptions.DataTransformError("Error transforming to sparse")


def base_input_pipeline(X, Y=None):
    data_info = {}
    data_info['d_x'] = get_flattened_shape(X)[1]
    data_info['d_y'] = get_flattened_shape(Y)[1]
    return X, Y, data_info

def base_ouput_pipeline(Y, data_info):
    return Y


def basic_classifier_input_pipeline(X, Y=None):
    try:
        X = get_2D_tensor(get_numpy_array(X))
        Y = get_numpy_array(Y)
        data_info = {}
        data_info['d_x'] = X.shape[1]
        if isinstance(Y, np.ndarray):
            logger.info("Y given, setting as training/testing")
            Y = get_2D_tensor(Y)
            if X.shape[0] != Y.shape[0]:
                raise exceptions.DataTransformError("Batch size mismatch")
            if is_sparse(Y):
                data_info['data_type'] = 'sparse'
                data_info['d_y'] = np.max(Y) + 1
            else:
                data_info['data_type'] = 'one-hot'
                data_info['d_y'] = Y.shape[1]
        else:
            logger.info("No Y given, setting as predicting")
        return X, Y, data_info
    except Exception as e:
        logger.error("Error with data pipeline: {}".format(e))
        raise exceptions.DataTransformError("Error in data pipeline")

def basic_classifier_output_pipeline(Y, data_info):
    if 'data_type' not in data_info:
        logger.error("data type not in config! assuming sparse output")
    try:
        sparse_output = np.argmax(Y, axis=1)
        if data_info.get('data_type', 'sparse') == 'sparse':
            return sparse_output
        else:
            return to_categorical(Y, data_info.get('d_y', None))
    except Exception as e:
        logger.error("Error with data pipeline: {}".format(e))
        raise exceptions.DataTranformError("Error in data pipeline")

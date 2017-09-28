from __future__ import absolute_import, division, print_function
import sys
import os
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path = [os.path.dirname(os.path.dirname(curr_path)), curr_path] + sys.path
curr_path = None
try:
    import cPickle as pickle
except:
    import pickle
import logging
import csv
import h5py
import numpy as np
import pandas as pd
import auto_deepnet.utils.exceptions as exceptions
from keras.models import load_model
import tarfile

logger = logging.getLogger("auto_deepnet")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

'''
function: verify_dir
inputs:
    - file_path: string pathname to verify
description:
    helper function to ensure that the directory of given file path exists, and creates one if it doesn't
'''
def verify_dir(file_path):
    try:
        dir_name, file_name = os.path.split(file_path)
    except Exception as e:
        logger.error("Error with file path {}: {}".format(file_path, e))
        raise exceptions.FileSaveError("Invalid file path")
    if len(dir_name) > 0 and not os.path.isdir(dir_name):
        logger.info("Directory {} does not exist. Creating...".format(dir_name))
        os.makedirs(dir_name)


'''
function: save_pickle_data
inputs:
    - file_path: string pathname to save data to
    - data_frame: pandas data_frame to save to disk in any picklable format
    - pandas_format (optional): whether to save as a pandas dataframe or as a numpy array
    - data_is_pandas (optional): whether the data to save is a pandas dataframe or not
    - append (optional): whether to append data to preexisting data. Requires data to be in the same format
    - mode (optional): The mode to open file as
description:
    helper function to save any data to disk via pickling
'''
def save_pickle_data(file_path, data_frame, **kwargs):
    logger.info("Opening pickle file {} to write data...".format(file_path))
    pandas_format = kwargs.get('pandas_format', True)
    data_is_pandas = kwargs.get('data_is_pandas', True)
    append = kwargs.get('append', False)
    mode = kwargs.get('mode', 'wb')
    if append and os.path.isfile(file_path) and data_is_pandas:
        logger.info("Opening file to append data...")
        try:
            data_frame = pd.concat((load_pickle_data(file_path), data_frame))
        except Exception as e:
            logger.error("Error appending data from {}: {}".format(file_path), e)
    if not data_is_pandas:
        with open(file_path, mode) as f:
            pickle.dump(data_frame, f)
    elif pandas_format:
        data_frame.to_pickle(file_path)
    else:
        with open(file_path, mode) as f:
            pickle.dump(data_frame.values, f)
    logger.info("Successfully saved pickle data")


'''
function: load_pickle_data
inputs:
    - file_path: string pathname to load data from
    - mode: the mode to open file as
description:
    helper function to load any pickled data from disk
'''
def load_pickle_data(file_path, **kwargs):
    mode = kwargs.get('mode', 'rb')
    logger.info("Opening pickle file {} to read...".format(file_path))
    with open(file_path, mode) as f:
        data = pickle.load(f)
    logger.info("Successfully read pickle data")
    return data


'''
function: save_hdf5_data
inputs:
    - file_path: string pathname to save data to
    - data_frame: the pandas dataframe to save to disk
    - key (optional): The name to call the dataset
    - pandas_format (optional): whether to save as a pandas structure or default hdf5
    - mode (optional): The mode to open file as
    - format (optional): whether to save as a table or fixed dataset
    - append (optional): Whether data should be appended or replaced
description:
    helper function to save pandas dataframe to an hdf5 file
'''
def save_hdf5_data(file_path, data_frame, **kwargs):
    pandas_format = kwargs.get('pandas_format', True)
    key = kwargs.get('key', 'data')
    mode = kwargs.get('mode', 'a')
    format = kwargs.get('format', 'table')
    append = kwargs.get('append', False)
    logger.info("Opening HDF5 file {} to write data...".format(file_path))
    if pandas_format:
        with pd.HDFStore(file_path, mode=mode) as f:
            if key in f and not append:
                f.remove(key)
            f.put(key=key, value=data_frame, format=format, append=append)
    else:
        if key == None:
            logger.error("Need a key when saving as default HDF5 format")
            raise exceptions.FileSaveError
        with h5py.File(file_path, mode) as f:
            if key in f:
                if append:
                    data_frame = pd.concat((pd.DataFrame(f[key]), data_frame))
                del f[key]
            f.create_dataset(key, data=data_frame.values)
    logger.info("Successfully saved hdf5 data")


'''
function: load_hdf5_file
inputs:
    - file_path: string pathname to load data from
    - key (optional): name of the dataset
    - pandas_format (optional): whether the file was saved in pandas format
    - mode (optional): The mode to open the file as
description:
    helper function to load an hdf5 file from disk
'''
def load_hdf5_data(file_path, **kwargs):
    key = kwargs.get('key', None)
    pandas_format = kwargs.get('pandas_format', True)
    mode = kwargs.get('mode', 'r')
    logger.info("Opening HDF5 file {} to read...".format(file_path))
    if pandas_format:
        data = pd.read_hdf(file_path, key=key, mode=mode)
    else:
        with h5py.File(file_path, mode) as f:
            data = f[key][()]
    logger.info("Successfully loaded HDF5 data")
    return data


'''
function: save_csv_data
inputs:
    - file_path: string pathname to load data from
    - data_frame: pandas data to save to csv
    - append: whether to append to preexisting data
    - mode (optional): The mode to open the file as
other inputs:
    - any inputs to pd.DataFrame.to_csv() (optional)
description:
    - helper function to save pandas dataframe to csv
'''
def save_csv_data(file_path, data_frame, **kwargs):
    logger.info("Opening CSV file {} to write data".format(file_path))
    if 'index' not in kwargs:
        kwargs['index'] = False
    if 'mode' not in kwargs:
        kwargs['mode'] = 'w'
    append = kwargs.pop('append', False)
    kwargs.pop('pandas_format', None)
    kwargs.pop('format', None)
    if append:
        data_frame.to_csv(file_path, index=False, mode='a', header=False)
    else:
        data_frame.to_csv(file_path, **kwargs)
    logger.info("Successfully saved CSV data")


'''
function: load_csv_data
inputs:
    - file_path: string pathname to load data from
other inputs:
    - any inputs used by pd.read_csv() (optional)
description:
    helper function to load csv data to a pandas dataframe
'''
def load_csv_data(file_path, **kwargs):
    kwargs.pop('pandas_format', None)
    kwargs.pop('mode', None)
    logger.info("Opening CSV file {} to read...".format(file_path))
    data = pd.read_csv(file_path, **kwargs)
    logger.info("Successfully loaded CSV data")
    return data


'''
function: save_data
inputs:
    - file_path: string pathname to save data to
    - data_frame: data to save to disk
    - save_format (optional): format to save to disk
    - overwrite (optional): whether to overwrite preexisting data
    - mode (optional): mode to open file in
    - key: The name to save the data as (required if hdf5 format, deprecated otherwise)
    - pandas_format (optional): whether to save as a pandas dataframe or as a numpy array
    - append (optional): whether to append data
additional inputs:
    - Any inputs that can be used by other saver functions
'''
def save_data(file_path, data_frame, save_format='hdf5', overwrite=False, mode='a', **kwargs):
    if 'key' not in kwargs and save_format == 'hdf5':
        logger.warning("No key specified, defaulting to 'data'")
        kwargs['key'] = 'data'
    if save_format != 'csv':
        if 'pandas_format' not in kwargs:
            kwargs['pandas_format'] = True
        if 'format' not in kwargs:
            kwargs['format'] = 'table'
    if 'append' not in kwargs:
        kwargs['append'] = False
    if 'index' not in kwargs:
        kwargs['index'] = False
    logger.info("Attempting to save data to {}...".format(file_path))
    try:
        verify_dir(file_path)
        if os.path.isfile(file_path):
            if not overwrite:
                logger.error("File {} already exists.".format(file_path))
                raise exceptions.FileSaveError
            if (mode == 'w' or save_format == 'pickle'):
                logger.warning("File {} will be overwritten".format(file_path))
                os.remove(file_path)
        if (mode == 'a' and save_format == 'pickle'):
            logger.warning("Can't use mode='a' for writing to pickle files. using mode='wb' instead...")
            mode = 'wb'
        
        saver = {
            'hdf5': save_hdf5_data,
            'csv': save_csv_data,
            'pickle': save_pickle_data
        }
        saver.get(save_format, save_hdf5_data)(file_path, data_frame, mode=mode, **kwargs)
    except Exception as e:
        logger.exception("Error saving file {}: {}".format(file_path, e))


'''
function: load_data
inputs:
    - file_path: string pathname to load data from
    - load_format: format to load data as
additional inputs:
    - any inputs used by other loader functions
'''
def load_data(file_path, load_format='hdf5', **kwargs):
    if 'key' not in kwargs and load_format == 'hdf5':
        kwargs['key'] = None
    if load_format != 'csv' and 'pandas_format' not in kwargs:
        kwargs['pandas_format'] = True
    if 'mode' not in kwargs:
        if load_format == 'pickle':
            kwargs['mode'] = 'rb'
        elif load_format == 'hdf5':
            kwargs['mode'] = 'r'
    logger.info("Attempting to load data from {}...".format(file_path))
    if not os.path.isfile(file_path):
        logger.error("File {} does not exist".format(file_path))
    loader = {
        'hdf5': load_hdf5_data,
        'csv': load_csv_data,
        'pickle': load_pickle_data
    }
    try:
        return loader.get(load_format, load_hdf5_data)(file_path, **kwargs)
    except Exception as e:
        logger.exception("Error loading file {}: {}".format(file_path, e))


'''
function: save_adn_model
inputs:
    - model_path: string pathname to save model to
    - adn_model: data structure holding the adn model to save to disk
description:
    saves the given model to the model path given as a tar file
'''
def save_adn_model(model_path, adn_model):
    try:
        config, model = adn_model
        verify_dir(model_path)
        dir_name, _ = os.path.split(model_path)
        if len(dir_name) == 0:
            dir_name = '.'
        config_path = os.path.join(dir_name, 'config.pkl')
        keras_model_path = os.path.join(dir_name, 'keras_model.h5')
        save_data(config_path, config, data_is_pandas=False, save_format='pickle', mode='wb', overwrite=True)
        model.save(keras_model_path)
        with tarfile.open(model_path, mode='w:gz') as f:
            f.add(config_path, arcname='config.pkl')
            f.add(keras_model_path, arcname='keras_model.h5')
        os.remove(config_path)
        os.remove(keras_model_path)
    except Exception as e:
        logger.exception("Error saving model {}: {}".format(model_path, e))


'''
function: load_adn_model
inputs:
    - model_path: string pathname to load model from
description:
    loads an adn model from disk given as a tar file
'''
def load_adn_model(model_path):
    try:
        dir_name, _ = os.path.split(model_path)
        if len(dir_name) == 0:
            dir_name = '.'
        config_path = os.path.join(dir_name, 'config.pkl')
        keras_model_path = os.path.join(dir_name, 'keras_model.h5')
        with tarfile.open(model_path, mode='r:gz') as f:
            f.extractall(dir_name)
        config = load_data(config_path, load_format='pickle', mode='rb')
        model = load_model(keras_model_path)
        os.remove(config_path)
        os.remove(keras_model_path)
        return config, model
    except Exception as e:
        logger.exception("Error loading model {}: {}".format(model_path, e))
        return None

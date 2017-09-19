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
import re
import auto_deepnet.utils.exceptions as exceptions

logger = logging.getLogger("auto_deepnet")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


'''
function: save_pickle_data
inputs:
    - file_path: string pathname to save data to
    - data_frame: pandas data_frame to save to disk in any picklable format
    - pandas_format: whether to save as a pandas dataframe or as a numpy array
description:
    helper function to save any data to disk via pickling
'''
def save_pickle_data(file_path, data_frame, **kwargs):
    logger.info("Opening pickle file {} to write data...".format(file_path))
    pandas_format = kwargs.get('pandas_format', True)
    try:
        if pandas_format:
            data_frame.to_pickle(file_path)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(data_frame.values, f)
    except Exception as e:
        logger.exception("Failed with Error {0}".format(e))
        raise exceptions.FileSaveError
    logger.info("Successfully saved pickle data")


'''
function: load_pickle_data
inputs:
    - file_path: string pathname to load data from
    helper function to load any pickled data from disk
'''
def load_pickle_data(file_path, **kwargs):
    logger.info("Opening pickle file {} to read...".format(file_path))
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        logger.exception("Failed with Error {0}".format(e))
        raise exceptions.FileLoadError
    logger.info("Successfully read pickle data")
    return data


'''
function: save_hdf5_data
inputs:
    - file_path: string pathname to save data to
    - data_frame: the pandas dataframe to save to disk
    - key: The name to call the dataset
    - pandas_format (optional): whether to save as a pandas structure or default hdf5
    - mode (optional): The mode to open file as
    - format (optional): whether to save as a table or fixed dataset
    - append (optional): Whether data should be appended or replaced
'''
def save_hdf5_data(file_path, data_frame, key, pandas_format=True, mode='a', format='table', append=False, **kwargs):
    if not key:
        logger.error("Need a key when saving as an HDF5 File")
        raise exceptions.FileSaveError
    logger.info("Opening HDF5 file {} to write data...".format(file_path))
    try:
        if pandas_format:
            with pd.HDFStore(file_path, mode=mode) as f:
                if key in f and not append:
                    f.remove(key)
                f.put(key=key, value=data_frame, format=format, append=append, **kwargs)
        else:
            if key == None:
                logger.error("Need a key when saving as default HDF5 format")
                raise exceptions.FileSaveError
            with h5py.File(file_path, mode) as f:
                if key in f:
                    del f[key]
                f.create_dataset(key, data=data_frame.values)
    except Exception as e:
        logger.exception("Failed with Error {0}".format(e))
        raise exceptions.FileSaveError
    logger.info("Successfully saved hdf5 data")


'''
function: load_hdf5_file
inputs:
    - file_path: string pathname to load data from
    - read_only (optional): whether to load file as a read only file
    - pandas_format (optional): whether the file was saved in pandas format
    - key (optional): name of the dataset
description:
    helper function to load an hdf5 file from disk
'''
def load_hdf5_data(file_path, key=None, pandas_format=True, mode='r', **kwargs):
    logger.info("Opening HDF5 file {} to read...".format(file_path))
    try:
        if pandas_format:
            data = pd.read_hdf(file_path, key=key, mode=mode, **kwargs)
        else:
            with h5py.File(file_path, mode) as f:
                data = f[key][()]
    except KeyError as e:
        logger.exception("Dataset {} does not exist".format(dataset))
        raise exceptions.FileLoadError("Dataset does not exist")
    except Exception as e:
        logger.exception("Problem loading dataset: {0}".format(e))
        raise exceptions.FileLoadError
    logger.info("Successfully loaded HDF5 data")
    return data


'''
function: save_csv_data
inputs:
    - file_path: string pathname to load data from
    - data_frame: pandas data to save to csv
'''
def save_csv_data(file_path, data_frame, **kwargs):
    logger.info("Opening CSV file {} to write data".format(file_path))
    try:
        data_frame.to_csv(file_path, **kwargs)
    except Exception as e:
        logger.exception("Problem saving dataset: {0}".format(e))
        raise exceptions.FileLoadError
    logger.info("Successfully saved CSV data")


'''
function: load_csv_data
inputs:
    - file_path: string pathname to load data from
'''
def load_csv_data(file_path, **kwargs):
    logger.info("Opening CSV file {} to read...".format(file_path))
    try:
        data = pd.read_csv(file_path, **kwargs)
    except Exception as e:
        logger.exception("Problem reading CSV: {0}".format(e))
        raise exceptiions.FileSaveError
    logger.info("Successfully loaded CSV data")
    return data


'''
function: save_data
inputs:
    - file_path: string pathname to save data to
    - data_frame: data to save to disk
    - key: The name to save the data as (required if hdf5 format, deprecated otherwise)
    - mode (optional): mode to open file in
    - save_format (optional): format to save to disk
    - overwrite (optional): whether to overwrite preexisting data
    - pandas_format (optional): whether to save as a pandas dataframe or as a numpy array
'''
def save_data(file_path, data_frame, save_format='hdf5', overwrite=False, mode='a', key='data', **kwargs):
    logger.info("Attempting to save data to {}...".format(file_path))
    try:
        dir_name, file_name = os.path.split(file_path)
    except Exception as e:
        logger.exception("Error with file path {}: {}".format(file_path, e))
        raise exceptions.FileSaveError("Invalid file path")
    if not os.path.isdir(dir_name):
        logger.info("Directory {} does not exist. Creating...".format(dir_name))
        os.makedirs(dir_name)
    if os.path.isfile(file_path):
        if not overwrite:
            logger.error("File {} already exists.".format(file_path))
            raise exceptions.FileSaveError
        if (mode == 'w' or save_format == 'pickle'):
            logger.warning("File {} will be overwritten".format(file_path))
            os.remove(file_path)
    if (mode == 'a' and save_format == 'pickle'):
        logger.warning("Can't append to pickle files. using mode='wb' instead...")
    
    saver = {
        'hdf5': save_hdf5_data,
        'csv': save_csv_data,
        'pickle': save_pickle_data
    }
    try:
        saver.get(save_format, save_hdf5_data)(file_path, data_frame, key, mode=mode, **kwargs)
    except Exception as e:
        logger.exceptions("Error saving file {}".format(file_path))
        raise exceptions.FileSaveError


def load_data(file_path, load_format='hdf5', **kwargs):
    logger.info("Attempting to load data from {}...".format(file_path))
    if not os.path.isfile(file_path):
        logger.error("File {} does not exist".format(file_path))
    loader = {
        'hdf5': load_hdf5_data,
        'csv': load_csv_data,
        'pickle': load_pickle_data
    }
    try:
        loader.get(load_format, load_hdf5_data)(file_path, **kwargs)
    except Exception as e:
        logger.exceptions("Error loading file {}".format(file_path))
        raise exceptions.FileLoadError

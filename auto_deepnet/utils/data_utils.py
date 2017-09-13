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
import auto_deepnet.utils.exceptions as exceptions

logger = logging.getLogger("auto_deepnet")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


'''
function: save_file_pickle
inputs:
    - file_path: string pathname to save data to
    - data: data to save to disk in any picklable format
    - overwrite (optional): whether a preexisting file should be overwritten
description:
    helper function to save any data to disk via pickling
'''
def save_file_pickle(file_path, data, overwrite=False):
    logger.info("Attempting to save data to {}...".format(file_path))
    try:
        dir_name, file_name = os.path.split(file_path)
    except Exception as e:
        logger.exception("Error with file path {}: {}".format(file_path, e))
        raise exceptions.FileSaveError("Invalid file path")
    if os.path.isdir(dir_name):
        logger.info("Directory {} does not exist. Creating...".format(dir_name))
        os.makedirs(dir_name)
    if os.path.isfile(file_path):
        if not overwrite:
            logger.error("File {} already exists".format(file_path))
            raise exceptions.FileSaveError("File already exists")
        logger.warning("File {} will be overwritten".format(file_path))
    logger.info("Opening file {} to write...".format(file_path))
    with open(file_path, "wb") as f:
        logger.info("Pickling and writing to disk...")
        try:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.exception("Failed with Error {0}".format(e))
            raise exceptions.FileSaveError
        logger.info("Successfully pickled and saved file")
        return
    logger.error("Something failed")
    raise exceptions.FileSaveError


'''
function: load_file_pickle
inputs:
    - file_path: string pathname to load data from
description:
    helper function to load any pickled data from disk
'''
def load_file_pickle(file_path):
    if not file_path:
        logger.error("Invalid file path")
        raise exceptions.FileLoadError("Invalid file path")
    logger.info("Attempting to load data from {}...".format(file_path))
    if not os.path.isfile(file_path):
        logger.error("File {} does not exist".format(file_path))
        raise exceptions.FileLoadError("File does not exist")
    logger.info("Opening file {} to read and unpickle...".format(file_path))
    with open(file_path, "rb") as f:
        try:
            data = pickle.load(f)
        except Exception as e:
            logger.exception("Failed with Error {0}".format(e))
            raise exceptions.FileLoadError
        logger.info("Successfully read and unpickled file")
        return data
    logger.error("Something failed")
    raise exceptions.FileLoadError


'''
function: get_hdf5_file
inputs:
    - file_path: string pathname to load data from
    - read_only (optional): whether to load file as a read only file
description:
    helper function to load an hdf5 file from disk
'''
def get_hdf5_file(file_path, read_only=True):
    if not file_path:
        logger.error("Invalid file path")
        raise exceptions.FileLoadError("Invalid file path")
    logger.info("Attempting to load data from {}...".format(file_path))
    mode = 'r' if read_only else 'a'
    if not os.path.isfile(file_path):
        if read_only:
            logger.error("File {} does not exist".format(file_path))
            raise exceptions.FileLoadError("File does not exist")
        logger.info("File {} does not exist. Creating...".format(file_path))
    else:
        logger.info("Opening File {}...".format(file_path))
    return h5py.File(file_path, mode)


'''
function: load_hdf5_dataset
inputs:
    - file_path: string pathname to load dataset from
    - dataset: name of the dataset
description:
    helper function to load a dataset from an hdf5 file in disk
'''
def load_hdf5_dataset(file_path, dataset):
    if not file_path:
        logger.error("Invalid file path")
        raise exceptions.FileLoadError("Invalid file path")
    logger.info("Attempting to load data from {}...".format(file_path))
    if not os.path.isfile(file_path):
        logger.error("File {} does not exist".format(file_path))
        raise exceptions.FileLoadError("File does not exist")
    logger.info("Opening File {}...".format(file_path))
    with h5py.File(file_path, 'r') as f:
        try:
            data = f[dataset][()]
        except KeyError as e:
            logger.exception("Dataset {} does not exist".format(dataset))
            raise exceptions.FileLoadError("Dataset does not exist")
        except Exception as e:
            logger.exception("Problem loading dataset: {0}".format(e))
            raise exceptions.FileLoadError
        return data

'''
function: save_hdf5_dataset
inputs:
    - file_path: string pathname to save dataset to
    - dataset: name of dataset
    - overwrite (optional): Whether a preexisting dataset should be overwritten
'''
def save_hdf5_dataset(file_path, dataset, data, overwrite=False):
    if not file_path:
        logger.error("Invalid file path")
        raise exceptions.FileSaveError("Invalid file path")
    logger.info("Attempting to save data to {}...".format(file_path))
    logger.info("Opening File {}...".format(file_path))
    with h5py.File(file_path, 'a') as f:
        if dataset in f:
            if not overwrite:
                logger.error("Dataset {} already exists!".format(dataset))
                raise exceptions.FileSaveError("Dataset already exists")
            logger.warning("Dataset {} already exists and will be overwritten".format(dataset))
            del f[dataset]
        try:
            dset = f.create_dataset(dataset, data=data)
        except Exception as e:
            logger.exception("Problem Creating Dataset: {0}".format(e))
            raise exceptions.FileSaveError


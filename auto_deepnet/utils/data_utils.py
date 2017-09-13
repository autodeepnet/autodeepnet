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
import auto_deepnet.utils.exceptions as exceptions

logger = logging.getLogger("auto_deepnet")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


'''
function: save_file_pickle
inputs:
    - file_path: string pathname to save data to
    - data: data to save to disk in any picklable format
    - overwrite: whether a preexisting file should be overwritten
description:
    helper function to save any data to disk via pickling
'''
def save_file_pickle(file_path, data, overwrite=False):
    if not file_path:
        logger.error("Invalid file path")
        raise exceptions.FileSaveError("Invalid file path")
    logger.info("Attempting to save data to {}...".format(file_path))
    dir_name, file_name = os.path.split(file_path)
    if len(dir_name) > 0 and not os.path.isdir(dir_name):
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
            pickle.dump(data, f)
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
        raise exceptions.FileSaveError("Invalid file path")
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

###TODO add logging and exception handling
def load_hdf5_dataset(file_path, dataset):
    with h5py.File(file_path, 'r') as hf:
        return hf[dataset]

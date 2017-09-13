from __future__ import absolute_import, division, print_function
import os
try:
    import cPickle as pickle
except:
    import pickle
import logging
import csv

logger = logging.getLogger("auto_deepnet")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%s(levelname)s:%(message)s', level=logging.DEBUG)

def save_file(file_path, data, overwrite=False):
    dir_name, file_name = os.path.split(file_path)
    if not os.path.isdir(dir_name):
        logger.info("Directory {} does not exist. Creating...".format(dir_name))
        os.path.mkdirs(dir_name)
    if os.path.isfile(file_path):
        if not overwrite:
            logger.error("File {} already exists. Returning...".format(file_path))
            return False
        logger.warning("File {} will be overwritten".format(file_path))
    logger.info("Opening File {} to write...".format(file_path))
    with open(file_path, "wb") as f:
        logger.info("Pickling and writing to disk...")
        try:
            pickle.dump(data, f)
        except Exception as e:
            logger.exception("Failed with Error {0}".format(e))
            return False
        logger.info("Successful")
        return True
    

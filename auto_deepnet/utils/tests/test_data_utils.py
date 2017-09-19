import unittest
import sys
import os
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path = [os.path.dirname(os.path.dirname(os.path.dirname(curr_path))), curr_path] + sys.path
os.environ['is_test_suite'] = 'True'
curr_path = None
import numpy as np
import h5py
import auto_deepnet.utils.data_utils as data_utils
import auto_deepnet.utils.exceptions as exceptions
import logging
import pandas as pd
#data_utils.logger.setLevel(logging.ERROR)

class TestPickle(unittest.TestCase):

    def setUp(self):
        self.s = pd.DataFrame(['hello world'])
        data_utils.save_pickle_data('s_pandas.pkl', self.s, pandas_format=True)
        data_utils.save_pickle_data('s.pkl', self.s, pandas_format=False)

    def tearDown(self):
        self.s = None
        os.remove('s.pkl')
        os.remove('s_pandas.pkl')
    
    def test_basic_read(self):
        self.assertEqual(self.s.values, data_utils.load_pickle_data('s.pkl', pandas_format=False))

    def test_pandas_read(self):
        np.testing.assert_array_equal(self.s.values, data_utils.load_pickle_data('s_pandas.pkl', pandas_format=True).values)

    def test_overwrite(self):
        a = pd.DataFrame(['foo'])
        data_utils.save_pickle_data('s.pkl', a, pandas_format=False)
        self.assertEqual(a.values, data_utils.load_pickle_data('s.pkl', pandas_format=False))

    def test_load_exceptions(self):
        with self.assertRaises(exceptions.FileLoadError):
            data_utils.load_pickle_data('s2.pkl')

    def test_append(self):
        a = pd.DataFrame(['foo'])
        data_utils.save_pickle_data('s_pandas.pkl', a, append=True, pandas_format=True)
        np.testing.assert_array_equal(np.concatenate((self.s.values, a.values)), data_utils.load_pickle_data('s_pandas.pkl', pandas_format=True))


class TestHDF5(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame(np.random.random((2, 2)))
        data_utils.save_hdf5_data('test.h5', self.data, pandas_format=False, mode='w', key='test_data')
        data_utils.save_hdf5_data('test_pandas.h5', self.data, pandas_format=True, mode='w', key='test_data')

    def tearDown(self):
        self.data = None
        if os.path.isfile('test.h5'):
            os.remove('test.h5')
        if os.path.isfile('test_pandas.h5'):
            os.remove('test_pandas.h5')
    
    def test_basic_read(self):
        np.testing.assert_array_equal(self.data.values, data_utils.load_hdf5_data('test.h5', pandas_format=False, key='test_data'))

    def test_pandas_read(self):
        np.testing.assert_array_equal(self.data.values, data_utils.load_hdf5_data('test_pandas.h5', pandas_format=True, key='test_data').values)

    def test_overwriting(self):
        data = pd.DataFrame(np.ones((2, 2)))
        data_utils.save_hdf5_data('test.h5', data, pandas_format=False, key='test_data')
        self.assertEqual(np.any(np.not_equal(self.data.values, data_utils.load_hdf5_data('test.h5', pandas_format=False, key='test_data'))), True)

    def test_pandas_overwriting(self):
        data = pd.DataFrame(np.ones((2, 2)))
        data_utils.save_hdf5_data('test_pandas.h5', data, pandas_format=True, key='test_data', append=False)
        self.assertEqual(np.any(np.not_equal(self.data.values, data_utils.load_hdf5_data('test_pandas.h5', pandas_format=True, key='test_data').values)), True)

    def test_pandas_append(self):
        data = pd.DataFrame(np.random.random((2, 2)))
        data_utils.save_hdf5_data('test_pandas.h5', data, pandas_format=True, key='test_data', append=True)
        newData = np.concatenate((self.data.values, data.values))
        np.testing.assert_array_equal(newData, data_utils.load_hdf5_data('test_pandas.h5', pandas_format=True, key='test_data').values)

    def test_saving_assertions(self):
        data = np.ones((2, 2))
        with self.assertRaises(exceptions.FileSaveError):
            data_utils.save_hdf5_data('test.h5', data, 'test_data')

    def test_loading_assertions(self):
        os.remove('test.h5')
        with self.assertRaises(exceptions.FileLoadError):
            data_utils.load_hdf5_data('test.h5', 'test_data')


class TestCSV(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame(np.random.random((2, 2)))
        data_utils.save_csv_data('test.csv', self.data)

    def tearDown(self):
        self.data = None
        if os.path.isfile('test.csv'):
            os.remove('test.csv')
    
    def test_basic_read(self):
        np.testing.assert_array_equal(self.data.values, data_utils.load_csv_data('test.csv').values)

    def test_overwriting(self):
        data = pd.DataFrame(np.ones((2, 2)))
        data_utils.save_csv_data('test.csv', data, mode='w')
        self.assertEqual(np.any(np.not_equal(self.data.values, data_utils.load_csv_data('test.csv').values)), True)
    
    def test_append(self):
        data = pd.DataFrame(np.ones((2, 2)))
        data_utils.save_csv_data('test.csv', data, mode='a', header=False)
        np.testing.assert_array_equal(np.concatenate((self.data.values, data.values)), data_utils.load_csv_data('test.csv').values)


'''
    def test_saving_assertions(self):
        data = np.ones((2, 2))
        with self.assertRaises(exceptions.FileSaveError):
            data_utils.save_hdf5_data('test.h5', data, 'test_data')

    def test_loading_assertions(self):
        os.remove('test.h5')
        with self.assertRaises(exceptions.FileLoadError):
            data_utils.load_hdf5_data('test.h5', 'test_data')
'''

    


class TestSave(unittest.TestCase):

    def test_hdf5(self):
        pass
    
    def test_pickle(self):
        pass

    def test_csv(self):
        pass

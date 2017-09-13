import unittest
import sys
import os
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path = [os.path.dirname(os.path.dirname(os.path.dirname(curr_path))), curr_path] + sys.path
os.environ['is_test_suite'] = 'True'
curr_path = None
import auto_deepnet.utils.data_utils as data_utils
import auto_deepnet.utils.exceptions as exceptions
import logging
#data_utils.logger.setLevel(logging.ERROR)

class TestPickle(unittest.TestCase):

    def setUp(self):
        self.s = 'hello world'
        data_utils.save_file_pickle('s.pkl', self.s, overwrite=True)

    def tearDown(self):
        self.s = None
        os.remove('s.pkl')
    
    def test_basic_pickle(self):
        self.assertEqual(self.s, data_utils.load_file_pickle('s.pkl'))

    def test_pickle_overwrite(self):
        a = 'foo'
        data_utils.save_file_pickle('s.pkl', a, overwrite=True)
        self.assertEqual(a, data_utils.load_file_pickle('s.pkl'))

    def test_pickle_save_exceptions(self):
        with self.assertRaises(exceptions.FileSaveError):
            data_utils.save_file_pickle('s.pkl', self.s, overwrite=False)
        with self.assertRaises(exceptions.FileSaveError):
            data_utils.save_file_pickle(None, self.s)

    def test_pickle_load_exceptions(self):
        with self.assertRaises(exceptions.FileLoadError):
            data_utils.load_file_pickle('s2.pkl')

class TestHDF5(unittest.TestCase):
    
    def test_basic(self):
        pass

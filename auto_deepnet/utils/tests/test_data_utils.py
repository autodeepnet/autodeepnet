import unittest
import sys
import os
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path = [os.path.dirname(os.path.dirname(curr_path)), curr_path] + sys.path
os.environ['is_test_suite'] = 'True'

import utils.data_utils as data_utils
import logging
data_utils.logger.setLevel(logging.ERROR)

class TestDataUtils(unittest.TestCase):
    
    def test_pickle(self):
        s = 'hello world'
        self.assertEqual(data_utils.save_file_pickle('s.pkl', s, overwrite=True), True)
        self.assertEqual(s, data_utils.load_file_pickle('s.pkl'))
        os.remove('s.pkl')

#suite = unittest.TestLoader().loadTestsFromTestCase(TestDataUtils)
#unittest.TextTestRunner(verbosity=2).run(suite)


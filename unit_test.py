import sys
import unittest
import numpy as np
from pairfinance.predictor import predictor

test_case_1 = {"id": 74545.0,
               "X1": -1.78,
               "X2": 2.4,
               "X3": 1.0
               }

test_case_2 = {"id": None,
               "X1": None,
               "X2": None,
               "X3": None
               }


class TestFinCrime(unittest.TestCase):

    def test_case_1_model(self):
        self.assertEqual({'time': 746.545654296875},
                         predictor(test_case_1)
                         )

    def test_case_2_model(self):
        self.assertEqual({'time': 816.1084594726562},
                         predictor(test_case_2)
                         )


if __name__ == '__main__':
    unittest.main()

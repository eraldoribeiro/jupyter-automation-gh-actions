import unittest
import importlib
import importnb
import numpy as np
from importnb import imports
from importnb import Notebook, get_ipython, imports


from unittest.mock import patch

import cvxopt
from cvxopt import matrix, printing


notebook_name = 'display_3D_solution.ipynb'


class TestJupyterNotebook(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.notebook  = Notebook.load_file(notebook_name)

    def red(self, text):
        print('\x1b[31m{}\x1b[0m'.format(text))

    @patch('matplotlib.pyplot.show')
    def test_get_translation_matrix(self, mock_show):
        mock_show.return_value = None

        actual = self.notebook.get_translation_matrix(0.1, 0.2, -0.1)

        expected = np.array([[ 1,  0,  0,  0.1],\
                             [ 0,  1,  0,  0.2],\
                             [ 0,  0,  1, -0.1],\
                             [ 0,  0,  0,  1.0]])



        assert np.array_equal(expected, actual)


    
    def test_get_rotation_matrix(self):

        actual = self.notebook.get_rotation_matrix(90, axis_name = 'x')

        expected = np.array([\
            [ 1.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],\
            [ 0.000000e+00,  0.000000e+00, -1.000000e+00,  0.000000e+00],\
            [ 0.000000e+00,  1.000000e+00,  0.000000e+00,  0.000000e+00],\
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])
        
        assert np.allclose(expected, actual)


    def test_get_scaling_matrix(self):

        actual = self.notebook.get_scaling_matrix(1.2, 1.5, 1.2)

        expected = np.array([\
                [1.2, 0. , 0. , 0. ],\
                [0. , 1.5, 0. , 0. ],\
                [0. , 0. , 1.2, 0. ],\
                [0. , 0. , 0. , 1. ]])                

        assert np.allclose(expected, actual)     


    def test_T1(self):
        
        # Rotation, Translation, and Scale (in this order)
        T1 = self.notebook.S @ self.notebook.T @ self.notebook.R

        Y1 = self.notebook.apply_transformation(self.notebook.Xt, T1)

        actual = np.sum(Y1[:,0:5])

        expected = 1.593428762862459               

        assert np.allclose(expected, actual)     

    def test_T2(self):
        
        # Translation, Rotation, and Scaling (in this order)
        T2 = self.notebook.S @ self.notebook.R @ self.notebook.T         
        Y2 = self.notebook.apply_transformation(self.notebook.Xt, T2)

        actual = np.sum(Y2[:,0:5])

        expected = -0.9565712371375411               

        assert np.allclose(expected, actual)     
        

    def test_T3(self):
        
        # Scaling, Translation, and Rotation (in this order)
        T3 = self.notebook.R @ self.notebook.T @ self.notebook.S         
        Y3 = self.notebook.apply_transformation(self.notebook.Xt, T3)

        actual = np.sum(Y3[:,0:5])

        expected = 2.8799194226041442               

        assert np.allclose(expected, actual)     
        




if __name__ == '__main__':
    unittest.main()

    







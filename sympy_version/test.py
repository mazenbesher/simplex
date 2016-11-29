#!/usr/bin/env python3

import unittest
import numpy as np
from sympy import Eq, symbols

from main import sympy_simplex, LP

aufgabe1 = LP( # Blatt 2
    np.matrix('2 0 6; -2 8 4; 3 6 5'),
    np.matrix('10; 12; 20'),
    np.matrix('2; 1; 3; 0; 0; 0'),
    [4, 5, 6])

kreise_example = LP( # Book Page 31
    np.matrix('-0.5 -5.5 -2.5 9; 0.5 -1.5 -0.5 1; 1 0 0 0'),  # A
    np.matrix('0; 0; 1'),  # b
    np.matrix('10; -57; -9; -24; 0; 0; 0'),  # c
    [5, 6, 7]
)

class TestSimplex(unittest.TestCase):
    def test_simplexAufgabe1(self):
        ziel = sympy_simplex(aufgabe1)[1]
        z = symbols('z')
        x3, x4, x6 = symbols('x3 x4 x6')
        self.assertEqual(ziel, Eq(z, -7*x3/3 - 3*x4/4 - x6/6 + 65/6))

    def test_simplexFractionsAndKreisen(self):
        # because of the fractions and int in lines 31, 32 in main.py
        ziel = sympy_simplex(kreise_example)[1]
        self.assertEqual(ziel.rhs.as_coefficients_dict()[1], 1) # Zielfunktionswert = 1


if __name__ == '__main__':
    unittest.main()

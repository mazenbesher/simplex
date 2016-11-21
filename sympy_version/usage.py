#!/usr/bin/env python3

import unittest
import numpy as np

from main import sympy_simplex, LP

print("""
    use jupyter notebook or qtconsole to see formatted results
    e.g. > jupyter qtconsole
    then in the console:
         > %load usage.py
    then press ENTER twice
""")

aufgabe1 = LP( # Blatt 2
    np.matrix('2 0 6; -2 8 4; 3 6 5'),
    np.matrix('10; 12; 20'),
    np.matrix('2; 1; 3; 0; 0; 0'),
    [4, 5, 6])

sympy_simplex(aufgabe1) 
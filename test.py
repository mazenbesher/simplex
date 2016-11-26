#!/usr/bin/env python3

import unittest
import numpy as np

from main import *

# Random examples ----------------------------------------
vorlesung = LP(  # Muesli Example
    np.matrix('2 3; 4 1; 1 1'),
    np.matrix('12000; 16000; 4300'),
    np.matrix('5; 4'),
)

blatt2_aufgabe1 = LP(
    np.matrix('2 0 6; -2 8 4; 3 6 5'),
    np.matrix('10; 12; 20'),
    np.matrix('2; 1; 3'),
)

blatt5_aufgabe1 = LP(  # extended muesli example
    np.matrix('2 3; 4 1; 1 1; 2 1'),  # A
    np.matrix('12000; 16000; 4300; 8200'),  # b
    np.matrix('5; 4'),  # c
)

# Specific examples ----------------------------------------
kreise_example = LP(  # Book Page 31
    np.matrix('-0.5 -5.5 -2.5 9; 0.5 -1.5 -0.5 1; 1 0 0 0'),  # A
    np.matrix('0; 0; 1'),  # b
    np.matrix('10; -57; -9; -24'),  # c
)

infeasible_start = LP(  # Blatt4 Aufgabe2
    np.matrix('-2 -1; -1 -2; -1 -1; 3 4; 1 0'),
    np.matrix('-12; -12; -10; 60; 12'),
    np.matrix('11; 8')
)

unbounden = LP(  # see https://www.desmos.com/calculator/7g2vcxlt7p
    np.matrix('-1 -1; 1 1'),
    np.matrix('-2; 1'),
    np.matrix('1; 1')
)


class TestIndexed(unittest.TestCase):
    def test_indexedOneColumn(self):
        self.assertTrue(np.array_equal(
            indexed(blatt2_aufgabe1.A, [1]),
            np.matrix('0; 8; 6')))

    def test_indexedTwoColumn(self):
        self.assertTrue(np.array_equal(
            indexed(blatt2_aufgabe1.A, [0, 2]),
            np.matrix('2 6; -2 4; 3 5')))


class TestSimplexSpecialCases(unittest.TestCase):
    def test_simplexKreisen(self):
        solu = simplex(kreise_example)[1]
        self.assertTrue(np.all(solu[0, 0:4] == [1, 0, 1, 0]))

    def test_simplexInfeasibleStart(self):
        # requires 2.Phase Simplex
        solu = simplex(infeasible_start)[1]
        x_1, x_2 = solu[0, 0], solu[0, 1]
        self.assertEqual(x_1, 12)
        self.assertEqual(x_2, 6)

    def test_simplexUnbounden(self):
        # requires 2.Phase Simplex
        self.assertIsNone(simplex(unbounden))


class TestSimplexRandomExamples(unittest.TestCase):
    def test_simplexVorlesung(self):
        solu = simplex(vorlesung)[1]
        x_1, x_2 = int(solu[0, 0]), int(solu[0, 1])
        self.assertEqual(x_1, 3900)
        self.assertEqual(x_2, 400)

    def test_simplexBlatt2Aufgabe1(self):
        solu = simplex(blatt2_aufgabe1)[1]
        x_1, x_2, x_3 = int(solu[0, 0]), solu[0, 1], int(solu[0, 2])
        self.assertEqual(x_1, 5)
        self.assertAlmostEqual(x_2, 0.83333333)
        self.assertEqual(x_3, 0)

    def test_simplexBlatt5Aufgabe1(self):
        solu = simplex(blatt5_aufgabe1)[1]
        x_1, x_2 = int(solu[0, 0]), int(solu[0, 1])
        self.assertEqual(x_1, 3900)
        self.assertEqual(x_2, 400)


if __name__ == '__main__':
    unittest.main()

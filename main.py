#!/usr/bin/env python3

import numpy as np
import copy
from numpy.linalg import inv

LIMIT = 1000 # maximale Anzahl der Durchläufe des Simplex-Algorithmus

# Hilfsfunktionen 
rows = lambda X: X.shape[0]  # Anzahl der Zeile einer Matrix
cols = lambda X: X.shape[1]  # Anzahl der Spalten einer Matrix
base_zero = lambda X: [i - 1 for i in X]  # Wandel eine Liste in Basis 1 um (für die LP-Basis)


# Eine Klasse um ein LP-Problem zu repräsentieren
class LP:
    def __init__(self, A, b, c, B):
        self.A = A
        self.b = b
        self.c = c
        self.B = B  # zur Basis 1 (nicht 0)


# Hilfsfunktion um die durch index definierten Spalten aus A zurückzugeben
# usage: siehe TestIndexed in test.py
def indexed(X, index):
    res = np.matrix(np.zeros((rows(X), len(index))))
    for i in range(len(index)): res[:, i] = copy.deepcopy(X[:, index[i]])
    return res


# Hauptfunktion
def simplex(lp, debug=False):
    """
    No cycling (kreise) will happen because of the selection of the sorting of
    the index sets (selecting the candidate with the smallest index - Bland's Rule)
    Example see test_simplexKreisen in test.py

    :param lp: LP linear program
    :param debug: debug mode (see last condition in this function)
    :return:
    """
    # Variablen definieren
    A, b, c, B = lp.A, lp.b, lp.c, lp.B
    B = base_zero(B)
    I = np.matrix(np.identity(rows(A)))
    AI = np.concatenate((A, I), 1)
    x = np.matrix(np.zeros(cols(AI)))
    c = np.matrix(np.transpose(c))  # just to make operations easier

    # 0. Init
    A_B = indexed(AI, B)
    N = [i for i in range(len(B)) if not i in B]
    x_B = np.dot(inv(A_B), b)

    # Update x
    B_pointer = 0
    for i in range(cols(AI)):
        if i in B:
            x[0, i] = x_B[B_pointer, 0]
            B_pointer += 1

    # Main-Loop
    for counter in range(LIMIT):
        # 1. BTRAN
        A_B = indexed(AI, B)
        c_B = indexed(c, B)
        ys = np.matrix(
            np.transpose(
                np.dot(c_B, inv(A_B))
            )
        )

        # 2. Pricing
        c_N = indexed(c, N)
        A_N = indexed(AI, N)
        cs_N = np.subtract(np.transpose(c_N), np.matrix(np.dot(np.transpose(A_N), ys)))
        if np.all(cs_N <= 0):
            return B, x # Optimale Lösung gefunden
        else:
            for i in range(len(cs_N)):
                if np.all(cs_N[i] > 0):
                    j = N[i]
                    break

        # 3. FTRAN
        A_j = indexed(AI, [j])
        w = np.matrix(np.dot(
            inv(A_B), A_j
        ))

        # 4. Ratio-Test
        if np.all(w <= 0):
            return "unbounded" # unbeschränkt
        else:
            arr = []
            for i in range(len(B)):
                k = B[i]
                x_k = x[0, k]
                w_k = w[i, 0]
                if (w_k > 0): arr.append(((x_k / w_k), k))

            t_star, i = min(arr)

        # 5. Update
        x_B = np.transpose(indexed(x, B))
        B_pointer = 0
        for l in range(cols(x)):
            if l in B:
                x[0, l] = x_B[B_pointer, 0] - (t_star * w[B_pointer, 0])
                B_pointer += 1

        x[0, j] = t_star

        B.remove(i)
        B.append(j)
        B.sort()

        N.remove(j)
        N.append(i)
        N.sort()

        counter += 1

        if debug:
            print("{}, x = {}, B = {}, N = {}".format(
                counter, x,
                # convert Basis und Nichtbasis to one-based
                [i + 1 for i in B],
                [i + 1 for i in N]
            ))

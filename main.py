#!/usr/bin/env python3

"""
Name: Mazen Bouchur
For usage and examples see `test.py`
"""

import numpy as np
import copy
from numpy.linalg import inv

LIMIT = 1000  # maximale Anzahl der Durchläufe des Simplex-Algorithmus
# in Theory we don't need this limiter anymore (since Balnd's rule is being used)

# Hilfsfunktionen 
rows = lambda X: X.shape[0]  # Anzahl der Zeile einer Matrix
cols = lambda X: X.shape[1]  # Anzahl der Spalten einer Matrix


# Eine Klasse um ein LP-Problem zu repräsentieren
class LP:
    # TODO: convert to standard form if not in it
    def __init__(self, A, b, c):
        """
        A class to represent a standard LP (max cTx s.t. A.x <= b and x >= 0)
        """
        self.A = A
        self.b = b
        self.c = c


# Hilfsfunktion um die durch index definierten Spalten aus A zurückzugeben
# usage: siehe TestIndexed in test.py
def indexed(X, index):
    res = np.matrix(np.zeros((rows(X), len(index))))
    for i in range(len(index)): res[:, i] = copy.deepcopy(X[:, index[i]])
    return res


# Hauptfunktion
def simplex(lp):
    """
    No cycling (kreisen) will happen because of the sorting of the index
     sets (selecting the candidate with the smallest index - Bland's Rule)
    Example see test_simplexKreisen in test.py

    :param lp: LP linear program (see LP class)
    :return: None if no optimal feasible solution otherwise (B, x)
        where is the optimal Basis solution and x is the solution
    """
    # Variablen definieren
    A, b, c = lp.A, lp.b, lp.c
    I = np.matrix(np.identity(rows(A)))
    AI = np.concatenate((A, I), 1)
    x = np.matrix(np.zeros(cols(AI)))
    c = np.concatenate((c, np.transpose(np.matrix(np.zeros(rows(b))))), axis=0)  # extend c for the slack variables
    c = np.matrix(np.transpose(c))  # use the transpose just to make operations easier

    # Search for feasible basis
    if np.all(b >= 0):
        # the origin (i.e. the zero basis) is feasible basis
        B = [cols(A) + i for i in range(rows(A))]
    else:
        # we need to search for a feasible basis => 2.Phase Simplex
        return __first_phase(lp)

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
    return __simplex_main_loop(AI, B, N, c, x)


# Hilfsfunktion
def __simplex_main_loop(AI, B, N, c, x):
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
            return B, x  # Optimale Lösung gefunden
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
            return None  # unbounded / unbeschränkt
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


# Hilfsfunktion, sucht nach einem zulaessigen Startbasisloesung falls
# das LP keine hat (d.h. NULL is nicht zuelassig am Anfang <=> not b >= 0)
def __first_phase(lp):
    """
    :return: a feasible start basis if exists or None if none exists
    """
    # das Hilfsproblem formulieren
    new_col = np.transpose(np.matrix(np.zeros(rows(lp.A)) - 1))
    A = np.concatenate((new_col, lp.A), axis=1)
    b = lp.b
    c = np.vstack([np.matrix('-1'), np.transpose(np.matrix(np.zeros(cols(A) - 1)))])

    # Pivotisierung
    # Variablen definieren
    I = np.matrix(np.identity(rows(A)))
    AI = np.concatenate((A, I), 1)
    x = np.matrix(np.zeros(cols(AI)))
    c = np.concatenate((c, np.transpose(np.matrix(np.zeros(rows(b))))), axis=0)  # extend c for the slack variables
    c = np.matrix(np.transpose(c))  # use the transpose just to make operations easier

    # the origin is now feasible basis in the hilfsproblem
    B = [cols(A) + i for i in range(rows(A))]

    # eintretende x_0
    j = 0

    # veralssende x_i mit max {|b_i|: b_i <0}
    i = max([(abs(i), abs(int(k))) for i, k in enumerate(b) if int(k) < 0],
            key=lambda x: x[1])[0]

    # Update Basis
    B[i] = j
    B.sort()

    # solve using simplex since now the basis is feasible
    # 0. Init
    A_B = indexed(AI, B)
    N = [i for i in range(rows(A) + cols(A)) if not i in B]
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
            # Optimale Lösung gefunden
            # Ueberpruefen ob der Zielfunktionswert = 0
            if np.dot(c, np.transpose(x)).item(0) != 0:
                return None
            else:
                # Hilfsproblem umformen
                # generate new problem (remove x_0 column and reformat z-row)
                # and solve it with simplex
                N.remove(0)
                x = x[:, 1:]
                AI = AI[:, 1:]

                # convert B, N to zero base
                N = [i - 1 for i in N]
                B = [i - 1 for i in B]

                # build c
                c = lp.c
                c = np.concatenate((c, np.transpose(np.matrix(np.zeros(rows(b))))),
                                   axis=0)  # extend c for the slack variables
                c = np.matrix(np.transpose(c))  # use the transpose just to make operations easier

                # Main-Loop
                return __simplex_main_loop(AI, B, N, c, x)
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
            return None  # unbounded / unbeschränkt
        else:
            arr = []
            for i in range(len(B)):
                k = B[i]
                x_k = x[0, k]
                w_k = w[i, 0]
                if (w_k > 0): arr.append(((x_k / w_k), k))
            # Taktik (immer x_0 die Basis zu verlassen wenn moeglich)
            for t_cand, i_cand in arr:
                if i_cand == 0 and t_cand == min(arr)[0]:
                    t_star, i = t_cand, i_cand
                    break
            else:  # else if the for-loop didn't break (Python only :)
                # x_0 is not candidate
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

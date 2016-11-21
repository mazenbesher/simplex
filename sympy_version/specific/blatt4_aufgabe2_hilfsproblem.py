"""1.Phase"""

from sympy import *
init_printing()

z, x0, x1, x2, x3, x4, x5, x6, x7 = symbols('z, x0, x1, x2, x3, x4, x5, x6, x7')

B = [x3, x4, x5, x6, x7]
N = [x0, x1, x2]

rows = [Eq(x3, -12 + 2 * x1 + 1 * x2 + x0),
        Eq(x4, -12 +     x1 + 2 * x2 + x0),
        Eq(x5, -10 +     x1 +     x2 + x0),
        Eq(x6, 60 - 3 *  x1 - 4 * x2 + x0),
        Eq(x7, 12 -      x1          + x0)]
ziel = Eq(z, - x0)

# -------------------------------------------------------------------------------
eintretende = x0

for i in range(10):
    # eintretende Variable finden
    # auswaehlen nach dem Teknik in der Vorlesung (d.h. var mit grosstem Koeffizeint)
    if i != 0: # nicht in erstem Durchlauf (da hier unzulaessig)
        eintretende = None
        max_eintretende = -oo
        for var, coeff in ziel.rhs.as_coefficients_dict().items():
            # 1 is the first coeff i.e. the value of the ziel function
            if var != 1 and  coeff > 0 and coeff > max_eintretende:
                    max_eintretende = coeff
                    eintretende = var

    # if no positiv costs => optimal
    if eintretende == None:
        break

    # verlassende Variable finden
    verlassende = None
    min_wert = +oo
    min_row = None

    if i == 0: # einfach definierne da im ersten Durchlauf Dich ist unzulaessig
        # verlassende = min([row.rhs.as_coefficients_dict()[1] for row in rows])
        verlassende = x3
        min_row = rows[0]
    else:
        for row in rows:
            if row.has(eintretende):
                new_row = row
                for nbv in N:
                    if nbv != eintretende:
                        new_row = new_row.subs(nbv, 0)
                wert = solve(new_row.rhs >= 0).as_set().right
                if wert < min_wert:
                    min_wert = wert
                    min_row = row
                    verlassende = row.lhs

    # die Formlen umsetzen und rows updaten
    new_formel = Eq(eintretende, solve(min_row, eintretende)[0])
    new_rows = [new_formel]
    for row in rows:
        if row.lhs != verlassende:
            new_rows.append(Eq(row.lhs, row.rhs.subs(eintretende, new_formel.rhs)))
    rows = new_rows

    # new ziel
    ziel = Eq(z, ziel.rhs.subs(eintretende, new_formel.rhs))
    pprint(latex(ziel))

    # update B, N
    B.remove(verlassende); B.append(eintretende)
    N.remove(eintretende); N.append(verlassende)

import numpy as np
from main import LP, simplex

# test = LP(
#     np.matrix('2 0 6; -2 8 4; 3 6 5'),
#     np.matrix('10; 12; 20'),
#     np.matrix('2; 1; 3; 0; 0; 0'),
#     [4, 5, 6])
# print(simplex(test, True)[1])

example = LP(
    np.matrix('-0.5 0.5 1; -5.5 -1.5 0; -2.5 -0.5 0; 9 1 0'), # A
    np.matrix('0; 0; 1'), # b
    np.matrix('10; -57; -9; -24; 0; 0; 0'), # c
    [5, 6, 7]
)
print(simplex(example))


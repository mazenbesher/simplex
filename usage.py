import numpy as np
from main import LP, simplex

test = LP(
    np.matrix('2 0 6; -2 8 4; 3 6 5'),
    np.matrix('10; 12; 20'),
    np.matrix('2; 1; 3; 0; 0; 0'),
    [4, 5, 6])
print(simplex(test, True)[1])

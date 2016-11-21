import numpy as np
from simplex.main import LP, simplex

deg = LP(
    np.matrix('0 0 2; 2 -4 6;-1 3 4'),
    np.matrix('1; 3; 2'),
    np.matrix('2; -1; 8; 0; 0; 0'),
    [4, 5, 6]
)
print(simplex(deg, True))

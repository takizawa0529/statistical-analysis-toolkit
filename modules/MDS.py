import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np


class DistanceMatrix:
    def __init__(self, pos):
        self.pos = pos
        self.check_is_Eucledian_distance()

    
    def distance_matrix(self):
        n = len(self.pos)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                diff = np.linalg.norm(self.pos[i] - self.pos[j])**2
                D[i][j] = D[j][i] = diff
        return D


    def check_is_Eucledian_distance(self):
        n = len(self.pos)
        D = self.distance_matrix()
        d = np.sqrt(D)

        for i in range(n):
            for j in range(i+1, n):
                for k in range(n):
                    if (k!=i) and (k!=j):
                        if d[i][j]+d[i][k]<d[j][k]:
                            raise ValueError(f"D is not Eucledian distance matrix")
        return True


class MultiDimensionalScaling:
    def __init__(self, D):
        self.D = D
        self.dim = len(self.D)

    def double_centering(self):
        V = self._I() - (1/self.dim)*self._J()
        return -(1/2) * (V @ self.D @ V)

    def _I(self):
        return np.identity(self.dim)

    def _J(self):
        return np.ones((self.dim, self.dim))

    def Eckart_Young_decomposition(self):
        B = self.double_centering()
        vals, U = np.linalg.eig(B)
        idx = vals > 0
        rank = np.sum(idx)

        vals, U = vals[idx].copy(), (U.T[idx]).T.copy()
        Lambda = np.diag(vals)

        X = U @ np.sqrt(Lambda)
        X = X.reshape(self.dim, rank).copy()
        return X

    def _same_array(self, a: np.ndarray, b: np.ndarray) -> bool:  
        a = a.reshape(-1)
        b = b.reshape(-1)      
        for idx in range(len(a)):
            if np.round(a[idx], 3) != np.round(b[idx], 3):
                return False
        return True

    def _judge_ability_to_construct_D(self, n: int) -> bool:
        X = (self.Eckart_Young_decomposition().T[:n]).T
        new_D = DistanceMatrix(X).distance_matrix()
        return self._same_array(new_D, self.D)
    
    def result(self):
        new_axis = self.Eckart_Young_decomposition()
        for i in range(len(new_axis[0])+1):
            if self._judge_ability_to_construct_D(i):
                print(f"データの次元を{len(new_axis[0])}から{i}に削減します")
                return new_axis.T[:i].T

        
        


    

if __name__=='__main__':
    D = np.array([[0, 5, 5, 16],
                  [5, 0, 4, 5],
                  [5, 4, 0, 5],
                  [16, 5, 5, 0],
                  ])

    #pos = np.array([[2, 5],
    #            [3, 1],
    #            [1, 6],])

    #D = DistanceMatrix(pos).distance_matrix()

    #print(DistanceMatrix(pos))
    #print(DistanceMatrix(pos).check_is_Eucledian_distance())
    print(MultiDimensionalScaling(D).result())
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def squared_distance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Return squared Euclidean distance matrix.
    X: (n, p)
    """
    X = np.asarray(X, dtype=float)
    G = X@X.T 
    diag = np.diag(G)
    D2 = diag[:, None]+diag[None, :] - 2.0*G
    
    D2[D2<0] = np.maximum(D2[D2<0], 0.0)
    np.fill_diagonal(D2, 0.0)
    return D2

def is_psd(A: np.ndarray, *, tol: float=1e-10) -> bool:
    A = 0.5 * (A + A.T)
    vals = np.linalg.eigvalsh(A)
    return bool(np.min(vals) >= -tol)

@dataclass(frozen=True)
class ClassicalMDS:
    D2: np.ndarray
    def __post_init__(self):
        D2 = np.asarray(self.D2, dtype=float)

        if D2.ndim != 2 or D2.shape[0] != D2.shape[1]:
            raise ValueError("D2 must be a square matrix")

        if not np.allclose(D2, D2.T, atol=1e-12):
            raise ValueError("D2 must be symmetric.")

        if not np.allclose(np.diag(D2), 0.0, atol=1e-12):
            raise ValueError("Diagonal entries of D2 must be 0.")

        if np.any(D2 < -1e-12):
            raise ValueError("D2 must be non-negative (up to numerical tolerance)")


    @property
    def n(self) -> int:
        return self.D2.shape[0]
    
    def gram_matrix(self) -> np.ndarray:
        n = self.n
        J = np.eye(n) - np.ones((n, n))/n
        return -0.5*J @ self.D2 @ J

    def is_euclidean(self, *, tol: float=1e-10) -> bool:
        return is_psd(self.gram_matrix(), tol=tol)

    def embed(self, k: int | None=None, *, tol: float=1e-12) -> np.ndarray:
        """
        Returns X: (n, k) classical MDS embedding.
        If k is None: use all positive eigenvalues.
        """
        B = self.gram_matrix()
        B = 0.5 * (B + B.T)

        vals, vecs = np.linalg.eigh(B)
        idx = np.argsort(vals)[::-1]
        vals, vecs = vals[idx], vecs[:, idx]

        tol_eff = max(tol, tol * max(1.0, vals[0]))

        pos = vals > tol_eff
        vals, vecs = vals[pos], vecs[:, pos]

        if k is not None:
            k = min(k, len(vals))
            vals, vecs = vals[:k], vecs[:, :k]

        # 念のため 0 未満を潰す（極小負の丸め）
        vals = np.maximum(vals, 0.0)

        return vecs * np.sqrt(vals)


    def reconstruction_error(self, k: int, *, rtol=1e-5, atol=1e-8) -> float:
        Xk = self.embed(k)
        D2_hat = squared_distance_matrix(Xk)
        num = np.linalg.norm(self.D2 - D2_hat)
        den = np.linalg.norm(self.D2) + 1e-15
        return num/den

    def minimal_dimension(self, *, eps: float=1e-6) -> int:
        X_full = self.embed(None)
        max_k = X_full.shape[1]
        for k in range(max_k+1):
            err = self.reconstruction_error(k)
            if err <= eps:
                return k
        return max_k


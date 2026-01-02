import sys
import pytest
from pathlib import Path
import numpy as np

# ===== import がコケないように、repo ルートを sys.path に入れる =====
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../statistical-analysis-toolkit
sys.path.insert(0, str(PROJECT_ROOT))

from modules.MDS import MultiDimensionalScaler, squared_distance_matrix


def pairwise_sq_dists_reference(X: np.ndarray) -> np.ndarray:
    """参照実装: (n,d) -> (n,n) squared euclidean distance matrix"""
    X = np.asarray(X, dtype=float)
    G = X @ X.T
    diag = np.diag(G)
    D2 = diag[:, None] + diag[None, :] - 2 * G
    D2 = np.maximum(D2, 0.0)
    np.fill_diagonal(D2, 0.0)
    return D2


def center_gram_from_D2(D2: np.ndarray) -> np.ndarray:
    """B = -1/2 * J D^2 J"""
    D2 = np.asarray(D2, dtype=float)
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * J @ D2 @ J


def pairwise_dists(X: np.ndarray) -> np.ndarray:
    return np.sqrt(pairwise_sq_dists_reference(X))


# --------------------------
# squared_distance_matrix のテスト
# --------------------------
def test_squared_distance_matrix_matches_reference():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    D2_ref = pairwise_sq_dists_reference(X)
    D2 = squared_distance_matrix(X)

    assert D2.shape == (10, 10)
    assert np.allclose(D2, D2_ref, atol=1e-12, rtol=1e-12)
    assert np.allclose(D2, D2.T, atol=1e-12)
    assert np.all(np.diag(D2) == 0.0)
    assert np.min(D2) >= -1e-12


# --------------------------
# MultiDimensionalScaler 入力バリデーション
# --------------------------
def test_init_rejects_non_square():
    D2 = np.zeros((3, 4))
    with pytest.raises(ValueError):
        MultiDimensionalScaler(D2)


def test_init_rejects_non_symmetric():
    D2 = np.array([[0.0, 1.0],
                   [2.0, 0.0]])
    with pytest.raises(ValueError):
        MultiDimensionalScaler(D2)


def test_init_rejects_negative_entries():
    D2 = np.array([[0.0, -1.0],
                   [-1.0, 0.0]])
    with pytest.raises(ValueError):
        MultiDimensionalScaler(D2)


# --------------------------
# gram_matrix の整合性
# --------------------------
def test_gram_matrix_matches_centering_formula():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 4))
    D2 = pairwise_sq_dists_reference(X)

    mds = MultiDimensionalScaler(D2)
    B1 = mds.gram_matrix()
    B2 = center_gram_from_D2(D2)

    assert np.allclose(B1, B2, atol=1e-10, rtol=1e-10)


# --------------------------
# embed(k) の距離再構成テスト
# --------------------------
def test_embed_reconstructs_distances_when_k_is_true_dim():
    rng = np.random.default_rng(2)
    n, true_dim = 20, 3
    X_true = rng.normal(size=(n, true_dim))
    D2 = pairwise_sq_dists_reference(X_true)

    mds = MultiDimensionalScaler(D2)
    X_hat = mds.embed(k=true_dim)

    # 距離が一致するか（回転・符号の不定性は距離なら問題なし）
    assert np.allclose(pairwise_dists(X_hat), np.sqrt(D2), atol=1e-6, rtol=1e-6)


def test_embed_invariant_under_rigid_transform_distance_level():
    rng = np.random.default_rng(3)
    n, d = 25, 2
    X = rng.normal(size=(n, d))

    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    t = np.array([10.0, -3.0])
    X2 = X @ R + t

    D2_1 = pairwise_sq_dists_reference(X)
    D2_2 = pairwise_sq_dists_reference(X2)

    X_hat1 = MultiDimensionalScaler(D2_1).embed(k=d)
    X_hat2 = MultiDimensionalScaler(D2_2).embed(k=d)

    assert np.allclose(pairwise_dists(X_hat1), pairwise_dists(X_hat2), atol=1e-6, rtol=1e-6)


# --------------------------
# is_euclidean / minimal_dimension
# --------------------------
def test_is_euclidean_true_for_euclidean_D2():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(15, 5))
    D2 = pairwise_sq_dists_reference(X)
    mds = MultiDimensionalScaler(D2)

    assert mds.is_euclidean(tol=1e-10)


def test_is_euclidean_false_for_non_euclidean_D2_example():
    # “必ず”非ユークリッドになる簡単例を作るのは意外と難しいので
    # Gram を明確に indefinite にしてから D2 に変換する。
    # 1次元の距離行列では必ずPSDなので、少なくとも3点以上で作る。
    rng = np.random.default_rng(5)
    n = 6

    # center された Gram を「明示的に不定」にする（負の固有値を入れる）
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eig = np.array([2.0, 1.0, 0.5, -0.2, -0.4, -1.0])  # 负を含める
    B = Q @ np.diag(eig) @ Q.T
    B = 0.5 * (B + B.T)

    # B -> D2 へ（D2_ij = B_ii + B_jj - 2 B_ij）
    diag = np.diag(B)
    D2 = diag[:, None] + diag[None, :] - 2 * B
    D2 = np.maximum(D2, 0.0)
    np.fill_diagonal(D2, 0.0)

    mds = MultiDimensionalScaler(D2)
    # 数値誤差もあるので tol は少し厳しすぎない値に
    assert mds.is_euclidean(tol=1e-12) is False


def test_minimal_dimension_equals_true_dim_when_exact():
    rng = np.random.default_rng(6)
    n, true_dim = 18, 2
    X_true = rng.normal(size=(n, true_dim))
    D2 = pairwise_sq_dists_reference(X_true)

    mds = MultiDimensionalScaler(D2)
    k = mds.minimal_dimension(eps=1e-10)
    assert k == true_dim


def test_reconstruction_error_monotone_non_increasing():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(16, 4))
    D2 = pairwise_sq_dists_reference(X)
    mds = MultiDimensionalScaler(D2)

    # k を増やすと再構成誤差は増えない（理論的には単調非増加）
    errs = [mds.reconstruction_error(k) for k in range(0, 5)]
    for a, b in zip(errs, errs[1:]):
        assert b <= a + 1e-12

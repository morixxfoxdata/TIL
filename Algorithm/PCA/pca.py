import numpy as np
from scipy.sparse.linalg import svds
class PCA:
    def __init__(self, n_components, tol=0.0, random_seed=0):
        # initialization
        """
        param
        ------
        n_components: 次元圧縮後の次元数
        tol: SVDを計算するライブラリに渡す値. 計算誤差の許容範囲?
        random_seed: 乱数のシード値
        """
        self.n_components = n_components
        self.tol = tol
        self.random_state_ = np.random.RandomState(random_seed)

    def fit(self, X):
        """
        args
        ------
        X: データ行列
        """
        # Xのshapeにおいて少ないほうのサイズで乱数生成(サイズは列数)
        v0 = self.random_state_.randn(min(X.shape))
        # 列ごとの平均を格納(サイズは列数)
        xbar = X.mean(axis=0)
        # Xの各行から平均値を引いたもの(サイズはXと同じ)
        Y = X - xbar
        # Sは共分散行列. サイズは列数×列数
        S = np.dot(Y.T, Y)
        U, Sigma, VT = svds(S, k=self.n_components, tol=self.tol, v0=v0)
        self.VT_ = VT[::-1, :]
    
    def transform(self, X):
        return self.VT_.dot(X.T).T


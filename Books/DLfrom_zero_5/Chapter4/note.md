# 4. 混合ガウスモデル

## 4.1 身の回りにある多峰性分布

**男女混合の身長の分布**
年齢と性別を限定した場合に正規分布になることが知られているが, 性別を限定しない場合はどうか？

男女で二つの山が存在することになる(二つの分布が存在する)

**アリの体長**
アリをサンプリングして体長を記録していくと多峰性分布になる(役割によってサイズが異なるから)

**間欠泉の噴出**
4.1.1 で用いる

### 4.1.1 多峰性分布のデータセット

```Python
import os
import numpy as np

path = os.path.join(os.path.dirname(__file__), 'old_faithful.txt')
xs = np.loadtxt(path)
print(xs.shape)
print(xs[0])
```

(272, 2)
[ 3.6 79. ]

データセットは np.ndarray として読み込まれる. 1 つ目のデータは噴出した時間が 3.6 分であり, 次の噴出までの間隔が 79 分だったことを示す.

散布図に示す.

![alt text](4_4.png)

## 4.2 混合ガウスモデルのデータ生成

複数の正規分布からなるモデルは**混合ガウスモデル**と呼ばれる.

1. モデル化: 観測データの分布が GMM によって表現できると仮定する
2. パラメータ推定: GMM のパラメータ推定を行う.

### 4.2.1 GMM によるデータ生成

1. 2 の正規分布の中から, ある確率分布に従って 1 つを選ぶ
2. 選んだ正規分布からデータを生成する

   この手順でデータを生成するコードをかく

### 4.2.2 データ生成を行うコード

```Python
import numpy as np
import matplotlib.pyplot as plt
# ====== 学習済パラメータ ======
# 平均行列(0: 1つ目の平均, 1: 2つ目の平均)
mus = np.array([[2.0, 54.50],
                [4.3, 80.0]])
# 共分散行列の行列
covs = np.array([[[0.07, 0.44],
                 [0.44, 33.7]],
                 [[0.17, 0.94],
                  [0.94, 36.00]]])
# 各正規分布が選ばれる確率
phis = np.array([0.35, 0.65])
# ===========================

def sample():
    z = np.random.choice(2, p=phis)
    mu, cov = mus[z], covs[z]
    x = np.random.multivariate_normal(mu, cov)
    return x

N = 500
xs = np.zeros((N, 2))
for i in range(N):
    xs[i] = sample()

# print(xs)
plt.scatter(xs[:, 0], xs[:, 1], color='orange', alpha=0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

生成されるデータは以下のような散布図になる.
二つの正規分布に従っている.

![alt text](4_6.png)

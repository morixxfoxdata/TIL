# 3. 多次元正規分布

## 3.1 Numpy と多次元配列

```Python
import numpy as np
x = np.array([1, 2, 3])

print(x.__class__)  # クラス名
print(x.shape)  # 形状
print(x.ndim)   # 次元数
```

要素ごとの演算を行う. 積はアダマール積と呼ばれる。
通常のベクトル積や行列積は np.dot で行う。

```Python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
y = np.dot(a, b)
print(y)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
Y = np.dot(A, B)
print(Y)
```

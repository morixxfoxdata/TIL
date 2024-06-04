# EM アルゴリズム

## KL ダイバージェンス

### 5.1.1 数式の表記

連続型の確率変数 $x$ があり, その確率密度が $p(x)$ で表されるものとする.
このとき, 関数 $f(x)$ の期待値は次の式で表す.

$$
\mathbb{E}_{p(x)}[f(x)] = \int f(x)p(x)dx
$$

確率分布 $q(x)$ に関する期待値であれば

$$
\mathbb{E}_{q(x)}[f(x)] = \int f(x)q(x)dx
$$

また, パラメータ $\theta$ を持つ確率分布は $p(x;\theta)$ を $p_\theta (x)$ という表記にする.
正規分布の場合は $\theta: \sigma, \mu$

### 5.1.2 KL ダイバージェンスの定義式

2 つの確率分布を測る尺度に **KL ダイバージェンス(Kullback-Leibler divergence)** がある.
2 つの確率分布 $p(x)$, $q(x)$ が与えられたとき

$$
D_{KL}(p\space\|\space q) = \int p(x)\log \frac{p(x)}{q(x)}dx
$$

上の式は, $x$ が連続型の確率変数の場合の KL ダイバージェンス. x が離散型の場合は

$$
D_{KL}(p\space\|\space q) = \sum_{x}p(x)\log \frac{p(x)}{q(x)}
$$

KL は以下の特性がある.

- 2 つの確率分布が異なるほど大きな値を示す.
- 0 以上の値を取り, 2 つの確率分布が同じ時のみ 0 をとる.
- 非対称な尺度であるため, $D_{KL}(p\space\|\space q) \ne D_{KL}(q\space\|\space p)$

これより, **KL ダイバージェンスは 2 つの確率分布がどれくらい異なるか** を表す尺度として利用できる.
この特性について, 具体例とともに考える.
コインの表と裏の出る確率が次のように決まっているとする.

| 事象         | 確率  |
| ------------ | ----- |
| 表の出る確率 | $70$% |
| 裏の出る確率 | $30$% |

これがコインの真の確率分布であり, ここでは $p$ という記号で表す. 続いてある人がこのコインの確率分布を次のように推定したとする.

| 事象         | 確率  |
| ------------ | ----- |
| 表の出る確率 | $50$% |
| 裏の出る確率 | $50$% |

この推定した確率を $q$ とする. この時, 真の確率分布と推定確率分布の KL ダイバージェンスは以下のように計算できる.

$$
D_{KL}(p \space\|\space q) = 0.7\log\frac{0.7}{0.5}+0.7\log\frac{0.3}{0.5} \\
= 0.021 \dots
$$

例えば別の人が以下のように推定したとすると, その時の KL ダイバージェンスは次の値になる.

$$
D_{KL}(p \space\|\space q) = 0.7\log\frac{0.7}{0.2}+0.7\log\frac{0.3}{0.8} \\
= 0.58 \dots
$$

真の確率分布から遠いほど KL の値が大きくなる.

### 5.1.3 KL ダイバージェンスと最尤推定の関係

ここで KL ダイバージェンスと最尤推定の関係について考える.

今, ここに「真の確率分布 $p_{*}(x)$ 」があり, サンプルデータ $\mathcal{D} = \lbrace{x^{(1)}, x^{(2)}, \dots , x^{(N)}}\rbrace$ を生成したとする.
我々が目指すことは, パラメータ $\theta$ で調整できる $p_{\theta}(x)$ を使って, $p_{*}(x)$ にできるだけ近い確率分布を作ること.

最尤推定では次の対数尤度を目的関数とする.

$$
\log \prod_{n=1}^{N}p_{\theta}(x^{(n)}) = \sum_{n=1}^{N}\log p_{\theta}(x^{(n)})
$$

そしてこの対数尤度を最大化するパラメータ $\theta$ は次の式で表される.

$$
\hat{\theta} = \argmax_{\theta}\sum_{n=1}^{N}\log p_{\theta}(x^{(n)})
$$

これは KL ダイバージェンスを最小にするという問題から導ける.

$$
D_{KL}(p_{*} \space\|\space p_{\theta}) = \int p_{*}(x)\log\frac{p_{*}(x)}{p_{\theta}(x)}dx
$$

上式を最小化するために計算するには全ての $x$ について積分する必要がある. しかし $p_{*}(x)$ の具体的な数式表現が不明なため, 計算不可能. ここで**モンテカルロ法**を用いて近似する.

モンテカルロ法は, 乱数を用いて複雑な確率分布や期待値などの近似値を計算するための手法.
ランダムに生成されたサンプルを用いて問題をシミュレートし, それらのサンプルから求めた結果の平均をとることで, 問題の解を近似する.

ここでは次の期待値をモンテカルロ法を用いて近似する.

$$
\mathbb{E}_{p_{*}(x)}[f(x)] = \int p_{*}(x)f(x)dx
$$

- $p_{*}(x)$ に基づいてサンプル $\lbrace{x^{(1)}, x^{(2)}, \dots , x^{(N)}}\rbrace$ を生成する
- 各データ $x^{(i)}$ における $f(x^{(i)})$ を求め, その平均を計算する.
  この手順により上記の積分を近似して表すことができる.

$$
\mathbb{E}_{p_{*}(x)}[f(x)] = \int p_{*}(x)f(x)dx
$$

$$
\approx \frac{1}{N}\sum_{n=1}^{N}f(x^{(n)})
$$

では先ほどの KL ダイバージェンス最小化問題について近似する.

$$
D_{KL}(p_{*} \space\|\space p_{\theta}) = \int p_{*}(x)\log\frac{p_{*}(x)}{p_{\theta}(x)}dx
$$

$$
\approx \frac{1}{N}\sum_{n=1}^{N}\log\frac{p_{*}(x^{(n)})}{p_{\theta}(x^{(n)})}
$$

$$
= \frac{1}{N}\sum_{n=1}^{N}\left(\log p_{*}(x^{(n)})- \log p_{\theta}(x^{(n)})\right)
$$

ここでの目標は $D_{KL}(p_{*} \space\|\space p_{\theta})$ を最小にする $\theta$ を求めることである.

つまり以下のように書ける.

$$
\argmin_{\theta} D_{KL}(p_{*} \space\|\space p_{\theta}) \approx \argmin_{\theta} \left(-\frac{1}{N}\sum_{n=1}^{N}\log p_{\theta}(x_n) \right)
$$

$$
= \argmin_{\theta} \left(-\sum_{n=1}^{N}\log p_{\theta}(x_n) \right)
$$

$$
= \argmax_{\theta}\sum_{n=1}^{N}\log p_{\theta}(x_n)
$$

以上から

$$
\argmin_{\theta}D_{KL}(p_{*} \space\|\space p_{\theta}) \approx \argmax_{\theta}\sum_{n=1}^{N}\log p_{\theta}(x_n)
$$

左辺が KL ダイバージェンスを最小化する $\theta$ で, 右辺は対数尤度が最大となる $\theta$ である. この二つが等しいことが示された.

## 5.2 EM アルゴリズムの導出 ①

目的：混合ガウスモデルのパラメータ推定

### 5.2.1 潜在変数を持つモデル

ここでは観測できる確率変数を $x$, 潜在変数を $z$, パラメータを $\theta$ で表す.
この時, 一つのデータに対する対数尤度は, 確率の周辺化により

$$
\log p_\theta(x) = \log\sum_{z}p_\theta(x, z)
$$

ここでは潜在変数が離散と仮定しているが, 連続な場合は積分になるだけ.

次に, サンプル $\mathcal{D} = \lbrace{x^{(1)}, x^{(2)}, \dots , x^{(N)}}\rbrace$ が得られた場合を考える.
この時の対数尤度は

$$
\log p_\theta(\mathcal{D}) = \sum_{n=1}^{N}\log p_{\theta}(x^{(n)})
$$

$$
= \sum_{n=1}^{N}\log \sum_{x^{(n)}}p_\theta(x^{(n)}, z^{(n)})
$$

上記の対数尤度を最大化したいが, "log-sum" の形になっており解析的に解けない.
EM アルゴリズムはこの問題を"sum-log"に変換することで解決する.

まずは 1 つのデータ $x$ に関する対数尤度について考える. 対数尤度が複雑な形をとる理由としては"log-sum"の形をしているから.
"log-sum"の形を解決するために確率の乗法定理を利用すると

$$
\log p_\theta(x) = \log \frac{p_\theta(x, z)}{p_\theta(z|x)}
$$

一見"log-sum"を解決しているように見えるが, 分母がベイズの定理より

$$
p_\theta(z|x) = \frac{p_\theta(x, z)}{\sum_z p_\theta(x, z)}
$$

結局"log-sum"の形となる.

### 5.2.2 任意の確率分布 $q(z)$

任意の確率分布を $p_\theta(z|x)$ の近似分布として導入する.

ここでは, $p_\theta(z|x)$ の代わりに $q(z)$ を使うために以下のようにする.

$$
\log p_\theta(x) = \log \frac{p_\theta(x, z)}{p_\theta(z|x)}
$$

$$
= \log \frac{p_\theta(x, z)q(z)}{p_\theta(z|x)q(z)}
$$

$$
= \log \frac{p_\theta(x, z)}{q(z)} + \log \frac{q(z)}{p_\theta(z|x)}
$$

第 1 項からは厄介な条件付き確率を消せたが, 第 2 項には残っている.

ここで第 2 項を KL ダイバージェンスの形式に変形することができれば先の見通しが立つ.

$$
\log p_\theta(x) = \log p_\theta(x)\sum_{z}q(z)
$$

$$
= \sum_{z}q(z)\log p_\theta(x)
$$

$$
= q(z) \left(\log \frac{p_\theta(x, z)}{q(z)} + \log \frac{q(z)}{p_\theta(z|x)} \right)
$$

$$
= \sum_{z}q(z)\log \frac{p_\theta(x, z)}{q(z)} + \sum_{z}q(z)\log \frac{q(z)}{p_\theta(z|x)}
$$

$$
= \sum_{z}q(z)\log \frac{p_\theta(x, z)}{q(z)} + D_{KL}(q(z) \| p_\theta (z|x))
$$

## 5.3 EM アルゴリズムの導出 ②

### 5.3.1 ELBO (エビデンスの下界)

$$
\sum_{z}q(z)\log \frac{p_\theta(x, z)}{q(z)} + D_{KL}(q(z) \| p_\theta (z|x))
$$

まず, 第 2 項の KL ダイバージェンスに注目する.
$q(z)$ は任意の確率分布であるがどのような確率分布であっても KL ダイバージェンスは常に 0 以上になるため

$$
\log p_\theta(x) = \sum_{z}q(z)\log \frac{p_\theta(x, z)}{q(z)} + D_{KL}(q(z) \| p_\theta (z|x))
$$

$$
\ge \sum_{z}q(z)\log \frac{p_{\theta}(x,z)}{q(z)}
$$

ここからわかることとして, KL ダイバージェンスが 0 以上であるが故に, 式第 1 項は必ず対数尤度以下の値をとる. それゆえ, この項を **ELBO(Evidence Lower BOund)** という.

- なぜエビデンス？
  対数尤度が大きくなることは求めたい $q, \theta$ が正しい方向を示している証拠となるから

ELBO は

$$
\text{ELBO}(x;q,\theta) = \sum_{z}q(z)\log \frac{p_{\theta}(x,z)}{q(z)}
$$

以下のような特徴がある.

- 対数尤度は常に ELBO 以上の値となる
- ELBO は"sum-log"の形になっていて解析しやすい

以上から, ELBO を大きくするようにパラメータを更新すれば対数尤度はそれ以上の値となる.
解析できない対数尤度の代わりに ELBO を最適化の対象とすることを考える.

### 5.3.2 EM アルゴリズムへ

ELBO には $q(z), \theta$ の二つのパラメータがあり, ELBO が大きくなるようにこれらのパラメータを最適化する.

一度に 2 変数を最適化するのは困難であるため, $\theta$ を固定して $q(z)$ を更新, $q(z)$ を固定して $\theta$ を更新という作業を繰り返す.

まず $\theta = \theta_{\text{old}}$ として $q(z)$ の最適化を考える.
$q(z)$ は任意の確率分布であり, どのような分布であっても対数尤度は ELBO 以上の値をとる.ただし, $q(z)$ の分布によって ELBO が対数尤度にどれくらい近づくかが変わる.
ここで注目すべきは以下の式である.

$$
\log p_{\theta}(x) = ELBO(x; q, \theta) + D_{KL}(q(z) \| p_\theta (z|x))
$$

これは, **KL ダイバージェンスと ELBO の和は $q(z)$ の値によらず一定であるということ**.
KL ダイバージェンスは二つの確率分布が等しい時に 0 になる.
つまり, $q(z) = p_{\theta}(z|x)$ である時に KL 項は 0 となり, ELBO が最大となる.

以上より, $q(z) = p_{\theta}(z|x)$ が更新式となる.

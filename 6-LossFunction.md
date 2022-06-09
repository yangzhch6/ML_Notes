# Loss Functions
# 分类损失
## 1. log loss
$$
L = \sum_{i=1}^{K}y_i log(p_i)  
$$

## 2. KL散度(相对熵) 与 交叉熵(corss entropy)
**熵**：对变量不确定性的度量(熵越大，不确定性越大，信息量越大)  
随机变量$x$，其取值为$x_1, x_2, \dots, x_K$, 其熵可表示为:

$$
H(x) = -E[p(x)logp(x)] = -\sum_{i=1}^{K} p(x_i) logp(x_i) 
$$

**KL散度(相对熵)**：对两个变量分布**距离**的度量(KL散度越大，两随机变量分布越远)   
若在$x_1, x_2, \dots, x_K$上，分布了俩随机变量$p, q$，其KL散度可表示为：

$$
D_{KL}(p||q) = \sum_{i=1}^{K}p(x_i)log\frac{p(x_i)}{q(x_i)}
$$   

对KL散度作变换：

$$
D_{KL}(p||q) = \sum_{i=1}^{K}p(x_i)logp(x_i) - \sum_{i=1}^{K}p(x_i)logq(x_i) \\
= -H(p) - (- H(p,q)) \\
= H(p, q) - H(p)
$$

也即，**KL散度 = 交叉熵 - 信息熵**

**交叉熵**：对两个概率分布**差异性**的度量(交叉熵越小，差异越小)

$$
H(p,q) = -\sum_{i=1}^{K}p(x_i)logq(x_i)
$$

`值得注意的是，分类任务中，最小化KL散度和最大化交叉熵是等价的。（为何我们用信息熵作为损失函数?）`  
考虑俩随机变量，$p$为真实标签的分布概率，而$q$为模型输出的分布概率。则此时，$p$的信息熵变成了一个常量，那么最小化KL散度$D_{KL}(p||q) = H(p, q) - H(p)$相当于最大小化交叉熵$H(p,q)$.

一般情况下，交叉熵损失函数为：

$$
L_{ce} = -\sum_{i=1}^{n}y_{ij} logp(p_{ij})\\
j \in 0, 1, \dots, K 
$$

其中$y_{ij}$表示第$i$个样本的标签是$j$，$p_{ij}$表示第$i$个样本在第$j$类的概率输出。

题外话:
1. 此外**最小化交叉熵完全等价于极大似然估计**：https://zhuanlan.zhihu.com/p/51099880
2. [贝叶斯派vs概率派](https://earyant.github.io/posts/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%9F%A5%E8%AF%86%E6%95%B4%E7%90%86/4-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80/%E5%9F%BA%E7%A1%80%E7%90%86%E8%AE%BA%20-%20%E9%A2%91%E7%8E%87%E6%B4%BE%20vs%20%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%B4%BE/#:~:text=%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%B4%BE%E8%AE%A4%E4%B8%BA%E4%B8%96%E7%95%8C,%E5%85%88%E9%AA%8C%E6%A6%82%E7%8E%87%E7%9A%84%E4%BF%AE%E6%AD%A3%E3%80%82) | [二者示例](https://blog.csdn.net/fq_wallow/article/details/104383057)

## 3. Balanced CE
其实就是根据样本权重分配loss权重

$$
L_{ce} = -\sum_{i=1}^{n}\alpha_jy_{ij} logp(p_{ij})\\
\alpha_j = \frac{n_j}{\sum_{k=1}^{K}n_k}\\
j \in 0, 1, \dots, K 
$$

其中$n_k$代表第$k$类样本数量

## 4. Focal loss
$\lambda$是focal loss里的可调节因子。
考虑$K$分类：

$$
L_{ce} = -\sum_{i=1}^{n}y_{ij} (1-p_{ij})^{\lambda} logp(p_{ij})\\
j \in 0, 1, \dots, K 
$$

**Focal Loss优点**:
相对于加入了样本均衡因素的CE，Focal Loss侧重于对难以分类的样本。因为$(1-p_{ij})^{\lambda}$中，若$p_{ij}$越小，代表该样本更难分类，因此对其赋予的loss更大。

-----
# 回归损失
## 1. MAE (L1 Loss)

$$
L_{MAE} = |y - f(x)|
$$

## 2. MSE (L2 Loss)

$$
L_{MSE} = |y - f(x)|^2
$$

**Comparison between MAE & MSE**:
1. 从优化的角度，MAE在极值点附近的梯度不接近0，因此不利于优化，相对于MSE而言难以收缩到极小值。 
2. 从异常值的角度来说，MAE对异常值和非异常值的梯度均是相同(线性)，一视同仁。而MSE从梯度角度而言，对异常值更敏感，更容易拟合异常值。

另一个角度是，MSE最优解位于均值处，MAE最优解位于中位数处，显然中位数解比均值解对异常值的鲁棒性更强。

## 3. Huber Loss
结合了MAE和MSE，设定了一个阈值$\delta$

$$
L_{Huber} = 
\begin{cases}
|y-f(x)|^2\hspace{5mm} & |y-f(x)|<\delta\\
2\delta|y-f(x)| + \delta^2 & else
\end{cases}
$$

**Huber Loss**的好处：
1. 在Loss较小的时候，使用MSE，这样在极值点处的梯度较小，容易优化。(修正了MAE的问题)
2. 在loss较大的时候(异常值)处，使用MAE，这样可以防止异常值梯度过大，影响模型的优化。

**上述三个Loss的选择**(个人认为)：
1. 如果异常值在任务中很重要：MSE > Huber Loss > MAE
2. 若异常值不重要且想降低其影响：Huber Loss > (MAE > MSE)

## 4. Hinge Loss (折页损失函数)

$$
L_{Hinge} = max(0, 1-y\hat{y})
$$

SVM里软间隔曾出现过。

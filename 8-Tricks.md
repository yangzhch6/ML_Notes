# Tricks
## 1. 

-----
## 2. Normalization
### 2.1 Batch Normalization
BN的理解重点在于它是针对整个Batch中的样本在同一维度特征在做处理。 

**在MLP中**，比如我们有10行5列数据。5列代表特征，10行代表10个样本。是对第一个特征这一列（对应10个样本）做一次处理，第二个特征（同样是一列）做一次处理，依次类推。

**在CNN中扩展**，我们的数据是N·C·H·W。其中N为样本数量也就是batch_size，C为通道数，H为高，W为宽，BN保留C通道数，在N,H,W上做操作。如果是RGB图像，只需3维normalization。

//后续需要补充公式，training和testing的区别，以及testing时方差的期望是怎么计算成$\frac{m}{m-1}$的

<ins>**BN优点**</ins>：
1. 加快网络的收敛
2. BN具有提高网络泛化能力的特性
3. 缓解过拟合
4. 保证了梯度的稳定，缓解梯度爆炸/消失

<ins>**BN缺点**</ins>   
1. batch_size较小的时候，效果差。因为batch size较小时，每个batch计算出的样本均值和方差波动较大，不易于训练。
2. 不太适用于RNN这种处理变长数据的任务/模型。实际上是可以用在RNN身上，但由于同一batch内长度分布不一致，容易导致效果不好

<ins>**BN核心思想**</ins>：BN的提出原本不是为了防止梯度消失或者防止过拟合，其核心是通过对系统参数搜索空间进行约束来增加系统鲁棒性，这种约束压缩了搜索空间，约束也改善了系统的结构合理性，这会带来一系列的性能改善，比如加速收敛，保证梯度，缓解过拟合等。
   
### 2.2 Layer Normalization

`BN和LN都有两个可学习权重`：$\lambda$ 和 $\beta$
$$
x' = \frac{x-E(x)}{\sqrt{Var(x) + \epsilon}}*\lambda + \beta
$$


### Q1: Normalization为什么有效
解决了Internal Covariate Shift。简单理解就是随着层数的增加，中间层的输出会发生“漂移。这会导致：
1. 每层神经元的输入数据不再是“独立同分布”。
2. 上层参数需要不断适应新的输入数据分布，降低学习速度。
3. 下层输入的变化可能趋向于变大或者变小，导致上层落入饱和区，使得学习过早停止。
4. 每层的更新都会影响到其它层，因此每层的参数更新策略需要尽可能的谨慎。

<!-- `LN其实也一定程度解决/缓解了ICT` -->

### Q2: 为什么CV用BN而不是LN，为什么NLP用LN而不是BN
BN放缩的同一特征下不同样本的数据，而LN放缩的是一行数据。而在CV中，没有必要对不同通道内的特征统一放缩。更通俗的讲，不同通道的特征分布本就不同，对其进行LN的行放缩没有意义。

而在NLP中，每个词的embedding是认为定义训练得到，对于句子而言做CV里的BN就显得没有意义。

### Q3: 为什么Layer Norm是对每个单词的Embedding做归一化？
因为不同句子长度不同，其在输入时会做padding，那么句子特征就会被padding的embedding干扰。



### Q4: 为什么BatchNorm和Dropout一起用会掉点？
参考链接3，4
简言之，在BN层前加Dropout，会导致BN层在training和testing阶段，其输入的方差不一致(方差偏移问题)。

解决方案：
1. 只在所有BN层的后面采用dropout层
2. 模型训练完后，固定参数，以测试模式对训练数据求BN的均值和方差，再对测试数据进行归一化，论文证明这种方法优于baseline。

相关链接：
1. [深度解析Batch normalization（批归一化）](https://zhuanlan.zhihu.com/p/435507061)
2. [样本方差的期望](https://www.cnblogs.com/yxzfscg/p/4959460.html)
3. [Batch Norm和Dropout的方差偏移问题](https://zhuanlan.zhihu.com/p/407767348)
4. [Batch Normalization和Dropout如何搭配使用？](https://blog.csdn.net/hexuanji/article/details/103635335)
5. https://arxiv.org/pdf/1801.05134.pdf

-----

## 3. 残差网络
`从反向传播和集成学习两个角度讲述为何残差网络反向传播能解决深层网络梯度问题`
https://zhuanlan.zhihu.com/p/452867110

-----
## 4. 初始化问题
随机初始化如果初始化值太大会出现饱和，进而在一些激活函数上导致梯度弥散；如果初始值太小也会导致梯度弥散。这是因为矩阵乘法的影响，在经过每一个线性层的时候，都会发生相同方向的方差放大或者缩小（雅可比矩阵连乘），因而会带来梯度爆炸和梯度消失的问题


为了防止以上问题的出现，我们坚持以下**原则(Glorot条件)：**
1. <ins>**forward时各层激活值分布保持一致，且需保证不处在梯度饱和区**</ins>
2. <ins>**backward时各层梯度分布保持一致**</ins>

### 4.0 全0初始化，或者全相同数值初始化
会导致反向传播时同层神经元梯度趋同，相当于同层神经元一直是一样的权重。相同的网络参数提取不到有用的特征，因此不会使用相同数值的初始化。

### 4.1 均匀分布/正态分布 初始化 (不考虑激活函数)
神经网络如果保持每层的信息流动是同一方差，那么会更加有利于优化。这也是normalization方法的意义所在。

考虑神经网络
$$y = \sum_{i=1}^{n}w_ix_i$$   
其中$n$为上一层神经网络的数量(当前层网络输入的维度)。**假设$x,w$独立同分布，且他们的均值均为0。**  

计算$y$的方差：

$$
\begin{align}
Var(y) &= \sum_{i=1}^{n}Var(w_ix_i)\\
&= \sum_{i=1}^{n} E[w_i]^2Var(x_i) + E[x_i]^2Var(w_i) + Var(x_i)Var(w_i)\\
&= \sum_{i=1}^{n}Var(x_i)Var(w_i) \\
&= nVar(w)Var(x)
\tag{4.1.1}
\end{align}
$$

也可以是（$x, y$相互独立且均值为0，则$D(xy) = D(x)D(y)$，直接带入即可得到）:

$$
Var(y) = \sum_{i=1}^{n}Var(w_ix_i) = nVar(w)Var(x)
\tag{4.1.2}
$$

我们希望神经网络训练过程中满足每层的方差保持不变

$$
Var(w) = \frac{1}{n}
\tag{4.1.3}
$$   

也即，在不考虑激活函数情况下，满足上式的初始化即可。  
在原本最简单的均匀分布/正态分布初始化中，并没有考虑backward的情况。

### 4.2 Xavier 初始化
考虑神经网络

$$
y^j_l = \sum_{i=1}^{n_l}w^{ij}_lx^i_l \\
x_{l+1} = f(y_l)
$$   

其中$n_l$为上一层层神经网络的输出维度(当前第$l$层网络输入的维度)，$f$为激活函数tanh。

首先介绍**Glorot假设**：
1. 输入的每个特征方差相同 $Var(x_i) = Var(x_j)$
2. 激活函数关于原点对称，这样其均值为0
3. $f'(0) = 1$
4. 初始时，激活函数的输入落在线性区域，也即$f'(y) = 1$

此外，我们还假设$x,w$独立同分布，且他们的均值均为0。    

$$
\begin{align}
Var(x_{l+1}) &= Var(f(y_l)) \\
&= Var[f(\sum_{i=1}^{n_l}w_lx_l)] ~~(由于初始化时，其落在线性区域)\\
&= \sum_{i=1}^{n_l}Var(w^i_lx^i_l)\\
&= \sum_{i=1}^{n_l} E[w^i_l]^2Var(x^i_l) + E[x^i_l]^2Var(w^i_l) + Var(x^i_l)Var(w^i_l)\\
&= \sum_{i=1}^{n_l}Var(x^i_l)Var(w^i_l) \\
&= n_lVar(w_l)Var(x_l)
\end{align}
\tag{4.2.1}
$$

也可以是（$x, y$相互独立且均值为0，则$D(xy) = D(x)D(y)$，直接带入即可得到）:

$$
Var(x_{l+1}) = \sum_{i=1}^{n_l}Var(w^i_lx^i_l) = n_lVar(w_l)Var(x_l)
\tag{4.2.2}
$$

我们希望神经网络forward过程中满足每层激活值的方差保持不变，也即$Var(x_{l+1}) = Var(x_l)$，我们可以得到：

$$
Var(w_l) = \frac{1}{n_l}
\tag{4.2.3}
$$   

但上述结论只考虑了forward.       
根据4.2.2式子，更进一步则有:

$$
Var(x_{l+1}) = Var(x_1)\prod_{i=1}^{l}n_iVar(w_i)
\tag{4.2-forward}
$$

其中，$Var(x_1)$为整个神经网络最开始的输入，我们一般会对其进行normalization，使得$Var(x_1) = 1$。进而可知，在Xavier初始化下，前相传播神经网络每一层激活输出的方差都接近1

接下来推导反向传播

$$
\frac{\partial L}{\partial w_l^{kj}} = \frac{\partial y_l}{\partial w_l^{kj}} \frac{\partial L}{\partial y_l^k} = x_l^j \frac{\partial L}{\partial y_l^k} 
\tag{4.2.4}
$$

接下来我们计算：

$$ 
\frac{\partial L}{\partial y_l^j} = \frac{\partial L}{\partial y_{l+1}} \frac{\partial y_{l+1}}{\partial y_l^j}
= \frac{\partial L}{\partial y_{l+1}} \frac{\partial y_{l+1}}{\partial x_{l+1}^j} \frac{\partial x_{l+1}^j}{\partial y_l^j} \\
= f'(y_l^j) (w_{l+1}^{[:,j]})^T \frac{\partial L}{\partial y_{l+1}} 
\tag{4.2.5}
$$

我们的主要目标是$Var(\frac{\partial L}{\partial w_{l+1}}) = Var(\frac{\partial L}{\partial w_l})$，而根据4.2.4式，我们需要先计算：

$$
\begin{align}
Var(\frac{\partial L}{\partial y_l^j}) &= Var(f'(y_l^j) (w_{l+1}^{[:,j]})^T \frac{\partial L}{\partial y_{l+1}} ) ~~~(由于处于激活函数线性区域) \\
&= Var((w_{l+1}^{[:,j]})^T \frac{\partial L}{\partial y_{l+1}} ) \\
&= Var(\sum_{i=1}^{n_{l+2}} w_{l+1}^{ij} \frac{\partial L}{\partial y_{l+1}^i})\\
&= n_{l+2}Var(w_{l+1}) Var(\frac{\partial L}{\partial y_{l+1}}) 
\tag{4.2.6}
\end{align}
$$

更进一步可以有：

$$
Var(\frac{\partial L}{\partial y_l^j})= Var(\frac{\partial L}{\partial y_d}) \prod_{i=l+1}^{d} n_{i+1}Var(w_i) 
\tag{4.2.7}
$$

其中$d$指该神经网络一共有$d$层。    
将4.2.7带入4.2.4则有：

$$
\begin{align}
Var(\frac{\partial L}{\partial w_l^{kj}}) &= Var(x_l^j \frac{\partial L}{\partial y_l^k}) \\
&= Var(x_l^j)Var(\frac{\partial L}{\partial y_l^k}) \\
&= Var(x_l^j) n_{l+2}Var(w_{l+1}) Var(\frac{\partial L}{\partial y_{l+1}}) \\
&= Var(x_l^j) Var(\frac{\partial L}{\partial y_d}) \prod_{i=l+1}^{d} n_{i+1}Var(w_i)
\end{align}
\tag{4.2.8}
$$

再将4.2-forward带入则有

$$
Var(\frac{\partial L}{\partial w_l^{kj}}) = Var(x_1) Var(\frac{\partial L}{\partial y_d}) \prod_{i=1}^{l-1}[n_iVar(w_i)] \prod_{i=l+1}^{d}[n_{i+1}Var(w_i)]
\tag{4.2.9}
$$

我们依然希望$Var(\frac{\partial L}{\partial w_{l_1}})=Var(\frac{\partial L}{\partial w_{l_2}})~ \forall l_1,l_2$其等价于$Var(\frac{\partial L}{\partial w_{l}})=Var(\frac{\partial L}{\partial w_{l+1}})$，因此有

$$
n_{l+2}Var(w_{l+1}) = n_{l}Var(w_l)
\tag{4.2-backward}
$$  

实际上，如果要同时满足**4.2-forward**和**4.2-backward**，是没有解的。在原文中，作者选择的是近似解。也即，先近似满足**4.2-forward**，这时每层权重符合**4.2.3**式。这时可以近似的认为各层的激活值方差相同$Var(x_l) = Var(x_{l+1})$，那么以此为基础我们重新考量**4.2.8**式:

$$
Var(\frac{\partial L}{\partial w_l^{kj}}) 
= Var(x_l^j) Var(\frac{\partial L}{\partial y_d}) \prod_{i=l+1}^{d} n_{i+1}Var(w_i) \\
Var(\frac{\partial L}{\partial w_{l+1}^{kj}}) 
= Var(x_{l+1}^j) Var(\frac{\partial L}{\partial y_d}) \prod_{i=l+2}^{d} n_{i+1}Var(w_i) \\
\Rightarrow n_{l+1}Var(w_{l}) = 1, \forall l 
$$

又由于其也要满足forward条件，综上，$w$的分布应该满足

$$
n_{l+1}Var(w_{l}) = 1 \\ 
n_{l}Var(w_{l}) = 1, \forall l 
$$

这时我们取两者中间值：

$$
Var(w_l) = \frac{1}{n_l + n_{l+1}}
$$

其中$n_l$是第$l$层神经网络的输出维度，而$n_{l+1}$是输出维度。
<!-- 按照我们在forward中推导出的$Var(x) = 1$，代入可得
$$
Var(\frac{\partial L}{\partial w_l^{kj}}) = n_{l+1}Var(w_{l+1}) Var(\frac{\partial L}{\partial y_{l+1}})
$$

&= Var(x_1)\prod_{i=1}^{l}[n_iVar(w_i)] ~~n_{l+1}Var(w_{l+1}) Var(\frac{\partial L}{\partial y_{l+1}})

正向传播时，上式是从前往后计算的，$Var(w) = \frac{1}{n_{in}}$，反向传播反过来了$Var(w) = \frac{1}{n_{out}}$。当$n_{in}$与$n_{out}$不相同时，我们取均值$n = \frac{n_{in}+n_{out}}{2}$，则$Var(w)$计算为：
$$
Var(w) = \frac{2}{n_{in}+n_{out}}
$$ -->

那么Xavier初始化只需满足以下两点：

$$
E(w) = 0 \\
Var(w) = \frac{2}{n_{in}+n_{out}}
$$

给出<ins>**Xavier初始化的两种形式**</ins>：
1. <ins>**均匀分布** (原文用的均匀分布)</ins>

$$
w \backsim U[-\frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}]]
$$

2. <ins>**正态分布**</ins>

$$
w\backsim N(0, \frac{2}{n_{in}+n_{out}})
$$

**Xavier初始化存在的问题**：主要是针对tanh激活函数设计的，且假设过多。当使用ReLU做激活函数时，因为Relu不关于0点对称，因此神经网络输出均值不为0，不满足推导条件。此外当假设条件不满足是，也不容易满足推导条件。

### 4.3 Kaiming 初始化
一些假设：
1. $E[w] = 0$
2. 且$w$分布关于0对称

还是考虑这个神经网络：

$$
每一层输出  ~~y = \sum_{i=1}^{n}w_ix_i \\
第_l层输入为_{l-1}层输出经\rm{ReLU}得到  ~~~x_l = \rm{ReLU}(y_{l-1})
$$   

我们推导Kaiming初始化：

$$
Var(y) = \sum_{i=1}^{n}Var(w_ix_i)\\
= \sum_{i=1}^{n} E[w_i]^2Var(x_i) + E[x_i]^2Var(w_i) + Var(x_i)Var(w_i)\\
$$

由于有假设$E[w] = 0$, 又由$Var(x) = E(x^2) - E(x)^2$，则有

$$
Var(y) 
= \sum_{i=1}^{n}E[x_i]^2Var(w_i) + Var(x_i)Var(w_i) \\
= \sum_{i=1}^{n}Var(w_i)(E[x_i]^2 + Var(x_i)) \\
= \sum_{i=1}^{n}Var(w_i)E(x_i^2) \\
= nVar(w)E(x^2)
\tag{4.3.1}
$$

考虑当前为第$l$层，记$x_l$的维度为$n_l$，则

$$
Var(y_l) = n_lVar(w_l)E(x_l^2)
\tag{4.3.2}
$$    

对于$E(x_l^2)$可以用$l-1$层来估计：

$$
E(x_l^2) = E(f^2(y_{l-1})) = \int_{-\infty}^{+\infty}p(y_{l-1})f^2(y_{l-1})dy_{l-1}\\
= \int_{0}^{+\infty}p(y_{l-1})y_{l-1}^2dy_{l-1}
\tag{4.3.3}
$$

其中$f$为RELU激活函数.    
又由假设$w$分布关于0对称，所以对任意的输入$x$，均有$y$关于0对称分布，则:

$$
E(y_{l-1}^2) = \int_{-\infty}^{+\infty}p(y_{l-1})t_{l-1}^2dy_{l-1} \\ 
= \int_{0}^{+\infty}p(y_{l-1})t_{l-1}^2dy_{l-1} + \int_{-\infty}^{0}p(y_{l-1})t_{l-1}^2dy_{l-1} \\
=2\int_{0}^{+\infty}p(y_{l-1})t_{l-1}^2dy_{l-1}
\tag{4.3.4}
$$

带入式4.3.3，得到

$$
\frac{1}{2}E(y_{l-1}^2) = E(x_l^2)
\tag{4.3.5}
$$

将其带入4.3.2，并根据$D(x) = E(x^2) - E(x)^2$, $E(y) = 0$则有

$$
Var(y_l) = \frac{1}{2}n_lVar(w_l)E(y_{l-1}^2) \\ 
= \frac{1}{2}n_lVar(w_l)[D(y_{l-1}) + E(y_{l-1})^2] \\ 
= \frac{1}{2}n_lVar(w_l)Var(y_{l-1}) 
\tag{4.3.6}
$$

我们可以不断迭代写出下式:

$$
Var(y_l) = Var(y_1) \prod_{i=1}^{l}\frac{1}{2}n_iVar(w_i)
$$

又由$y_1 = w_1x_1$，且输入的$x_1$经过归一化后$E(x) = 0, D(x) = 1$，带入4.3.1式有

$$
Var(y_1) = n_1Var(w_1) 
$$

```
值得注意的是，我们本质是希望每层激活值的方差分布一致，在Xavier推导中就是如此。但其实如能保证激活函数输入的均值和方差保持一致，也能保证激活函数的输出保持一致。如上式所示，为了方便数学推导，Kaiming初始化推导过程是使用的后者。
```

故有

$$
\frac{1}{2}n_lVar(w_l) = 1 \\
Var(w_l) = \frac{2}{n_l}
\tag{4.3.7}
$$

此处为forward，下面推导反向传播backword

$$
\frac{\partial L}{\partial y_l} = \frac{\partial L}{\partial x_{l+1}} \frac{\partial x_{l+1}}{\partial y_l} \\
= \frac{\partial L}{\partial x_{l+1}} f'(y_l)
\tag{4.3.8}
$$

假设$f'(y_l)$和$\frac{\partial L}{\partial x_{l+1}}$相互独立，接下来计算：

$$
E(\frac{\partial L}{\partial y_l}) = E(\frac{\partial L}{\partial x_{l+1}})E(f'(y_l)) 
= \frac{1}{2}E(\frac{\partial L}{\partial x_{l+1}})
$$

我们假设$E(\frac{\partial L}{\partial x_{l+1}}) = 0$，则有$E(\frac{\partial L}{\partial y_l}) = 0$，
进而则有：

$$
Var(\frac{\partial L}{\partial y_l}) = E[(\frac{\partial L}{\partial y_l})^2] - E[\frac{\partial L}{\partial y_l}]^2 = E[(\frac{\partial L}{\partial y_l})^2]
$$

记$\triangledown x = \frac{\partial L}{\partial x}$，然后：

$$
Var(\triangledown y_l) = var(f'(y_l) \triangledown x_{l+1})\\
= \frac{1}{2}\int_{-\infin}^{0} \triangledown x_{l+1}^2 f_{\triangledown y_l}(\triangledown y_l)d\triangledown y_l + \frac{1}{2}\int_{0}^{+\infin}\triangledown x_{l+1}^2f_{\triangledown y_l}(\triangledown y_l)d\triangledown y_l \\
= \frac{1}{2} \int_{-\infin}^{+\infin}\triangledown x_{l+1}^2f_{\triangledown x_{l+1}}(\triangledown x_{l+1})d\triangledown x_{l+1} \\
= \frac{1}{2} Var(\triangledown x_{l+1})
\tag{4.3.9}
$$

接下来我们进一步计算

$$
Var(\triangledown x_l^k) = Var(\sum_{i=1}^{n_{l+1}}\triangledown y_l^i \frac{\partial y_l^i}{\partial x_l^k}) = Var(\sum_{i=1}^{n_{l+1}}\triangledown y_l^i w_l^{ki}) \\
= n_{l+1}Var(\triangledown y_l)Var(w_l)
$$

将**4.3.9**带入，则有：

$$
Var(\triangledown x_l) = \frac{1}{2} n_{l+1}Var(w_l) Var(\triangledown x_{l+1})
$$

为尽可能满足$Var(\triangledown x_l) = Var(\triangledown x_{l+1})$，故所以，反向传播时则有

$$
Var(w_l) = \frac{2}{n_{l+1}}
$$

```
这一部分为何没有折中还没有搞清楚
```

如果使用正态分布初始化，则为：

$$
w\backsim N(0, \frac{2}{n})
$$

相关链接：
1. [神经网络权重矩阵初始化的意义？](https://www.zhihu.com/question/291032522/answer/605843215)
2. [神经网络中的权值初始化：从最基本的方法到Kaiming方法一路走来的历程](https://mp.weixin.qq.com/s?__biz=Mzg5ODAzMTkyMg==&mid=2247487002&idx=1&sn=864b245384862e42ce1e8529bc7eb62a&chksm=c0698647f71e0f51fe7fdddfc0d2dd717df8af51c0bcca2373537bc9959e5b7ae212c9956384&scene=27#wechat_redirect)
3. [Kaiming初始化的推导](https://zhuanlan.zhihu.com/p/305055975)
4. [Xavier初始化的推导](https://blog.csdn.net/victoriaw/article/details/73000632)
5. [https://arxiv.org/pdf/1502.01852.pdf](https://arxiv.org/pdf/1502.01852.pdf)

-----
## 5. 多卡训练
一般而言，多卡训练分成两种：
1. 模型太大放不进一个GPU里，做成**模型并行**，梯度信息在多卡间传递（消耗时间很大），一般不常见。
2. **数据并行**：每张卡里都有一个模型，并具有不同的数据，每一次iteration后都要进行梯度计算和gpu间通信，以确保每张卡内的模型都经过同样的梯度更新。(常见形式)

接下来介绍下Pytorch里，数据并行的两种实现形式

### 5.1 Data Parallel (单进程，Tree Reduce)

### 5.2 Distributed Data Parallel (多进程，Ring Reduce)

相关链接：
1. [PyTorch 源码解读之 DP & DDP：模型并行和分布式训练解析
](https://zhuanlan.zhihu.com/p/343951042)
2. [DDP系列第一篇：入门教程](https://zhuanlan.zhihu.com/p/178402798)
3. [Pytorch 分布式训练](https://zhuanlan.zhihu.com/p/76638962)
4. [pytorch DDP每个进程都读一整份数据，数据量太大的情况下用什么方法解决内存爆掉的问题？](https://www.zhihu.com/question/423201889/answer/1498519328)
5. [【分布式训练】单机多卡的正确打开方式（一）：理论基础](https://zhuanlan.zhihu.com/p/72939003) (最为推荐)
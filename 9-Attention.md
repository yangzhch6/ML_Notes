# Attention

## 1. 加性和乘性attention
### 1.1 加性attention (Bahdanau Attention)

### 1.2 乘性attention (Luong Attention) 

## 2. transformer
### 2.1 embedding
self attention对位置不敏感，因此需要在embedding中添加位置信息。而position embeddin需要考虑几点:
1. 能在全局上体现谁先谁后
2. 避免局部先后顺序的信息差被总体长度稀释，具有一定的不变性。
3. 需要有值域的范围限制。

### 2.2 transformer layer
#### **Encoder**
#### **Decoder**

### Q1. transformer中的attention为什么scaled?
其实该链接讲的很清晰:
[transformer中的attention为什么scaled?-知乎](https://www.zhihu.com/question/339723385/answer/782509914)

总结就是：
1. 当softmax的输入数值变大，由于指数的作用，数值间的差距会被迅速拉大。
2. 由于数值间差距被迅速拉大，导致后续梯度更新很困难，甚至梯度为0.

而在原文中有：假设向量 q 和 k 的各个分量是互相独立的随机变量，均值是0，方差是1，那么点积$q\cdot k$的均值是0，方差是$d_k$。
$$
E(XY) = E(X)E(Y) = 0 \\
D(XY) = E(X^2\cdot Y^2)-E(XY)^2 \\
=E(X^2\cdot Y^2)-[E(X)E(Y)]^2 \\
=1
$$

则均值为$E(q\cdot k) = \sum_{i=1}^{d_k}E(q_i k_i) = 0$，  
方差为$D(q\cdot k) = \sum_{i=1}^{d_k}D(q_i k_i) = d_k$
因此放缩$\sqrt{d_k}$可以将$q\cdot k$放缩到方差为1


### Q2：<ins>**为什么在其他 softmax 的应用场景，不需要做 scaled？**</ins>
简言之，这个问题涉及到加性attention和乘性attention区别。参考链接3

相关链接：
1. [一文看懂 Bahdanau 和 Luong 两种 Attention 机制的区别](https://zhuanlan.zhihu.com/p/129316415)
2. [Attention机制（Bahdanau attention & Luong Attention）](https://blog.csdn.net/sinat_34072381/article/details/106728056)
3. [为什么在其他 softmax 的应用场景，不需要做 scaled](https://www.zhihu.com/question/339723385/answer/811341890)
4. [如何理解Transformer论文中的positional encoding，和三角函数有什么关系？](https://www.zhihu.com/question/347678607/answer/864217252)
5. [Transformer中的Position Embedding](https://zhuanlan.zhihu.com/p/360539748)
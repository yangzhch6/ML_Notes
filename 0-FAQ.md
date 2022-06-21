## 1. 如何提升泛化性？
数据角度：
1. 更多的数据
2. 数据增强：图片可以放大缩小，翻转等等。文本可以做同义词替换，随机删除，随机交换位置，回译等等

2. 更大批次： 每批次采用更多的数据将有助于模型更好的学习到正确的模式，模型输出结果也会更加稳定
3. 



## 怎么处理过拟合？

## 怎么处理欠拟合？

## . 样本不均衡怎么办？

## . Gradient Vanish & Explode
二者都是由于网络层数加深，雅可比矩阵(都大于1或小于1)连乘导致的问题

### 1.1 Gradient Explode (梯度爆炸) 解决方案
1. 剪裁（**需要注意的是该方法很难解决梯度消失，主要是面向梯度爆炸提出的**）
2. Normalization
3. pre-training+fine-tunning
4. weithts regularization（权重正则化）防止权重过大，从而导致连乘时梯度爆炸

### 1.2 Gradient Vanish
1. LSTM
2. ReLU (<ins>存在疑惑：对RNN难以起到作用？</ins>)
3. Normalization
4. 残差 (<ins>存在疑惑：从推导来看解决的是梯度消失，无法解决梯度爆炸</ins>)
5. pre-training+fine-tunning
# Pytorch计算机制

## is_leaf

## AutoGrad
核心就是pytorch会自动跟踪构建一个动态图，该动态图记录的了**变量**及其**运算符**，反向传播时，会在根节点(loss值)处开始，向后按照链式法则进行反向传播计算各节点梯度。最后在auto_grad=True的节点更新梯度。

## Pytorch和Tensorflow区别

## 相关资料
1. [PyTorch 的 Autograd](https://zhuanlan.zhihu.com/p/69294347)
2. [一文搞懂 PyTorch 内部机制](https://zhuanlan.zhihu.com/p/338256656)
3. [PyTorch 源码解读之 torch.autograd：梯度计算详解](https://zhuanlan.zhihu.com/p/321449610)
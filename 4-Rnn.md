## 1. 为什么RNN会有梯度爆炸/消失?
当序列长度太长，反向传播求导时会有连乘。

## 2. RNN和CNN对比，RNN对文本的时间序列的优点？   
1. 适合解决线性序列问题；
2. 可以接纳不定长输入；
3. LSTM三门的引入，捕获长距离依赖能力加强。
  
[CNN/RNN/Transformer比较](https://www.jianshu.com/p/67666ada573b)    
[为什么Rnn用的ReLU作激活函数效果没有提升？](https://www.zhihu.com/question/61265076)

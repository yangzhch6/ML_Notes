# Model
## Deep & Wide
## ESMM

# Feature

# Debias
## 采样纠偏
### **为何采样？** 
在CTR等任务中，正样本系数，占比极小。当数据规模较大时候，需要对负样本进行采样，用以加速模型训练。
### **采样后如何对结果进行修正？**
为了解此问题，首先需要对logit及odds概念比较清楚：
- Odds指的是 某事件发生的概率 与 某事件不发生的概率 之比。可以理解为正类概率比负类概率。
- logit：实际上是log(odds)，Logit的一个很重要的特性就是没有上下限。一般来说，很多模型都会把神经网络的输出直接作为logit，比如逻辑回归中的logit就是一层先行层的输出。  

该问题的具体解决方法：
1. serving纠偏：正常训练，但在serving进行纠偏。该方法有缺陷：不同时间段的负采样率可能不同，或者混合数据流的负采样率不同，会导致serving纠偏失效。
2. **training纠偏**：训练时，将模型输出进行调整后再拟合训练数据；serving截断无需调整。

$
\frac{P(y=1|x)}{P(y=-1|x)} = \frac{P(y=1, x)}{P(y=-1, x)} = \frac{P(x|y=1)P(y=1)}{P(x|y=-1)P(y=-1)}
$

假设正负样本采样率分别为$\sigma^+$, $\sigma^-$，则$P'(y=1) = \sigma^+ P(y=1)$  
且假设采样不影响特征分布$P(x|y=1)=P'(x|y=1)$，则

$
odds = \frac{P(y=1|x)}{P(y=-1|x)} = \frac{P'(x|y=1)P(y=1)}{P'(x|y=-1)P(y=-1)} = \frac{P'(x|y=1)P'(y=1)\sigma^-}{P'(x|y=-1)P'(y=-1)\sigma^+} = odds'\frac{\sigma^-}{\sigma^+}
$

$
logit = log(odds) = log(odds') + log(\frac{\sigma^-}{\sigma^+}) = logit' + log(\frac{\sigma^-}{\sigma^+})
$

可以得到：  

$
logit' = logit - log(\frac{\sigma^-}{\sigma^+})
$

在训练过程中，按上述公式进行纠偏


# Indicators
## AUC
### AUC计算
### AUC是对采样鲁棒

参考：https://blog.csdn.net/Leon_winter/article/details/104673047

# Optimizers

$\theta_{i-1}$ is the parameter of neural network $f$ at $_{i-1}$ step.

$g_i = \nabla L[f(\theta_{i-1}, x), y]$ is the gradient at $i$ step.   

------
## 1. SGD (Stochastic Gradient Descent)
$\alpha$ is the learning rate.  
the descent gradient $\eta_i$ at $i$ step can be formulated as:
$$
\eta_i = \alpha g_i
$$

update the parameter:
$$
\theta_i = \theta_{i-1} - \eta_i
$$

```
Drawbacks: 梯度更新次数较多，由不同batch决定的梯度方向改变比较频繁，从图上来讲就是参数优化的路径呈现频繁波动的锯齿状。导致模型优化速度慢，且容易陷入局部最优值。
```
-------
## 2. Momentum SGD
$\alpha$ is the learning rate.  
$v_{i-1}$ is the descent gradient of $_{i-1}$ step and is initialized as $v_0$.  
$\beta$ is the parameter of Momentum SGD algorithm.  
$$
v_i = \beta v_{i-1} - g_i \\   
r = r + 
\theta_i = \theta_{i-1} + \alpha v_i
$$

```
Advantages: 引入动量平滑了梯度更新方向。从图上来讲，减少了参数优化方向的震动，使得优化方向更加平滑。因此优点主要在于两点：1）不像SGD那样容易走入局部极小值点。2）模型收敛速度更快。
```
-------
## 3. NAG (Nesterov Accelerated Gradient)
$\alpha$ is the learning rate.      
$\beta$ is the parameter of NAG.   
initialize $d_0$   
$$
d_i = \beta d_{i-1} + g(\theta_{i-1}-\alpha\beta d_{i-1})\\
\theta_i = \theta_{i-1} - \alpha d_i
$$

```
优点：实际上，NAG相当于利用牛顿法对模型进行二阶优化。因此，理论上能够更快更好的收敛。

链接【https://zhuanlan.zhihu.com/p/22810533】详细证明了这一结论
```
另附上泰勒展开公式
$$
f(x) = f(x_0) + \frac{f^{(1)}(x_0)}{1!}(x-x_0) + \frac{f^{(2)}(x_0)}{2!}(x-x_0)^2 + \dots + \frac{f^{(n)}(x_0)}{n!}(x-x_0)^n + R_n(x)
$$

```
为何没有三阶或更高阶优化？个人认为现实世界里用神经网络所拟合的函数，很多都难以被描述成多阶可导的，因此泰勒展开公式在高阶不成立，就不可以用其进行优化。
```

## 4. Adagrad
$\alpha$ is the learning rate.      
$\delta$ is is a very small constant.   
Accumulative gradient sum of squares is denoted as $r$ 
$$
r = r + g^2 \\
\theta_i = \theta_{i-1} - \alpha \frac{1}{\delta + \sqrt{r}} g
$$

```
Adagrad 通过使用累计梯度平方和，来降低每个batch优化时梯度的抖动程度。平缓参数优化方向，加快收敛速度，比SGD更难陷入局部最优。

Drawbacks: 1) 从梯度大小的角度，但随着步长的增加，累计梯度平方和变大，会使得下降的梯度收缩并最终会变得非常小。2) 从信息的角度，简单的累计梯度平方和对每一步的梯度都相同程度对待，不论远近
```

-----
## 5. RMSProp
$\alpha$ is the learning rate.   
$\beta$ is the parameter of RMSProp.     
$\delta$ is is a very small constant.   
Accumulative gradient sum of squares is denoted as $r$ 
$$
r = \beta r + (1-\beta)g^2 \\
\theta_i = \theta_{i-1} - \alpha \frac{1}{\delta + \sqrt{r}} g
$$

```
RMSProp优化算法和AdaGrad算法唯一的不同，就在于累积平方梯度的求法不同。其缓解了AdaGrad梯度下降收缩太快的问题。
```
-----
## 6. Adam  
$\alpha$ is the learning rate.   
$\beta_1$ and $\beta_2$ are the parameter of Adam.   
$\delta$ is is a very small constant.     
$r$ is the accumulative gradient sum   
initialize $v_0$   
$$
\begin{align}
v_i &= \beta_1 v_{i-1} + (1-\beta_1) g_i\\
r_t &= \beta_2 r + (1-\beta_2) g_i^2 \\
\hat v_i &= \frac{v_i}{1-\beta_1^i} ~~~~(bias-corrected)\\
\hat r_i &= \frac{r_i}{1-\beta_2^i} ~~~~(bias-corrected)\\
\theta_i &= \theta_{i-1} - \alpha \frac{1}{\delta + \sqrt{\hat r_i}} \hat v_i
\end{align}
\tag{Adam}
$$

```
Adam综合了RMSProp和Momentum的优点。
此外Adam还解决了动量方法和梯度累加方法存在的一个问题，也即动量v和梯度平方r和一开始均被初始化为0，因此在最开始的几步都很小。因此，为了降低偏差对训练初期的影响，Adam方法对动量和梯度累加和进行了纠偏操作。
```
-----
## 7. Adam + L2  
如果在损失函数中添加正则项(regularization):
$$
L_{reg} = L + \frac{\lambda}{2n} \sum_{i=1}^{n}w_i^2 \\
L_{reg}' = L' + \frac{\lambda}{n} \sum_{i=1}^{n}w_i 
$$

Adam优化器中加入L2正则项：

$\alpha$ is the learning rate.   
$\beta_1$ and $\beta_2$ are the parameter of Adam.   
$\delta$ is is a very small constant.     
$r$ is the accumulative gradient sum   
$\lambda$ is the L2 penalty    
initialize $v_0$   
$$
\begin{align}
g_i &= g_i + \lambda \theta_{i-1} \\
v_i &= \beta_1 v_{i-1} + (1-\beta_1)g_i\\
r_i &= \beta_2 r + (1-\beta_2) g_i^2 \\
\hat v_i &= \frac{v_i}{1-\beta_1^i} ~~~~(bias-corrected)\\
\hat r_t &= \frac{r_i}{1-\beta_2^i} ~~~~(bias-corrected)\\
\theta_i &= \theta_{i-1} - \alpha \frac{1}{\delta + \sqrt{\hat r_i}} \hat v_i
\end{align}
\tag{Adam+L2}
$$

-----
## 8. AdamW  
Adam+L2存在一个问题，也即当梯度的计算加入L2后，其会除去掉一个梯度累加平方和$r_i$。在考虑L2正则项的损失函数中，我们本意是希望模型权重$w$越大，梯度越大，但是Adam+L2中计算出的梯度累加平方和$r_i$也会更大，这就使得减去的正则项偏小。AdamW改进了这一点，将正则项绕过梯度累加步骤，放到最后进行计算。

$\alpha$ is the learning rate.   
$\beta_1$ and $\beta_2$ are the parameter of Adam.   
$\delta$ is is a very small constant.     
$r$ is the accumulative gradient sum   
$\lambda$ is the L2 penalty    
initialize $v_0$   
$$
\begin{align}
v_i &= \beta_1 v_{i-1} + (1-\beta_1)g_i\\
r_i &= \beta_2 r + (1-\beta_2) g_i^2 \\
\hat v_i &= \frac{v_i}{1-\beta_1^i} ~~~~(bias-corrected)\\
\hat r_t &= \frac{r_i}{1-\beta_2^i} ~~~~(bias-corrected)\\
\theta_i &= \theta_{i-1} - \alpha \frac{1}{\delta + \sqrt{\hat r_i}} \hat v_i - \alpha \lambda \theta_{i-1}
\end{align}
\tag{Adam+L2}
$$

------
我认为的讲优化器讲的比较好的：https://blog.csdn.net/bvl10101111/article/details/72616378

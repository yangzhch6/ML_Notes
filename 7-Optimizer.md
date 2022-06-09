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
v_i = \beta_1 v_{i-1} + (1-\beta_1) g_i\\
r = \beta_2 r + (1-\beta_2) g_i^2 \\
\theta_i = \theta_{i-1} - \alpha \frac{1}{\delta + \sqrt{r}} v_i
$$

------
我认为的讲优化器讲的比较好的：https://blog.csdn.net/bvl10101111/article/details/72616378
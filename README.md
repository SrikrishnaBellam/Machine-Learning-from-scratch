# Machine Learning from scratch
 A try to code the ML algorithms from scratch

 

## Linear Regression (from scratch)

Linear Regression models a continuous target as a **linear function** of the input:
$$\hat{y} = m x + b$$
It learns the parameters (m) (slope) and (b) (bias) by minimizing **Mean Squared Error (MSE)**:
$$\mathcal{L}*\text{MSE} = \frac{1}{N}\sum*{i=1}^{N}(\hat{y}_i - y_i)^2$$
Using batch gradient descent, the parameter updates are:
$$\frac{\partial \mathcal{L}}{\partial m}= \frac{2}{N}\sum ( \hat{y}-y) x,\quad
\frac{\partial \mathcal{L}}{\partial b}= \frac{2}{N}\sum ( \hat{y}-y)$$
This repo implements a minimal training loop with `forward → loss → backprop → update`, plus simple input standardization for stable optimization. Try a small learning rate (e.g., `1e-3`) and monitor loss to ensure convergence.


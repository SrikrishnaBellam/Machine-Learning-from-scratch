# Machine Learning from scratch
 A try to code the ML algorithms from scratch

 

## Linear Regression (from scratch)

Linear Regression models a continuous target as a **linear function** of the input:

$\hat{y} = m x + b$

It learns the parameters (m) (slope) and (b) (bias) by minimizing **Mean Squared Error (MSE)**

Using batch gradient descent, the parameter updates are:

$\frac{\partial \mathcal{L}}{\partial m}= \frac{2}{N}\sum ( \hat{y}-y) x,\quad
\frac{\partial \mathcal{L}}{\partial b}= \frac{2}{N}\sum ( \hat{y}-y)$

This repo implements a minimal training loop with `forward → loss → backprop → update`, plus simple input standardization for stable optimization. Try a small learning rate (e.g., `1e-3`) and monitor loss to ensure convergence.

## Logistic Regression – Mathematical Framework

---

### 1. Hypothesis (Forward Propagation)

Given input features  
$X \in \mathbb{R}^{m \times n}, \quad y \in \{0,1\}^m$

Weights and bias:  
$w \in \mathbb{R}^{n \times 1}, \quad b \in \mathbb{R}$

Linear combination:  
$z^{(i)} = w^\top x^{(i)} + b$

Sigmoid activation:  
$\hat{y}^{(i)} = \sigma(z^{(i)}) = \frac{1}{1 + e^{-z^{(i)}}}$

---

### 2. Cost Function (Binary Cross-Entropy)

For $m$ training samples, the loss is:  

$J(w, b) = -\frac{1}{m} \sum_{i=1}^m \Big[ y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)}) \Big]$

---

### 3. Gradients (Backward Propagation)

The partial derivatives are:  

$\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^m \big( \hat{y}^{(i)} - y^{(i)} \big) x^{(i)}$

$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m \big( \hat{y}^{(i)} - y^{(i)} \big)$

---

### 4. Parameter Updates (Gradient Descent)

Using learning rate \( \alpha \):  

$w := w - \alpha \frac{\partial J}{\partial w}$

$b := b - \alpha \frac{\partial J}{\partial b}$

---

### 5. Prediction Rule

For classification:  

$$
\hat{y}^{(i)} =
\begin{cases}
1 & \text{if } \sigma(z^{(i)}) \geq 0.5 \\
0 & \text{if } \sigma(z^{(i)}) < 0.5
\end{cases}
$$

---

### 6. Accuracy

The accuracy metric is:  

$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}} \times 100$



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

---
---

## K Nearest Neighbor

### 🧮 1. Mathematical Formulation

Given a training dataset
$\mathcal{D} = {(x_i, y_i)}_{i=1}^{N}, \quad x_i \in \mathbb{R}^d, ; y_i \in \mathcal{Y}$

and a new test point ($x'$), the KNN algorithm predicts the label of ($x'$) as follows:

---

### **Step 1: Compute distance**

For every ($x_i$) in the dataset, compute its distance to the test sample ($x'$).

Commonly we use **Euclidean distance**:
$D(x', x_i) = \sqrt{\sum_{j=1}^{d} (x'*j - x*{ij})^2}$

Other options: Manhattan ($(L_1)$), Minkowski, Cosine, etc.

---

### **Step 2: Sort and select neighbors**

Find the **K smallest distances**, i.e., the **K closest points**:
$\mathcal{N}_K(x') = \text{indices of } K \text{ smallest } D(x', x_i)$

---

### **Step 3: Voting / Averaging**

* **For classification:**
  $\hat{y} = \text{mode}({y_i : i \in \mathcal{N}_K(x')})$

* **For regression:**
  $\hat{y} = \frac{1}{K} \sum_{i \in \mathcal{N}_K(x')} y_i$

---

### **Step 4: Optional weighting by distance**

Weighted voting can use:
$w_i = \frac{1}{D(x', x_i) + \epsilon}$
and then use weighted majority or weighted average.

---






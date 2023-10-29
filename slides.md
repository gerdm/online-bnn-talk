---
theme: default
background: https://raw.githubusercontent.com/gerdm/qmul-fire-talk-0323/main/public/cover-lofi-dalle.png
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Online neural network training
title: Online training of Bayesian neural networks
mdc: true
---

# Online training of Bayesian neural networks
Adaptation, uncertainty, **scalability**

Gerardo Duran-Martin

---

## What do we mean by online learning?

Let ${\cal D}_t = \{y_t, x_t\}$, $D_{1:t} = ({\cal D}_{1}, \ldots, {\cal D}_t)$, and suppose $y_t \sim p(\cdot | \boldsymbol\theta_t, x_t)$. We estimate

$$
p(\theta_t  \vert {\cal D}_{1:t}) \propto p(y_t \vert \theta_t, x_t)p(\theta_t \vert {\cal D}_{1:t-1})
$$

---

## Why online learning?

- Cost of retraining models becomes expensive and practically infeasible for modern neural network architectures
- In time series datasets, e.g. in Finance, changes in the world and static parameters result in lower model performance.
- Adaptation to the non-iid setting: gradual or abrupt changes

---

## Online learning in an ML context

1. Bandits
2. Continual learning
3. hyperparameter optimisation (BayesOpt)
4. High-dimensional regression
5. Online classification
6. Reinforcement learning

---

# State-space models (SSMs)

$$
\begin{aligned}
p(\boldsymbol\theta_t\vert\boldsymbol\theta_{1:t-1}) &= p(\boldsymbol\theta_t\vert\boldsymbol\theta_{t-1})\\
p({\bf y}_t \vert \boldsymbol\theta_t, {\bf y}_{t-1})& = p({\bf y}_t \vert \boldsymbol\theta_t)
\end{aligned}
$$

---

# Filtering an SSM

In a filtering problem, we estimate the posterior distribution of the latent state given all past observations $y_{1:t}$

We seek to estimate $p(\boldsymbol\theta_t \vert y_{1:t})$ — filtering.

$$
p(\boldsymbol\theta_t \vert {\cal D}_{1:t}) \propto
\underbrace{
p({\cal D}_t \vert \theta_t)
\overbrace{
\int p(\boldsymbol\theta_{t-1}\vert{\cal D}_{1:t-1}) p(\boldsymbol\theta_t\vert\boldsymbol\theta_{t-1})d\boldsymbol\theta_{t-1}}^\text{predict step}}_\text{update step}
$$

---

## The linear setting

$$
\begin{aligned}
p(\boldsymbol\theta_t \vert \boldsymbol\theta_{t-1}) &= {\cal N}(\boldsymbol\theta_t \vert {\bf A}_t\boldsymbol\theta_{t-1}, {\bf Q}_t)\\
p({\bf y}_t \vert \boldsymbol\theta_t) &=
{\cal N}({\bf y}_t \vert {\bf H}_t\boldsymbol\theta_t, \boldsymbol{\bf R}_t)
\end{aligned}
$$

with

- $\boldsymbol\theta_t \in \mathbb{R}^D$ The latent variable
- ${\bf y}_t\in\mathbb{R}^C$ the target variable
- ${\bf A} _t\in\mathbb{R}^{D\times D}$ the latent transition matrix
- ${\bf H}_t\in\mathbb{R}^{C\times D}$ the projection matrix
- ${\bf Q}_t$ the dynamic’s covariance
- ${\bf R}_t$ the emission covariance

---

# The Kalman filter equations

Solution in the example above is given by the **Kalman Filter (KF) equations.**

### Predict step

$$
\begin{aligned}
\bar{\bf m}_t &= {\bf A}_t{\bf m}_{t-1}\\
\bar{\bf P}_t &= {\bf A}_t{\bf P}_{t-1}{\bf A}_t^\intercal + {\bf Q}_t
\end{aligned}
$$

### Update step

$$
\begin{aligned}
{\bf e}_t &= {\bf y}_t - {\bf H}_t\bar{\bf m}_t\\
{\bf S}_t &= {\bf H}_t\bar{\bf P}_t{\bf H}_{t}^\intercal + {\bf R}_t\\
{\bf K}_t &= \bar{\bf P}_t{\bf H}_{t}^\intercal{\bf S}_t^{-1}\\ \\
{\bf m}_t &= \bar{\bf m}_t + {\bf K}_t{\bf e}_t\\
{\bf P}_t &= \bar{\bf P}_t - {\bf K}_t{\bf S}_t {\bf K}_t^\intercal
\end{aligned}
$$

---

# Filtering
At time $t$,

$$
p(\boldsymbol\theta_t \vert y_{1:t}) ={\cal N}(\boldsymbol\theta_t \vert {\bf m}_t, {\bf P}_t)
$$

---

## Example: tracking an object in 2d

See §8.2.1.1 in Murphy, K. P. (2023). *Probabilistic machine learning: Advanced topics*. MIT press.Chicago

### Setup

Estimate the position of a moving object with position $(x_{t,1}, x_{t,2})$ and velocity $(\dot{x}_{1}, \dot{x}_{t,2})$. Assume fix and known $R_t$ and $Q_t$, then

- Hidden (latent) state $\boldsymbol\theta_t = (x_{t,1}, x_{t,2}, \dot{x}_{t,1}, \dot{x}_{t,2})$
- Observed variable $y_t$

---

## Example (cont'd)

- Transition matrix

$$
A_t = \begin{bmatrix}
1 & 0 & \Delta & 0\\
0 & 1 & 0 & \Delta \\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{bmatrix}
$$

- Projection matrix

$$
{\bf H}_t = \begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0
\end{bmatrix}
$$

---

### Tracked positions ($x_{t,1}, x_{t,2})$

Observed position, true position, and filtered position.

<img class="horizontal-center" width=500
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/Untitled.png">

---

# Linear regression as a state-space model

## Setup

Suppose $y_t = 1 + 2x_t + \epsilon_t$ with $\epsilon_t\sim{\cal N}(0, 1)$

<img class="horizontal-center" width=500
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/Untitled%201.png">

---

## Rewriting the model as an SSM

Let ${\bf H}_t = {\bf x}_t^\intercal$ with ${\bf x}_t = \begin{bmatrix}1 &x_t\end{bmatrix}^\intercal$, $Q_t = \gamma{\bf I}_D$, and ${\bf R}_t = \beta\in\mathbb{R}^+$.

$$
\begin{aligned}
p(\boldsymbol\theta_t \vert \boldsymbol\theta_{t-1}) &= {\cal N}(\boldsymbol\theta_t \vert \boldsymbol\theta_{t-1}, \gamma{\bf I}_D)\\
p(y_t \vert \boldsymbol\theta_t) &= {\cal N}(y_t \vert {\bf x}_t^\intercal\boldsymbol\theta_t, \beta)
\end{aligned}
$$

---

## Filtering results

Take $\gamma = 0$ (fixed latent state assumption) and $\beta = 1$ (known variance)

<img class="horizontal-center" width=500
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/Untitled%202.png">

<img class="horizontal-center" width=500
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/Untitled%203.png">

---

# From linear regression to non-linear regression

Via neural networks

$$
\begin{aligned}
p(\boldsymbol\theta_t \vert \boldsymbol\theta_{t-1}) &= {\cal N}(\boldsymbol\theta_t \vert \boldsymbol\theta_{t-1}, \gamma{\bf I}_D)\\
p(y_t \vert \boldsymbol\theta_t) &= {\cal N}(y_t \vert f(\boldsymbol\theta_t, {\bf x}_t), \beta)
\end{aligned}
$$

Where $f:\mathbb{R}^D\times\mathbb{R}^M\to\mathbb{R}^C$ is a non-linear function, e.g., a neural network, $\boldsymbol\theta_t$ are the parameters of the neural network, and ${\bf x}_t$ are covariates observed at time $t$.

---

## Extended Kalman filter (EKF)

Linearising a neural network.

At time $t$, make a first-order Taylor expansion around the previous mean ${\bf m}_{t-1}$ so that

$$
\begin{aligned}
f(\boldsymbol\theta_t, {\bf x}_t) &\approx f({\bf m}_{t-1}, {\bf x}_t) + \nabla f({\bf m}_{t-1}, {\bf x}_t)^\intercal(\boldsymbol\theta_t - {\bf m}_{t-1})\\
&= 
\underbrace{
\nabla f({\bf m}_{t-1}, {\bf x}_t)^\intercal}_{ {\bf H}_t}\boldsymbol\theta_t 
+
\overbrace{
\left(f({\bf m}_{t-1}, {\bf x}_t) - \nabla f({\bf m}_{t-1}, {\bf x}_t)^\intercal{\bf m}_{t-1}\right)}^{ {\bf c}_t}
\end{aligned}
$$

We can make use of the KF equations!

$$
\begin{aligned}
p(\boldsymbol\theta_t \vert \boldsymbol\theta_{t-1}) &= {\cal N}(\boldsymbol\theta_t \vert \boldsymbol\theta_{t-1}, {\bf Q}_t)\\
p({\bf y}_t \vert \boldsymbol\theta_t) &=
{\cal N}({\bf y}_t \vert {\bf H}_t\boldsymbol\theta_t + {\bf c}_t, \boldsymbol{\bf R}_t)
\end{aligned}
$$

---

### Example: EKF for neural network training

Consider a three-hidden-layer MLP with 6 units in each layer and ELU activation unit. We show

- Posterior predictive mean (purple line)
- Samples from the posterior predictive mean (gray lines)
- Observation at time $t$: white dots
- Past observed points: blue dots

--- 

### Example (cont'd)
    
 <img class="horizontal-center" width=500
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/ekf.gif">
    

---

### What about…

- $y_t \in \{0, 1\}$ — Bernoulli distribution
- $y_t \in \{C_1, \ldots, C_K\}$ — Multinomial distribution
- $y_t \in \mathbb{R}^+$ — Gamma distribution
- $y_t \in [0, 1]$ — Beta distribution
- $y_t \in \{0, 1, 2, \ldots\}$ — Poisson distribution

---

## Online inference of model parameters using any member of the exponential family

A random variable $y\in\mathbb{R}^S$ is a member of the exponential-family if its probability density function can be written as

$$
\log\text{expfam}(y \vert \eta(\boldsymbol\theta)) = \eta(\boldsymbol\theta)^\intercal \text{suffstat}(y) + b(y) + a(\eta(\boldsymbol\theta))
$$

With

- $\boldsymbol\theta\in\mathbb{R}^D$ the model parameters
- $\eta: \mathbb{R}^D\to\mathbb{R}^S$ the link function from model parameters to natural parameters
- $\text{suffstat}: \mathbb{R}^C \to \mathbb{R}^S$ the sufficient statistics for the random variable $y_t$
- $a: \mathbb{R}^S\to\mathbb{R}$ the log-partition function
- $b: \mathbb{R}^C \to \mathbb{R}$ the log-base measure

---

### SSMs for Gaussian latent variable and expfam target variable

$$
\begin{aligned}
p(\boldsymbol\theta_t \vert \boldsymbol\theta_{t-1}) &= {\cal N}(\boldsymbol\theta_t \vert \boldsymbol\theta_{t-1}, \gamma{\bf I}_D)\\
p(y_t \vert \boldsymbol\theta_t) &= \text{expfam}(y_t \vert \eta(\boldsymbol\theta_t, {\bf x}_t))
\end{aligned}
$$

---

### Moment-matched Extended Kalman filter (expfamEKF)

See [Online Natural Gradient as a Kalman Filter \[1703.00209\]](https://www.notion.so/Online-Natural-Gradient-as-a-Kalman-Filter-1703-00209-841bd5d825eb4db690bb26b594b6fe8f?pvs=21)

Idea: Let target variable $y_t$ be the sufficient statistics  $\text{suffstat}(y_t)$ where $\text{suffstat}$ are the sufficient statistics for a random variable $y_t\sim\text{expfam}(\cdot)$ and $\eta(\boldsymbol\theta, {\bf x}_t)$ is the link function from model parameters to natural parameters.

---

### Expfam EKF equations

We make use of the fact that the log-partition $a(\cdot)$ satisfies

$$
\begin{aligned}
\mathbb{E}[y_t\vert \eta_t] &= \frac{\partial}{\partial \eta_t} a(\eta_t)\\
\text{Cov}(y_t\vert \eta_t) &= \frac{\partial^2}{\partial \eta_t^2}a(\eta_t)
\end{aligned}
$$

So that

$$
\begin{aligned}
{\color{crimson}{\bf e}_t} &= \text{suffstat}(y_t) - \mathbb{E}[y_t \vert \eta_t]\\
{\color{teal}{\bf R}_t} &= \text{Cov}(y_t\vert\eta_t)\\
{\bf S}_t &= {\bf H}_t\bar{\bf P}_t{\bf H}_{t}^\intercal + {\color{teal}{\bf R}_t}\\
{\bf K}_t &= \bar{\bf P}_t{\bf H}_{t}^\intercal{\bf S}_t^{-1}\\ \\
{\bf m}_t &= \bar{\bf m}_t + {\bf K}_t{\color{crimson}{\bf e}_t}\\
{\bf P}_t &= \bar{\bf P}_t - {\bf K}_t{\bf S}_t {\bf K}_t^\intercal
\end{aligned}
$$

---

## Example: BernKF on the Moon’s dataset

Model $y_t\sim\text{Bern}(f(\boldsymbol\theta_t, {\bf x}_t))$

<img class="horizontal-center" width=500
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/bern-ekf_(1).gif">

---

# What’s the catch?

- **Method is not scalable —** EKF takes $O(D^2)$ memory.

For MNIST classification, using a 3-layered MLP with 300 units sized 32bits, a single update (one observation) would require more than 10,000gb in memory.

- **Slow** — EKF takes $O(D^3)$ in time (matrix inversion)

- **Moment-matched EKF not always stable:** Heteroskedastic Gaussian

- **Neural networks are often overparameterised**: Needing redundant compuations

- **The rate of change in parameters $Q_t$ is not necessarily fixed and set to zero**, e.g., in continual learning problems.

---

# Our work

1. **Memory constraints**: hardware dependency
2. **Time constraints**: dependency to application
3. Make use of more **modern neural networks architectures**: CNNs, Transformers, …
4. **Changing environments**: considering $Q_t \neq 0$
5. **One-pass-learning constraint**

---

# The methods

Setup

- Let $D$ be the total number of parameters in the neural network, $d \ll D$ a low-rank dimension.
- We constraint the training of the model to a single sweep of the data

---

# Intrinsic dimension hypothesis

(See [Measuring the Intrinsic Dimension of Objective Landscapes \[1804.08838\]](https://www.notion.so/Measuring-the-Intrinsic-Dimension-of-Objective-Landscapes-1804-08838-18efd94d0c804054b176426c9aefebbc?pvs=21))

Training within a **random, low-dimensional affine subspace** can suffice to reach high training and test accuracies on a variety of tasks, provided the training dimension exceeds a threshold that called the intrinsic dimension

---

# Subspace EKF

See [Efficient Online Bayesian Inference for Neural Bandits \[2112.00195\]](https://www.notion.so/Efficient-Online-Bayesian-Inference-for-Neural-Bandits-2112-00195-f9f3b4c28e4d4112a846a6f2cc40efff?pvs=21) 

### The intrinsic dimension in subspace training

Fix ${\bf A} \in\mathbb{R}^{D\times d}$ such that ${\bf A}_{i,j}\sim{\cal N}(m,1)$, ${\bf z}_t \in \mathbb{R}^d$ and define $\boldsymbol\theta_t = {\bf Az}_t$

$$
\begin{aligned}
p({\bf z}_t \vert {\bf z}_{t-1}) &= {\cal N}({\bf z}_t \vert {\bf z}_{t-1}, \gamma{\bf I}_D)\\
p(y_t \vert {\bf z}_t) &= {\cal N}(y_t \vert f({\bf Az}_t, {\bf x}_t), \beta)
\end{aligned}
$$

We estimate $p({\bf z}_t \vert {\cal D}_{1:t})$ at every timestep using the Kalman filter equations

Update cost in memory is $O(d^2 + Dd)$

---

## Example: intrinsic dimension on the moon’s dataset

<img class="horizontal-center" width=500
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/Untitled%204.png">

---

### Setup

- Multilayered perceptron (MLP) with three layers, 50 units per layer, and ReLU activation
- MLP has `5301` units
- We compare full-EKF to subspace training $p({\bf z}_t \vert {\cal D}_{1:t})$ for varying dimensions.
- Train using 500 samples / Test using 300 samples

---

## Result (I)

Subspace dimension v.s. test accuracy

<img class="horizontal-center" width=500
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/Untitled%205.png">

---

## Result(II)

% of total parameters v.s. % underperformance to Full-covariance EKF (no subspace)

<img class="horizontal-center" width=500
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/Untitled%206.png">

---

# Benchmark: Online classification of **Fashion MNIST using a CNN**

Dataset example

<img class="horizontal-center" width=400
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/Untitled%207.png">

---

### Setup:

- Multinomial-type problem: 10 outputs
- 10,000 training examples
- 5,000 test examples (evaluated at every timestep)
- LeNet 5 with relu activation function: Total number of parameters: 61706
- (stress test) Subspace dimension: 100 (0.16% of total parameters)

---

## Online Fashion MNIST (I)

Let ${\bf A}_{i,j}\sim{\cal N}(0,1)$

<img class="horizontal-center" width=500
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/Untitled%208.png">

---

## The choice of projection: Lottery subspace approach

See [How many degrees of freedom do we need to train deep networks: a loss landscape perspective \[2107.05802\]](https://www.notion.so/How-many-degrees-of-freedom-do-we-need-to-train-deep-networks-a-loss-landscape-perspective-2107-05-fd7472302b2844e2a67fa92544c78847?pvs=21)

- Obtain better performance by *********informing********* the space over which the subspace is moving through.
- Train network in full space (using an unseen ********warmup******** dataset). Store parameters at each step and define the matrix ${\bf A} \in \mathbb{R}^{D\times d}$ whose $d$ columns are top $d$ principal components of the trayectory of parameters.

---

### Example

The warmup dynamics for Fashion MNIST on LeNet5

- Consider a warmup dataset ${\cal D}^\text{warmup}$ with 2000 (unseen) samples

```python
d = 100
w = [...] # Size D
params = [w]
for n in n_warmup:
	w = update_step(w, warmup)
  params = [params, w]
A = svd(params)[:, :d]
```

---

### Projected weights to $d=3$

<img class="horizontal-center" width=400
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/Untitled%209.png">

---

## Online Fashion MNIST (II)

Let ${\bf A}$ be the SVD-driven projection matrix

<img class="horizontal-center" width=500
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/Untitled%2010.png">

---

# Can we do better?

---

# PULSE: Projection-based unification of subspace and last-layer training

See [Detecting Toxic Flow](https://www.notion.so/Detecting-Toxic-Flow-d5bf396630824b5da367eceda165c445?pvs=21)

Leveraging the **intrinsic dimension** of neural networks + **last-layer** methods

Write

$$
f(\boldsymbol\theta_t,{\bf x}_t) = {\bf w}_t^\intercal\,h({\bf A z}_t; {\bf x}_t)
$$

with $\boldsymbol\theta_t = ({\bf w}_t, {\bf z}_t)$, ${\bf w}_t \in \reals^{d_\text{last}}$ the output layer of the neural network, and ${\bf z}\in\reals^{d_\text{hidden}}$ the projected (hidden) units.

Update cost in memory is $O(d_\text{hidden}^2 + d_\text{last}^2+ Dd_\text{hidden})$

---

## Online Fashion MNIST (III)

Let ${\bf A}_{i,j} \sim {\cal N}(0, 1)$

<img class="horizontal-center" width=500
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/Untitled%2011.png">

---

## Online fashion MNIST (IV)

Let ${\bf A}$ be the SVD-driven projection matrix

<img class="horizontal-center" width=500
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/Untitled%2012.png">

---

# Can we do better?

---

# LoFi: Low-rank (extended) Kalman filter

See [Low-rank extended Kalman filtering for online learning of neural networks from streaming data \[2305.19535\]](https://www.notion.so/Low-rank-extended-Kalman-filtering-for-online-learning-of-neural-networks-from-streaming-data-2305--8336aa5be8754ed3a9d63bfa84f03648?pvs=21)

Embed the intrinsic-dimension hypothesis in the target distribution. At every timestep $t$, we estimate

$$
q_t(\boldsymbol\theta_t) = {\cal N}(\boldsymbol\theta_t \vert {\bf m}_t, ({\bf A}_t{\bf A}_t^\intercal + \Upsilon_t)^{-1})
$$

with, 

- ${\bf m}_t\in\mathbb{R}^D$ — mean,
- ${\bf A}_t\in\mathbb{R}^{D\times d}$ — Low-rank part of covariance matrix
- $\Upsilon_t = \text{diag}(\upsilon_1, \ldots, \upsilon_D)$ — Diagonal part of covariance matrix

---

## Online Fashion MNIST (V)

<img class="horizontal-center" width=500
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/Untitled%2013.png">

---

# What about SGD?

## Online SGD

Update of all parameters after a single step

$$
\boldsymbol\theta_t = \boldsymbol\theta_{t-1} - \alpha_t\nabla_{\boldsymbol\theta}\log p(y_t | \boldsymbol\theta_{t-1}, {\bf x}_t) 
$$

## Replay-buffer SGD

Store a buffer of the last $k$ datapoints

$$
\boldsymbol\theta_t = \boldsymbol\theta_{t-1} - \frac{\alpha_t}{k}\sum_{\tau=1}^k\nabla_{\boldsymbol\theta}\log p(y_{t-\tau} | \boldsymbol\theta_{t-1}, {\bf x}_{t-\tau})
$$

---

## Example: Full-covariance EKF v.s. online gradient descent

We compare posterior predictive mean (EKF) on a neural network with $Q_t = 0 \cdot {\bf I}$ to (i) online gradient descent (left) and (ii) replay-buffer gradient descent with buffer of 10 (right)

Despite simplicity, a good contender to EKF.

<div class="float-left">
<img class="horizontal-center" width=400
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/ekf-osgd1.gif">
</div>

<div class="float-right">
<img class="horizontal-center" width=400
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/ekf-osgd10.gif">
</div>

---

## Online fashion MNIST (VI) — Final benchmark

<img class="horizontal-center" width=500
     src="/One-pass%20learning%20methods%20for%20training%20Bayesian%20ne%20a9133bf9e9574c49b8243d163414d447/Untitled%2014.png">

---

# The takeaway

- Online learning of Bayesian neural networks can be formulated as a filtering problem using state-space models

---

# Online BNNs
### gerdm.github.io/online-bnn-talk

<div class="centered">
  <h2 style="line-height:100px;"> <a href="https://grrddm.notion.site/One-pass-learning-methods-for-training-Bayesian-neural-networks-a9133bf9e9574c49b8243d163414d447?pvs=4">📑 Notion notes </a></h2>
  <h2> <a href="https://github.com/gerdm/rebayes-mini">💻 rebayes-mini</a></h2>
</div>

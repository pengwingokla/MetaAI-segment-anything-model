
## Overview

This directory covers fundamental machine learning concepts including probability theory, linear algebra, and optimization algorithms. The assignment is split into two Jupyter notebooks implementing mathematical foundations and practical algorithms from scratch.

## Files

- `1-probability-dist-linear-algebra` - Problems 1 & 2: Probability distributions and linear algebra
- `2-stochastic-gradient-descent` - Problems 3 & 4: Stochastic gradient descent implementations
- `README.md` - This documentation file

## Assignment Problems

### Problem 1: Simulation
**Bivariate Normal Distribution Analysis**

- **Part A:** Simulate bivariate normal distribution with 200 samples
  - Mean: [0.6, 0.6]
  - Covariance matrix: [[0.1, 0], [0, 0.1]]
  - Generate scatter plot similar to Figure 6.8b from Math for ML book

- **Part B:** Overlay contour plots with simulated points
  - Uses `scipy.stats.multivariate_normal` for theoretical contours
  - Visualizes probability density function with color mapping

### Problem 2: Projection
**Principal Component Analysis using SVD**

Implementation of 3D Gaussian projection onto principal component subspace:
- **Covariance Matrix:** 4×4×3 symmetric matrix with complex correlations
- **Method:** Singular Value Decomposition (SVD) for dimensionality reduction
- **Analysis:** Correlation interpretation through principal component signs

**Key Concepts:**
- Principal components determined by U matrix columns from SVD
- Correlation signs indicate positive/negative feature relationships
- Projection preserves maximum variance in reduced dimensions

### Problem 3: Stochastic Gradient Descent
**Baseline SGD Implementation**

From-scratch implementation for polynomial regression on sinusoidal data:
- **Dataset:** 15 noisy samples from sin(2πx) function
- **Features:** Polynomial degree 15 with bias term
- **Learning Schedule:** Adaptive rate t₀/(t + t₁) where t₀=5, t₁=50
- **Epochs:** 50 with random sample selection per iteration

**Implementation Details:**
- Manual gradient computation: 2 * X^T * (Xθ - y)
- Random parameter initialization with reproducible seed
- Loss tracking via Mean Squared Error calculation

### Problem 4: SGD Enhancements
**Advanced Optimization Algorithms**

Comparison of three optimization methods:

#### Momentum SGD 
- **Formula:** V_t = βV_{t-1} + ηg_t
- **Hyperparameter:** momentum β = 0.9
- **Benefit:** Accelerates convergence by maintaining velocity

#### Adam Optimizer  
- **First moment:** m_t = β₁m_{t-1} + (1-β₁)g_t
- **Second moment:** v_t = β₂v_{t-1} + (1-β₂)g_t²
- **Bias correction:** Applied to both moments
- **Hyperparameters:** β₁=0.9, β₂=0.999, ε=1e-8

## Technical Implementation

### Key Functions
- `create_toy_data()` - Generates noisy sinusoidal training data
- `create_polynomial_features()` - Constructs polynomial feature matrix
- `learning_schedule()` - Implements adaptive learning rate decay
- `polynomial_sgd()` - Baseline SGD with parameter tracking
- `advanced_polynomial_sgd()` - Enhanced optimizers (Momentum, Adam)

## Results & Analysis

The implementations demonstrate:
1. **Bivariate distributions** show proper sampling and contour alignment
2. **SVD projection** preserves correlation structure in reduced dimensions  
3. **SGD convergence** achieved within 50 epochs for all optimizers
4. **Adam optimizer** shows fastest and most stable convergence
5. **Momentum SGD** provides smooth acceleration over baseline

## Learning Outcomes

- Hands-on experience with probability theory and multivariate statistics
- Understanding of linear algebra applications in dimensionality reduction
- Implementation of gradient-based optimization from mathematical foundations
- Comparison of modern optimization techniques in practical scenarios

# Bilinear Layer for Modular Arithmetic with Tensor Network Analysis

This project implements a bilinear layer trained on modular arithmetic and analyzes the resulting tensor network properties, demonstrating the interpretability benefits of tensor networks over traditional neural networks.

## Overview

The implementation focuses on two key tensor network properties:

1. **Global Structure**: The bilinear layer can be contracted into a 3rd-order tensor, allowing us to apply SVD to find global structure in the model.

2. **Analytic Relationships**: Unlike MLPs where relationships between layers must be approximated with data, tensor networks provide exact analytic solutions through weight multiplication.

## Problem Setup

- **Task**: Modular arithmetic `a + b = c (mod 113)`
- **Input**: One-hot vectors for `a` and `b` (dimension 226)
- **Output**: One-hot vector for `c` (dimension 113)
- **Architecture**: Bilinear layer `y = x^T W x + b` where `W` is a 3rd-order tensor

## Key Features

### Bilinear Layer Implementation
- Custom bilinear layer that computes `x^T W x` for a 3rd-order tensor `W`
- Proper weight initialization and scaling
- Weight decay regularization as required

### Dataset Generation
- Modular arithmetic dataset with configurable modulus (default P=113)
- One-hot encoding for inputs and outputs
- Train/validation split with proper data loading

### Analysis Components

1. **Interaction Matrices**: 
   - Computes interaction matrices showing how inputs `a` and `b` interact
   - Visualizes cross-interactions and diagonal patterns
   - Based on formula from bilinear layer paper (top of page 3)

2. **3rd-Order Tensor SVD**:
   - Performs SVD on the unfolded tensor along different modes
   - Analyzes top eigenvectors and singular values
   - Based on Section 3.3 of the bilinear layer paper

3. **Modular Structure Analysis**:
   - Examines learned patterns for modular arithmetic
   - Compares learned patterns with ideal modular structure
   - Visualizes how the model learns the `(a + b) mod P` relationship

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Running the Analysis
```bash
python run_analysis.py
```

Or directly:
```bash
python bilinear_modular_arithmetic.py
```

### Expected Output
The script will generate several plots:
- `training_curves.png`: Training loss and validation accuracy
- `interaction_matrices.png`: Learned interaction patterns
- `tensor_svd_analysis.png`: SVD analysis of the 3rd-order tensor
- `modular_structure_analysis.png`: Analysis of learned modular patterns

## Key Results

The bilinear layer should learn to:
1. Recognize the modular arithmetic pattern `(a + b) mod P`
2. Show clear interaction matrices with diagonal structure
3. Exhibit interpretable tensor decomposition with meaningful eigenvectors
4. Achieve high accuracy on the modular arithmetic task

## Technical Details

### Bilinear Layer Formula
For input `x` and 3rd-order tensor `W`:
```
y_k = sum_i sum_j x_i * W_{i,j,k} * x_j + b_k
```

### Interaction Matrix Analysis
The interaction matrices reveal how the model learns to combine inputs:
- `W_ab`: Cross-interactions between `a` and `b`
- `W_aa`, `W_bb`: Self-interactions
- Diagonal patterns indicate learned modular structure

### Tensor Network Benefits
1. **Global Structure**: The entire model can be represented as a single contracted tensor
2. **Analytic Relationships**: Exact relationships between different parts of the model
3. **Interpretability**: Clear mathematical structure that can be analyzed with linear algebra

## References

- [LessWrong: Interpreting Modular Addition in MLPs](https://www.lesswrong.com/posts/cbDEjnRheYn38Dpc5/interpreting-modular-addition-in-mlps)
- [Bilinear Layer Paper](https://arxiv.org/pdf/2410.08417)
- Tensor Network Theory and Applications
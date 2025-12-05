# randomized_optimizaiton

This repository contains a full benchmarking study of four randomized optimization algorithms applied to both discrete optimization problems and a neural network classification task.

The algorithms evaluated are:

- Randomized Hill Climbing (RHC)
- Simulated Annealing (SA)
- Genetic Algorithm (GA)
- MIMIC
- (Comparison with Backpropagation for Neural Network)

The complete methodology and experimental results are documented in the accompanying PDF report.

---

## Repository Structure

```
project_root/
│
├── randomized_optimization_part1.py      # Discrete optimization experiments (FlipFlop, N-Queens, FourPeaks)
├── randomized_optimization_part2.py      # Neural network optimization (RHC, SA, GA vs Backprop)
├── Randomized_optimization_report.pdf    # Full project analysis and results
└── README.md                             # Documentation
```

---

## Discrete Optimization Experiments (Part 1)

The following optimization problems were implemented:

- Flip-Flop  
- N-Queens  
- Four Peaks  

For each problem, the four randomized optimization algorithms (RHC, SA, GA, MIMIC) were tested with systematic hyperparameter searches, and the following metrics were recorded:

- Best fitness achieved  
- Runtime (training time)  
- Fitness curves over iterations  
- Effect of hyperparameters (restart count, decay schedule, mutation probability, population size, keep_pct)

All results, graphs, and interpretations for these problems are described in the report.

---

## Neural Network Optimization Experiments (Part 2)

A neural network classifier was trained using four optimization algorithms:

- Randomized Hill Climbing
- Simulated Annealing
- Genetic Algorithm
- Backpropagation (baseline)

The dataset used is the **WiFi Localization dataset** (multiclass classification with 4 classes).  
It can be downloaded from:

https://archive.ics.uci.edu/dataset/422/wireless+indoor+localization

Part 2 evaluates:

- Hyperparameter tuning for RHC, SA, and GA  
- Accuracy comparison across algorithms  
- Training time comparison  
- Loss curves for all optimization methods  
- Strengths and weaknesses of randomized optimization in continuous NN weight spaces

All plots and analysis appear in the final section of the report.

---

## How to Run

Install required packages:

```bash
pip install numpy pandas mlrose-hiive matplotlib scikit-learn
```

Run discrete optimization experiments:

```bash
python randomized_optimization_part1.py
```

Run neural network optimization experiments:

```bash
python randomized_optimization_part2.py
```

Outputs (plots and printed metrics) are saved automatically in the working directory.

---

## Summary of Findings

- For discrete problems, **GA and MIMIC** often achieve the highest fitness but require significantly more computation time.
- **RHC and SA** perform competitively while being far more computationally efficient.
- For neural network training with the WiFi dataset, **Backpropagation clearly outperforms randomized optimization** in both accuracy and runtime.
- Randomized optimization may still be useful for difficult discrete search spaces or non-differentiable objective functions.

For full details, refer to the project report.

---

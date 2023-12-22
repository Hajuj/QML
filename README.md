# Quantum Advantage Actor-Critic for Reinforcement Learning

## Overview

This project involves implementing and evaluating a Quantum Advantage Actor-Critic (QA2C) for reinforcement learning tasks. The QA2C is designed with different rotation gate architectures and a hybrid model incorporating post-processing Neural Networks (NN). The goal is to explore the performance of pure and hybrid quantum circuits in reinforcement learning and compare them with classical approaches and a random agent baseline.

## Replication Instructions

To replicate the results from our paper, follow these steps:

1. **Quantum and Hybrid A2C Algorithms:**
   - Navigate to the `arch3` directory.
   - Run the pure VQC implementation (`a2c_4`) and the hybrid VQC with post-processing NN (`a2c_5_ppnn`).

2. **Classical A2C Algorithms:**
   - For the pure classical A2C with a 4-layer hidden architecture, navigate to `arch1/a2c_4` and run the `a2c.py` script.
   - For the classical A2C with a 5-layer hidden architecture and post-processing NN, navigate to `arch1/a2c_5_ppnn` and run the `a2c.py` script.

3. **Random Agent:**
   - Navigate to the `random` folder.
   - Run the `random_agent.py` script.

  ## Results and Analysis
- Each architecture directory (arch1, arch2, arch3) contains respective results.
- The gradients folder contains scripts to visualize the gradient flow in the models.
- The hyperparameters_exp folder houses the exploration data for tuning the models.
- The paper folder contains the processed data in CSV format, detailing the performance and outcomes of the different models.
- The plots folder showcases the graphical representations of the data.
- Please refer to the paper for more details on the methodology, results, and analysis.

## Project Structure
- arch1/: Contains VQC with RX RY RZ rotations.
  - a2c_4: Pure VQC implementation.
  - a2c_5_ppnn: Hybrid VQC with post-processing NN.
- arch2/: Contains VQC with RY RZ RY rotations.
  - a2c_4: Pure VQC implementation.
  - a2c_5_ppnn: Hybrid VQC with post-processing NN.
- arch3/: Contains VQC with RZ RY RZ rotations used in the paper.
  - a2c_4: Pure VQC implementation.
  - a2c_5_ppnn: Hybrid VQC with post-processing NN.
- gradients/: Scripts for plotting the gradients of the model.
- hyperparameters_exp/: Experiments to determine the best hyperparameters.
- paper/: CSV files containing results used in the paper.
- plots/: Generated plots used in the paper.
- random/: Implementation of the random agent.
- plot_stats.py: Script to plot the performance of the agents.
- plot_stats_withsteps.py: Script to plot the performance of the agents with steps

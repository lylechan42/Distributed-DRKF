This repository contains a MATLAB implementation of a Distributed  Distributionally Robust Kalman Filter framework for state estimation across networked systems. The framework supports multiple estimation methods and diffusion strategies to optimize performance in distributed sensor networks.

## Overview

The DRKF framework implements various distributed estimation techniques for multi-agent systems where each agent has access to different sensor measurements. The implementation features:

- Distributed Kalman Filter (DKF) with multiple diffusion strategies
- Distributionally Robust Optimization (DRO) variants
- Network topology configuration
- Comprehensive Monte Carlo simulation capabilities
- Performance evaluation and visualization tools

## Network Configuration

The default implementation uses a 4-node square network topology where:

- Each node connects to two adjacent nodes
- Nodes have different sensor configurations
- The system allows testing various estimation and fusion methods

## Estimation Methods

The framework implements multiple estimation approaches:

1. **Standard DKF**: Classical Distributed Kalman Filter
2. DRO-based methods
   - KL Divergence-based robust estimation
   - Wasserstein distance-based robust estimation
   - Moment-based robust estimation

## Diffusion Strategies

Three diffusion (information fusion) approaches are implemented:

1. **No Diffusion**: Each node relies solely on its local measurements
2. **Average Diffusion**: Simple averaging of state estimates from neighboring nodes
3. **Covariance Intersection (CI)**: More sophisticated fusion that accounts for correlation between estimates

## Key Files

- `main.m`: Main execution script that runs all experiments
- `utils/`: Directory containing utility functions:
  - `getSys.m`: System model initialization
  - `getCorrData.m`: Data generation with correlated noise
  - `calculate_mse.m`: Performance calculation functions
  - `DRO.m`: Implementation of distributionally robust optimization
  - `initialize_nodes.m`: Network configuration
  - `plot_MSEs_new.m`: Visualization tools

# AquaOpt: Sea Lice Management Optimization

A Julia-based project for optimizing sea lice management strategies in aquaculture using various reinforcement learning and control algorithms.

## Project Structure

```
AquaOpt/
├── data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed data files
├── results/
│   ├── figures/               # Generated plots and visualizations
│   ├── policies/              # Saved policy files
│   └── data/                  # Simulation results
├── src/
│   ├── cleaning.jl            # Data cleaning and preprocessing
│   ├── SeaLicePOMDP.jl        # POMDP model definition
│   ├── SimulationPOMDP.jl     # Simulation environment
│   ├── kalmanFilter.jl        # Kalman filter implementation
│   ├── optimization.jl        # Policy optimization algorithms
│   └── plot_views.jl          # Visualization utilities
└── scripts/
    └── main.jl                # Main execution script
```

## Overview

This project implements various algorithms for optimizing sea lice management in aquaculture facilities. It uses Partially Observable Markov Decision Processes (POMDPs) to model the sea lice population dynamics and treatment decisions.

## Key Components

### 1. POMDP Model (`SeaLicePOMDP.jl`)
- Defines the sea lice management problem as a POMDP
- States: Sea lice levels (0.0 to 10.0)
- Actions: Treatment or No Treatment
- Observations: Noisy measurements of sea lice levels
- Rewards: Balance between treatment costs and sea lice levels

### 2. Simulation Environment (`SimulationPOMDP.jl`)
- Implements the simulation environment for testing policies
- Uses stochastic dynamics for sea lice growth
- Includes observation noise modeling

### 3. State Estimation (`kalmanFilter.jl`)
- Implements Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF)
- Used for state estimation in the POMDP framework
- Handles uncertainty in sea lice measurements

### 4. Optimization Algorithms (`optimization.jl`)
The project implements several algorithms for policy optimization:

1. **Value Iteration (VI)**
   - Classic dynamic programming approach
   - Solves the underlying MDP

2. **SARSOP**
   - Point-based POMDP solver
   - Efficient for large state spaces

3. **QMDP**
   - Approximate POMDP solver
   - Assumes full observability for next step

4. **Heuristic Policy**
   - Simple threshold-based policy
   - Uses belief state for decision making

### 5. Data Processing (`cleaning.jl`)
- Handles data preprocessing and cleaning
- Converts raw data into format suitable for analysis
- Implements data transformation utilities

### 6. Visualization (`plot_views.jl`)
- Generates various plots for analysis:
  - Sea lice levels over time
  - Policy comparison plots
  - Belief state visualization
  - Cost vs. sea lice level trade-offs

## Usage

1. Install required Julia packages:
```julia
using Pkg
Pkg.add(["POMDPs", "POMDPTools", "POMDPModels", "QMDP", "NativeSARSOP", "DiscreteValueIteration", "Plots", "StatsPlots", "JLD2", "CSV", "DataFrames"])
```

2. Run the main script:
```julia
julia scripts/main.jl --run
```

## Configuration

The main configuration parameters can be found in `scripts/main.jl`:
- Lambda values: Trade-off between treatment cost and sea lice levels
- Number of episodes: For policy evaluation
- Steps per episode: Simulation horizon
- Heuristic thresholds: For the heuristic policy

## Results

Results are saved in the `results/` directory:
- Policies are saved as JLD2 files
- Plots are saved as PNG files
- Simulation data is saved for further analysis

## Dependencies

- Julia 1.6+
- POMDPs.jl ecosystem
- Plots.jl for visualization
- JLD2 for data storage
- CSV and DataFrames for data handling
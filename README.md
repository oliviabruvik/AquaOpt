# AquaOpt: Sea Lice Management Optimization

A Julia-based project for optimizing sea lice management strategies in aquaculture using various reinforcement learning and control algorithms.

## Project Structure

```
AquaOpt/
├── data/
│   ├── raw/                    # Raw data files
│   │   ├── fish_disease.csv    # Fish disease data
│   │   ├── lice_treatments.csv # Treatment data
│   │   ├── licedata.csv        # Sea lice data
│   │   └── salmon_lice.csv     # Salmon lice data
│   └── processed/              # Processed data files
│       ├── bayesian_data.csv   # Bayesian analysis data
│       ├── sealice_data.csv    # Processed sea lice data
│       └── combined_*.csv      # Combined datasets
├── results/
│   ├── experiments/            # Experiment results by date
│   ├── figures/               # Generated plots and visualizations
│   ├── policies/              # Saved policy files
│   ├── data/                  # Simulation results
│   └── sensitivity_analysis/  # Sensitivity analysis results
├── src/
│   ├── AquaOpt.jl             # Main module entry point
│   ├── Models/                # POMDP model definitions
│   │   ├── SeaLicePOMDP.jl    # Base POMDP model
│   │   ├── SeaLiceLogPOMDP.jl # Log-space POMDP model
│   │   ├── SimulationPOMDP.jl # Simulation environment
│   │   ├── SimulationLogPOMDP.jl # Log-space simulation
│   │   └── KalmanFilter.jl    # State estimation filters
│   ├── Algorithms/            # Algorithm implementations
│   │   ├── Policies.jl        # Policy implementations
│   │   ├── Simulation.jl      # Simulation engine
│   │   ├── Evaluation.jl      # Policy evaluation and metrics
│   │   └── SensitivityAnalysis.jl # Sensitivity analysis
│   ├── Plotting/              # Visualization modules
│   │   ├── Timeseries.jl      # Time series plots and treatment distributions
│   │   ├── Comparison.jl      # Policy comparison plots
│   │   ├── Heatmaps.jl        # Heatmap visualizations
│   │   └── Convergence.jl     # Convergence analysis plots
│   ├── Data/                  # Data handling
│   │   ├── Loading.jl         # Data loading utilities
│   │   └── Cleaning.jl        # Data preprocessing
│   ├── Utils/                 # Utility modules
│   │   ├── Config.jl          # Configuration management
│   │   ├── Logging.jl         # Logging utilities
│   │   └── Validation.jl      # Input validation
│   └── MLE/                   # Maximum Likelihood Estimation
├── notebooks/                 # Jupyter notebooks for analysis
├── scripts/                   # Execution scripts
├── tests/                     # Test suite
├── Project.toml               # Package dependencies
└── Manifest.toml              # Dependency lock file
```

## Overview

This project implements various algorithms for optimizing sea lice management in aquaculture facilities. It uses Partially Observable Markov Decision Processes (POMDPs) to model the sea lice population dynamics and treatment decisions. The system considers multiple treatment options including chemical treatments, thermal treatments, and no-treatment scenarios.

## Key Components

### 1. Models (`src/Models/`)
- **SeaLicePOMDP.jl**: Base POMDP model for sea lice management
- **SeaLiceLogPOMDP.jl**: Log-space POMDP model for better numerical stability
- **SimulationPOMDP.jl**: Simulation environment for policy testing
- **KalmanFilter.jl**: State estimation using EKF and UKF

### 2. Algorithms (`src/Algorithms/`)
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

5. **Random Policy**
   - Baseline policy for comparison

### 3. Plotting (`src/Plotting/`)
- **Timeseries.jl**: 
  - Sea lice levels over time with confidence intervals
  - Treatment distribution comparisons
  - Policy belief evolution
  - Treatment cost analysis
  - Population dynamics visualization
- **Comparison.jl**: Policy comparison and Pareto frontiers
- **Heatmaps.jl**: Treatment decision heatmaps
- **Convergence.jl**: Algorithm convergence analysis

### 4. Data Handling (`src/Data/`)
- **Loading.jl**: Data loading and preprocessing utilities
- **Cleaning.jl**: Data cleaning and transformation functions

### 5. Utilities (`src/Utils/`)
- **Config.jl**: Configuration management and validation
- **Logging.jl**: Structured logging and progress tracking
- **Validation.jl**: Input validation and error handling

## Key Features

### Treatment Analysis
- **Multiple Treatment Types**: Chemical treatments, thermal treatments, and no-treatment scenarios
- **Treatment Distribution Analysis**: Bar charts comparing treatment frequencies between policies
- **Cost-Benefit Analysis**: Treatment costs vs. sea lice control effectiveness

### Population Dynamics
- **Sea Lice Stages**: Adult, motile, and sessile sea lice tracking
- **Regulatory Compliance**: Monitoring against regulatory limits (0.5 adult female lice per fish)
- **Belief State Evolution**: Kalman filter-based state estimation

### Policy Comparison
- **Multi-Policy Evaluation**: Comparison across VI, SARSOP, QMDP, and heuristic policies
- **Statistical Analysis**: Confidence intervals and statistical significance testing
- **Sensitivity Analysis**: Parameter sensitivity across different scenarios

## Usage

### Basic Usage

Run the main optimization script:
```julia
julia scripts/main.jl
```

### Policy Comparison

Compare specific policies (e.g., Heuristic vs SARSOP):
```julia
using AquaOpt

# Load your parallel data and config
# Then create comparison plots
plot_heuristic_vs_sarsop_sealice_levels_over_time(parallel_data, config)
plot_treatment_distribution_comparison(parallel_data, config)
```

### Sensitivity Analysis

Run sensitivity analysis across multiple parameters:
```julia
julia -e "using AquaOpt; main_sensitivity()"
```

### Custom Configuration

Create a custom configuration:
```julia
using AquaOpt

# Define algorithms
algorithms = [
    Algorithm(solver_name="Heuristic_Policy"),
    Algorithm(solver=ValueIterationSolver(max_iterations=30), solver_name="VI_Policy")
]

# Define configuration
config = Config(num_episodes=1000, steps_per_episode=52)
pomdp_config = POMDPConfig(log_space=true)

# Run optimization
results = test_optimizer(algorithms[1], config, pomdp_config)
```

## Configuration

### Main Configuration (`Config`)
- `lambda_values`: Trade-off between treatment cost and sea lice levels
- `num_episodes`: Number of episodes for policy evaluation
- `steps_per_episode`: Simulation horizon
- `heuristic_threshold`: Threshold for heuristic policy
- `process_noise`: Process noise for simulation
- `observation_noise`: Observation noise for simulation

### POMDP Configuration (`POMDPConfig`)
- `costOfTreatment`: Cost of treatment action
- `growthRate`: Sea lice growth rate
- `discount_factor`: Discount factor for future rewards
- `log_space`: Whether to use log-space model

## Results

Results are automatically saved in the `results/` directory:

- **Policies**: Saved as JLD2 files in `results/policies/`
- **Simulation Data**: Saved in `results/data/`
- **Plots**: Generated in `results/figures/`
  - Time series plots in `research_plots/sealice_time_plots/`
  - Treatment distribution plots in `research_plots/treatment_plots/`
  - Policy comparison plots in `research_plots/`
- **Sensitivity Analysis**: Results in `results/sensitivity_analysis/`

## Dependencies

### Core Dependencies
- **POMDPs.jl**: POMDP framework
- **POMDPTools**: POMDP utilities and tools
- **POMDPModels**: POMDP model implementations
- **DiscreteValueIteration**: Value iteration solver
- **NativeSARSOP**: SARSOP solver
- **QMDP**: QMDP solver

### Data and Visualization
- **DataFrames**: Data manipulation
- **Plots**: Plotting framework
- **JLD2**: Data serialization
- **CSV**: CSV file handling

### Utilities
- **Parameters**: Parameterized structs
- **GaussianFilters**: Kalman filter implementations
- **Distributions**: Probability distributions
- **Statistics**: Statistical functions

## Mathematical Formulation of the Sea Lice POMDP

### 1. Problem Definition

We formulate sea lice management in salmon aquaculture as a Partially Observable Markov Decision Process (POMDP) with the following components:

#### 1.1 State Space
The state $s_t \in \mathcal{S}$ at time $t$ is defined as:
$$s_t = (L_t^A, L_t^M, L_t^S, T_t, W_t^P, W_t^A, N_t, W_t^F, S_t)$$

where:
- $L_t^A$: Adult sea lice level (lice per fish)
- $L_t^M$: Motile sea lice level (lice per fish)  
- $L_t^S$: Sessile sea lice level (lice per fish)
- $T_t$: Water temperature (°C)
- $W_t^P$: Production week (weeks since production start)
- $W_t^A$: Annual week (week of the year, 1-52)
- $N_t$: Number of fish in the pen
- $W_t^F$: Average fish weight (kg)
- $S_t$: Water salinity (psu, constant)

#### 1.2 Action Space
The action space $\mathcal{A}$ consists of three treatment options:
$$\mathcal{A} = \{a_0, a_1, a_2\}$$

where:
- $a_0$: No treatment
- $a_1$: Chemical treatment (cost: 10 MNOK, effectiveness: 75-84%)
- $a_2$: Thermal treatment (cost: 15 MNOK, effectiveness: 70-88%)

#### 1.3 Observation Space
The observation $o_t \in \mathcal{O}$ is:
$$o_t = (\hat{L}_t^A, \hat{L}_t^M, \hat{L}_t^S, \hat{T}_t, W_t^P, W_t^A, \hat{N}_t, \hat{W}_t^F, S_t)$$

where $\hat{\cdot}$ denotes observed (noisy) values.

### 2. State Transition Dynamics

#### 2.1 Sea Lice Population Dynamics
The sea lice levels evolve according to a stage-structured population model:

**Treatment Effect:**
$$L_t^{A'} = L_t^A \cdot (1 - \rho_a(a))$$
$$L_t^{M'} = L_t^M \cdot (1 - \rho_m(a))$$
$$L_t^{S'} = L_t^S \cdot (1 - \rho_s(a))$$

where $\rho_i(a)$ is the treatment effectiveness for stage $i \in \{A, M, S\}$.

**Population Growth:**
$$L_{t+1}^S = s_1 \cdot L_t^{S'}$$
$$L_{t+1}^M = s_3 \cdot (1 - d_2(T_t)) \cdot L_t^{M'} + s_2 \cdot d_1(T_t) \cdot L_t^{S'}$$
$$L_{t+1}^A = s_4 \cdot L_t^{A'} + d_2(T_t) \cdot 0.5 \cdot (s_3 + s_4) \cdot L_t^{M'}$$

where:
- $s_1 = 0.49$: sessile survival rate
- $s_2 = 2.3$: sessile to motile scaling
- $s_3 = 0.88$: motile survival rate  
- $s_4 = 0.61$: adult survival rate
- $d_1(T) = \frac{1}{1 + \exp(-(-2.4 + 0.37(T - 9)))}$: sessile to motile development rate
- $d_2(T) = \frac{1}{1 + \exp(-(-2.1 + 0.037(T - 9)))}$: motile to adult development rate

#### 2.2 Temperature Dynamics
$$T_{t+1} = T_{mean} + T_{amp} \cdot \cos\left(\frac{2\pi(W_t^A - W_{peak})}{52}\right) + \epsilon_T$$

where:
- $T_{mean} = 12.0°C$: mean annual temperature
- $T_{amp} = 4.5°C$: temperature amplitude
- $W_{peak} = 27$: week of peak temperature
- $\epsilon_T \sim \mathcal{N}(0, \sigma_T^2)$: temperature noise

#### 2.3 Fish Population and Growth
**Fish Growth (von Bertalanffy-like):**
$$W_{t+1}^F = W_t^F + k_0(T_t) \cdot (W_{max} - W_t^F)$$

where:
- $k_0(T) = \max(k_{growth} \cdot (1 + \alpha_T(T - 10)), 0)$: temperature-dependent growth rate
- $W_{max} = 5.0$ kg: asymptotic weight
- $k_{growth} = 0.01$: base growth rate
- $\alpha_T = 0.03$: temperature sensitivity

**Fish Mortality:**
$$N_{t+1} = N_t \cdot (1 - \mu_{nat}) \cdot (1 - \mu_{trt}(a)) - H_t + M_t^{in} - M_t^{out}$$

where:
- $\mu_{nat} = 0.0008$: natural mortality rate
- $\mu_{trt}(a)$: treatment-induced mortality
- $H_t$: harvest
- $M_t^{in}, M_t^{out}$: fish movements

### 3. Observation Model

#### 3.1 Sea Lice Counting
The observed sea lice counts follow negative binomial distributions:

$$C_t^A \sim \text{NegBin}(r_A, p_A)$$
$$C_t^M \sim \text{NegBin}(r_M, p_M)$$
$$C_t^S \sim \text{NegBin}(r_S, p_S)$$

where:
- $r_i = n \cdot \rho_i$: dispersion parameter
- $p_i = \frac{n \cdot \rho_i}{n \cdot \rho_i + \mu_i}$: success probability
- $\mu_i = n \cdot p_{scount} \cdot L_t^i$: expected count
- $n = 20$: number of fish sampled
- $\rho_A = 0.175, \rho_M = 0.187, \rho_S = 0.037$: aggregation parameters
- $p_{scount} = \frac{1}{1 + \exp(-(\beta_0 + \beta_1(W_t^F - W_0)))}$: under-counting correction

The observed lice levels are:
$$\hat{L}_t^i = \frac{C_t^i}{n}$$

#### 3.2 Other Observations
$$\hat{T}_t = T_t + \epsilon_T, \quad \epsilon_T \sim \mathcal{N}(0, \sigma_T^2)$$
$$\hat{N}_t = N_t + \epsilon_N, \quad \epsilon_N \sim \mathcal{N}(0, \sigma_N^2)$$
$$\hat{W}_t^F = W_t^F + \epsilon_W, \quad \epsilon_W \sim \mathcal{N}(0, \sigma_W^2)$$

### 4. Reward Function

The reward function $R(s_t, a_t, s_{t+1})$ is defined as:

$$R(s_t, a_t, s_{t+1}) = -(\lambda_{trt} \cdot C_{trt}(a_t) + \lambda_{reg} \cdot P_{reg}(s_t, a_t) + \lambda_{bio} \cdot L_{bio}(s_t, s_{t+1}) + \lambda_{health} \cdot D_{health}(a_t, s_t))$$

where:

**Treatment Cost:**
$$C_{trt}(a) = \begin{cases}
0 & \text{if } a = a_0 \\
10 & \text{if } a = a_1 \\
15 & \text{if } a = a_2
\end{cases}$$

**Regulatory Penalty:**
$$P_{reg}(s, a) = \begin{cases}
1 & \text{if } L_t^A > L_{limit} \text{ and } a = a_0 \\
0 & \text{otherwise}
\end{cases}$$

**Biomass Loss:**
$$L_{bio}(s_t, s_{t+1}) = \max(B_t - B_{t+1}, 0)$$
where $B_t = W_t^F \cdot N_t$ is the total biomass.

**Fish Health Penalty:**
$$D_{health}(a, s) = D_{disease}(a) \cdot L_t^A$$

**Weight Parameters:**
- $\lambda_{trt} = 0.8$: treatment cost weight
- $\lambda_{reg} = 0.1$: regulatory penalty weight  
- $\lambda_{bio} = 0.05$: biomass loss weight
- $\lambda_{health} = 0.05$: health penalty weight

### 5. Belief State

The belief state $b_t$ represents the probability distribution over states:
$$b_t(s) = P(s_t = s | o_{1:t}, a_{1:t-1})$$

The belief is updated using Bayes' rule:
$$b_{t+1}(s_{t+1}) = \eta \cdot P(o_{t+1} | s_{t+1}, a_t) \sum_{s_t} P(s_{t+1} | s_t, a_t) b_t(s_t)$$

where $\eta$ is a normalization constant.

### 6. Policy and Value Function

A policy $\pi$ maps belief states to actions:
$$\pi: \mathcal{B} \rightarrow \mathcal{A}$$

The value function for a policy $\pi$ is:
$$V^\pi(b) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, \pi(b_t), s_{t+1}) | b_0 = b\right]$$

The optimal value function satisfies the Bellman equation:
$$V^*(b) = \max_{a \in \mathcal{A}} \left[\sum_{s} b(s) \sum_{s'} P(s' | s, a) R(s, a, s') + \gamma \sum_{o} P(o | a, b) V^*(b') \right]$$

where $\gamma = 0.95$ is the discount factor.

### 7. Objective

The objective is to find the optimal policy that maximizes the expected discounted cumulative reward:
$$\pi^* = \arg\max_{\pi} V^\pi(b_0)$$

This formulation captures the key trade-offs in sea lice management: treatment costs, regulatory compliance, fish health, and biomass preservation, while accounting for the uncertainty in sea lice observations and population dynamics.



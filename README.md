# AquaOpt: Sea Lice Management Optimization

AquaOpt is a Julia package for designing and assessing sea lice mitigation strategies on Norwegian salmon farms. The codebase couples low‑dimensional POMDP solvers with a biologically detailed simulator, making it possible to compare classic control baselines against modern POMDP methods while keeping the ecological and regulatory assumptions explicit.

## Repository Layout

```
.
├── src/                     # Core package (models, solvers, plotting, utilities)
├── scripts/                 # Reproducible analysis + manuscript tables
├── results/                 # Auto-generated experiment folders (policies, histories, plots)
├── final_results/, logs/    # Convenience exports from long experiments
├── paper_plots.jl           # Helper for manuscript-ready plots
├── Project.toml/Manifest.toml
└── runs.txt                 # Notes about previously executed runs
```

`src/AquaOpt.jl` wires every module together and exposes `main`, `run_experiments`, and plotting utilities so they can be called from scripts or notebooks.

## Modeling Summary

AquaOpt defines three nested models:

1. `SeaLicePOMDP`: 1-D adult abundance model in natural space.
2. `SeaLiceLogPOMDP`: identical dynamics solved in log space for numerical stability.
3. `SeaLiceSimPOMDP`: the high-fidelity simulator used for evaluation with explicit stage structure, biomass, temperature, and farm operations.

All models share actions, rewards, and observation assumptions.

### Temperature Forcing

Water temperature for location ℓ ∈ {north, west, south} is a cosine prior:

```
T(week, ℓ) = T_mean[ℓ] + T_amp[ℓ] * cos(2π * (week - peak_week[ℓ]) / 52)
```

Location parameters (stored in `src/Utils/Config.jl`) enforce realistic profiles, e.g. `north: T_mean=12 °C, T_amp=4.5, peak_week=27`, while the south site reaches `20 °C` on average. The same parameter set also defines development coefficients (`d1_*`, `d2_*`), survival rates (`s1…s4`), and external larval influx.

### Stage-Structured Dynamics

Both the simulator and the abstract POMDP transition through the helper `predict_next_abundances` (`src/Utils/Utils.jl`). For adult (`A_t`), motile (`M_t`), and sessile (`S_t`) lice, with water temperature `T_t`, location-specific parameters, and reproduction rate `ρ`:

```
d₁(T) = 1 / (1 + exp(-(d1_intercept + d1_temp_coef * (T - T_mean))))
d₂(T) = 1 / (1 + exp(-(d2_intercept + d2_temp_coef * (T - T_mean))))

S_{t+1} = s1_sessile * S_t + ρ * A_t + external_influx
M_{t+1} = s3_motile * (1 - d₂) * M_t + s2_scaling * d₁ * S_t
A_{t+1} = s4_adult * A_t + d₂ * 0.5 * (s3_motile + s4_adult) * M_t
```

Treatments apply multiplicative kill rates to `(A_t, M_t, S_t)` before the equations above. Adult lice are capped at 10 per fish to prevent numerical explosions. Default reproduction is `ρ = 2.0` new sessile larvae per adult female per week.

### Treatments and Action Set

Actions are defined in `src/Utils/SharedTypes.jl`:

| Action                | Cost (MNOK) | Adult↓ | Motile↓ | Sessile↓ | Mortality | Fish health penalty |
|-----------------------|-------------|--------|---------|----------|-----------|---------------------|
| `NoTreatment`         | 0.0         | 0%     | 0%      | 0%       | 0.00      | 0.0                 |
| `MechanicalTreatment` | 10.0        | 75%    | 84%     | 74%      | 0.05      | 10.0                |
| `ChemicalTreatment`   | 7.0         | 60%    | 58%     | 37%      | 0.04      | 7.0                 |
| `ThermalTreatment`    | 13.0        | 88%    | 87%     | 70%      | 0.07      | 15.0                |

Kill rates derive from Aldrin et al. (2023). Mortality and fish health penalties are expressed in percentage points and relative stress units, respectively, and propagate through the reward.

### Observation and Belief Update

Counts follow the Aldrin et al. (2023) sampling design: 20 fish are inspected per week (`n_sample = 20`). True lice per fish (`x`) are converted to total counts `μ_total = n_sample * p_scount * x`, where the optional under-reporting factor `p_scount = logistic(β₀ + β₁(W - 0.1))`.

Negative binomial parameters use `k = n_sample * ρ_stage`, `r = k`, and `p = k / (k + μ_total)`. Default dispersions are `ρ_adult = 0.175`, `ρ_motile = 0.187`, `ρ_sessile = 0.037`. `SeaLicePOMDP` and `SeaLiceLogPOMDP` discretize that likelihood onto their state grids; `SeaLiceSimPOMDP` samples counts directly.

During simulation the belief is maintained by an Unscented or Extended Kalman Filter (`src/Models/KalmanFilter.jl`) whose four-state vector is `[Adult, Motile, Sessile, Temperature]`. The process model is exactly the biological transition plus the temperature recursion, and the observation model is identity with diagonal Gaussian noise derived from simulator standard deviations.

### Reward Structure

Rewards are shared by all models (`SeaLicePOMDP.reward`, `SeaLiceLogPOMDP.reward`, `SeaLiceSimPOMDP.reward`) and use configurable weights `λ = [λ_trt, λ_reg, λ_bio, λ_health, λ_lice]`:

```
R = - (λ_trt * Cost_treat
       + λ_reg * RegulatoryPenalty
       + λ_bio * BiomassLoss
       + λ_health * TreatmentStress
       + λ_lice * LiceBurden)
```

- `RegulatoryPenalty`: 100 when adult lice exceed the 0.5 lice/fish limit (provides a large deterrent and can be customized).
- `BiomassLoss`: mortality from treatment (fraction * 400 tonnes) plus lice-driven growth reduction above 0.5 lice/fish in the low-dimensional models; the simulator additionally reduces growth through its state transition.
- `TreatmentStress`: action-specific `fish_disease` penalty (table above).
- `LiceBurden`: `Adult * (1 + 0.2 * max(0, Adult - 0.5))`, capturing non-linear damage.

Solver defaults (see `SolverConfig`) weight treatment cost heavily, while simulation reward weights (`SimulationConfig.sim_reward_lambdas`) emphasize welfare.

## Solvers and Policies

Defined in `src/Algorithms/Policies.jl` and instantiated via `define_algorithms`:

- `NeverTreat_Policy` / `AlwaysTreat_Policy`: deterministic baselines.
- `Random_Policy`: draws uniformly over the four actions.
- `Heuristic_Policy`: compares the posterior probability of exceeding a lice threshold (default 0.4 adult lice/fish) against action-specific belief thresholds (0.3 / 0.35 / 0.4 for mechanical/chemical/thermal).
- `NUS_SARSOP_Policy`: SARSOP from the `SARSOP.jl` package fed with the discretized `SeaLice(Log)POMDP`.
- `VI_Policy`: discrete value iteration from `DiscreteValueIteration.jl`.
- `QMDP_Policy`: myopic QMDP approximation.

Low-fidelity policies are wrapped by `AdaptorPolicy` (for simulator rollouts) or `LOFIAdaptorPolicy` (when simulator = solver model) to convert Gaussian beliefs into the discrete belief vectors expected by POMDPTools. For full-observability sanity checks there is also a `FullObservabilityAdaptorPolicy` that feeds true simulator states to the policy.

## Simulation, Evaluation, and State Variables

`SeaLiceSimPOMDP` (see `src/Models/SimulationPOMDP.jl`) augments lice stages with farm context:

```
EvaluationState = (
    SeaLiceLevel_pred, Adult, Motile, Sessile, Temperature,
    ProductionWeek, AnnualWeek, NumberOfFish, AvgFishWeight, Salinity
)
```

Key transition steps per week:

1. Apply action-specific kill rates to `Adult/Motile/Sessile`.
2. Advance biology with `predict_next_abundances`.
3. Update temperature using the cosine model (location dependent).
4. Update average weight with a von Bertalanffy-style rule:
   `W_{t+1} = W_t + k(T, lice) * (w_max - W_t)` where `k = max(0, k_growth * (1 + temp_sensitivity * (T - 10)) * lice_growth_factor)` and `lice_growth_factor = 1 / (1 + exp(5*(Adult - 0.5)))`.
5. Update fish counts with natural mortality (0.08%/week) and treatment mortality, plus any scripted harvest/move events.
6. Add Gaussian process noise to lice stages and temperature, then sample noisy observations via the negative binomial model.

The simulator’s reward tracks the same components described earlier, but biomass loss is derived from the fish counts and weights instead of a static approximation.

Evaluation is orchestrated in `src/Algorithms/Simulation.jl` and `Evaluation.jl`:

1. `generate_mdp_pomdp_policies` solves each policy for every λ in `ExperimentConfig.lambda_values` and stores them under `results/experiments/<exp>/<algo>/pomdp_mdp`.
2. `simulate_all_policies` loads those policies, wraps them for simulator compatibility, and runs `num_episodes × steps_per_episode` rollouts with shared seeds.
3. `extract_reward_metrics` computes treatment counts, average lice levels, biomass loss, regulatory exceedances, and fish disease tallies per episode/history.
4. `evaluate_simulation_results` aggregates by λ and writes CSV/JLD summaries for manuscript tables.

Histories and plots are written inside `results/experiments/<exp>/` so every run remains self-contained.

## Plotting and Reports

`src/Plotting/` contains utilities for the manuscript and diagnostics:

- `Timeseries.jl`, `Comparison.jl`, `Heatmaps.jl`: lice trajectories, treatment distributions, policy comparisons, and action heatmaps.
- `ParallelPlots.jl`, `PlosOnePlots.jl`: figure panels tailored for the manuscript (multiple policies per chart, treatment stack plots, etc.).
- `PlotUtils.jl`: shared styling and PGFPlotsX helpers.

`paper_plots.jl` shows how to load an experiment’s `parallel_data` and replay the plotting pipeline. Additional scripts:

- `scripts/policy_analysis.jl`: runs `main` in “paper” mode for a chosen location.
- `scripts/reward_analysis.jl`: sweeps reward weights to produce sensitivity plots.
- `scripts/region_analysis.jl`: contrasts north/west/south parameterizations.
- `scripts/aggregate_lambda_summaries.jl`: merges per-λ metrics into manuscript tables.
- `scripts/generate_methodology_tables.jl`: exports LaTeX tables describing the exact equations above (useful when drafting the Methods section).
- `scripts/simulation_tests.jl`: quick regression checks on simulator settings.

## Running Experiments

1. **Install dependencies**
   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```
2. **Configure reward weights** – supply `reward_lambdas` (solver) and `sim_reward_lambdas` (high-fidelity simulator). Example values from the manuscript:
   ```julia
   solver_weights = [0.8, 0.2, 0.0, 0.0, 0.0]
   sim_weights = [0.7, 0.2, 0.1, 0.9, 2.0]
   ```
3. **Run the pipeline**
   ```bash
   julia --project=. -e '
       using AquaOpt;
       main(first_step_flag="solve",
            log_space=true,
            experiment_name="paper_run",
            mode="paper",
            location="south",
            ekf_filter=true,
            plot=true,
            reward_lambdas=solver_weights,
            sim_reward_lambdas=sim_weights)
   '
   ```
   `mode` selects solver hyperparameters (`debug`, `light`, `paper` presets in `setup_experiment_configs`).
4. **Inspect outputs** – metrics (`results/experiments/.../avg_results.csv`), treatment breakdowns (`extract_reward_metrics`), and figures (`figures/`).

To skip solving and reuse existing policies, point `get_latest_matching_config` (in `ExperimentTracking.jl`) to a prior experiment and run with `first_step_flag="simulate"` or `first_step_flag="plot"`.

## Notes for Manuscript Preparation

- All methodology equations referenced above come directly from `predict_next_abundances` (biological drift), `get_temperature` (forcing), the action table (`ACTION_CONFIGS`), and the shared reward functions in `src/Models/`. The same functions are reused everywhere to avoid solver/simulator mismatch.
- Parameter tables (`scripts/generate_methodology_tables.jl`) pull from the exact structs documented here, so citing those outputs in the manuscript will remain consistent with the code.
- Solver metadata, config snapshots, and tracking CSVs are automatically written to `results/experiments/experiments.csv`, which is useful for the Methods section when listing run settings or seeds.

For any additional provenance, consult `runs.txt` or the `logs/` directory to see when large experiments were executed.

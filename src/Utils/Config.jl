using Parameters

# TODO: checkout confparser.jl (read in TOML files)

# ----------------------------
# Location-specific biological parameters
# ----------------------------
@with_kw struct LocationParams
    # Temperature model parameters
    T_mean::Float64         # Average annual temperature (°C)
    T_amp::Float64          # Temperature amplitude (°C)
    peak_week::Int          # Week of peak temperature

    # Development rate parameters (logistic function coefficients)
    d1_intercept::Float64   # Intercept for sessile → motile development
    d1_temp_coef::Float64   # Temperature coefficient for sessile → motile
    d2_intercept::Float64   # Intercept for motile → adult development
    d2_temp_coef::Float64   # Temperature coefficient for motile → adult

    # Weekly survival probabilities
    s1_sessile::Float64     # Sessile stage survival
    s2_scaling::Float64     # Sessile → motile scaling factor
    s3_motile::Float64      # Motile stage survival
    s4_adult::Float64       # Adult stage survival

    # External larval pressure
    external_influx::Float64  # Weekly influx of sessile larvae from external sources
end

# Shared biological parameters across all locations.
# Only T_mean and external_influx differ between north/west/south.
const SHARED_BIO_PARAMS = (
    T_amp        = 4.5,
    peak_week    = 27,
    d1_intercept = -2.4,
    d1_temp_coef = 0.37,
    d2_intercept = -2.1,
    d2_temp_coef = 0.037,
    s1_sessile   = 0.49,
    s2_scaling   = 2.3,
    s3_motile    = 0.88,
    s4_adult     = 0.61,
)

# Pre-built location parameter sets keyed by location name
const LOCATION_PARAMS = Dict(
    "north" => LocationParams(;
        SHARED_BIO_PARAMS...,
        T_mean = 8.0,
        external_influx = 0.1,
    ),
    "west" => LocationParams(;
        SHARED_BIO_PARAMS...,
        T_mean = 10.0,
        external_influx = 0.12,
    ),
    "south" => LocationParams(;
        SHARED_BIO_PARAMS...,
        T_mean = 12.0,
        external_influx = 0.15,
    ),
)

"""
Look up biological parameters for a given location ("north", "west", or "south").
"""
function get_location_params(location::String)
    if !haskey(LOCATION_PARAMS, location)
        error("Invalid location: $location. Must be 'north', 'west', or 'south'")
    end
    return LOCATION_PARAMS[location]
end

# ----------------------------
# Solver Configuration (affects policy structure)
# ----------------------------
@with_kw struct SolverConfig
    # POMDP structure parameters (affect the policy being solved)
    reproduction_rate::Float64 = 4.0
    discount_factor::Float64 = 0.95
    adult_sd::Float64 = 0.05
    log_space::Bool = false
    regulation_limit::Float64 = 0.5
    season_regulation_limits::Vector{Float64} = [0.2, 0.5, 0.5, 0.5]  # [Spring, Summer, Autumn, Winter]
    location::String = "north"
    discretization_step::Float64 = 0.1
    full_observability_solver::Bool = false

    # Financial parameters — country-specific, passed to all POMDPs
    salmon_price_MNOK_per_tonne::Float64 = 0.07  # ~70 NOK/kg Norwegian salmon spot price
    regulatory_violation_cost_MNOK::Float64 = 10.0  # Forced emergency treatment + production disruption + license risk
    welfare_cost_MNOK::Float64 = 1.0  # Per stress-score unit
    chronic_lice_cost_MNOK::Float64 = 0.5  # Per burden-unit/week

    # Reward weights for solving
    reward_lambdas::Vector{Float64} = [0.8, 0.2, 0.0, 0.0, 0.0] # [treatment, regulatory, biomass, health, sea lice]

    # Solver algorithm parameters
    sarsop_max_time::Float64 = 150.0
    VI_max_iterations::Int = 30
    QMDP_max_iterations::Int = 30

    # Heuristic parameters
    heuristic_threshold::Float64 = 0.5
    heuristic_belief_threshold_mechanical::Float64 = 0.3
    heuristic_belief_threshold_chemical::Float64 = 0.35
    heuristic_belief_threshold_thermal::Float64 = 0.4
    heuristic_rho::Float64 = 0.8
end

# ----------------------------
# Simulation Configuration (for evaluating policies)
# ----------------------------
@with_kw struct SimulationConfig
    # Simulation run parameters
    num_episodes::Int = 10
    steps_per_episode::Int = 20
    ekf_filter::Bool = true
    step_through::Bool = false
    verbose::Bool = false
    high_fidelity_sim::Bool = true

    # SimPOMDP parameters (stochasticity in simulation)
    adult_mean::Float64 = 0.13
    motile_mean::Float64 = 0.47
    sessile_mean::Float64 = 0.12

    # Observation noise (biological variability in transitions)
    adult_obs_sd::Float64 = 0.17 
    motile_obs_sd::Float64 = 0.327
    sessile_obs_sd::Float64 = 0.10

    # Observation noise (measurement uncertainty from Negative Binomial sampling)
    adult_sd::Float64 = 0.1     # 0.05
    motile_sd::Float64 = 0.29   # 0.1
    sessile_sd::Float64 = 0.16  # 0.05
    temp_sd::Float64 = 0.3

    # Observation parameters from Aldrin et al. 2023
    n_sample::Int = 20
    ρ_adult::Float64 = 0.175
    ρ_motile::Float64 = 0.187
    ρ_sessile::Float64 = 0.037

    # Under-reporting parameters from Aldrin et al. 2023
    use_underreport::Bool = false
    beta0_Scount_f::Float64 = -1.535
    beta1_Scount::Float64 = 0.039
    mean_fish_weight_kg::Float64 = 1.5
    W0::Float64 = 0.1

    # Reward weights for simulation evaluation (can differ from solving)
    sim_reward_lambdas::Vector{Float64} = [0.7, 0.2, 0.1, 0.9, 2.0]
end

# ----------------------------
# Experiment struct (combines solver and simulation configs)
# ----------------------------
@with_kw struct ExperimentConfig
    # Configurations
    solver_config::SolverConfig = SolverConfig()
    simulation_config::SimulationConfig = SimulationConfig()

    # File management
    experiment_name::String = "exp"
    policies_dir::String = joinpath("results", "experiments", experiment_name,"policies")
    simulations_dir::String = joinpath("results", "experiments", experiment_name, "simulation_histories")
    results_dir::String = joinpath("results", "experiments", experiment_name, "avg_results")
    figures_dir::String = joinpath("results", "experiments", experiment_name, "figures")
    experiment_dir::String = joinpath("results", "experiments", experiment_name)
end

# ----------------------------
# Algorithm struct
# ----------------------------
struct HeuristicSolver <: Solver end

@with_kw struct Algorithm{S<:Union{Solver, Nothing}}
    solver::S = HeuristicSolver()
    solver_name::String = "Heuristic_Policy"
    solver_config::SolverConfig = SolverConfig()
end

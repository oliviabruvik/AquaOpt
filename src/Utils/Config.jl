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

# Define location-specific parameter sets
function get_location_params(location::String)
    if location == "north"
        return LocationParams(
            T_mean = 12.0,
            T_amp = 4.5,
            peak_week = 27,
            d1_intercept = -2.4,
            d1_temp_coef = 0.37,
            d2_intercept = -2.1,
            d2_temp_coef = 0.037,
            s1_sessile = 0.49,
            s2_scaling = 2.3,
            s3_motile = 0.88,
            s4_adult = 0.61,
            external_influx = 0.1
        )
    elseif location == "west"
        return LocationParams(
            T_mean = 16.0,
            T_amp = 4.5,
            peak_week = 27,
            d1_intercept = -1.5,
            d1_temp_coef = 0.5,
            d2_intercept = -1.0,
            d2_temp_coef = 0.1,
            s1_sessile = 0.6,
            s2_scaling = 3.0,
            s3_motile = 0.90, # Reduced from 0.95
            s4_adult = 0.70,
            external_influx = 0.12
        )
    elseif location == "south"
        return LocationParams(
            T_mean = 20.0,
            T_amp = 4.5,
            peak_week = 27,
            d1_intercept = -1.5,
            d1_temp_coef = 0.5,
            d2_intercept = -1.0,
            d2_temp_coef = 0.1,
            s1_sessile = 0.7,      # Reduced from 0.8
            s2_scaling = 3.5,      # Reduced from 5.0
            s3_motile = 0.92,      # Reduced from 0.99
            s4_adult = 0.85,       # Reduced from 0.99
            external_influx = 0.15 # Reduced from 0.2
        )
    else
        error("Invalid location: $location. Must be 'north', 'west', or 'south'")
    end
end

# ----------------------------
# Solver Configuration (affects policy structure)
# ----------------------------
@with_kw struct SolverConfig
    # POMDP structure parameters (affect the policy being solved)
    costOfTreatment::Float64 = 10.0
    growthRate::Float64 = 0.15
    reproduction_rate::Float64 = 2.0
    discount_factor::Float64 = 0.95
    raw_space_sampling_sd::Float64 = 0.5
    log_space::Bool = false
    regulation_limit::Float64 = 0.5
    location::String = "north"
    discretization_step::Float64 = 0.1
    full_observability_solver::Bool = false

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
@with_kw mutable struct SimulationConfig
    # Simulation run parameters
    num_episodes::Int = 10
    steps_per_episode::Int = 20
    ekf_filter::Bool = true
    step_through::Bool = false
    verbose::Bool = false
    high_fidelity_sim::Bool = true

    # SimPOMDP parameters (stochasticity in simulation)
    adult_mean::Float64 = 0.125
    motile_mean::Float64 = 0.25
    sessile_mean::Float64 = 0.25
    adult_sd::Float64 = 0.05
    motile_sd::Float64 = 0.1
    sessile_sd::Float64 = 0.1
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
@with_kw mutable struct ExperimentConfig
    # Configurations
    solver_config::SolverConfig = SolverConfig()
    simulation_config::SimulationConfig = SimulationConfig()

    # Algorithm parameters
    lambda_values::Vector{Float64} = [0.6]

    # File management
    experiment_name::String = "exp"
    policies_dir::String = joinpath("results", "experiments", experiment_name,"policies")
    simulations_dir::String = joinpath("results", "experiments", experiment_name, "simulation_histories")
    results_dir::String = joinpath("results", "experiments", experiment_name, "avg_results")
    figures_dir::String = joinpath("results", "experiments", experiment_name, "figures")
    experiment_dir::String = joinpath("results", "experiments", experiment_name)
end

# ----------------------------
# Heuristic config struct
# ----------------------------
@with_kw struct HeuristicConfig
    raw_space_threshold::Float64 = 0.4
    belief_threshold_mechanical::Float64 = 0.3
    belief_threshold_chemical::Float64 = 0.35
    belief_threshold_thermal::Float64 = 0.4
    rho::Float64 = 0.8
end

# ----------------------------
# Algorithm struct
# ----------------------------
@with_kw struct Algorithm{S<:Union{Solver, Nothing}}
    solver::S = nothing # TODO: set to heuristic solver
    solver_name::String = "Heuristic_Policy"
    heuristic_config::HeuristicConfig = HeuristicConfig()
end

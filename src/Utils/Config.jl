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
    d1_temp_offset::Float64 # Temperature offset for sessile → motile
    d2_intercept::Float64   # Intercept for motile → adult development
    d2_temp_coef::Float64   # Temperature coefficient for motile → adult
    d2_temp_offset::Float64 # Temperature offset for motile → adult

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
            d1_temp_offset = 9.0,
            d2_intercept = -2.1,
            d2_temp_coef = 0.037,
            d2_temp_offset = 9.0,
            s1_sessile = 0.49,
            s2_scaling = 2.3,
            s3_motile = 0.88,
            s4_adult = 0.61,
            external_influx = 0.01
        )
    elseif location == "west"
        return LocationParams(
            T_mean = 16.0,
            T_amp = 4.5,
            peak_week = 27,
            d1_intercept = -1.5,
            d1_temp_coef = 0.5,
            d1_temp_offset = 16.0,
            d2_intercept = -1.0,
            d2_temp_coef = 0.1,
            d2_temp_offset = 16.0,
            s1_sessile = 0.6,
            s2_scaling = 3.0,
            s3_motile = 0.95,
            s4_adult = 0.70,
            external_influx = 0.1
        )
    elseif location == "south"
        return LocationParams(
            T_mean = 20.0,
            T_amp = 4.5,
            peak_week = 27,
            d1_intercept = -1.5,
            d1_temp_coef = 0.5,
            d1_temp_offset = 20.0,
            d2_intercept = -1.0,
            d2_temp_coef = 0.1,
            d2_temp_offset = 20.0,
            s1_sessile = 0.8,
            s2_scaling = 5.0,
            s3_motile = 0.99,
            s4_adult = 0.99,
            external_influx = 0.2
        )
    else
        error("Invalid location: $location. Must be 'north', 'west', or 'south'")
    end
end

# ----------------------------
# Experiment struct
# ----------------------------
@with_kw mutable struct ExperimentConfig

    # Simulation parameters
    num_episodes::Int = 10
    steps_per_episode::Int = 20
    ekf_filter::Bool = true
    step_through::Bool = false
    verbose::Bool = false
    high_fidelity_sim::Bool = true
    full_observability_solver::Bool = false
    discretization_step::Float64 = 0.1

    # POMDP parameters
    costOfTreatment::Float64 = 10.0
    growthRate::Float64 = 0.15 # 0.3 #1.26 # "The growth rate of sea lice is 0.3 per day." Costello (2006)
    reproduction_rate::Float64 = 2.0 # Number of sessile larvae produced per adult female per week (fecundity)
    discount_factor::Float64 = 0.95
    raw_space_sampling_sd::Float64 = 0.5
    log_space::Bool = false
    regulation_limit::Float64 = 0.5
    location::String = "north" # Location for temperature and biological model: "north", "west", or "south"

    # SimPOMDP parameters
    adult_mean::Float64 = 0.125
    motile_mean::Float64 = 0.25
    sessile_mean::Float64 = 0.25
    adult_sd::Float64 = 0.05
    motile_sd::Float64 = 0.1
    sessile_sd::Float64 = 0.1
    temp_sd::Float64 = 0.3

    # Observation parameters from Aldrin et al. 2023
    n_sample::Int = 20                      # number of fish counted (ntc)
    ρ_adult::Float64 = 0.175                # aggregation parameter for adults
    ρ_motile::Float64 = 0.187               # aggregation parameter for motile
    ρ_sessile::Float64 = 0.037              # aggregation parameter for sessile

    # Under-reporting parameters from Aldrin et al. 2023
    use_underreport::Bool = false           # toggle logistic under-count correction
    beta0_Scount_f::Float64 = -1.535        # farm-specific intercept (can vary by farm)
    beta1_Scount::Float64 = 0.039           # common weight slope
    mean_fish_weight_kg::Float64 = 1.5      # mean fish weight (kg)
    W0::Float64 = 0.1                       # weight centering (kg)

    # Algorithm parameters
    lambda_values::Vector{Float64} = [0.6] # collect(0.0:0.2:1.0)
    reward_lambdas::Vector{Float64} = [0.8, 0.2, 0.0, 0.0, 0.0] # [treatment, regulatory, biomass, health, sea lice]
    sim_reward_lambdas::Vector{Float64} = [0.7, 0.2, 0.1, 0.9, 2.0] # for high-fidelity sim
    sarsop_max_time::Float64 = 150.0
    VI_max_iterations::Int = 30
    QMDP_max_iterations::Int = 30

    # Heuristic parameters
    heuristic_threshold::Float64 = 0.5  # In absolute space
    heuristic_belief_threshold_mechanical::Float64 = 0.3
    heuristic_belief_threshold_thermal::Float64 = 0.4
    heuristic_rho::Float64 = 0.8

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
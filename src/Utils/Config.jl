using Parameters

# TODO: checkout confparser.jl (read in TOML files)

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
    
    # POMDP parameters
    costOfTreatment::Float64 = 10.0
    growthRate::Float64 = 0.15 # 0.3 #1.26 # "The growth rate of sea lice is 0.3 per day." Costello (2006)
    rho::Float64 = 0.95 # "The treatment kills off 95% on all stages." DOI: 10.1016/j.aquaculture.2019.734329
    discount_factor::Float64 = 0.95
    raw_space_sampling_sd::Float64 = 0.5
    log_space::Bool = false
    regulation_limit::Float64 = 0.5

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
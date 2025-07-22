using Parameters

# ----------------------------
# Experiment struct
# ----------------------------
@with_kw struct ExperimentConfig

    # Simulation parameters
    num_episodes::Int = 10
    steps_per_episode::Int = 20
    process_noise::Float64 = 0.5
    observation_noise::Float64 = 0.5
    ekf_filter::Bool = false
    step_through::Bool = false
    verbose::Bool = false
    
    # POMDP parameters
    costOfTreatment::Float64 = 10.0
    growthRate::Float64 = 0.3 #1.26 # "The growth rate of sea lice is 0.3 per day." Costello (2006)
    rho::Float64 = 0.95 # "The treatment kills off 95% on all stages." DOI: 10.1016/j.aquaculture.2019.734329
    discount_factor::Float64 = 0.95
    log_space::Bool = false
    skew::Bool = false

    # Algorithm parameters
    lambda_values::Vector{Float64} = collect(0.0:0.2:1.0) # collect(0.0:0.05:1.0)
    sarsop_max_time::Float64 = 150.0
    VI_max_iterations::Int = 30
    QMDP_max_iterations::Int = 30

    # Heuristic parameters
    heuristic_threshold::Float64 = 5.0  # In absolute space
    heuristic_belief_threshold::Float64 = 0.5
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
    raw_space_threshold::Float64 = 5.0
    belief_threshold::Float64 = 0.5
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

# ----------------------------
# POMDP config struct
# ----------------------------
@with_kw struct POMDPConfig
    costOfTreatment::Float64 = 10.0
    growthRate::Float64 = 1.26
    rho::Float64 = 0.7
    discount_factor::Float64 = 0.95
    log_space::Bool = false
    skew::Bool = false
end
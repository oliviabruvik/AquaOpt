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

    # POMDP parameters
    costOfTreatment::Float64 = 10.0
    growthRate::Float64 = 1.26
    rho::Float64 = 0.7
    discount_factor::Float64 = 0.95
    log_space::Bool = false
    
    # Algorithm parameters
    lambda_values::Vector{Float64} = collect(0.0:0.05:1.0)
    heuristic_threshold::Float64 = 5.0  # In absolute space
    heuristic_belief_threshold::Float64 = 0.5
    heuristic_rho::Float64 = 0.8

    # File management
    policies_dir::String = joinpath("results", "policies")
    figures_dir::String = joinpath("results", "figures")
    data_dir::String = joinpath("results", "data")
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
    convert_to_mdp::Bool = false
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
end
using Parameters

# ----------------------------
# Configuration struct
# ----------------------------
@with_kw struct Config
    lambda_values::Vector{Float64} = collect(0.0:0.05:1.0)
    num_episodes::Int = 10
    steps_per_episode::Int = 20
    heuristic_threshold::Float64 = 5.0  # In absolute space
    heuristic_belief_threshold::Float64 = 0.5
    process_noise::Float64 = 0.5
    observation_noise::Float64 = 0.5
    policies_dir::String = joinpath("results", "policies")
    figures_dir::String = joinpath("results", "figures")
    data_dir::String = joinpath("results", "data")
    ekf_filter::Bool = false
end

# ----------------------------
# Algorithm struct
# ----------------------------
@with_kw struct Algorithm{S<:Union{Solver, Nothing}}
    solver::S = nothing # TODO: set to heuristic solver
    convert_to_mdp::Bool = false
    solver_name::String = "Heuristic_Policy"
    heuristic_threshold::Union{Float64, Nothing} = nothing # set to heuristic threshold
    heuristic_belief_threshold::Union{Float64, Nothing} = nothing
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
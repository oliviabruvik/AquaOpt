include("../Models/KalmanFilter.jl")
include("../Models/SeaLicePOMDP.jl")
include("../Models/SeaLiceLogPOMDP.jl")
include("../Models/SimulationPOMDP.jl")
include("../Models/SimulationLogPOMDP.jl")
include("../Utils/Config.jl")
include("Policies.jl")
include("../Data/Loading.jl")
include("Simulation.jl")

using GaussianFilters
using POMDPs
using POMDPModels
using POMDPTools
using DataFrames
using JLD2
using Plots
using Distributions
using Parameters

# ----------------------------
# Optimizer Wrapper
# ----------------------------
function test_optimizer(algorithm, config)

    results = DataFrame(
        lambda=Float64[],
        avg_treatment_cost=Float64[],
        avg_sealice=Float64[],
        avg_reward=Float64[],
        state_hists=Vector{Any}[],
        action_hists=Vector{Any}[],
        belief_hists=Vector{Any}[]
    )

    # Create directory for simulation histories and results
    histories_dir = joinpath(config.data_dir, "simulation_histories", "$(algorithm.solver_name)")
    mkpath(histories_dir)
    results_dir = joinpath(config.data_dir, "avg_results")
    mkpath(results_dir)
    policies_dir = joinpath(config.data_dir, "policies", "$(algorithm.solver_name)")
    mkpath(policies_dir)

    # Generate policies for each lambda
    for λ in config.lambda_values

        # Generate POMDP and MDP
        if config.log_space
            pomdp = SeaLiceLogMDP(
                lambda=λ,
                costOfTreatment=config.costOfTreatment,
                growthRate=config.growthRate,
                rho=config.rho,
                discount_factor=config.discount_factor,
                skew=config.skew
            )
        else
            pomdp = SeaLiceMDP(
                lambda=λ,
                costOfTreatment=config.costOfTreatment,
                growthRate=config.growthRate,
                rho=config.rho,
                discount_factor=config.discount_factor,
                skew=config.skew
            )
        end
    
        # Get the underlying MDP
        mdp = UnderlyingMDP(pomdp)

        # Generate policy
        policy = generate_policy(algorithm, pomdp, mdp)

        # Run simulation to calculate average cost and average sea lice level
        r_total_hists, action_hists, state_hists, measurement_hists, reward_hists, belief_hists = run_simulation(policy, mdp, pomdp, config, algorithm)
        avg_reward, avg_cost, avg_sealice = calculate_averages(config, pomdp, action_hists, state_hists, reward_hists)

        # Calculate the average reward, cost, and sea lice level
        push!(results, (λ, avg_cost, avg_sealice, avg_reward, state_hists, action_hists, belief_hists))

        # Save all histories for this lambda
        histories = Dict(
            "r_total_hists" => r_total_hists,
            "action_hists" => action_hists,
            "state_hists" => state_hists,
            "measurement_hists" => measurement_hists,
            "reward_hists" => reward_hists,
            "belief_hists" => belief_hists,
            "lambda" => λ,
            "avg_reward" => avg_reward,
            "avg_cost" => avg_cost,
            "avg_sealice" => avg_sealice
        )
        
        # Save histories to file
        history_filename = "hists_$(λ)_lambda"
        @save joinpath(histories_dir, "$(history_filename).jld2") histories
        CSV.write(joinpath(histories_dir, "$(history_filename).csv"), DataFrame(histories))

        # Save policy, pomdp, and mdp to file
        policy_pomdp_mdp_filename = "policy_pomdp_mdp_$(λ)_lambda"
        @save joinpath(policies_dir, "$(policy_pomdp_mdp_filename).jld2") policy pomdp mdp
    end

    # Save results
    avg_results_filename = "$(algorithm.solver_name)_avg_results"
    @save joinpath(results_dir, "$(avg_results_filename).jld2") results
    
    return results
end

# ----------------------------
# Evaluate Policy
# ----------------------------
function evaluate_policy(lambda_values; mdp, solver, episodes=100, steps_per_episode=50)
    results = DataFrame(lambda=Float64[], avg_treatment_cost=Float64[], avg_sealice=Float64[])

    for λ in lambda_values
        mdp = mdp(lambda=λ)
        policy = solve(solver, mdp)

        total_cost = 0.0
        total_sealice = 0.0
        total_steps = episodes * steps_per_episode

        for _ in 1:episodes
            s = rand(initialstate(mdp))
            for _ in 1:steps_per_episode
                a = action(policy, s)
                total_cost += (a == Treatment ? mdp.costOfTreatment : 0.0)
                total_sealice += s.SeaLiceLevel
                s = rand(transition(mdp, s, a))
            end
        end

        avg_cost = total_cost / total_steps
        avg_sealice = total_sealice / total_steps

        push!(results, (λ, avg_cost, avg_sealice))
    end

    rename!(results, [:lambda, :avg_treatment_cost, :avg_sealice])
    return results
end
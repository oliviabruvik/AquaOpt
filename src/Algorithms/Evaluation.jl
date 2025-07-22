include("../Models/KalmanFilter.jl")
include("../Models/SeaLicePOMDP.jl")
include("../Models/SeaLiceLogPOMDP.jl")
include("../Models/SimulationPOMDP.jl")
include("../Models/SimulationLogPOMDP.jl")
include("../Utils/Config.jl")
include("Policies.jl")
include("Simulation.jl")

using GaussianFilters
using POMDPs
using POMDPModels
using POMDPTools
using POMDPXFiles
using DataFrames
using JLD2
using Plots
using Distributions
using Parameters

# ----------------------------
# Calculate averages
# ----------------------------
function evaluate_simulation_results(config, algorithm, histories)

    # Create directory for simulation histories
    histories_dir = joinpath(config.data_dir, "simulation_histories", "$(algorithm.solver_name)")
    histories_filename = "$(algorithm.solver_name)_histories"

    avg_results_dir = joinpath(config.data_dir, "avg_results")
    mkpath(avg_results_dir)
    
    @load joinpath(histories_dir, "$(histories_filename).jld2") histories

    avg_results = DataFrame(
        lambda=Float64[],
        avg_treatment_cost=Float64[],
        avg_sealice=Float64[],
        avg_reward=Float64[],
    )

    for λ in config.lambda_values

        # Get histories for this lambda
        histories_lambda = histories[histories.lambda .== λ, :]

        # Get action, state, and reward histories
        action_hists = histories_lambda.action_hists[1]
        state_hists = histories_lambda.state_hists[1]
        reward_hists = histories_lambda.reward_hists[1]

        avg_reward, avg_treatment_cost, avg_sealice = calculate_averages(config, action_hists, state_hists, reward_hists)

        # Calculate the average reward, cost, and sea lice level
        push!(avg_results, (λ, avg_treatment_cost, avg_sealice, avg_reward))

    end

    # Save results
    avg_results_filename = "$(algorithm.solver_name)_avg_results"
    @save joinpath(avg_results_dir, "$(avg_results_filename).jld2") avg_results
    
    return avg_results
end
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
# Calculate average 
# ----------------------------
function evaluate_simulation_results(config, algorithm, histories)

    # Create directory for simulation histories
    histories_dir = joinpath(config.simulations_dir, "$(algorithm.solver_name)")
    histories_filename = "$(algorithm.solver_name)_histories"
    
    @load joinpath(histories_dir, "$(histories_filename).jld2") histories

    avg_results = DataFrame(
        lambda=Float64[],
        avg_treatment_cost=Float64[],
        avg_sealice=Float64[],
        avg_reward=Float64[],
    )

    for λ in config.lambda_values

        # Get histories for this lambda
        histories_lambda = histories[λ]

        episode_costs, episode_abundances, episode_rewards = [], [], []

        for episode in 1:config.num_episodes

            # Get episode history
            episode_history = histories_lambda[episode]

            # Get action, state, and reward histories
            actions = collect(action_hist(episode_history))
            states = collect(state_hist(episode_history))
            rewards = collect(reward_hist(episode_history))

            # Get total treatment cost
            episode_cost = sum(a == Treatment for a in actions) * config.costOfTreatment
            
            # Get mean abundance
            episode_abundance = config.log_space ? mean(exp(s.SeaLiceLevel) for s in states) : mean(s.SeaLiceLevel for s in states)

            # Get mean reward
            episode_reward = mean(rewards)

            # Add to episode lists
            push!(episode_costs, episode_cost)
            push!(episode_abundances, episode_abundance)
            push!(episode_rewards, episode_reward)
        end

        # Calculate the average reward, cost, and sea lice level
        avg_reward, avg_treatment_cost, avg_abundance = mean(episode_rewards), mean(episode_costs), mean(episode_abundances)
        push!(avg_results, (λ, avg_treatment_cost, avg_abundance, avg_reward))

    end

    # Save results
    mkpath(config.results_dir)
    @save joinpath(config.results_dir, "$(algorithm.solver_name)_avg_results.jld2") avg_results
    
    return avg_results
end

# ----------------------------
# Calculate average sea lice level across all ep
# ----------------------------
# function calculate_average_sealice(config, history)



function get_histories_for_algorithm(config, algorithm)

    histories_dir = joinpath(config.simulations_dir, "$(algorithm.solver_name)")
    histories_filename = "$(algorithm.solver_name)_histories"
    @load joinpath(histories_dir, "$(histories_filename).jld2") histories
    return histories
end

# ----------------------------
# Calculate Averages
# ----------------------------
# function calculate_averages(config, action_hists, state_hists, reward_hists)

#     total_steps = config.num_episodes * config.steps_per_episode
#     total_cost, total_sealice, total_reward = 0.0, 0.0, 0.0

#     for i in 1:config.num_episodes
#         total_cost += sum(a == Treatment for a in action_hists[i]) * config.costOfTreatment
#         # Handle both regular and log space states
#         total_sealice += if typeof(state_hists[i][1]) <: SeaLiceLogState
#             sum(exp(s.SeaLiceLevel) for s in state_hists[i])
#         else
#             sum(s.SeaLiceLevel for s in state_hists[i])
#         end
#         total_reward += sum(reward_hists[i])
#     end

#     return total_reward / total_steps, total_cost / total_steps, total_sealice / total_steps
# end
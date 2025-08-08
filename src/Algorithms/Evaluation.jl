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
            episode_abundance = mean(s.SeaLiceLevel for s in states)

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
# Print all histories to a text file
# Data stores the histories in a dataframe with the following columns:
# reward, n_steps, history, policy, lambda, seed
# Creates a simulation_steps folder with a text file for each episode.
# The text file contains the history for each step in the episode.
# The text file is named lambda_<lambda>_seed_<seed>_simulation_history.txt
# The history is a tuple with the following elements:
# - action
# - state
# - reward
# - observation
# - belief
# ----------------------------
function print_histories(data, config)

    # Get all unique policies
    policies = unique(data.policy)

    for policy in policies

        # Get histories for this policy
        data_policy = filter(row -> row.policy == policy, data)

        # Create policy folder
        mkpath(joinpath(config.simulations_dir, policy))

        # Get all unique lambdas
        lambdas = unique(data_policy.lambda)

        for lambda in lambdas

            # Get histories for this lambda
            data_lambda = filter(row -> row.lambda == lambda, data_policy)

            # Get all unique seeds
            seeds = unique(data_lambda.seed)

            for seed in seeds

                # Get histories for this seed
                data_seed = filter(row -> row.seed == seed, data_lambda)
                h = data_seed.history[1]
                episode = 1

                filename = "lambda_$(lambda)_seed_$(seed)_simulation_history.txt"
                filepath = joinpath(config.simulations_dir, policy, filename)

                # Create file if it doesn't exist
                if !isfile(filepath)
                    open(filepath, "w") do file
                        println(file, "Simulation History")
                        println(file, "--------------------------------")
                    end
                end

                for (s, a, r, o, b, bp, sp) in eachstep(h, "(s, a, r, o, b, bp, sp)")

                    # State
                    s_adult = round(s.Adult, digits=2)
                    s_motile = round(s.Motile, digits=2)
                    s_sessile = round(s.Sessile, digits=2)
                    s_temp = round(s.Temperature, digits=2)
                    s_pred = round(s.SeaLiceLevel, digits=2)

                    # Observation
                    o_adult = round(o.Adult, digits=2)
                    o_motile = round(o.Motile, digits=2)
                    o_sessile = round(o.Sessile, digits=2)
                    o_temp = round(o.Temperature, digits=2)
                    o_pred = round(o.SeaLiceLevel, digits=2)

                    # New state
                    sp_adult = round(sp.Adult, digits=2)
                    sp_motile = round(sp.Motile, digits=2)
                    sp_sessile = round(sp.Sessile, digits=2)
                    sp_temp = round(sp.Temperature, digits=2)
                    sp_pred = round(sp.SeaLiceLevel, digits=2)

                    # Belief
                    b_adult = round(b.μ[1], digits=2)
                    b_motile = round(b.μ[2], digits=2)
                    b_sessile = round(b.μ[3], digits=2)
                    b_temp = round(b.μ[4], digits=2)

                    # New belief
                    bp_adult = round(bp.μ[1], digits=2)
                    bp_motile = round(bp.μ[2], digits=2)
                    bp_sessile = round(bp.μ[3], digits=2)
                    bp_temp = round(bp.μ[4], digits=2)

                    # Create table with state, observation, belief, and new state information
                    # Save to file with filepath
                    open(filepath, "a") do file
                        println(file, "--------------------------------")
                        println(file, "\nEpisode $episode:")
                        println(file, "   Took action: $a")
                        println(file, "   Received reward: $(round(r, digits=2))")
                        println(file, "┌─────────┬──────────┬─────────────┬──────────┬─────────────┬─────────────┐")
                        println(file, "│ Variable│   State  │ Observation │  Belief  │  New Belief │  New State  │")
                        println(file, "├─────────┼──────────┼─────────────┼──────────┼─────────────┼─────────────┤")
                        println(file, "│ Adult   │ $(s_adult)    │ $(o_adult)       │ $(b_adult)    │ $(bp_adult)       │ $(sp_adult)       │")
                        println(file, "│ Motile  │ $(s_motile)    │ $(o_motile)       │ $(b_motile)    │ $(bp_motile)       │ $(sp_motile)       │")
                        println(file, "│ Sessile │ $(s_sessile)    │ $(o_sessile)       │ $(b_sessile)    │ $(bp_sessile)       │ $(sp_sessile)       │")
                        println(file, "│ Pred    │ $(s_pred)    │ $(o_pred)       │ $(b_temp)    │ $(bp_temp)       │ $(sp_pred)       │")
                        println(file, "└─────────┴──────────┴─────────────┴──────────┴─────────────┴─────────────┘")
                    end

                    episode += 1
                end
            end
        end
    end
end
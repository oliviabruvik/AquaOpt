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
using Statistics
using Printf
using CSV

# ----------------------------
# Mean and confidence interval function
# ----------------------------
function mean_and_ci(x)
    m = mean(x)
    ci = 1.96 * std(x) / sqrt(length(x))  # 95% confidence interval
    return (mean = m, ci = ci)
end

function _expected_biomass_shortfall(sim_params::SeaLiceSimPOMDP, s, sp)
    ideal_survival_rate = 1 - sim_params.nat_mort_rate
    expected_fish = max(s.NumberOfFish * ideal_survival_rate, 0.0)
    k0_base = sim_params.k_growth * (1.0 + sim_params.temp_sensitivity * (s.Temperature - 10.0))
    ideal_k0 = max(k0_base, 0.0)
    expected_weight = s.AvgFishWeight + ideal_k0 * (sim_params.w_max - s.AvgFishWeight)
    expected_weight = clamp(expected_weight, sim_params.weight_bounds...)
    expected_biomass = biomass_tons(expected_weight, expected_fish)
    next_biomass = biomass_tons(sp)
    return max(expected_biomass - next_biomass, 0.0)
end

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

        for episode in 1:config.simulation_config.num_episodes

            # Get episode history
            episode_history = histories_lambda[episode]

            # Get action, state, and reward histories
            actions = collect(action_hist(episode_history))
            states = collect(state_hist(episode_history))
            rewards = collect(reward_hist(episode_history))

            # Get total treatment cost
            episode_cost = sum(get_treatment_cost(a) for a in actions)
            
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
    CSV.write(joinpath(config.results_dir, "$(algorithm.solver_name)_avg_results.csv"), avg_results)
    
    return avg_results
end

# ----------------------------
# Extract histories for a specific algorithm from parallel simulation results
# Takes in config, algorithm and parallel data
# Returns a histories object where the key is the lambda value and the value is a vector of episode histories
# ----------------------------
function extract_simulation_histories(config, algorithm, parallel_data)

    histories = Dict{Float64, Vector{Any}}()

    for λ in config.lambda_values

        # Get histories for this lambda
        data_lambda = filter(row -> row.lambda == λ && row.policy == algorithm.solver_name, parallel_data)

        episode_histories = Vector{Any}()

        for episode in 1:config.simulation_config.num_episodes

            # Get histories for this episode
            data_episode = filter(row -> row.episode_number == episode, data_lambda)

            if nrow(data_episode) == 1
                push!(episode_histories, data_episode.history[1])
            else
                error("Expected 1 history for lambda=$(λ), policy=$(algorithm.solver_name), seed=$(episode), but found $(nrow(data_episode))")
            end
        end

        histories[λ] = episode_histories
    end

    # Create directory for simulation histories
    histories_dir = joinpath(config.simulations_dir, "$(algorithm.solver_name)")
    mkpath(histories_dir)
    histories_filename = "$(algorithm.solver_name)_histories"
    histories_filepath = joinpath(histories_dir, "$(algorithm.solver_name)_histories.jld2")
    @save histories_filepath histories

    return histories
end

# ----------------------------
# Display the mean and confidence interval for each lambda and each policy
# ----------------------------
function display_rewards_across_policies(parallel_data, config)

    # Display the mean and confidence interval for each lambda and each policy
    for λ in config.lambda_values

        println("Lambda: $(λ)")

        # Filter data for current lambda
        data_filtered = filter(row -> row.lambda == λ, parallel_data)
        data_grouped_by_policy = groupby(data_filtered, :policy)
        result = combine(data_grouped_by_policy, :reward => mean_and_ci => AsTable)

        # Order by mean reward
        result = sort(result, :mean, rev=true)
        println(result)
    
    end
end

# ----------------------------
# Display the best lambda for each policy
# ----------------------------
function display_best_lambda_for_each_policy(parallel_data, algorithms)

    # Get all unique policies
    policies = unique(parallel_data.policy)

    for algo in algorithms

        println("Policy: $(algo.solver_name)")

        # Filter data for current policy
        data_filtered = filter(row -> row.policy == algo.solver_name, parallel_data)
        data_grouped_by_lambda = groupby(data_filtered, :lambda)
        result = combine(data_grouped_by_lambda, :reward => mean_and_ci => AsTable)
        println(result)
    end
end

# ----------------------------
# Extract the number of treatments, regulatory penalties, lost biomass, and fish disease for each policy from the histories and add as columns to the parallel_data dataframe
# Returns a new DataFrame without modifying the input
# ----------------------------
function extract_reward_metrics(data, config)

    # Create a copy of the data to avoid mutating the input
    processed_data = copy(data)

    # Add new columns to the DataFrame copy
    processed_data.mean_rewards_across_sims = zeros(Float64, nrow(processed_data))
    processed_data.treatment_cost = zeros(Float64, nrow(processed_data))
    processed_data.treatments = Vector{Dict{Action, Int}}(undef, nrow(processed_data))
    processed_data.num_regulatory_penalties = zeros(Float64, nrow(processed_data))
    processed_data.fish_disease = zeros(Float64, nrow(processed_data))
    processed_data.lost_biomass_1000kg = zeros(Float64, nrow(processed_data))
    processed_data.mean_adult_sea_lice_level = zeros(Float64, nrow(processed_data))

    # For each episode, extract the number of treatments, regulatory penalties, lost biomass, and fish disease
    sim_params = SeaLiceSimPOMDP(location=config.solver_config.location)

    for (i, row) in enumerate(eachrow(processed_data))

        # Get the history
        h = row.history
        states = collect(h[:s])
        actions = collect(h[:a])
        rewards = collect(h[:r])

        # Get distribution of treatments
        treatments = Dict{Action, Int}()
        for a in actions
            treatments[a] = get(treatments, a, 0) + 1
        end

        # Get total lost biomass
        lost_biomass_1000kg = 0.0
        if length(states) > 1
            n_pairs = min(length(states) - 1, length(actions))
            for t in 1:n_pairs
                lost_biomass_1000kg += _expected_biomass_shortfall(sim_params, states[t], states[t+1])
            end
        end

        # Get total fish disease
        fish_disease = sum(get_fish_disease(a) + 100.0 * s.SeaLiceLevel for (s, a) in zip(states, actions))

        # Add to dataframe copy
        processed_data.treatment_cost[i] = sum(get_treatment_cost(a) for a in actions)
        processed_data.treatments[i] = treatments
        processed_data.num_regulatory_penalties[i] = sum(s.Adult > config.solver_config.regulation_limit ? 1.0 : 0.0 for s in states)
        processed_data.lost_biomass_1000kg[i] = lost_biomass_1000kg
        processed_data.fish_disease[i] = fish_disease
        processed_data.mean_adult_sea_lice_level[i] = mean(s.Adult for s in states)
        processed_data.mean_rewards_across_sims[i] = mean(rewards)
    end

    return processed_data

end

# ----------------------------
# Display the mean and confidence interval for each lambda and each policy
# ----------------------------
function display_reward_metrics(parallel_data, config, display_ci=false, print_sd=false)

    # Display the mean and confidence interval for each lambda and each policy
    for λ in config.lambda_values

        println("Lambda: $(λ)")

        # Filter data for current lambda
        data_filtered = filter(row -> row.lambda == λ, parallel_data)
        data_grouped_by_policy = groupby(data_filtered, :policy)

        # Check if treatments column exists in the data
        if :treatments in names(data_filtered)
            # Process each column separately to avoid duplicate column names
            if display_ci
                result = combine(
                    data_grouped_by_policy,
                    :reward => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_reward,
                    :reward => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_reward,
                    :mean_rewards_across_sims => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_mean_rewards_across_sims,
                    :mean_rewards_across_sims => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_mean_rewards_across_sims,
                    :treatment_cost => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_treatment_cost,
                    :treatment_cost => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_treatment_cost,
                    :mean_adult_sea_lice_level => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_mean_adult_sea_lice_level,
                    :mean_adult_sea_lice_level => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_mean_adult_sea_lice_level,
                    :num_regulatory_penalties => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_num_regulatory_penalties,
                    :num_regulatory_penalties => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_num_regulatory_penalties,
                    :lost_biomass_1000kg => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_lost_biomass_1000kg,
                    :lost_biomass_1000kg => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_lost_biomass_1000kg,
                    :fish_disease => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_fish_disease,
                    :fish_disease => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_fish_disease,
                    :treatments => (x -> round(mean_and_ci([get(t[1], NoTreatment, 0) for t in x]).mean, digits=2)) => :mean_num_NoTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], NoTreatment, 0) for t in x]).ci, digits=2)) => :ci_num_NoTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], MechanicalTreatment, 0) for t in x]).mean, digits=2)) => :mean_num_MechanicalTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], MechanicalTreatment, 0) for t in x]).ci, digits=2)) => :ci_num_MechanicalTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], ChemicalTreatment, 0) for t in x]).mean, digits=2)) => :mean_num_ChemicalTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], ChemicalTreatment, 0) for t in x]).ci, digits=2)) => :ci_num_ChemicalTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], ThermalTreatment, 0) for t in x]).mean, digits=2)) => :mean_num_ThermalTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], ThermalTreatment, 0) for t in x]).ci, digits=2)) => :ci_num_ThermalTreatment
                )
            else
                result = combine(
                    data_grouped_by_policy,
                    :reward => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_reward,
                    :mean_rewards_across_sims => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_mean_rewards_across_sims,
                    :treatment_cost => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_treatment_cost,
                    :mean_adult_sea_lice_level => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_mean_adult_sea_lice_level,
                    :num_regulatory_penalties => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_num_regulatory_penalties,
                    :lost_biomass_1000kg => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_lost_biomass_1000kg,
                    :fish_disease => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_fish_disease,
                    :treatments => (x -> round(mean_and_ci([get(t[1], NoTreatment, 0) for t in x]).mean, digits=2)) => :mean_num_NoTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], MechanicalTreatment, 0) for t in x]).mean, digits=2)) => :mean_num_MechanicalTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], ChemicalTreatment, 0) for t in x]).mean, digits=2)) => :mean_num_ChemicalTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], ThermalTreatment, 0) for t in x]).mean, digits=2)) => :mean_num_ThermalTreatment
                )
            end
        else
            # Process without treatments column
            if display_ci
                result = combine(
                    data_grouped_by_policy,
                    :reward => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_reward,
                    :reward => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_reward,
                    :mean_rewards_across_sims => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_mean_rewards_across_sims,
                    :mean_rewards_across_sims => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_mean_rewards_across_sims,
                    :treatment_cost => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_treatment_cost,
                    :treatment_cost => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_treatment_cost,
                    :num_regulatory_penalties => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_num_regulatory_penalties,
                    :num_regulatory_penalties => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_num_regulatory_penalties,
                    :mean_adult_sea_lice_level => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_mean_adult_sea_lice_level,
                    :mean_adult_sea_lice_level => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_mean_adult_sea_lice_level,
                    :lost_biomass_1000kg => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_lost_biomass_1000kg,
                    :lost_biomass_1000kg => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_lost_biomass_1000kg,
                    :fish_disease => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_fish_disease,
                    :fish_disease => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_fish_disease
                )
            else
                result = combine(
                    data_grouped_by_policy,
                    :reward => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_reward,
                    :mean_rewards_across_sims => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_mean_rewards_across_sims,
                    :treatment_cost => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_treatment_cost,
                    :num_regulatory_penalties => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_num_regulatory_penalties,
                    :mean_adult_sea_lice_level => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_mean_adult_sea_lice_level,
                    :lost_biomass_1000kg => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_lost_biomass_1000kg,
                    :fish_disease => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_fish_disease
                )
            end
        end

        # Order by mean reward
        result = sort(result, :mean_reward, rev=true)
        println(result)

        if print_sd
            # Create pivot table format
            println("\n" * "="^80)
            println("LAMBDA: $(λ)")
            println("="^80)
            
            # Print header
            println(@sprintf("%-20s %12s %12s %12s %12s %12s", 
                            "Policy", "Mean Reward", "Treatment Cost", "Reg. Penalties", "Sea Lice Level", "Fish Disease"))
            println("-"^80)
            
            # Print each policy's results
            display_ci = true
            result = combine(
                data_grouped_by_policy,
                :reward => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_reward,
                :reward => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_reward,
                :mean_rewards_across_sims => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_mean_rewards_across_sims,
                :mean_rewards_across_sims => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_mean_rewards_across_sims,
                :treatment_cost => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_treatment_cost,
                :treatment_cost => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_treatment_cost,
                :num_regulatory_penalties => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_num_regulatory_penalties,
                :num_regulatory_penalties => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_num_regulatory_penalties,
                :mean_adult_sea_lice_level => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_mean_adult_sea_lice_level,
                :mean_adult_sea_lice_level => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_mean_adult_sea_lice_level,
                :lost_biomass_1000kg => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_lost_biomass_1000kg,
                :lost_biomass_1000kg => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_lost_biomass_1000kg,
                :fish_disease => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_fish_disease,
                :fish_disease => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_fish_disease
            )
            for row in eachrow(result)
                policy_name = row.policy
                mean_reward = display_ci ? @sprintf("%.3f±%.3f", row.mean_reward, row.ci_reward) : @sprintf("%.3f", row.mean_reward)
                treatment_cost = display_ci ? @sprintf("%.3f±%.3f", row.mean_treatment_cost, row.ci_treatment_cost) : @sprintf("%.3f", row.mean_treatment_cost)
                reg_penalties = display_ci ? @sprintf("%.3f±%.3f", row.mean_num_regulatory_penalties, row.ci_num_regulatory_penalties) : @sprintf("%.3f", row.mean_num_regulatory_penalties)
                sea_lice = display_ci ? @sprintf("%.3f±%.3f", row.mean_mean_adult_sea_lice_level, row.ci_mean_adult_sea_lice_level) : @sprintf("%.3f", row.mean_mean_adult_sea_lice_level)
                fish_disease = display_ci ? @sprintf("%.1f±%.1f", row.mean_fish_disease, row.ci_fish_disease) : @sprintf("%.1f", row.mean_fish_disease)

                println(@sprintf("%-20s %12s %12s %12s %12s %12s", 
                            policy_name, mean_reward, treatment_cost, reg_penalties, sea_lice, fish_disease))
            end
            
            # If treatments column exists, print treatment distribution
            if :treatments in names(data_filtered)
                println("\nTreatment Distribution:")
                println("-"^50)
                println(@sprintf("%-20s %12s %12s %12s %12s", "Policy", "No Treatment", "Mechanical", "Chemical", "Thermal"))
                println("-"^50)
                
                for row in eachrow(result)
                    policy_name = row.policy
                    no_treatment = display_ci ? @sprintf("%.1f±%.1f", row.mean_num_NoTreatment, row.ci_num_NoTreatment) : @sprintf("%.1f", row.mean_num_NoTreatment)
                    mechanical = display_ci ? @sprintf("%.1f±%.1f", row.mean_num_MechanicalTreatment, row.ci_num_MechanicalTreatment) : @sprintf("%.1f", row.mean_num_MechanicalTreatment)
                    chemical = display_ci ? @sprintf("%.1f±%.1f", row.mean_num_ChemicalTreatment, row.ci_num_ChemicalTreatment) : @sprintf("%.1f", row.mean_num_ChemicalTreatment)
                    thermal = display_ci ? @sprintf("%.1f±%.1f", row.mean_num_ThermalTreatment, row.ci_num_ThermalTreatment) : @sprintf("%.1f", row.mean_num_ThermalTreatment)
                    
                    println(@sprintf("%-20s %12s %12s %12s %12s", 
                                policy_name, no_treatment, mechanical, chemical, thermal))
                end
            end
        end
        
        println("\n")

        # Save results to csv
        mkpath(config.results_dir)
        CSV.write(joinpath(config.results_dir, "reward_metrics_lambda_$(λ).csv"), result)
    
    end
end

function print_reward_metrics_for_vi_policy(data, config)

     # Add new columns to the DataFrame
     data.mean_rewards_across_sims = zeros(Float64, nrow(data))
     data.treatment_cost = zeros(Float64, nrow(data))
     data.treatments = Vector{Dict{Action, Int}}(undef, nrow(data))
     data.num_regulatory_penalties = zeros(Float64, nrow(data))
 
     # For each episode, extract the number of treatments, regulatory penalties, lost biomass, and fish disease
     for (i, row) in enumerate(eachrow(data))
 
         # Get the history
         h = row.history
         states = collect(h[:s])
         actions = collect(h[:a])
         rewards = collect(h[:r])
 
         # Get distribution of treatments
         treatments = Dict{Action, Int}()
         for a in actions
             treatments[a] = get(treatments, a, 0) + 1
         end
         # Add to dataframe
         data.treatment_cost[i] = sum(get_treatment_cost(a) for a in actions)
         data.treatments[i] = treatments
         data.num_regulatory_penalties[i] = sum(s.SeaLiceLevel > config.solver_config.regulation_limit ? 1.0 : 0.0 for s in states)
         data.mean_rewards_across_sims[i] = mean(rewards)
     end

     parallel_data = data
     display_ci = true
 
    # Display the mean and confidence interval for each lambda and each policy
    for λ in config.lambda_values

        println("Lambda: $(λ)")

        # Filter data for current lambda
        data_filtered = filter(row -> row.lambda == λ, parallel_data)
        data_grouped_by_policy = groupby(data_filtered, :policy)

        # Check if treatments column exists in the data
        if :treatments in names(data_filtered)
            # Process each column separately to avoid duplicate column names
            if display_ci
                result = combine(
                    data_grouped_by_policy,
                    :reward => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_reward,
                    :reward => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_reward,
                    :mean_rewards_across_sims => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_mean_rewards_across_sims,
                    :mean_rewards_across_sims => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_mean_rewards_across_sims,
                    :treatment_cost => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_treatment_cost,
                    :treatment_cost => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_treatment_cost,
                    :num_regulatory_penalties => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_num_regulatory_penalties,
                    :num_regulatory_penalties => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_num_regulatory_penalties,
                    :treatments => (x -> round(mean_and_ci([get(t[1], NoTreatment, 0) for t in x]).mean, digits=2)) => :mean_num_NoTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], NoTreatment, 0) for t in x]).ci, digits=2)) => :ci_num_NoTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], MechanicalTreatment, 0) for t in x]).mean, digits=2)) => :mean_num_MechanicalTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], MechanicalTreatment, 0) for t in x]).ci, digits=2)) => :ci_num_MechanicalTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], ChemicalTreatment, 0) for t in x]).mean, digits=2)) => :mean_num_ChemicalTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], ChemicalTreatment, 0) for t in x]).ci, digits=2)) => :ci_num_ChemicalTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], ThermalTreatment, 0) for t in x]).mean, digits=2)) => :mean_num_ThermalTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], ThermalTreatment, 0) for t in x]).ci, digits=2)) => :ci_num_ThermalTreatment
                )
            else
                result = combine(
                    data_grouped_by_policy,
                    :reward => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_reward,
                    :mean_rewards_across_sims => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_mean_rewards_across_sims,
                    :treatment_cost => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_treatment_cost,
                    :num_regulatory_penalties => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_num_regulatory_penalties,
                    :treatments => (x -> round(mean_and_ci([get(t[1], NoTreatment, 0) for t in x]).mean, digits=2)) => :mean_num_NoTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], MechanicalTreatment, 0) for t in x]).mean, digits=2)) => :mean_num_MechanicalTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], ChemicalTreatment, 0) for t in x]).mean, digits=2)) => :mean_num_ChemicalTreatment,
                    :treatments => (x -> round(mean_and_ci([get(t[1], ThermalTreatment, 0) for t in x]).mean, digits=2)) => :mean_num_ThermalTreatment
                )
            end
        else
            # Process without treatments column
            if display_ci
                result = combine(
                    data_grouped_by_policy,
                    :reward => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_reward,
                    :reward => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_reward,
                    :mean_rewards_across_sims => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_mean_rewards_across_sims,
                    :mean_rewards_across_sims => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_mean_rewards_across_sims,
                    :treatment_cost => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_treatment_cost,
                    :treatment_cost => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_treatment_cost,
                    :num_regulatory_penalties => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_num_regulatory_penalties,
                    :num_regulatory_penalties => (x -> round(mean_and_ci(x).ci, digits=2)) => :ci_num_regulatory_penalties,
                )
            else
                result = combine(
                    data_grouped_by_policy,
                    :reward => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_reward,
                    :mean_rewards_across_sims => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_mean_rewards_across_sims,
                    :treatment_cost => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_treatment_cost,
                    :num_regulatory_penalties => (x -> round(mean_and_ci(x).mean, digits=2)) => :mean_num_regulatory_penalties,
                )
            end
        end

        # Order by mean reward
        result = sort(result, :mean_reward, rev=true)
        println(result)

        # Create pivot table format
        println("\n" * "="^80)

        # Save results to csv
        mkpath(config.results_dir)
        CSV.write(joinpath(config.results_dir, "reward_metrics_lambda_$(λ).csv"), result)

    end
end

# ----------------------------
# Print the count of each treatment for each policy
# The treatment count is the number of times each treatment is taken for each policy
# The treatment count is stored in the data.treatments column
# ----------------------------
function print_treatment_frequency(data, config)

    for lambda in config.lambda_values

        println("Lambda: $(lambda)")

        # Get histories for this lambda
        data_lambda = filter(row -> row.lambda == lambda, data)

        treatment_counts = Dict{String, Dict{String, Vector{Int}}}()

        # Get all unique policies
        policies = unique(data_lambda.policy)

        for policy in policies

            # Initialize the inner dictionary for this policy
            treatment_counts[policy] = Dict{String, Vector{Int}}(
                "NoTreatment" => [],
                "MechanicalTreatment" => [],
                "ChemicalTreatment" => [],
                "ThermalTreatment" => [],
            )

            # Get histories for this policy
            data_policy = filter(row -> row.policy == policy, data_lambda)

            # Get all unique seeds
            seeds = unique(data_policy.seed)

            for seed in seeds

                # Get histories for this seed
                data_seed = filter(row -> row.seed == seed, data_policy)

                # Get the treatment count from the data.treatments column
                treatment_count = data_seed.treatments[1]  # Get the first (and only) treatment count dictionary

                # Count the number of times each treatment is taken
                push!(treatment_counts[policy]["NoTreatment"], get(treatment_count, NoTreatment, 0))
                push!(treatment_counts[policy]["MechanicalTreatment"], get(treatment_count, MechanicalTreatment, 0))
                push!(treatment_counts[policy]["ChemicalTreatment"], get(treatment_count, ChemicalTreatment, 0))
                push!(treatment_counts[policy]["ThermalTreatment"], get(treatment_count, ThermalTreatment, 0))

            end
        end

        # Print the treatment counts for each policy as a table
        # Create a DataFrame from the treatment counts
        # Add mean and confidence interval to the dataframe
        treatment_data = []
        for (policy, counts) in treatment_counts
            push!(treatment_data, (
                policy = policy,
                NoTreatment = mean_and_ci(counts["NoTreatment"]).mean,
                MechanicalTreatment = mean_and_ci(counts["MechanicalTreatment"]).mean,
                ChemicalTreatment = mean_and_ci(counts["ChemicalTreatment"]).mean,
                ThermalTreatment = mean_and_ci(counts["ThermalTreatment"]).mean,
            ))
        end
        
        # Convert to DataFrame and display
        if !isempty(treatment_data)
            treatment_df = DataFrame(treatment_data)
            println(treatment_df)
        else
            println("No treatment data available for lambda $(lambda)")
        end

        # Save treatment data to csv
        mkpath(config.results_dir)
        CSV.write(joinpath(config.results_dir, "treatment_data_lambda_$(lambda).csv"), treatment_df)
    end
end


# ----------------------------
# Print all histories to a text file
# Data stores the histories in a dataframe with the following columns:
# reward, n_steps, history, policy, lambda, seed
# Creates a simulation_steps folder with a text file for each episode.
# The text file contains the history for each step in the episode.
# The text file is named lambda_<lambda>_seed_<seed>_simulation_history.txt
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

using GaussianFilters
using POMDPs
using POMDPModels
using POMDPTools
using DataFrames
using JLD2
using Plots
using Distributions
using Parameters
using Statistics
using Base.Sys
using StatsBase: std
using Random

include("../../src/Utils/Utils.jl")
include("../../src/Models/SeaLiceLogPOMDP.jl")
include("../../src/Models/SeaLicePOMDP.jl")
include("../../src/Models/KalmanFilter.jl")
include("../../src/Algorithms/Policies.jl")

# ----------------------------
# Initialize belief function
# ----------------------------
function initialize_belief(sim_pomdp, config)
    """
    Create initial belief state based on POMDP type and configuration.
    Returns a tuple of Normal distributions for each state component.
    """
    if typeof(sim_pomdp) <: SeaLiceLogSimMDP
        return (
            # TODO: fix this for log space
            Normal(sim_pomdp.log_lice_initial_mean, sim_pomdp.sampling_sd), # adult
            Normal(sim_pomdp.motile_mean, sim_pomdp.motile_sd), # motile
            Normal(sim_pomdp.sessile_mean, sim_pomdp.sessile_sd), # sessile
            Normal(temperature_model(sim_pomdp.production_start_week), sim_pomdp.sampling_sd) # temperature
        )
    else
        return (
            sim_pomdp.adult_mean + sim_pomdp.adult_dist,
            sim_pomdp.motile_mean + sim_pomdp.motile_dist,
            sim_pomdp.sessile_mean + sim_pomdp.sessile_dist,
            temperature_model(sim_pomdp.production_start_week) + sim_pomdp.temp_dist,
        )
    end
end

# ----------------------------
# Mean and confidence interval function
# ----------------------------
function mean_and_ci(x)
    m = mean(x)
    ci = 1.96 * std(x) / sqrt(length(x))  # 95% confidence interval
    return (mean = m, ci = ci)
end

# ----------------------------
# Create Sim POMDP based on whether we're in log space
# ----------------------------
function create_sim_pomdp(config, λ)    # TODO: fix this for log space

   if config.log_space
        return SeaLiceLogSimMDP(
            lambda=λ,
            costOfTreatment=config.costOfTreatment,
            rho=config.rho,
            discount_factor=config.discount_factor,
            skew=config.skew,
            sampling_sd=abs(log(config.raw_space_sampling_sd))
        )
    else
        return SeaLiceSimMDP(
            lambda=λ,
            costOfTreatment=config.costOfTreatment,
            rho=config.rho,
            discount_factor=config.discount_factor,
            skew=config.skew,
            # SimPOMDP parameters
            adult_mean=config.adult_mean,
            motile_mean=config.motile_mean,
            sessile_mean=config.sessile_mean,
            adult_sd=config.adult_sd,
            motile_sd=config.motile_sd,
            sessile_sd=config.sessile_sd,
            temp_sd=config.temp_sd,
        )
    end
end

# ----------------------------
# Simulate policy
# ----------------------------
function simulate_policy(algorithm, config)

    # Create directory for simulation histories
    histories_dir = joinpath(config.simulations_dir, "$(algorithm.solver_name)")
    mkpath(histories_dir)

    # Create directory for policies
    policies_dir = joinpath(config.policies_dir, "$(algorithm.solver_name)")
    mkpath(policies_dir)

    histories = Dict{Float64, Any}()

    # Simulate policy
    for λ in config.lambda_values

        # Load policy, pomdp, and mdp
        policy_pomdp_mdp_filename = "policy_pomdp_mdp_$(λ)_lambda"
        @load joinpath(policies_dir, "$(policy_pomdp_mdp_filename).jld2") policy pomdp mdp

        # Create adaptor policy
        adaptor_policy = AdaptorPolicy(policy)

        # Simulate policy
        histories[λ] = run_simulation(adaptor_policy, mdp, pomdp, config, algorithm)

        # Run all episodes in parallel
        data = run_all_episodes(adaptor_policy, mdp, pomdp, config, algorithm)

    end

    # Save results
    histories_filename = "$(algorithm.solver_name)_histories"
    @save joinpath(histories_dir, "$(histories_filename).jld2") histories
    
    return histories
end


# ----------------------------
# Simulation & Evaluation
# ----------------------------
function run_simulation(policy, mdp, pomdp, config, algorithm)

    # Store all histories
    histories = []

    # Create simulator POMDP based on whether we're in log space
    sim_pomdp = create_sim_pomdp(config, pomdp.lambda)

    # Create simulator
    # sim = RolloutSimulator(max_steps=config.steps_per_episode)
    hr = HistoryRecorder(max_steps=config.steps_per_episode)
    kf = build_kf(sim_pomdp, ekf_filter=config.ekf_filter)
    updater = KalmanUpdater(kf)

    # Run simulation for each episode
    for episode in 1:config.num_episodes

        # Get initial belief from initial mean and sampling sd
        initial_belief = initialize_belief(sim_pomdp, config)

        hist = simulate(hr, sim_pomdp, policy, updater, initial_belief, rand(initialstate(sim_pomdp)))
        push!(histories, hist)
    end

    # Return averages
    return histories
end

# ----------------------------
# Simulate all policies in parallel
# ----------------------------
function run_all_episodes(policy, mdp, pomdp, config, algorithm)

    # Defining parameters for parallel simulation
    starting_seed = 1

    # Create simulator POMDP based on whether we're in log space
    sim_pomdp = create_sim_pomdp(config, pomdp.lambda)

    # Create simulator
    # sim = RolloutSimulator(max_steps=config.steps_per_episode)
    hr = HistoryRecorder(max_steps=config.steps_per_episode)
    kf = build_kf(sim_pomdp, ekf_filter=config.ekf_filter)
    updater = KalmanUpdater(kf)

    # Get initial belief from initial mean and sampling sd
    initial_belief = initialize_belief(sim_pomdp, config)

    # Create the list of Sim objects
    sim_list = []

    # Add Sim objects for each episode
    for sim_number in 1:config.num_episodes
        seed = starting_seed + sim_number

        # Create Sim object following POMDPs.jl documentation format with custom updater
        push!(sim_list, Sim(
            sim_pomdp,           # POMDP
            policy,              # Policy
            updater,             # Custom updater
            initial_belief,      # Initial belief
            rand(initialstate(sim_pomdp));  # Initial state
            rng=MersenneTwister(seed),
            max_steps=config.steps_per_episode,
            metadata=Dict(:policy => algorithm.solver_name, :lambda => pomdp.lambda, :seed => sim_number)
        ))
    end

    # Run the simulations in parallel
    data = run_parallel(sim_list, proc_warn=false) do sim, hist
        return (
            reward = discounted_reward(hist),
            n_steps = n_steps(hist),
            history = hist,  # Store the full history
            policy = sim.metadata[:policy],
            lambda = sim.metadata[:lambda],
            seed = sim.metadata[:seed]
        )
    end

    # Calculate the mean and confidence interval for each policy
    grouped_df = groupby(data, :policy)
    result = combine(grouped_df, :reward => mean_and_ci => AsTable)

    return data
end

# ----------------------------
# Simulate policy
# ----------------------------
function simulate_all_policies(algorithms, config)

    # Defining parameters for parallel simulation
    starting_seed = 1

    # Create the list of Sim objects
    sim_list = []

    # Simulate policy
    for λ in config.lambda_values

        # Create simulator POMDP based on whether we're in log space
        sim_pomdp = create_sim_pomdp(config, λ)

        # Create simulator
        # sim = RolloutSimulator(max_steps=config.steps_per_episode)
        hr = HistoryRecorder(max_steps=config.steps_per_episode)
        kf = build_kf(sim_pomdp, ekf_filter=config.ekf_filter)
        updater = KalmanUpdater(kf)

        # Get initial belief from initial mean and sampling sd
        initial_belief = initialize_belief(sim_pomdp, config)

        # Load policy, pomdp, and mdp
        for algo in algorithms
            policy_pomdp_mdp_filename = "policy_pomdp_mdp_$(λ)_lambda"
            @load joinpath(config.policies_dir, "$(algo.solver_name)", "$(policy_pomdp_mdp_filename).jld2") policy pomdp mdp

            # Create adaptor policy
            adaptor_policy = AdaptorPolicy(policy)

            # Add Sim objects for each episode
            for sim_number in 1:config.num_episodes
                seed = starting_seed + sim_number

                # Create Sim object following POMDPs.jl documentation format with custom updater
                push!(sim_list, Sim(
                    sim_pomdp,           # POMDP
                    adaptor_policy,              # Policy
                    updater,             # Custom updater
                    initial_belief,      # Initial belief
                    rand(initialstate(sim_pomdp));  # Initial state
                    rng=MersenneTwister(seed),
                    max_steps=config.steps_per_episode,
                    metadata=Dict(:policy => algo.solver_name, :lambda => pomdp.lambda, :seed => sim_number)
                ))
            end
        end
    end

    # Run the simulations in parallel
    data = run_parallel(sim_list, proc_warn=false) do sim, hist
        return (
            reward = discounted_reward(hist),
            n_steps = n_steps(hist),
            history = hist,  # Store the full history
            policy = sim.metadata[:policy],
            lambda = sim.metadata[:lambda],
            seed = sim.metadata[:seed]
        )
    end

    # Calculate the mean and confidence interval for each lambda and each policy
    for λ in config.lambda_values

        println("Lambda: $(λ)")

        # Filter data for current lambda
        data_filtered = filter(row -> row.lambda == λ, data)
        data_grouped_by_policy = groupby(data_filtered, :policy)
        result = combine(data_grouped_by_policy, :reward => mean_and_ci => AsTable)

        # Order by mean reward
        result = sort(result, :mean, rev=true)
        println(result)
    
    end

    # Show the best lambda for each policy
    for algo in algorithms

        println("Policy: $(algo.solver_name)")

        # Filter data for current policy
        data_filtered = filter(row -> row.policy == algo.solver_name, data)
        data_grouped_by_lambda = groupby(data_filtered, :lambda)
        result = combine(data_grouped_by_lambda, :reward => mean_and_ci => AsTable)
        println(result)
    
    end

    # Save data
    mkpath(config.simulations_dir)
    @save joinpath(config.simulations_dir, "all_policies_simulation_data.jld2") data
    println("Saved data to $(config.simulations_dir)/all_policies_simulation_data.jld2")

    return data

end
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



# ----------------------------
# Initialize belief function
# We need an initial belief for the simulation because our state
# and observation variables are different.
# Returns a tuple of Normal distributions for each state component.
# ----------------------------
function initialize_belief(sim_pomdp, config)
    if config.high_fidelity_sim
        return (
            sim_pomdp.adult_mean + sim_pomdp.adult_dist, # adult
            sim_pomdp.motile_mean + sim_pomdp.motile_dist, # motile
            sim_pomdp.sessile_mean + sim_pomdp.sessile_dist, # sessile
            get_temperature(sim_pomdp.production_start_week, sim_pomdp.location) + sim_pomdp.temp_dist, # temperature
        )
    else
        return initialstate(sim_pomdp)
    end
end

# ----------------------------
# Create Sim POMDP
# ----------------------------
function create_sim_pomdp(config, λ)
    # Simulate policies on a POMDP with a larger state space
    # for a realistic evaluation of performance. 
    if config.high_fidelity_sim
        return SeaLiceSimPOMDP(
            lambda=λ,
            reward_lambdas=config.sim_reward_lambdas,
            costOfTreatment=config.costOfTreatment,
            rho=config.rho,
            discount_factor=config.discount_factor,
            # SimPOMDP parameters
            adult_mean=config.adult_mean,
            motile_mean=config.motile_mean,
            sessile_mean=config.sessile_mean,
            adult_sd=config.adult_sd,
            motile_sd=config.motile_sd,
            sessile_sd=config.sessile_sd,
            temp_sd=config.temp_sd,
            location=config.location,
        )
    else
        # Use the same POMDP type that policies were trained on
        if config.log_space
            return SeaLiceLogPOMDP(
                lambda=λ,
                reward_lambdas=config.sim_reward_lambdas,
                costOfTreatment=config.costOfTreatment,
                growthRate=config.growthRate,
                rho=config.rho,
                discount_factor=config.discount_factor,
                discretization_step=config.discretization_step,
                adult_sd=abs(log(config.raw_space_sampling_sd)),
                regulation_limit=config.regulation_limit,
                full_observability_solver=config.full_observability_solver,
            )
        else
            return SeaLicePOMDP(
                lambda=λ,
                reward_lambdas=config.sim_reward_lambdas,
                costOfTreatment=config.costOfTreatment,
                growthRate=config.growthRate,
                rho=config.rho,
                discount_factor=config.discount_factor,
                discretization_step=config.discretization_step,
                adult_sd=config.raw_space_sampling_sd,
                regulation_limit=config.regulation_limit,
                full_observability_solver=config.full_observability_solver,
            )
        end
    end
end

# ----------------------------
# Simulate policy
# Calls run_all_episodes and run_simulation
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
        adaptor_policy = AdaptorPolicy(policy, pomdp, config.location)

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

    # Create simulator POMDP
    sim_pomdp = create_sim_pomdp(config, pomdp.lambda)

    # Create simulator
    # sim = RolloutSimulator(max_steps=config.steps_per_episode)
    hr = HistoryRecorder(max_steps=config.steps_per_episode)
    updater = build_kf(sim_pomdp, ekf_filter=config.ekf_filter)

    # Run simulation for each episode
    for episode in 1:config.num_episodes

        # Get initial belief and state
        initial_belief = initialize_belief(sim_pomdp, config)
        initial_state = rand(initialstate(sim_pomdp))

        hist = simulate(hr, sim_pomdp, policy, updater, initial_belief, initial_state)
        push!(histories, hist)
    end

    # Return averages
    return histories
end

# ----------------------------
# Simulate one policy in parallel
# ----------------------------
function run_all_episodes(policy, mdp, pomdp, config, algorithm)

    # Defining parameters for parallel simulation
    starting_seed = 1

    # Create simulator POMDP
    sim_pomdp = create_sim_pomdp(config, pomdp.lambda)

    # Create simulator
    # sim = RolloutSimulator(max_steps=config.steps_per_episode)
    hr = HistoryRecorder(max_steps=config.steps_per_episode)
    updater = build_kf(sim_pomdp, ekf_filter=config.ekf_filter)

    # Create the list of Sim objects
    sim_list = []

    # Add Sim objects for each episode
    for sim_number in 1:config.num_episodes
        seed = starting_seed + sim_number

        # Get initial belief and state
        initial_belief = initialize_belief(sim_pomdp, config)
        initial_state = rand(initialstate(sim_pomdp))

        # Create Sim object following POMDPs.jl documentation format with custom updater
        push!(sim_list, Sim(
            sim_pomdp,           # POMDP
            policy,              # Policy
            updater,             # Custom updater
            initial_belief,      # Initial belief
            initial_state;       # Initial state
            rng=MersenneTwister(seed),
            max_steps=config.steps_per_episode,
            metadata=Dict(:policy => algorithm.solver_name, :lambda => pomdp.lambda, :seed => seed, :episode_number => sim_number)
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
# Simulate all policies in parallel
# ----------------------------
function simulate_all_policies(algorithms, config)
    rng = MersenneTwister(1)

    # Defining parameters for parallel simulation
    starting_seed = 1

    # Create the list of Sim objects
    sim_list = []

    # Simulate policy
    for λ in config.lambda_values

        # Create simulator POMDP
        sim_pomdp = create_sim_pomdp(config, λ)
        @info "Created simulator POMDP with reward lambdas: $(sim_pomdp.reward_lambdas)"

        # Create simulator
        hr = HistoryRecorder(max_steps=config.steps_per_episode)
        if config.high_fidelity_sim
            updater = build_kf(sim_pomdp, ekf_filter=config.ekf_filter)
        else
            updater = DiscreteUpdater(sim_pomdp)
        end

        # Load policy, pomdp, and mdp
        for algo in algorithms
            policy_pomdp_mdp_filename = "policy_pomdp_mdp_$(λ)_lambda"
            policy_pomdp_mdp_filepath = joinpath(config.policies_dir, "$(algo.solver_name)", "$(policy_pomdp_mdp_filename).jld2")
            # @info "Loading policy, pomdp, and mdp from $(policy_pomdp_mdp_filepath)"
            @load policy_pomdp_mdp_filepath policy pomdp mdp

            # Create adaptor policy
            if config.high_fidelity_sim
                adaptor_policy = AdaptorPolicy(policy, pomdp, config.location)
            else
                adaptor_policy = LOFIAdaptorPolicy(policy, pomdp)
            end

            # Add Sim objects for each episode
            for sim_number in 1:config.num_episodes
                seed = starting_seed + sim_number

                # Get initial belief and state
                initial_belief = initialize_belief(sim_pomdp, config)
                initial_state = rand(initialstate(sim_pomdp))

                # Create Sim object following POMDPs.jl documentation format with custom updater
                push!(sim_list, Sim(
                    sim_pomdp,                      # POMDP
                    adaptor_policy,                 # Policy
                    updater,                        # Custom updater
                    initial_belief,                 # Initial belief
                    initial_state;                  # Initial state
                    rng=Random.seed!(copy(rng), seed),
                    max_steps=config.steps_per_episode,
                    metadata=Dict(:policy => algo.solver_name, :lambda => λ, :seed => seed, :episode_number => sim_number)
                ))
            end
        end
    end
   # Main.@infiltrate

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

    # Save data
    mkpath(config.simulations_dir)
    data_filepath = joinpath(config.simulations_dir, "all_policies_simulation_data.jld2")
    @save data_filepath data
    @info "Saved parallel simulation data to $(data_filepath)"

    return data

end

# ----------------------------
# Simulate all policies in parallel
# ----------------------------
function simulate_all_policies_on_mdp(algorithms, config)

    # Defining parameters for parallel simulation
    starting_seed = 1

    # Create the list of Sim objects
    sim_list = []

    # Simulate policy
    for λ in config.lambda_values

        # Load policy, pomdp, and mdp
        for algo in algorithms
            policy_pomdp_mdp_filename = "policy_pomdp_mdp_$(λ)_lambda"
            @load joinpath(config.policies_dir, "$(algo.solver_name)", "$(policy_pomdp_mdp_filename).jld2") policy pomdp mdp

            # Create simulator
            hr = HistoryRecorder(max_steps=config.steps_per_episode)

            # Add Sim objects for each episode
            for sim_number in 1:config.num_episodes
                seed = starting_seed + sim_number

                # Get initial belief and state
                initial_state = rand(initialstate(mdp))

                push!(sim_list, Sim(
                    mdp,                      # MDP
                    policy,                 # Policy
                    initial_state;                  # Initial state
                    rng=MersenneTwister(seed),
                    max_steps=config.steps_per_episode,
                    metadata=Dict(:policy => algo.solver_name, :lambda => λ, :seed => seed, :episode_number => sim_number)
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

    # Save data
    mkpath(config.simulations_dir)
    @save joinpath(config.simulations_dir, "all_policies_simulation_data.jld2") data
    println("Saved data to $(config.simulations_dir)/all_policies_simulation_data.jld2")

    return data

end


# ----------------------------
# Simulate VI policy on a high fidelity MDP with full observability
# ----------------------------
function simulate_vi_policy_on_hifi_mdp(algorithms, config)

    # Defining parameters for parallel simulation
    starting_seed = 1

    # Create the list of Sim objects
    sim_list = []

    # Simulate policy
    for λ in config.lambda_values

        # Load policy, pomdp, and mdp
        for algo in algorithms

            # Only test VI policy
            if algo.solver_name == "VI_Policy"
                policy_pomdp_mdp_filename = "policy_pomdp_mdp_$(λ)_lambda"
                @load joinpath(config.policies_dir, "$(algo.solver_name)", "$(policy_pomdp_mdp_filename).jld2") policy pomdp mdp

                # Create simulator
                hr = HistoryRecorder(max_steps=config.steps_per_episode)

                # Create simulator POMDP
                sim_pomdp = create_sim_pomdp(config, λ)
                sim_mdp = UnderlyingMDP(sim_pomdp)

                # Create adaptor policy
                adaptor_policy = AdaptorPolicy(policy, pomdp, config.location)

                # Add Sim objects for each episode
                for sim_number in 1:config.num_episodes
                    seed = starting_seed + sim_number

                    # Get initial belief and state
                    initial_state = rand(initialstate(sim_mdp))

                    push!(sim_list, Sim(
                        sim_mdp,                      # MDP
                        adaptor_policy,                 # Policy
                        initial_state;                  # Initial state
                        rng=MersenneTwister(seed),
                        max_steps=config.steps_per_episode,
                        metadata=Dict(:policy => algo.solver_name, :lambda => λ, :seed => seed, :episode_number => sim_number)
                    ))
                end
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

    # Save data
    mkpath(config.simulations_dir)
    @save joinpath(config.simulations_dir, "all_policies_simulation_data.jld2") data
    println("Saved data to $(config.simulations_dir)/all_policies_simulation_data.jld2")

    return data

end
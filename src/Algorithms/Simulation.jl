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
    if config.simulation_config.high_fidelity_sim
        return (
            adult = sim_pomdp.adult_mean + sim_pomdp.adult_dist,
            motile = sim_pomdp.motile_mean + sim_pomdp.motile_dist,
            sessile = sim_pomdp.sessile_mean + sim_pomdp.sessile_dist,
            temperature = get_temperature(sim_pomdp.production_start_week, sim_pomdp.location) + sim_pomdp.temp_dist,
        )
    else
        return initialstate(sim_pomdp)
    end
end

# ----------------------------
# Create Sim POMDP
# ----------------------------
function create_sim_pomdp(config)
    # Simulate policies on a POMDP with a larger state space
    # for a realistic evaluation of performance.
    sc = config.solver_config
    if config.simulation_config.high_fidelity_sim
        return SeaLiceSimPOMDP(
            reward_lambdas=config.simulation_config.sim_reward_lambdas,
            reproduction_rate=sc.reproduction_rate,
            discount_factor=sc.discount_factor,
            # SimPOMDP parameters
            adult_mean=config.simulation_config.adult_mean,
            motile_mean=config.simulation_config.motile_mean,
            sessile_mean=config.simulation_config.sessile_mean,
            adult_sd=config.simulation_config.adult_sd,
            motile_sd=config.simulation_config.motile_sd,
            sessile_sd=config.simulation_config.sessile_sd,
            temp_sd=config.simulation_config.temp_sd,
            location=sc.location,
            # Country-specific parameters
            season_regulation_limits=sc.season_regulation_limits,
            salmon_price_MNOK_per_tonne=sc.salmon_price_MNOK_per_tonne,
            regulatory_violation_cost_MNOK=sc.regulatory_violation_cost_MNOK,
            welfare_cost_MNOK=sc.welfare_cost_MNOK,
            chronic_lice_cost_MNOK=sc.chronic_lice_cost_MNOK,
        )
    else
        # Use the same POMDP type that policies were trained on
        sim_cfg = config.simulation_config
        adult_mean = max(sim_cfg.adult_mean, 1e-6)
        motile_ratio = sim_cfg.motile_mean / adult_mean
        sessile_ratio = sim_cfg.sessile_mean / adult_mean
        base_temperature = get_location_params(sc.location).T_mean

        if sc.log_space
            return SeaLiceLogPOMDP(
                reward_lambdas=config.simulation_config.sim_reward_lambdas,
                discount_factor=sc.discount_factor,
                discretization_step=sc.discretization_step,
                adult_sd=sc.adult_sd,
                regulation_limit=sc.regulation_limit,
                season_regulation_limits=sc.season_regulation_limits,
                full_observability_solver=sc.full_observability_solver,
                location=sc.location,
                reproduction_rate=sc.reproduction_rate,
                motile_ratio=motile_ratio,
                sessile_ratio=sessile_ratio,
                base_temperature=base_temperature,
                salmon_price_MNOK_per_tonne=sc.salmon_price_MNOK_per_tonne,
                regulatory_violation_cost_MNOK=sc.regulatory_violation_cost_MNOK,
                welfare_cost_MNOK=sc.welfare_cost_MNOK,
                chronic_lice_cost_MNOK=sc.chronic_lice_cost_MNOK,
            )
        else
            return SeaLicePOMDP(
                reward_lambdas=config.simulation_config.sim_reward_lambdas,
                discount_factor=sc.discount_factor,
                discretization_step=sc.discretization_step,
                adult_sd=sc.adult_sd,
                regulation_limit=sc.regulation_limit,
                season_regulation_limits=sc.season_regulation_limits,
                full_observability_solver=sc.full_observability_solver,
                salmon_price_MNOK_per_tonne=sc.salmon_price_MNOK_per_tonne,
                regulatory_violation_cost_MNOK=sc.regulatory_violation_cost_MNOK,
                welfare_cost_MNOK=sc.welfare_cost_MNOK,
                chronic_lice_cost_MNOK=sc.chronic_lice_cost_MNOK,
            )
        end
    end
end

# ----------------------------
# Simulate all policies in parallel
# ----------------------------
function simulate_all_policies(algorithms, config, all_policies)
    rng = MersenneTwister(1)

    # Defining parameters for parallel simulation
    starting_seed = 1

    # Create the list of Sim objects
    sim_list = []

    # Create simulator POMDP
    sim_pomdp = create_sim_pomdp(config)
    @info "Created simulator POMDP with reward lambdas: $(sim_pomdp.reward_lambdas)"

    # Create updater
    if config.simulation_config.high_fidelity_sim
        kf_updater = build_kf(sim_pomdp, ekf_filter=config.simulation_config.ekf_filter)
        updater = EnrichedUpdater(kf_updater)
    else
        updater = DiscreteUpdater(sim_pomdp)
    end

    for algo in algorithms
        (; policy, pomdp) = all_policies[algo.solver_name]

        # Create adaptor policy
        if config.simulation_config.high_fidelity_sim
            adaptor_policy = AdaptorPolicy(policy, pomdp, config.solver_config.location, config.solver_config.reproduction_rate)
        else
            adaptor_policy = LOFIAdaptorPolicy(policy, pomdp)
        end

        # Add Sim objects for each episode
        for sim_number in 1:config.simulation_config.num_episodes
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
                max_steps=config.simulation_config.steps_per_episode,
                metadata=Dict(:policy => algo.solver_name, :seed => seed, :episode_number => sim_number)
            ))
        end
    end

    # Run the simulations in parallel
    data = run_parallel(sim_list, proc_warn=false) do sim, hist
        return (
            reward = discounted_reward(hist),
            n_steps = n_steps(hist),
            history = hist,
            policy = sim.metadata[:policy],
            seed = sim.metadata[:seed]
        )
    end

    # Save data
    mkpath(config.simulations_dir)
    data_filepath = joinpath(config.simulations_dir, "all_policies_simulation_data.jld2")
    @save data_filepath data
    @info "Saved parallel simulation data to $(data_filepath)"

    return data, sim_pomdp

end


using DataFrames, CSV, Dates

# ----------------------------
# Save experiment configuration
# ----------------------------
function save_experiment_config(config::ExperimentConfig, heuristic_config::HeuristicConfig, first_step_flag::String, csv_path="results/experiments/experiments.csv")
    df = DataFrame(
        # Experiment parameters
        experiment_name = config.experiment_name,
        timestamp = Dates.now(),

        # Simulation parameters
        num_episodes = config.num_episodes,
        steps_per_episode = config.steps_per_episode,
        process_noise = 0.0,
        observation_noise = 0.0,
        ekf_filter = config.ekf_filter,

        # POMDP parameters
        costOfTreatment = config.costOfTreatment,
        growthRate = config.growthRate,
        rho = config.rho,
        discount_factor = config.discount_factor,
        log_space = config.log_space,

        # Algorithm parameters
        lambda_values = string(config.lambda_values),  # store as string
        reward_lambdas = string(config.reward_lambdas),
        sarsop_max_time = config.sarsop_max_time,
        VI_max_iterations = config.VI_max_iterations,
        QMDP_max_iterations = config.QMDP_max_iterations,

        # Heuristic parameters
        heuristic_threshold = heuristic_config.raw_space_threshold,
        heuristic_belief_threshold = heuristic_config.belief_threshold,
        heuristic_rho = heuristic_config.rho,

        # Run management
        experiment_dir = config.experiment_dir,
        first_step_flag = first_step_flag,
    )
    if isfile(csv_path)
        CSV.write(csv_path, df; append=true, writeheader=false)
    else
        CSV.write(csv_path, df)
    end
end

# ----------------------------
# Find latest config with same policy and (optionally) simulation parameters
# ----------------------------
function get_latest_matching_config(config::ExperimentConfig, heuristic_config::HeuristicConfig, also_match_simulation_params::Bool, csv_path="results/experiments/experiments.csv")
    df = CSV.read(csv_path, DataFrame)

    # Filter for matching config values
    # POMDP parameters
    mask = (df.costOfTreatment .== config.costOfTreatment) .&
            (df.growthRate .== config.growthRate) .&
            (df.rho .== config.rho) .&
            (df.discount_factor .== config.discount_factor) .&
            (df.log_space .== config.log_space) .&
            (df.reward_lambdas .== string(config.reward_lambdas)) .&
            (df.regulation_limit .== config.regulation_limit) .&
        # Algorithm parameters
            (df.lambda_values .== string(config.lambda_values)) .&
            (df.sarsop_max_time .== config.sarsop_max_time) .&
            (df.VI_max_iterations .== config.VI_max_iterations) .&
            (df.QMDP_max_iterations .== config.QMDP_max_iterations) .&

        # Heuristic parameters
            (df.heuristic_threshold .== heuristic_config.raw_space_threshold) .&
            (df.heuristic_belief_threshold .== heuristic_config.belief_threshold) .&
            (df.heuristic_rho .== heuristic_config.rho) .&

        # Run management
            ((df.first_step_flag .== "solve") .|| (df.first_step_flag .== "simulate"))

    if also_match_simulation_params
        # If the also_match_simulation_params flag is true, we want to match simulation parameters
        # Because we are plotting, we don't need access to the policies
        mask = mask .& (df.num_episodes .== config.num_episodes) .&
           (df.steps_per_episode .== config.steps_per_episode) .&
           (df.process_noise .== config.process_noise) .&
           (df.observation_noise .== config.observation_noise) .&
           (df.ekf_filter .== config.ekf_filter)
    else
        # If the also_match_simulation_params flag is false, we will also run simulations, so we
        # need access to the policies themselves, so we need a run where the first step is solve
        mask = mask .& (df.first_step_flag .== "solve")
    end

    matches = df[mask, :]
    if nrow(matches) == 0
        @info "No matching experiment found. Run solve first."
        return nothing
    end
    # Sort by timestamp descending and return the experiment_name of the most recent
    sort!(matches, :timestamp, rev=true)
    @info "Latest matching config: $(matches.experiment_dir[1])"
    @load joinpath(matches.experiment_dir[1], "config", "experiment_config.jld2") config
    return config
end
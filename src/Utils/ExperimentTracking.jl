using DataFrames, CSV, Dates

function save_experiment_config(config::ExperimentConfig, heuristic_config::HeuristicConfig, csv_path="results/experiments/experiments.csv")
    df = DataFrame(
        # Experiment parameters
        experiment_name = config.experiment_name,
        timestamp = Dates.now(),

        # Simulation parameters
        num_episodes = config.num_episodes,
        steps_per_episode = config.steps_per_episode,
        process_noise = config.process_noise,
        observation_noise = config.observation_noise,
        ekf_filter = config.ekf_filter,

        # POMDP parameters
        costOfTreatment = config.costOfTreatment,
        growthRate = config.growthRate,
        rho = config.rho,
        discount_factor = config.discount_factor,
        log_space = config.log_space,
        skew = config.skew,

        # Algorithm parameters
        lambda_values = string(config.lambda_values),  # store as string
        sarsop_max_time = config.sarsop_max_time,
        VI_max_iterations = config.VI_max_iterations,
        QMDP_max_iterations = config.QMDP_max_iterations,

        # Heuristic parameters
        heuristic_threshold = heuristic_config.raw_space_threshold,
        heuristic_belief_threshold = heuristic_config.belief_threshold,
        heuristic_rho = heuristic_config.rho,

        # Run parameters
        # mode = config.mode,
        # raw_space = config.raw_space,
        # solve = config.solve,
        # simulate = config.simulate,
        # plot = config.plot,
    )
    if isfile(csv_path)
        CSV.write(csv_path, df; append=true, writeheader=false)
    else
        CSV.write(csv_path, df)
    end
end

function find_latest_experiment(config::ExperimentConfig, heuristic_config::HeuristicConfig, csv_path="results/experiments/experiments.csv")
    df = CSV.read(csv_path, DataFrame)
    # Filter for matching config values
    # Experiment parameters
    mask = (df.num_episodes .== config.num_episodes) .&
           (df.steps_per_episode .== config.steps_per_episode) .&
           (df.process_noise .== config.process_noise) .&
           (df.observation_noise .== config.observation_noise) .&
           (df.ekf_filter .== config.ekf_filter) .&

        # POMDP parameters
           (df.costOfTreatment .== config.costOfTreatment) .&
           (df.growthRate .== config.growthRate) .&
           (df.rho .== config.rho) .&
           (df.discount_factor .== config.discount_factor) .&
           (df.log_space .== config.log_space) .&
           (df.skew .== config.skew) .&

        # Algorithm parameters
           (df.lambda_values .== string(config.lambda_values)) .&
           (df.sarsop_max_time .== config.sarsop_max_time) .&
           (df.VI_max_iterations .== config.VI_max_iterations) .&
           (df.QMDP_max_iterations .== config.QMDP_max_iterations) .&

        # Heuristic parameters
           (df.heuristic_threshold .== heuristic_config.raw_space_threshold) .&
           (df.heuristic_belief_threshold .== heuristic_config.belief_threshold) .&
           (df.heuristic_rho .== heuristic_config.rho)
    matches = df[mask, :]
    if nrow(matches) == 0
        @info "No matching experiment found"
        return nothing
    end
    # Sort by timestamp descending and return the experiment_name of the most recent
    sort!(matches, :timestamp, rev=true)
    @info "Latest experiment: $(matches.experiment_name[1])"
    return matches.experiment_name[1]
end

# function find_latest_policy_config(config::ExperimentConfig, heuristic_config::HeuristicConfig, csv_path="results/experiments/experiments.csv")
#     df = CSV.read(csv_path, Dat aFrame)
#     # Filter for matching config values
#     # Experiment parameters
#     mask = (df.num_episodes .== config.num_episodes) .&
#            (df.steps_per_episode .== config.steps_per_episode) .&
#            (df.process_noise .== config.process_noise) .&
#            (df.observation_noise .== config.observation_noise) .&
#            (df.ekf_filter .== config.ekf_filter) .&
# end
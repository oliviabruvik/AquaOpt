using DataFrames, CSV, Dates

# ----------------------------
# Save experiment configuration
# ----------------------------
function save_experiment_config(config::ExperimentConfig, heuristic_config::HeuristicConfig, csv_path="results/experiments/experiments.csv")
    df = DataFrame(
        # Experiment parameters
        experiment_name = config.experiment_name,
        timestamp = Dates.now(),

        # Simulation parameters
        num_episodes = config.simulation_config.num_episodes,
        steps_per_episode = config.simulation_config.steps_per_episode,
        process_noise = 0.0,
        observation_noise = 0.0,
        ekf_filter = config.simulation_config.ekf_filter,

        # POMDP parameters
        costOfTreatment = config.solver_config.costOfTreatment,
        growthRate = config.solver_config.growthRate,
        discount_factor = config.solver_config.discount_factor,
        log_space = config.solver_config.log_space,

        # Algorithm parameters
        lambda_values = string(config.lambda_values),  # store as string
        reward_lambdas = string(config.solver_config.reward_lambdas),
        sarsop_max_time = config.solver_config.sarsop_max_time,
        VI_max_iterations = config.solver_config.VI_max_iterations,
        QMDP_max_iterations = config.solver_config.QMDP_max_iterations,

        # Heuristic parameters
        heuristic_threshold = heuristic_config.raw_space_threshold,
        heuristic_belief_threshold_mechanical = heuristic_config.belief_threshold_mechanical,
        heuristic_belief_threshold_thermal = heuristic_config.belief_threshold_thermal,
        heuristic_rho = heuristic_config.rho,

        # Run management
        experiment_dir = config.experiment_dir,
    )
    if isfile(csv_path)
        CSV.write(csv_path, df; append=true, writeheader=false)
    else
        CSV.write(csv_path, df)
    end
end
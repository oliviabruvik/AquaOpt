# -------------------------
# Include shared types first
# -------------------------
include("Utils/SharedTypes.jl")

# -------------------------
# Include other files
# -------------------------
include("Algorithms/Evaluation.jl")
include("Algorithms/Policies.jl")
include("Algorithms/Simulation.jl")
include("Data/Cleaning.jl")
include("Models/SeaLicePOMDP.jl")
include("Plotting/Heatmaps.jl")
include("Plotting/Timeseries.jl")
include("Plotting/Comparison.jl")
include("Plotting/ParallelPlots.jl")
include("Plotting/PlosOnePlots.jl")
include("Utils/Config.jl")
include("Utils/ExperimentTracking.jl")

# Environment variables
ENV["PLOTS_BROWSER"] = "true"
ENV["PLOTS_BACKEND"] = "plotlyjs"

# Import required packages
using Logging
using DiscreteValueIteration
using GridInterpolations
using NativeSARSOP
using SARSOP
using POMDPs
using POMDPTools
using POMDPXFiles
using Plots: plot, plot!, scatter, scatter!, heatmap, heatmap!, histogram, histogram!, savefig
using LocalFunctionApproximation
using LocalApproximationValueIteration
using Dates

plotlyjs()  # Activate Plotly backend

# ----------------------------
# Simulate policies
# ----------------------------
function simulate_policies(algorithms, config)
    @info "Simulating policies"
    parallel_data = simulate_all_policies(algorithms, config)

    if config.high_fidelity_sim
        #Simulate policies
        for algo in algorithms
            println("Simulating $(algo.solver_name)")
            histories = simulate_policy(algo, config)
            avg_results = evaluate_simulation_results(config, algo, histories)
        end
    end
    return parallel_data
end

# ----------------------------
# Plot Plos One Plots
# ----------------------------
function plot_plos_one_plots(parallel_data, config)
    @info "Plotting Plos One Plots"
    plos_one_plot_kalman_filter_belief_trajectory(parallel_data, "NUS_SARSOP_Policy", config, 0.6)
    # plot_kalman_filter_trajectory_with_uncertainty(parallel_data, "NUS_SARSOP_Policy", config, 0.6)
    plos_one_sealice_levels_over_time(parallel_data, config)
    plos_one_combined_treatment_probability_over_time(parallel_data, config)
    plos_one_sarsop_dominant_action(parallel_data, config, 0.6)
    @info "Saved all plos one plots to $(config.figures_dir)"
end

function run_experiments(;first_step_flag="solve", log_space=true, experiment_name="exp", mode="light")

    main(first_step_flag=first_step_flag, log_space=true, experiment_name="log_space_ekf", mode="paper", ekf_filter=true)
    main(first_step_flag=first_step_flag, log_space=false, experiment_name="raw_space_ekf", mode="paper", ekf_filter=false)
    main(first_step_flag=first_step_flag, log_space=true, experiment_name="log_space_noekf", mode="paper", ekf_filter=false)
    main(first_step_flag=first_step_flag, log_space=false, experiment_name="raw_space_noekf", mode="paper", ekf_filter=true)

end

# ----------------------------
# Main function
# ----------------------------
function main(;first_step_flag="solve", log_space=true, experiment_name="exp", mode="light", ekf_filter=true)

    @info "Running experiment: $experiment_name, log_space: $log_space, ekf_filter: $ekf_filter, mode: $mode"

    config, heuristic_config = setup_experiment_configs(experiment_name, log_space, ekf_filter, mode)
    algorithms = define_algorithms(config, heuristic_config)

    if first_step_flag == "solve"
        @info "Solving policies"
        for algo in algorithms
            println("Solving $(algo.solver_name)")
            generate_mdp_pomdp_policies(algo, config)
        end
        @info "Simulating policies"
        parallel_data = simulate_policies(algorithms, config)
    end

    if first_step_flag == "simulate"
        @info "Skipping policy solving"
        parallel_data = simulate_policies(algorithms, config)
    end

    if first_step_flag == "plot"
        @info "Skipping policy solving and simulation"
        @load joinpath(config.simulations_dir, "all_policies_simulation_data.jld2") data
        parallel_data = data
    end

    if config.high_fidelity_sim == false
        print_reward_metrics_for_vi_policy(parallel_data, config)
        exit()
    end

    # Extract reward metrics
    processed_data = extract_reward_metrics(parallel_data, config)

    # Display reward metrics
    display_reward_metrics(processed_data, config, false)

    # Compare VI policy on high fidelity MDP with full observability
    # vi_parallel_data = simulate_vi_policy_on_hifi_mdp(algorithms, config)
    # print_reward_metrics_for_vi_policy(vi_parallel_data, config)

    # Plot the results
    # plot_plos_one_plots(parallel_data, config)

    # Log experiment configuration in experiments.csv file with all experiments
    @info "Saved experiment configuration to $(config.experiment_dir)/config/experiment_config.jld2"
    save_experiment_config(config, heuristic_config, first_step_flag)

    # Save config to file in current directory for easy access
    mkpath(joinpath(config.experiment_dir, "config"))
    @save joinpath(config.experiment_dir, "config", "experiment_config.jld2") config
    open(joinpath(config.experiment_dir, "config", "experiment_config.txt"), "w") do io
        for field in fieldnames(typeof(config))
            value = getfield(config, field)
            println(io, "$field: $value")
        end
    end

end


# ----------------------------
# Set up and save experiment configuration
# ----------------------------
function setup_experiment_configs(experiment_name, log_space, ekf_filter=true, mode="light")

    # Define experiment configuration
    exp_name = string(Dates.today(), "/", Dates.now(), "_", experiment_name, "_mode_", mode)

    @info "Setting up experiment configuration for experiment: $exp_name"

    if mode == "light"
        config = ExperimentConfig(
            num_episodes=1000,
            steps_per_episode=104,
            log_space=log_space,
            ekf_filter=ekf_filter,
            experiment_name=exp_name,
            verbose=false,
            step_through=false,
            # reward_lambdas=[0.7, 0.2, 0.1, 0.05, 0.1], # [treatment, regulatory, biomass, health, sea lice level]
            reward_lambdas=[0.7, 0.2, 0.1, 0.1, 0.8], # [treatment, regulatory, biomass, health, sea lice level]
            sarsop_max_time=5.0,
            VI_max_iterations=10,
            QMDP_max_iterations=10,
            policies_dir = joinpath("NorthernNorway", "policies"),
            simulations_dir = joinpath("NorthernNorway", "simulation_histories"),
            results_dir = joinpath("NorthernNorway", "avg_results"),
            figures_dir = joinpath("NorthernNorway", "figures"),
            experiment_dir = joinpath("NorthernNorway"),
        )
    elseif mode == "debug"
        config = ExperimentConfig(
            num_episodes=10,
            steps_per_episode=104,
            log_space=log_space,
            ekf_filter=ekf_filter,
            experiment_name=exp_name,
            verbose=false,
            step_through=false,
            reward_lambdas=[0.7, 0.2, 0.1, 0.1, 0.8], # [treatment, regulatory, biomass, health, sea lice]
            sarsop_max_time=5.0,
            VI_max_iterations=10,
            QMDP_max_iterations=10,
            discount_factor = 0.95,
            high_fidelity_sim = false, # Toggles whether we simulate policies on sim (true) or solver (false) POMDP
            full_observability_solver = false, # Toggles whether we have full observability in the observation function or not (false). Pairs with high_fidelity_sim = false.
        )
    elseif mode == "fullobs"
        config = ExperimentConfig(
            num_episodes=10,
            steps_per_episode=104,
            log_space=log_space,
            ekf_filter=ekf_filter,
            experiment_name=exp_name,
            verbose=false,
            step_through=false,
            reward_lambdas=[0.7, 0.2, 0.1, 0.1, 0.8], # [treatment, regulatory, biomass, health, sea lice]
            sarsop_max_time=5.0,
            VI_max_iterations=10,
            QMDP_max_iterations=10,
            discount_factor = 0.95,
            high_fidelity_sim = false, # Toggles whether we simulate policies on sim (true) or solver (false) POMDP
            full_observability_solver = false, # Toggles whether we have full observability in the observation function or not (false). Pairs with high_fidelity_sim = false.
        )
    elseif mode == "paper"
        config = ExperimentConfig(
            num_episodes=100,
            steps_per_episode=104,
            log_space=log_space,
            ekf_filter=ekf_filter,
            experiment_name=exp_name,
            verbose=false,
            step_through=false,
            reward_lambdas=[0.7, 0.2, 0.1, 0.1, 0.8], # [treatment, regulatory, biomass, health, sea lice]
            sarsop_max_time=50.0,
            VI_max_iterations=50,
            QMDP_max_iterations=100,
            discount_factor = 0.95,
            high_fidelity_sim = false,
        )
    end
        
    heuristic_config = HeuristicConfig(
        raw_space_threshold=config.heuristic_threshold,
        belief_threshold_mechanical=config.heuristic_belief_threshold_mechanical,
        belief_threshold_thermal=config.heuristic_belief_threshold_thermal,
        rho=config.heuristic_rho
    )

    @info "Heuristic config: $heuristic_config"
    @info "Config: $config"

    return config, heuristic_config

end

# ----------------------------
# Define algorithms
# ----------------------------
function define_algorithms(config, heuristic_config)

    @info "Defining solvers"
    native_sarsop_solver = NativeSARSOP.SARSOPSolver(max_time=config.sarsop_max_time) #, verbose=false)
    
    @info "Defining NUS SARSOP solver"
    nus_sarsop_solver = SARSOP.SARSOPSolver(
        timeout=config.sarsop_max_time,
        verbose=false,
        policy_filename=joinpath(config.policies_dir, "NUS_SARSOP_Policy/policy.out"),
        pomdp_filename=joinpath(config.experiment_dir, "pomdp_mdp/pomdp.pomdpx")
    )
    
    @info "Defining VI solver"
    vi_solver = ValueIterationSolver(max_iterations=config.VI_max_iterations, belres=1e-10, verbose=false)

    @info "Defining QMDP solver"
    qmdp_solver = QMDPSolver(max_iterations=config.QMDP_max_iterations)

    
    algorithms = [
        # Algorithm(solver_name="NeverTreat_Policy"),
        # Algorithm(solver_name="AlwaysTreat_Policy"),
        # Algorithm(solver_name="Random_Policy"),
        # Algorithm(solver_name="Heuristic_Policy", heuristic_config=heuristic_config),
        Algorithm(solver=nus_sarsop_solver, solver_name="NUS_SARSOP_Policy"),
        Algorithm(solver=vi_solver, solver_name="VI_Policy"),
        Algorithm(solver=qmdp_solver, solver_name="QMDP_Policy"),
    ]
    return algorithms
end

# ----------------------------
# Plot Parallel Plots
# ----------------------------
function plot_parallel_plots(parallel_data, config)

    # Print treatment frequency
    print_treatment_frequency(parallel_data, config)

    # Print histories
    print_histories(parallel_data, config)

    # Plot heuristic vs sarsop sealice levels over time
    plot_heuristic_vs_sarsop_sealice_levels_over_time_latex(parallel_data, config)

    # Plot one simulation with all state variables over time
    plot_one_simulation_with_all_state_variables_over_time(parallel_data, config, "NUS_SARSOP_Policy")
    plot_one_simulation_with_all_state_variables_over_time(parallel_data, config, "Heuristic_Policy")

    # Plot treatment distribution comparison
    plot_treatment_distribution_comparison_latex(parallel_data, config)
    
    # Plot SARSOP policy action heatmap
    plot_sarsop_policy_action_heatmap(config, 0.6)
    
    # Plot Heuristic policy action heatmap
    plot_heuristic_policy_action_heatmap(config, 0.6)
    
    # Plot combined policy action heatmaps side by side
    plot_combined_policy_action_heatmaps(config, 0.6)

    plot_beliefs_over_time(parallel_data, "NUS_SARSOP_Policy", config, 0.6)
    plot_beliefs_over_time(parallel_data, "Heuristic_Policy", config, 0.6)

    # Plot combined treatment probability over time
    plot_combined_treatment_probability_over_time(parallel_data, config)
end

# ----------------------------
# Plot results
# ----------------------------
function plot_results(algorithms, config)

    @info "Plotting results"
    for algo in algorithms

        # Load histories
        histories_dir = joinpath(config.simulations_dir, "$(algo.solver_name)")
        histories_path = joinpath(histories_dir, "$(algo.solver_name)_histories.jld2")
        if !isfile(histories_path)
            @error "Histories file not found at $histories_path. Run simulation first. with --simulate"
            continue
        end
        @load histories_path histories

        # Load data from parallel simulations
        data_path = joinpath(config.simulations_dir, "all_policies_simulation_data.jld2")
        if !isfile(data_path)
            @error "Data file not found at $data_path. Run simulation first. with --simulate"
            continue
        end
        @load data_path data

        # Load avg results
        avg_results_path = joinpath(config.results_dir, "$(algo.solver_name)_avg_results.jld2")
        if !isfile(avg_results_path)
            @error "Avg results file not found at $avg_results_path. Run simulation first. with --simulate"
            continue
        end
        @load avg_results_path avg_results

        # Plot belief means and variances
        plot_beliefs_over_time(data, algo.solver_name, config, 0.6)

        # Plot policy cost vs sealice
        plot_policy_cost_vs_sealice(histories, avg_results, algo.solver_name, config)

        # Plot policy belief levels
        plot_policy_belief_levels(histories, algo.solver_name, config, 0.6)

        # Plot treatment heatmap
        plot_treatment_heatmap(algo, config)

        # Plot simulation treatment heatmap
        # plot_simulation_treatment_heatmap(algo, config; use_observations=false, n_bins=50)

        plot_algo_sealice_levels_over_time(config, algo.solver_name, 0.6)
        plot_algo_adult_predicted_over_time(config, algo.solver_name, 0.6)

    end
    
    # Plot comparison plots
    plot_all_cost_vs_sealice(config)
    plot_policy_sealice_levels_over_lambdas(config)
    plot_policy_treatment_cost_over_lambdas(config)
    plot_policy_sealice_levels_over_time(config, 0.6)
    plot_policy_treatment_cost_over_time(config, 0.6)
    plot_policy_reward_over_lambdas(config)

    @info "Saved all plots to $(config.figures_dir)"
end


if abspath(PROGRAM_FILE) == @__FILE__

    first_step_flag = "solve" # "solve", "simulate", "plot"
    log_space_flag = true
    experiment_name_flag = "exp"
    mode_flag = "light"

    for arg in ARGS
        if occursin("--experiment_name=", arg)
            global experiment_name_flag = split(arg, "=")[2]
        elseif occursin("--mode=", arg)
            global mode_flag = split(arg, "=")[2]
        elseif occursin("--first_step=", arg)
            global first_step_flag = String(split(arg, "=")[2])
        elseif arg == "--raw_space"
            global log_space_flag = false
        end
    end

    @info "Running with mode: $mode_flag, log_space: $log_space_flag, experiment_name: $experiment_name_flag"

    # run_experiments(first_step_flag=first_step_flag, log_space=log_space_flag, experiment_name=experiment_name_flag, mode=mode_flag)
    main(first_step_flag=first_step_flag, log_space=log_space_flag, experiment_name=experiment_name_flag, mode=mode_flag)
end
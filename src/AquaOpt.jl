module AquaOpt

# -------------------------
# Include utilities and configuration first
# -------------------------
include("Utils/SharedTypes.jl")
include("Utils/Config.jl")  # Config must come before Utils since Utils uses get_location_params
include("Utils/Utils.jl")

# -------------------------
# Include model files
# -------------------------
include("Models/KalmanFilter.jl")
include("Models/SeaLiceLogPOMDP.jl")
include("Models/SeaLicePOMDP.jl")
include("Models/SimulationPOMDP.jl")

# -------------------------
# Include algorithm files
# -------------------------
include("Algorithms/Policies.jl")
include("Algorithms/Simulation.jl")
include("Algorithms/Evaluation.jl")

# -------------------------
# Include plotting files
# -------------------------
include("Plotting/Heatmaps.jl")
include("Plotting/Timeseries.jl")
include("Plotting/Comparison.jl")
include("Plotting/ParallelPlots.jl")
include("Plotting/PlosOnePlots.jl")
include("Plotting/Plots.jl")

# -------------------------
# Include experiment tracking
# -------------------------
include("Utils/ExperimentTracking.jl")

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
using JLD2
import PGFPlotsX

# Initialize plotting backend - this function runs when module is loaded
function __init__()
    # Set environment variables
    ENV["PLOTS_BROWSER"] = "true"
    ENV["PLOTS_BACKEND"] = "plotlyjs"

    # Activate Plotly backend
    # Users can call this or set their own backend
    try
        Plots.plotlyjs()
    catch e
        @warn "Could not set plotlyjs backend" exception=e
    end

    # Configure PGFPlotsX preamble for LaTeX plotting
    try
        PGFPlotsX.DEFAULT_PREAMBLE = [
            raw"\usepackage{pgfplots}",
            raw"\usepgfplotslibrary{fillbetween}",
            raw"\usepgfplotslibrary{groupplots}",
            raw"\usetikzlibrary{intersections}",
            raw"\pgfplotsset{compat=newest}",
            raw"\pgfplotsset{legend style={text=white,fill=none,draw=none}}"
        ]
    catch e
        @debug "Could not configure PGFPlotsX" exception=e
    end
end

function run_experiments(mode)
    main(first_step_flag="solve", log_space=true, experiment_name="log_space_ekf", mode=mode, ekf_filter=true)
    main(first_step_flag="solve", log_space=false, experiment_name="raw_space_ekf", mode=mode, ekf_filter=false)
    main(first_step_flag="solve", log_space=true, experiment_name="log_space_noekf", mode=mode, ekf_filter=false)
    main(first_step_flag="solve", log_space=false, experiment_name="raw_space_noekf", mode=mode, ekf_filter=true)
    return
end

# ----------------------------
# Main function
# ----------------------------
function main(;first_step_flag="solve", log_space=true, experiment_name="exp", mode="light", ekf_filter=true, plot=false)

    config, heuristic_config = setup_experiment_configs(experiment_name, log_space, ekf_filter, mode)
    algorithms = define_algorithms(config, heuristic_config)

    @info """\n
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                         NEW EXPERIMENT RUN                              ║
    ╠════════════════════════════════════════════════════════════════════════╣
    ║  Mode:            $mode_flag
    ║  Log Space:       $log_space_flag
    ║  EKF Filter:      $ekf_filter
    ║  ExperimentDir: $(config.experiment_dir)
    ╚════════════════════════════════════════════════════════════════════════╝
    """

    # Log experiment configuration in experiments.csv file with all experiments
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

    @info "Solving policies"
    for algo in algorithms
        generate_mdp_pomdp_policies(algo, config)
    end

    @info "Simulating policies"
    parallel_data = simulate_all_policies(algorithms, config)

    # If we are simulating on high fidelity model, we want to evaluate the simulation results
    if config.simulation_config.high_fidelity_sim
        for algo in algorithms
            histories = extract_simulation_histories(config, algo, parallel_data)
            evaluate_simulation_results(config, algo, histories)
        end
    else
        print_reward_metrics_for_vi_policy(parallel_data, config)
        exit()
    end

    # Extract reward metrics
    processed_data = extract_reward_metrics(parallel_data, config)

    # Display reward metrics
    display_reward_metrics(processed_data, config, false)

    if plot
        # Plot the results
        plot_plos_one_plots(processed_data, config)
    end

    # Treatment frequency
    print_treatment_frequency(processed_data, config)

end

# ----------------------------
# Set up and save experiment configuration
# ----------------------------
function setup_experiment_configs(experiment_name, log_space, ekf_filter=true, mode="light")

    # Define experiment configuration
    exp_name = string(Dates.today(), "/", Dates.now(), "_", experiment_name, "_mode_", mode)

    @info "Setting up experiment configuration for experiment: $exp_name"

    if mode == "light"
        solver_cfg = SolverConfig(
            log_space=log_space,
            reward_lambdas=[0.7, 0.2, 0.1, 0.1, 0.8], # [treatment, regulatory, biomass, health, sea lice level]
            sarsop_max_time=5.0,
            VI_max_iterations=10,
            QMDP_max_iterations=10,
        )
        sim_cfg = SimulationConfig(
            num_episodes=1000,
            steps_per_episode=104,
            ekf_filter=ekf_filter,
            verbose=false,
            step_through=false,
        )
        config = ExperimentConfig(
            solver_config=solver_cfg,
            simulation_config=sim_cfg,
            experiment_name=exp_name,
            policies_dir = joinpath("NorthernNorway", "policies"),
            simulations_dir = joinpath("NorthernNorway", "simulation_histories"),
            results_dir = joinpath("NorthernNorway", "avg_results"),
            figures_dir = joinpath("NorthernNorway", "figures"),
            experiment_dir = joinpath("NorthernNorway"),
        )
    elseif mode == "debug"
        solver_cfg = SolverConfig(
            log_space=log_space,
            reward_lambdas=[1.0, 3.0, 0.5, 0.005, 0.0], # [treatment, regulatory, biomass, health, sea lice]
            sarsop_max_time=5.0,
            VI_max_iterations=10,
            QMDP_max_iterations=10,
            discount_factor = 0.95,
            location = "north", # "north", "west", or "south"
            full_observability_solver = false, # Toggles whether we have full observability in the observation function or not (false). Pairs with high_fidelity_sim = false.
        )
        sim_cfg = SimulationConfig(
            num_episodes=100,
            steps_per_episode=52,
            ekf_filter=ekf_filter,
            sim_reward_lambdas = [1.0, 3.0, 0.5, 0.01, 0.0]  # [treatment, regulatory, biomass, health, sea_lice]
        )
        config = ExperimentConfig(
            solver_config=solver_cfg,
            simulation_config=sim_cfg,
            experiment_name=exp_name,
        )
    elseif mode == "paper"
        solver_cfg = SolverConfig(
            log_space=log_space,
            reward_lambdas=[1.0, 3.0, 0.5, 0.01, 0.0], # [treatment, regulatory, biomass, health, sea lice]
            sarsop_max_time=300.0,
            VI_max_iterations=100,
            QMDP_max_iterations=100,
            discount_factor = 0.95,
            location = "north", # "north", "west", or "south"
            full_observability_solver = false, # Toggles whether we have full observability in the observation function or not (false). Pairs with high_fidelity_sim = false.
        )
        sim_cfg = SimulationConfig(
            num_episodes=1000,
            steps_per_episode=104,
            ekf_filter=ekf_filter,
            sim_reward_lambdas = [1.0, 3.0, 0.5, 0.005, 0.0]  # [treatment, regulatory, biomass, health, sea_lice]
        )
        config = ExperimentConfig(
            solver_config=solver_cfg,
            simulation_config=sim_cfg,
            experiment_name=exp_name,
        )
    end
        
    heuristic_config = HeuristicConfig(
        raw_space_threshold=config.solver_config.heuristic_threshold,
        belief_threshold_mechanical=config.solver_config.heuristic_belief_threshold_mechanical,
        belief_threshold_thermal=config.solver_config.heuristic_belief_threshold_thermal,
        rho=config.solver_config.heuristic_rho
    )

    return config, heuristic_config

end

# ----------------------------
# Define algorithms
# ----------------------------
function define_algorithms(config, heuristic_config)

    native_sarsop_solver = NativeSARSOP.SARSOPSolver(max_time=config.solver_config.sarsop_max_time) #, verbose=false)

    nus_sarsop_solver = SARSOP.SARSOPSolver(
        timeout=config.solver_config.sarsop_max_time,
        verbose=false,
        policy_filename=joinpath(config.policies_dir, "NUS_SARSOP_Policy/policy.out"),
        pomdp_filename=joinpath(config.experiment_dir, "pomdp_mdp/pomdp.pomdpx")
    )

    vi_solver = ValueIterationSolver(max_iterations=config.solver_config.VI_max_iterations, belres=1e-10, verbose=false)

    qmdp_solver = QMDPSolver(max_iterations=config.solver_config.QMDP_max_iterations)

    algorithms = [
        Algorithm(solver_name="NeverTreat_Policy"),
        Algorithm(solver_name="AlwaysTreat_Policy"),
        Algorithm(solver_name="Random_Policy"),
        Algorithm(solver_name="Heuristic_Policy", heuristic_config=heuristic_config),
        Algorithm(solver=nus_sarsop_solver, solver_name="NUS_SARSOP_Policy"),
        Algorithm(solver=vi_solver, solver_name="VI_Policy"),
        Algorithm(solver=qmdp_solver, solver_name="QMDP_Policy"),
    ]
    return algorithms
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

    # main(first_step_flag=first_step_flag, log_space=log_space_flag, experiment_name=experiment_name_flag, mode=mode_flag)
    run_experiments(mode_flag)
end

# -------------------------
# Export main functions for use in notebooks/scripts
# -------------------------
# Main workflow functions
export main, run_experiments, setup_experiment_configs, define_algorithms

# Policy generation functions
export generate_mdp_pomdp_policies, create_pomdp_mdp, generate_policy

# Simulation functions
export simulate_policy, simulate_all_policies, create_sim_pomdp, initialize_belief

# Evaluation functions
export evaluate_simulation_results, extract_simulation_histories
export extract_reward_metrics, display_reward_metrics
export print_reward_metrics_for_vi_policy

# Plotting functions
export plot_plos_one_plots, plot_parallel_plots, plot_results
export plot_treatment_heatmap, plot_simulation_treatment_heatmap
export plot_beliefs_over_time, plot_beliefs_over_time_plotsjl
export plot_sealice_levels_over_time
export plot_policy_sealice_levels_over_time, plot_policy_treatment_cost_over_time

export plos_one_plot_kalman_filter_belief_trajectory
export plos_one_sealice_levels_over_time
export plos_one_combined_treatment_probability_over_time
export plos_one_sarsop_dominant_action
export plot_kalman_filter_trajectory_with_uncertainty
export plot_kalman_filter_belief_trajectory_two_panel
export plos_one_algo_sealice_levels_over_time
export plos_one_treatment_distribution_comparison
export plos_one_episode_sealice_levels_over_time

# Configuration types
export ExperimentConfig, SolverConfig, SimulationConfig, HeuristicConfig, Algorithm, LocationParams, get_location_params

# Utility functions
export predict_next_abundances, get_temperature

end # AquaOpt
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
include("Plotting/PlotUtils.jl")
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

# ----------------------------
# Main function
# ----------------------------
function main(;log_space=true, experiment_name="exp", mode="debug", location="south", ekf_filter=true, plot=true,
    reward_lambdas::Vector{Float64}=[0.46, 0.12, 0.12, 0.18, 0.12],
    sim_reward_lambdas::Vector{Float64}=[0.46, 0.12, 0.12, 0.18, 0.12],
    season_regulation_limits::Vector{Float64}=[0.2, 0.5, 0.5, 0.5],
    regulatory_violation_cost_MNOK::Float64=10.0,
    salmon_price_MNOK_per_tonne::Float64=0.07,
    welfare_cost_MNOK::Float64=1.0,
    chronic_lice_cost_MNOK::Float64=0.5)

    # Set up experiment config and log in experiments.csv file
    config = setup_experiment_configs(experiment_name, log_space, ekf_filter, mode, location;
        reward_lambdas=reward_lambdas, sim_reward_lambdas=sim_reward_lambdas,
        season_regulation_limits=season_regulation_limits,
        regulatory_violation_cost_MNOK=regulatory_violation_cost_MNOK,
        salmon_price_MNOK_per_tonne=salmon_price_MNOK_per_tonne,
        welfare_cost_MNOK=welfare_cost_MNOK,
        chronic_lice_cost_MNOK=chronic_lice_cost_MNOK)
    save_experiment_config(config)
    
    @info """\n
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                         NEW EXPERIMENT RUN                              ║
    ╠════════════════════════════════════════════════════════════════════════╣
    ║  Mode:            $mode
    ║  Log Space:       $log_space
    ║  EKF Filter:      $ekf_filter
    ║  Reward lambdas:  $reward_lambdas
    ║  Sim R lambdas:   $sim_reward_lambdas
    ║  Location:        $location
    ║  Reg limits:      $season_regulation_limits
    ║  Violation cost:  $(regulatory_violation_cost_MNOK) MNOK
    ║  Salmon price:    $(salmon_price_MNOK_per_tonne) MNOK/t
    ║  ExperimentDir: $(config.experiment_dir)
    ╚════════════════════════════════════════════════════════════════════════╝
    """

    # Define algorithms
    algorithms = define_algorithms(config)

    @info "Solving policies"
    all_policies = solve_policies(algorithms, config)

    @info "Simulating policies"
    parallel_data, sim_pomdp = simulate_all_policies(algorithms, config, all_policies)

    # Extract reward metrics
    processed_data = extract_reward_metrics(parallel_data, config, sim_pomdp)

    # Display reward metrics
    display_reward_metrics(processed_data, config, true, true)

    if config.simulation_config.high_fidelity_sim && plot
        plot_plos_one_plots(processed_data, config, algorithms)
    end

end

# ----------------------------
# Set up and save experiment configuration
# ----------------------------
function setup_experiment_configs(experiment_name, log_space, ekf_filter=true, mode="debug", location="south";
    reward_lambdas::Vector{Float64}=[1.0, 3.0, 0.5, 0.01, 0.0],
    sim_reward_lambdas::Vector{Float64}=[1.0, 3.0, 0.5, 0.01, 0.0],
    season_regulation_limits::Vector{Float64}=[0.2, 0.5, 0.5, 0.5],
    regulatory_violation_cost_MNOK::Float64=10.0,
    salmon_price_MNOK_per_tonne::Float64=0.07,
    welfare_cost_MNOK::Float64=1.0,
    chronic_lice_cost_MNOK::Float64=0.5)

    # Define experiment configuration
    exp_name = string(Dates.today(), "/", Dates.now(), "_", experiment_name, "_", mode, "_", location, "_", reward_lambdas)

    @info "Setting up experiment configuration for experiment: $exp_name"

    # Mode-specific overrides
    if mode == "debug"
        sarsop_time, vi_iters, qmdp_iters = 5.0, 10, 10
        n_episodes, n_steps = 100, 52
    elseif mode == "paper"
        sarsop_time, vi_iters, qmdp_iters = 3000.0, 800, 800
        n_episodes, n_steps = 1000, 100
    else
        error("Invalid mode: $mode. Must be 'debug' or 'paper'")
    end

    solver_cfg = SolverConfig(
        log_space=log_space,
        reward_lambdas=reward_lambdas,
        sarsop_max_time=sarsop_time,
        VI_max_iterations=vi_iters,
        QMDP_max_iterations=qmdp_iters,
        discount_factor=0.95,
        location=location,
        season_regulation_limits=season_regulation_limits,
        salmon_price_MNOK_per_tonne=salmon_price_MNOK_per_tonne,
        regulatory_violation_cost_MNOK=regulatory_violation_cost_MNOK,
        welfare_cost_MNOK=welfare_cost_MNOK,
        chronic_lice_cost_MNOK=chronic_lice_cost_MNOK,
        heuristic_belief_threshold_mechanical=0.45,
        heuristic_belief_threshold_chemical=0.4,
        heuristic_belief_threshold_thermal=0.475,
        full_observability_solver=false,
    )
    sim_cfg = SimulationConfig(
        num_episodes=n_episodes,
        steps_per_episode=n_steps,
        ekf_filter=ekf_filter,
        high_fidelity_sim=true,
        sim_reward_lambdas=sim_reward_lambdas,
    )
    config = ExperimentConfig(
        solver_config=solver_cfg,
        simulation_config=sim_cfg,
        experiment_name=exp_name,
    )
        
    return config

end

# ----------------------------
# Define algorithms
# ----------------------------
function define_algorithms(config)

    nus_sarsop_solver = SARSOP.SARSOPSolver(
        timeout=config.solver_config.sarsop_max_time,
        verbose=false,
        policy_filename=joinpath(config.policies_dir, "NUS_SARSOP_Policy/policy.out"),
        pomdp_filename=joinpath(config.experiment_dir, "pomdp_mdp/pomdp.pomdpx")
    )

    native_sarsop_solver = NativeSARSOP.SARSOPSolver(
        max_time=config.solver_config.sarsop_max_time,
        verbose=false,
    )

    vi_solver = ValueIterationSolver(max_iterations=config.solver_config.VI_max_iterations, belres=1e-10, verbose=false)

    qmdp_solver = QMDPSolver(max_iterations=config.solver_config.QMDP_max_iterations)

    algorithms = [
        Algorithm(solver_name="NeverTreat_Policy"),
        Algorithm(solver_name="AlwaysTreat_Policy"),
        Algorithm(solver_name="Random_Policy"),
        Algorithm(solver_name="Heuristic_Policy", solver_config=config.solver_config),
        # Algorithm(solver=nus_sarsop_solver, solver_name="NUS_SARSOP_Policy"),
        Algorithm(solver=native_sarsop_solver, solver_name="Native_SARSOP_Policy"),
        Algorithm(solver=vi_solver, solver_name="VI_Policy"),
        Algorithm(solver=qmdp_solver, solver_name="QMDP_Policy"),
    ]
    return algorithms
end


if abspath(PROGRAM_FILE) == @__FILE__

    log_space_flag = true
    experiment_name_flag = "exp"
    mode_flag = "debug"
    location_flag = "north"

    for arg in ARGS
        if occursin("--experiment_name=", arg)
            global experiment_name_flag = split(arg, "=")[2]
        elseif occursin("--mode=", arg)
            global mode_flag = split(arg, "=")[2]
        elseif occursin("--location=", arg)
            global location_flag = split(arg, "=")[2]
        elseif arg == "--raw_space"
            global log_space_flag = false
        end
    end

    main(log_space=log_space_flag, experiment_name=experiment_name_flag, mode=mode_flag, location=location_flag)
end

# -------------------------
# Export main functions for use in notebooks/scripts
# -------------------------
# Main workflow functions
export main, setup_experiment_configs, define_algorithms

# Policy generation functions
export solve_policies, create_pomdp_mdp, generate_policy

# Simulation functions
export simulate_all_policies, create_sim_pomdp, initialize_belief

# Evaluation functions
export extract_reward_metrics, display_reward_metrics

# Plotting functions
export plot_plos_one_plots
export plos_one_plot_kalman_filter_belief_trajectory
export plos_one_sealice_levels_over_time
export plos_one_reward_over_time, plos_one_biomass_loss_over_time
export plos_one_regulatory_penalty_over_time, plos_one_fish_disease_over_time
export plos_one_treatment_cost_over_time
export plos_one_combined_treatment_probability_over_time
export plos_one_sarsop_dominant_action
export plot_kalman_filter_trajectory_with_uncertainty
export plot_kalman_filter_belief_trajectory_two_panel
export plos_one_algo_sealice_levels_over_time
export plos_one_treatment_distribution_comparison
export plos_one_episode_sealice_levels_over_time

# Configuration types
export ExperimentConfig, SolverConfig, SimulationConfig, Algorithm, LocationParams, get_location_params

# Utility functions
export predict_next_abundances, get_temperature


end # AquaOpt

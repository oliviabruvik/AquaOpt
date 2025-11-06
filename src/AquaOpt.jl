module AquaOpt

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
include("Models/SeaLicePOMDP.jl")
include("Plotting/Heatmaps.jl")
include("Plotting/Timeseries.jl")
include("Plotting/Comparison.jl")
include("Plotting/ParallelPlots.jl")
include("Plotting/PlosOnePlots.jl")
include("Utils/Config.jl")
include("Utils/ExperimentTracking.jl")
include("Plotting/Plots.jl")

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
using JLD2
using Profile
using ProfileView
using PProf

plotlyjs()  # Activate Plotly backend

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
function main(;first_step_flag="solve", log_space=true, experiment_name="exp", mode="light", ekf_filter=true)

    @info "Running experiment: $experiment_name, log_space: $log_space, ekf_filter: $ekf_filter, mode: $mode"
    config, heuristic_config = setup_experiment_configs(experiment_name, log_space, ekf_filter, mode)
    algorithms = define_algorithms(config, heuristic_config)

    @info "Solving policies"
    for algo in algorithms
        generate_mdp_pomdp_policies(algo, config)
    end

    @info "Simulating policies"
    parallel_data = simulate_all_policies(algorithms, config)

    # If we are simulating on high fidelity model, we want to evaluate the simulation results
    if config.high_fidelity_sim
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
    elseif mode == "VIdebug"
        config = ExperimentConfig(
            num_episodes=1000,
            steps_per_episode=104,
            log_space=log_space,
            ekf_filter=ekf_filter,
            experiment_name=exp_name,
            verbose=false,
            step_through=false,
            reward_lambdas=[0.7, 0.2, 0.1, 0.1, 0.8], # [treatment, regulatory, biomass, health, sea lice]
            sarsop_max_time=500.0,
            VI_max_iterations=100,
            QMDP_max_iterations=100,
            discount_factor = 0.95,
            high_fidelity_sim = false, # Toggles whether we simulate policies on sim (true) or solver (false) POMDP
            full_observability_solver = false, # Toggles whether we have full observability in the observation function or not (false). Pairs with high_fidelity_sim = false.
        )
    elseif mode == "debug"
        config = ExperimentConfig(
            num_episodes=10,
            steps_per_episode=20,
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
            location = "north", # "north", "west", or "south"
            high_fidelity_sim = true, # Toggles whether we simulate policies on sim (true) or solver (false) POMDP
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
            num_episodes=1000,
            steps_per_episode=104,
            log_space=log_space,
            ekf_filter=ekf_filter,
            experiment_name=exp_name,
            verbose=false,
            step_through=false,
            reward_lambdas=[0.7, 0.2, 0.1, 0.1, 0.8], # [treatment, regulatory, biomass, health, sea lice]
            sarsop_max_time=1000.0,
            VI_max_iterations=500,
            QMDP_max_iterations=500,
            discount_factor = 0.95,
            location = "south",
            high_fidelity_sim = false,
        )
    end
        
    heuristic_config = HeuristicConfig(
        raw_space_threshold=config.heuristic_threshold,
        belief_threshold_mechanical=config.heuristic_belief_threshold_mechanical,
        belief_threshold_thermal=config.heuristic_belief_threshold_thermal,
        rho=config.heuristic_rho
    )

    return config, heuristic_config

end

# ----------------------------
# Define algorithms
# ----------------------------
function define_algorithms(config, heuristic_config)

    native_sarsop_solver = NativeSARSOP.SARSOPSolver(max_time=config.sarsop_max_time) #, verbose=false)
    
    nus_sarsop_solver = SARSOP.SARSOPSolver(
        timeout=config.sarsop_max_time,
        verbose=false,
        policy_filename=joinpath(config.policies_dir, "NUS_SARSOP_Policy/policy.out"),
        pomdp_filename=joinpath(config.experiment_dir, "pomdp_mdp/pomdp.pomdpx")
    )
    
    vi_solver = ValueIterationSolver(max_iterations=config.VI_max_iterations, belres=1e-10, verbose=false)

    qmdp_solver = QMDPSolver(max_iterations=config.QMDP_max_iterations)

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

    @info "Running with mode: $mode_flag, log_space: $log_space_flag, experiment_name: $experiment_name_flag"

    # @profview main(first_step_flag=first_step_flag, log_space=log_space_flag, experiment_name=experiment_name_flag, mode=mode_flag)
    @time main(first_step_flag=first_step_flag, log_space=log_space_flag, experiment_name=experiment_name_flag, mode=mode_flag)

    @time run_experiments(mode_flag)
    # Print text summary of hotspots
    @info "\n=== Profile Summary (Top Functions by Time) ==="
    #Profile.print(format=:flat, sortedby=:count, maxdepth=20, mincount=10)

end
end # AquaOpt
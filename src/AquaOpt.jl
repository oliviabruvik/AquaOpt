include("Algorithms/Evaluation.jl")
include("Algorithms/Policies.jl")
include("Algorithms/Simulation.jl")
include("Data/Cleaning.jl")
include("Models/SeaLicePOMDP.jl")
include("Plotting/Heatmaps.jl")
include("Plotting/Timeseries.jl")
include("Plotting/Comparison.jl")
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
# Solve policies
# ----------------------------
function solve_policies(algorithms, config)
    @info "Solving policies"
    for algo in algorithms
        println("Solving $(algo.solver_name)")
        generate_mdp_pomdp_policies(algo, config)
    end
end

# ----------------------------
# Simulate policies
# ----------------------------
function simulate_policies(algorithms, config)
    @info "Simulating policies"
    for algo in algorithms
        println("Simulating $(algo.solver_name)")
        histories = simulate_policy(algo, config)
        avg_results = evaluate_simulation_results(config, algo, histories)
    end
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

        # Load avg results
        avg_results_path = joinpath(config.results_dir, "$(algo.solver_name)_avg_results.jld2")
        if !isfile(avg_results_path)
            @error "Avg results file not found at $avg_results_path. Run simulation first. with --simulate"
            continue
        end
        @load avg_results_path avg_results

        # Plot policy cost vs sealice
        plot_policy_cost_vs_sealice(histories, avg_results, algo.solver_name, config)

        # Plot policy belief levels
        plot_policy_belief_levels(histories, algo.solver_name, config, 0.6)

        # Plot treatment heatmap
        plot_treatment_heatmap(algo, config)

        # Plot simulation treatment heatmap
        plot_simulation_treatment_heatmap(algo, config; use_observations=false, n_bins=50)

    end
    
    # Plot comparison plots
    plot_all_cost_vs_sealice(config)
    plot_policy_sealice_levels_over_lambdas(config)
    plot_policy_treatment_cost_over_lambdas(config)
    plot_policy_sealice_levels_over_time(config, 0.6)
    plot_policy_treatment_cost_over_time(config, 0.6)
    plot_policy_reward_over_lambdas(config)

    # Generate Pareto frontier
    plot_pareto_frontier(config)

    @info "Saved all plots to $(config.figures_dir)"
end


# ----------------------------
# Main function
# ----------------------------
function main(;solve=true, simulate=true, plot=true, log_space=true, experiment_name="exp", skew=false, mode="light")

    config, heuristic_config = setup_experiment_configs(experiment_name, log_space, skew, mode)
    algorithms = define_algorithms(config, heuristic_config)

    if solve
        solve_policies(algorithms, config)
    # elseif simulate
    #     @info "Skipping policy solving"
    #     # If we skip solving but want to simulate, we need to find the latest 
    #     # experiment with the same policy parameters and use that experiment's config
    #     latest_experiment = find_latest_experiment(config, heuristic_config)
    #     @info "Latest experiment: $latest_experiment"
    #     config, heuristic_config = load_experiment_config(latest_experiment)
    # elseif plot
    #     @info "Skipping policy solving and simulation"
    #     # If we skip solving and simulation but want to plot, we need to find the latest 
    #     # experiment with the same policy and simulation parameters and use that experiment's config
    #     latest_experiment = find_latest_policy_config(config, heuristic_config)
    #     @info "Latest experiment: $latest_experiment"
    #     config, heuristic_config = load_experiment_config(latest_experiment)
    #     # Load the policies
    #     load_policies(algorithms, config)
    end

    if simulate
        simulate_policies(algorithms, config)
    end

    if plot
        plot_results(algorithms, config)
    end

end


# ----------------------------
# Set up and save experiment configuration
# ----------------------------
function setup_experiment_configs(experiment_name, log_space, skew=false, mode="light")

    # Define experiment configuration
    exp_name = string(Dates.today(), "/", Dates.now(), "_", experiment_name, "_mode_", mode)

    @info "Setting up experiment configuration for experiment: $exp_name"

    if mode == "light"
        config = ExperimentConfig(
            num_episodes=10,
            steps_per_episode=52,
            log_space=log_space,
            skew=skew,
            experiment_name=exp_name,
            verbose=false,
            step_through=false,
            lambda_values=[0.4, 0.6],
            sarsop_max_time=5.0,
            VI_max_iterations=10,
            QMDP_max_iterations=10,
        )
    elseif mode == "debug"
        config = ExperimentConfig(
            num_episodes=1,
            steps_per_episode=20,
            log_space=log_space,
            skew=skew,
            experiment_name=exp_name,
            verbose=true,
            step_through=true,
            lambda_values=[0.2, 0.4, 0.6, 0.8],
            sarsop_max_time=5.0,
            VI_max_iterations=10,
            QMDP_max_iterations=10,
        )
    elseif mode == "full"
        config = ExperimentConfig(
            num_episodes=10,
            steps_per_episode=52,
            log_space=log_space,
            skew=skew,
            experiment_name=exp_name,
            verbose=false,
            step_through=false,
        )
    end
        
    heuristic_config = HeuristicConfig(
        raw_space_threshold=config.heuristic_threshold,
        belief_threshold=config.heuristic_belief_threshold,
        rho=config.heuristic_rho
    )

    # Write config to file
    mkpath(joinpath(config.experiment_dir, "config"))
    @save joinpath(config.experiment_dir, "config", "experiment_config.jld2") config
    open(joinpath(config.experiment_dir, "config", "experiment_config.txt"), "w") do io
        for field in fieldnames(typeof(config))
            value = getfield(config, field)
            println(io, "$field: $value")
        end
    end

    # Log experiment configuration
    save_experiment_config(config, heuristic_config)

    return config, heuristic_config

end

# ----------------------------
# Define algorithms
# ----------------------------
function define_algorithms(config, heuristic_config)

    @info "Defining solvers"
    native_sarsop_solver = NativeSARSOP.SARSOPSolver(max_time=config.sarsop_max_time, verbose=false)
    nus_sarsop_solver = SARSOP.SARSOPSolver(
        timeout=config.sarsop_max_time,
        verbose=false,
        policy_filename=joinpath(config.policies_dir, "NUS_SARSOP_Policy/policy.out"),
        pomdp_filename=joinpath(config.experiment_dir, "pomdp_mdp/pomdp.pomdpx")
    )
    vi_solver = ValueIterationSolver(max_iterations=config.VI_max_iterations)
    qmdp_solver = QMDPSolver(max_iterations=config.QMDP_max_iterations)

    @info "Defining algorithms"
    algorithms = [
        Algorithm(solver_name="NeverTreat_Policy"),
        Algorithm(solver_name="AlwaysTreat_Policy"),
        Algorithm(solver_name="Random_Policy"),
        Algorithm(solver_name="Heuristic_Policy", heuristic_config=heuristic_config),
        Algorithm(solver=native_sarsop_solver, solver_name="SARSOP_Policy"),
        Algorithm(solver=nus_sarsop_solver, solver_name="NUS_SARSOP_Policy"),
        Algorithm(solver=vi_solver, solver_name="VI_Policy"),
        Algorithm(solver=qmdp_solver, solver_name="QMDP_Policy"),
    ]
    return algorithms
end


if abspath(PROGRAM_FILE) == @__FILE__

    solve_flag = true
    simulate_flag = true
    plot_flag = true
    log_space_flag = true
    skew_flag = false
    experiment_name_flag = "exp"
    mode_flag = "light"

    for arg in ARGS
        if occursin("--experiment_name=", arg)
            global experiment_name_flag = split(arg, "=")[2]
        elseif occursin("--mode=", arg)
            global mode_flag = split(arg, "=")[2]
        elseif arg == "--no-solve"
            global solve_flag = false
        elseif arg == "--no-simulate"
            global simulate_flag = false
        elseif arg == "--no-plot"
            global plot_flag = false
        elseif arg == "--raw_space"
            global log_space_flag = false
        end
    end

    @info "Running with mode: $mode_flag, log_space: $log_space_flag, skew: $skew_flag, experiment_name: $experiment_name_flag"

    main(solve=solve_flag, simulate=simulate_flag, plot=plot_flag, log_space=log_space_flag, experiment_name=experiment_name_flag, mode=mode_flag, skew=skew_flag)
end
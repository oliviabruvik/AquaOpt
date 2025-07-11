include("Algorithms/Evaluation.jl")
include("Algorithms/Policies.jl")
include("Algorithms/Simulation.jl")
include("Data/Cleaning.jl")
include("Models/SeaLicePOMDP.jl")
include("Plotting/Heatmaps.jl")
include("Plotting/Timeseries.jl")
include("Plotting/Comparison.jl")
include("Utils/Config.jl")

# Environment variables
ENV["PLOTS_BROWSER"] = "true"
ENV["PLOTS_BACKEND"] = "plotlyjs"

# Import required packages
using Logging
using DiscreteValueIteration
using GridInterpolations
using NativeSARSOP: SARSOPSolver
using POMDPs
using POMDPTools
using Plots: plot, plot!, scatter, scatter!, heatmap, heatmap!, histogram, histogram!, savefig
using LocalFunctionApproximation
using LocalApproximationValueIteration
using Dates

plotlyjs()  # Activate Plotly backend

# ----------------------------
# Main function
# ----------------------------
function main(;run_algorithms=true, log_space=true, experiment_name="exp")

    EXPERIMENT_CONFIG, HEURISTIC_CONFIG = setup_configs(experiment_name, log_space)

    # Define algorithms
    algorithms = [
        Algorithm(solver_name="NoTreatment_Policy"),
        Algorithm(solver_name="Random_Policy"),
        Algorithm(solver_name="Heuristic_Policy", heuristic_config=HEURISTIC_CONFIG),
        Algorithm(solver=ValueIterationSolver(max_iterations=EXPERIMENT_CONFIG.VI_max_iterations), solver_name="VI_Policy"),
        # time out
        Algorithm(solver=SARSOPSolver(max_time=EXPERIMENT_CONFIG.sarsop_max_time, verbose=true), solver_name="SARSOP_Policy"),
        Algorithm(solver=QMDPSolver(max_iterations=EXPERIMENT_CONFIG.QMDP_max_iterations), solver_name="QMDP_Policy")
    ]

    # Solve POMDPs and simulate policies
    if run_algorithms
        all_results = solve_and_simulate_algorithms(algorithms, EXPERIMENT_CONFIG)
    else
        @info "Skipping algorithms"
        all_results = nothing
    end

    # Plot results
    plot_results(all_results, algorithms, EXPERIMENT_CONFIG)

end

# ----------------------------
# Solve and simulate algorithms
# ----------------------------
function solve_and_simulate_algorithms(algorithms, EXPERIMENT_CONFIG)

    all_results = Dict{String, DataFrame}()
    for algo in algorithms
        @info "Running $(algo.solver_name)"
        results = test_optimizer(algo, EXPERIMENT_CONFIG)
        all_results[algo.solver_name] = results
    end

    # Save all results
    mkpath(joinpath(EXPERIMENT_CONFIG.data_dir, "avg_results"))
    @save joinpath(EXPERIMENT_CONFIG.data_dir, "avg_results", "All_policies_all_results.jld2") all_results

    return all_results

end

# ----------------------------
# Plot results
# ----------------------------
function plot_results(all_results, algorithms, EXPERIMENT_CONFIG)

    @info "Generating result plots"

    if all_results == nothing
        all_results = load_results(EXPERIMENT_CONFIG)
    end

    # Plot individual policy plots
    for algo in algorithms
        results = all_results[algo.solver_name]

        # Plot policy cost vs sealice
        plot_policy_cost_vs_sealice(results, algo.solver_name, EXPERIMENT_CONFIG)

        # Plot policy belief levels
        plot_policy_belief_levels(results, algo.solver_name, EXPERIMENT_CONFIG, 0.6)

        # Plot treatment heatmap
        # plot_treatment_heatmap(algo, EXPERIMENT_CONFIG)

        # Plot simulation treatment heatmap
        # plot_simulation_treatment_heatmap(algo, EXPERIMENT_CONFIG; use_observations=false, n_bins=50)

    end
    
    # Plot comparison plots
    plot_all_cost_vs_sealice(EXPERIMENT_CONFIG)
    plot_policy_sealice_levels_over_lambdas(EXPERIMENT_CONFIG)
    plot_policy_treatment_cost_over_lambdas(EXPERIMENT_CONFIG)
    plot_policy_sealice_levels_over_time(EXPERIMENT_CONFIG, 0.6)
    plot_policy_treatment_cost_over_time(EXPERIMENT_CONFIG, 0.6)
    plot_policy_reward_over_lambdas(EXPERIMENT_CONFIG)

    # Generate Pareto frontier
    plot_pareto_frontier(EXPERIMENT_CONFIG)

end

# ----------------------------
# Load all results from file
# ----------------------------
function load_results(EXPERIMENT_CONFIG)

    results_file_path = joinpath(EXPERIMENT_CONFIG.data_dir, "avg_results", "All_policies_all_results.jld2")

    if isfile(results_file_path)
        @load results_file_path all_results
    else
        @info "Results file not found at $results_file_path, running algorithms and simulations"
        solve_and_simulate_algorithms(algorithms, EXPERIMENT_CONFIG)
        @load results_file_path all_results
    end

    return all_results
end

# ----------------------------
# Set up and save experiment configuration
# ----------------------------
function setup_configs(experiment_name, log_space)

    # Define experiment configuration
    exp_name = experiment_name * "_" * string(log_space) * "_log_space" * "_" * string(Dates.now())
    EXPERIMENT_CONFIG = ExperimentConfig(
        num_episodes=1, #10,
        steps_per_episode=52,
        log_space=log_space,
        experiment_name=exp_name,
    )

    # Define heuristic configuration
    HEURISTIC_CONFIG = HeuristicConfig(
        raw_space_threshold=EXPERIMENT_CONFIG.heuristic_threshold,
        belief_threshold=EXPERIMENT_CONFIG.heuristic_belief_threshold,
        rho=EXPERIMENT_CONFIG.heuristic_rho
    )

    # Write EXPERIMENT_CONFIG to file
    mkpath(joinpath(EXPERIMENT_CONFIG.data_dir, "config"))
    @save joinpath(EXPERIMENT_CONFIG.data_dir, "config", "experiment_config.jld2") EXPERIMENT_CONFIG
    open(joinpath(EXPERIMENT_CONFIG.data_dir, "config", "experiment_config.txt"), "w") do io
        for field in fieldnames(typeof(EXPERIMENT_CONFIG))
            value = getfield(EXPERIMENT_CONFIG, field)
            println(io, "$field: $value")
        end
    end

    return EXPERIMENT_CONFIG, HEURISTIC_CONFIG

end

if abspath(PROGRAM_FILE) == @__FILE__

    run_algorithms_flag = true
    log_space_flag = true
    experiment_name_flag = "exp"

    for arg in ARGS
        if occursin("--experiment_name=", arg)
            global experiment_name_flag = split(arg, "=")[2]
        elseif arg == "--no-run_algorithms"
            global run_algorithms_flag = false
        elseif arg == "--no-log_space"
            global log_space_flag = false
        end
    end

    main(run_algorithms=run_algorithms_flag, log_space=log_space_flag, experiment_name=experiment_name_flag)
end
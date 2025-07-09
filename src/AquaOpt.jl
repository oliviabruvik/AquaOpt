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

plotlyjs()  # Activate Plotly backend

# ----------------------------
# Main function
# ----------------------------
function main(;run_algorithms=true, run_plots=true, log_space=true)

    @info "Loading and cleaning data"
    df = CSV.read(joinpath("data", "processed", "sealice_data.csv"), DataFrame)

    # Define experiment configuration
    EXPERIMENT_CONFIG = ExperimentConfig(num_episodes=3, steps_per_episode=52, log_space=log_space)

    # Define POMDP configuration
    POMDP_CONFIG = POMDPConfig(
        log_space=EXPERIMENT_CONFIG.log_space,
        costOfTreatment=EXPERIMENT_CONFIG.costOfTreatment,
        growthRate=EXPERIMENT_CONFIG.growthRate,
        rho=EXPERIMENT_CONFIG.rho,
        discount_factor=EXPERIMENT_CONFIG.discount_factor
    )

    # Define heuristic configuration
    HEURISTIC_CONFIG = HeuristicConfig(
        raw_space_threshold=EXPERIMENT_CONFIG.heuristic_threshold,
        belief_threshold=EXPERIMENT_CONFIG.heuristic_belief_threshold,
        rho=EXPERIMENT_CONFIG.heuristic_rho
    )

    # Print EXPERIMENT_CONFIG, POMDP_CONFIG, and HEURISTIC_CONFIG values
    show(EXPERIMENT_CONFIG)
    show(POMDP_CONFIG)
    show(HEURISTIC_CONFIG)

    # Define algorithms
    algorithms = [
        Algorithm(solver_name="NoTreatment_Policy"),
        Algorithm(solver_name="Random_Policy"),
        Algorithm(solver_name="Heuristic_Policy", heuristic_config=HEURISTIC_CONFIG),
        Algorithm(solver=ValueIterationSolver(max_iterations=30), solver_name="VI_Policy"),
        Algorithm(solver=SARSOPSolver(max_time=10.0), solver_name="SARSOP_Policy"),
        Algorithm(solver=QMDPSolver(max_iterations=30), solver_name="QMDP_Policy")
    ]

    if run_algorithms
        solve_and_simulate_algorithms(algorithms, EXPERIMENT_CONFIG, POMDP_CONFIG)
    end

    if run_plots
        plot_results(algorithms, EXPERIMENT_CONFIG, POMDP_CONFIG)
    end

end

function solve_and_simulate_algorithms(algorithms, EXPERIMENT_CONFIG, POMDP_CONFIG)

    all_results = Dict{String, DataFrame}()
    for algo in algorithms
        @info "Running $(algo.solver_name)"
        results = test_optimizer(algo, EXPERIMENT_CONFIG, POMDP_CONFIG)
        all_results[algo.solver_name] = results
    end

    # Save all results
    mkpath(joinpath(EXPERIMENT_CONFIG.data_dir, "avg_results", "All_policies"))
    @save joinpath(EXPERIMENT_CONFIG.data_dir, "avg_results", "All_policies", "all_results_$(POMDP_CONFIG.log_space)_log_space_$(EXPERIMENT_CONFIG.num_episodes)_episodes_$(EXPERIMENT_CONFIG.steps_per_episode)_steps.jld2") all_results

end

function plot_results(algorithms, EXPERIMENT_CONFIG, POMDP_CONFIG)

    @info "Generating result plots"

    # Load all results but check if file exists first
    results_file_path = joinpath(EXPERIMENT_CONFIG.data_dir, "avg_results", "All_policies", "all_results_$(POMDP_CONFIG.log_space)_log_space_$(EXPERIMENT_CONFIG.num_episodes)_episodes_$(EXPERIMENT_CONFIG.steps_per_episode)_steps.jld2")
    if isfile(results_file_path)
        @load results_file_path all_results
    else
        @info "Results file not found at $results_file_path, running algorithms and simulations"
        solve_and_simulate_algorithms(algorithms, EXPERIMENT_CONFIG, POMDP_CONFIG)
        @load results_file_path all_results
    end
    
    # Plot individual policy plots
    for algo in algorithms
        results = all_results[algo.solver_name]

        # Plot policy cost vs sealice
        plot_policy_cost_vs_sealice(results, algo.solver_name, EXPERIMENT_CONFIG, POMDP_CONFIG)

        # Plot policy belief levels
        plot_policy_belief_levels(results, algo.solver_name, EXPERIMENT_CONFIG, POMDP_CONFIG, 0.6)

        # Plot treatment heatmap
        plot_treatment_heatmap(algo, EXPERIMENT_CONFIG, POMDP_CONFIG)

        # Plot simulation treatment heatmap
        plot_simulation_treatment_heatmap(algo, EXPERIMENT_CONFIG, POMDP_CONFIG; use_observations=false, n_bins=50)

    end
    
    # Plot comparison plots
    plot_all_cost_vs_sealice(EXPERIMENT_CONFIG, POMDP_CONFIG)
    plot_policy_sealice_levels_over_lambdas(EXPERIMENT_CONFIG, POMDP_CONFIG)
    plot_policy_treatment_cost_over_lambdas(EXPERIMENT_CONFIG, POMDP_CONFIG)
    plot_policy_sealice_levels_over_time(EXPERIMENT_CONFIG, POMDP_CONFIG, 0.6)
    plot_policy_treatment_cost_over_time(EXPERIMENT_CONFIG, POMDP_CONFIG, 0.6)
    plot_policy_reward_over_lambdas(EXPERIMENT_CONFIG, POMDP_CONFIG)

    # Generate Pareto frontier
    plot_pareto_frontier(EXPERIMENT_CONFIG, POMDP_CONFIG)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main(run_algorithms=true, run_plots=true, log_space=true)
end
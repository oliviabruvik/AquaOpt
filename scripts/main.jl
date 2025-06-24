include("../src/cleaning.jl")
include("../src/SeaLicePOMDP.jl")
include("../src/plot_views.jl")
include("../src/optimization.jl")

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
using Plots
using LocalFunctionApproximation
using LocalApproximationValueIteration

plotlyjs()  # Activate Plotly backend

# TODO: STD should be decreasing over time
# TODO: belief plot diff lambda

# ----------------------------
# Main function
# ----------------------------
function main(;run_algorithms=true, run_plots=true, log_space=true)

    @info "Loading and cleaning data"
    df = CSV.read(joinpath("data", "processed", "sealice_data.csv"), DataFrame)

    CONFIG = Config(num_episodes=1000, steps_per_episode=52)
    POMDP_CONFIG = POMDPConfig(log_space=log_space)

    algorithms = [
        Algorithm(solver_name="Random_Policy"),
        Algorithm(heuristic_threshold=CONFIG.heuristic_threshold, heuristic_belief_threshold=CONFIG.heuristic_belief_threshold),
        Algorithm(solver=ValueIterationSolver(max_iterations=30), convert_to_mdp=true, solver_name="VI_Policy"),
        Algorithm(solver=SARSOPSolver(max_time=10.0), solver_name="SARSOP_Policy"),
        Algorithm(solver=QMDPSolver(max_iterations=30), solver_name="QMDP_Policy")
    ]

    if run_algorithms
        solve_and_simulate_algorithms(algorithms, CONFIG, POMDP_CONFIG)
    end

    if run_plots
        plot_results(algorithms, CONFIG, POMDP_CONFIG)
    end

end

function solve_and_simulate_algorithms(algorithms, CONFIG, POMDP_CONFIG)

    all_results = Dict{String, DataFrame}()
    for algo in algorithms
        @info "Running $(algo.solver_name)"
        results = test_optimizer(algo, CONFIG, POMDP_CONFIG)
        all_results[algo.solver_name] = results
    end

    # Save all results
    mkpath(joinpath(CONFIG.data_dir, "avg_results", "All_policies"))
    @save joinpath(CONFIG.data_dir, "avg_results", "All_policies", "all_results_$(POMDP_CONFIG.log_space)_log_space_$(CONFIG.num_episodes)_episodes_$(CONFIG.steps_per_episode)_steps.jld2") all_results

end

function plot_results(algorithms, CONFIG, POMDP_CONFIG)

    @info "Generating result plots"

    # Load all results but check if file exists first
    results_file_path = joinpath(CONFIG.data_dir, "avg_results", "All_policies", "all_results_$(POMDP_CONFIG.log_space)_log_space_$(CONFIG.num_episodes)_episodes_$(CONFIG.steps_per_episode)_steps.jld2")
    if isfile(results_file_path)
        @load results_file_path all_results
    else
        @info "Results file not found at $results_file_path, running algorithms and simulations"
        solve_and_simulate_algorithms(algorithms, CONFIG, POMDP_CONFIG)
        @load results_file_path all_results
    end
    
    # Plot individual policy plots
    for algo in algorithms
        results = all_results[algo.solver_name]

        # Plot policy cost vs sealice
        plot_policy_cost_vs_sealice(results, algo.solver_name, CONFIG, POMDP_CONFIG)

        # Plot policy belief levels
        plot_policy_belief_levels(results, algo.solver_name, CONFIG, POMDP_CONFIG, 0.6)

        # Plot treatment heatmap
        plot_treatment_heatmap(algo, CONFIG, POMDP_CONFIG)

        # Plot simulation treatment heatmap
        plot_simulation_treatment_heatmap(algo, CONFIG, POMDP_CONFIG; use_observations=false, n_bins=50)

        # if algo.solver_name == "VI_Policy"  # TODO: Make this more general
        #     plot_value_iteration_convergence(algo, CONFIG, POMDP_CONFIG, 0.6)
        # else
        #     plot_simulation_convergence(algo, CONFIG, POMDP_CONFIG, 0.6; window_size=5)
        # end
    end
    

    # Plot comparison plots
    plot_all_cost_vs_sealice(CONFIG, POMDP_CONFIG)
    plot_policy_sealice_levels_over_lambdas(CONFIG, POMDP_CONFIG)
    plot_policy_treatment_cost_over_lambdas(CONFIG, POMDP_CONFIG)
    plot_policy_sealice_levels_over_time(CONFIG, POMDP_CONFIG, 0.6)
    plot_policy_treatment_cost_over_time(CONFIG, POMDP_CONFIG, 0.6)
    plot_policy_reward_over_lambdas(CONFIG, POMDP_CONFIG)

    mkpath(joinpath(CONFIG.figures_dir, "research_plots"))

    # Generate Pareto frontier
    plot_pareto_frontier(CONFIG, POMDP_CONFIG)

    
    
    # Run sensitivity analysis for key parameters
    param_values = [0.5, 0.7, 0.9]  # Example values for treatment effectiveness
    # plot_sensitivity_analysis(config, pomdp_config, "treatment_effectiveness", param_values)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main(run_algorithms=false, run_plots=true, log_space=true)
end
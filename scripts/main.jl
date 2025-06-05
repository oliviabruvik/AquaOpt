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
function main(run_algorithms=true)

    @info "Loading and cleaning data"
    df = CSV.read(joinpath("data", "processed", "sealice_data.csv"), DataFrame)

    CONFIG = Config()
    POMDP_CONFIG = POMDPConfig()

    if run_algorithms
        algorithms = [
            Algorithm(heuristic_threshold=CONFIG.heuristic_threshold, heuristic_belief_threshold=CONFIG.heuristic_belief_threshold),
            Algorithm(solver=ValueIterationSolver(max_iterations=30), convert_to_mdp=true, solver_name="VI_Policy"),
            Algorithm(solver=SARSOPSolver(max_time=10.0), solver_name="SARSOP_Policy"),
            Algorithm(solver=QMDPSolver(max_iterations=30), solver_name="QMDP_Policy")
        ]

        for algo in algorithms
            @info "Running $(algo.solver_name)"
            results = test_optimizer(algo, CONFIG, POMDP_CONFIG)
            CSV.write(joinpath(CONFIG.data_dir, "$(algo.solver_name)/results.csv"), results)
        end
    end

    @info "Generating result plots"
    overlay_plot = plot_mdp_results_overlay(CONFIG.num_episodes, CONFIG.steps_per_episode)
    comparison_plot = plot_policy_sealice_levels(CONFIG.num_episodes, CONFIG.steps_per_episode)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main("--run" in ARGS)
end
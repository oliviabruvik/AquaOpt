include("../src/cleaning.jl")
include("../src/optimize_mdp.jl")
include("../src/plot_views.jl")
include("../src/optimization.jl")

# Environment variables
ENV["PLOTS_BROWSER"] = "true"
ENV["PLOTS_BACKEND"] = "plotlyjs"

# Import required packages
using Logging
using DiscreteValueIteration
using NativeSARSOP: SARSOPSolver
using POMDPs
using Plots

plotlyjs()  # Activate Plotly backend

# ----------------------------
# Configuration
# ----------------------------
const CONFIG = Dict(
    :lambda_values => 0.0:0.2:1.0, # 0.0:0.05:1.0
    :num_episodes => 100, # 1000,
    :steps_per_episode => 50 # 100
)

# Create results directories
mkpath("results/figures")
mkpath("results/data")

# ----------------------------
# Main function
# ----------------------------
function main(run_algorithms=false)

    @info "Loading and cleaning data"
    df = load_and_clean("data/raw/licedata.csv")
    sealice_levels_over_time_plot = plot_sealice_levels_over_time(df)

    if run_algorithms
        policies = [
            ("Heuristic Policy", ValueIterationSolver(max_iterations=30), true),
            ("VI Policy", ValueIterationSolver(max_iterations=30), true),
            ("SARSOP Policy", SARSOPSolver(max_time=10.0), false),
            ("QMDP Policy", QMDPSolver(max_iterations=30), false)
        ]

        for (policy_name, solver, convert_to_mdp) in policies
            @info "Running $policy_name"
            test_optimizer(
                CONFIG[:lambda_values],
                solver,
                episodes=CONFIG[:num_episodes],
                steps_per_episode=CONFIG[:steps_per_episode],
                convert_to_mdp=convert_to_mdp,
                plot_name=policy_name
            )
        end
    end

    @info "Generating result plots"
    overlay_plot = plot_mdp_results_overlay(CONFIG[:num_episodes], CONFIG[:steps_per_episode])
    comparison_plot = plot_policy_sealice_levels(CONFIG[:num_episodes], CONFIG[:steps_per_episode])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main("--run" in ARGS)
end
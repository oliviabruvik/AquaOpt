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
using Plots
using LocalFunctionApproximation
using LocalApproximationValueIteration

plotlyjs()  # Activate Plotly backend

# ----------------------------
# Configuration
# ----------------------------
struct Config
    lambda_values::Vector{Float64} # 0.0:0.05:1.0
    num_episodes::Int # 1000
    steps_per_episode::Int # 100
    policies_dir::String
    figures_dir::String
    data_dir::String
end

const CONFIG = Config(
    collect(0.0:0.2:1.0), # 0.0:0.05:1.0
    1000, #10, # 1000
    50, #5,  # 100
    joinpath("results", "policies"),
    joinpath("results", "figures"),
    joinpath("results", "data")
)

# Create results directories
mkpath(CONFIG.policies_dir)
mkpath(CONFIG.figures_dir)
mkpath(CONFIG.data_dir)

# Create interpolation points for sea lice levels (0.0 to 10.0)
grid = RectangleGrid(range(0.0, 10.0, length=100))
interp = LocalGIFunctionApproximator(grid)

# ----------------------------
# Main function
# ----------------------------
function main(run_algorithms=true)

    @info "Loading and cleaning data"
    # df = load_and_clean("data/raw/licedata.csv")
    df = CSV.read(joinpath("data", "processed", "sealice_data.csv"), DataFrame)
    # sealice_levels_over_time_plot = plot_sealice_levels_over_time(df)

    run_algorithms = true
    println("run_algorithms: $run_algorithms")
    if run_algorithms
        algorithms = [
            Algorithm(ValueIterationSolver(max_iterations=30), true, "Heuristic_Policy", 0.5),
            Algorithm(ValueIterationSolver(max_iterations=30), true, "VI_Policy", nothing),
            # Algorithm(LocalApproximationValueIterationSolver(interp, verbose=true, max_iterations=1000, is_mdp_generative=false), true, "VI_Policy", nothing),
            Algorithm(SARSOPSolver(max_time=10.0), false, "SARSOP_Policy", nothing),
            Algorithm(QMDPSolver(max_iterations=30), false, "QMDP_Policy", nothing)
        ]

        for algo in algorithms
            @info "Running $(algo.solver_name)"
            test_optimizer(algo, CONFIG)
        end
    end

    @info "Generating result plots"
    overlay_plot = plot_mdp_results_overlay(CONFIG.num_episodes, CONFIG.steps_per_episode)
    comparison_plot = plot_policy_sealice_levels(CONFIG.num_episodes, CONFIG.steps_per_episode)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main("--run" in ARGS)
end
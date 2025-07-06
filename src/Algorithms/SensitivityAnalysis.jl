using POMDPs
using POMDPModels
using POMDPTools
using DataFrames
using JLD2
using Plots
using CSV
using Statistics
using Parameters

include("Evaluation.jl")
include("Policies.jl")
include("Simulation.jl")
include("../Models/SeaLicePOMDP.jl")
include("../Models/SeaLiceLogPOMDP.jl")
include("../Models/SimulationPOMDP.jl")
include("../Models/SimulationLogPOMDP.jl")
include("../AquaOpt.jl")

# ----------------------------
# Sensitivity Analysis
# ----------------------------

@with_kw struct SensitivityConfig
    heuristic_thresholds::Vector{Float64} = collect(0.5:0.5:1.0) # collect(0.5:0.5:3.0)
    belief_thresholds::Vector{Float64} = collect(0.3:0.2:0.5) # collect(0.3:0.2:0.9)
    growth_rates::Vector{Float64} = collect(1.0:0.1:1.1) # collect(1.0:0.1:1.4)
    lambdas::Vector{Float64} = collect(0.3:0.1:0.4) # collect(0.3:0.1:0.7)
    process_noises::Vector{Float64} = collect(0.3:0.1:0.4) # collect(0.3:0.1:0.7)
    observation_noises::Vector{Float64} = collect(0.3:0.1:0.4)
    num_episodes::Int = 10
    steps_per_episode::Int = 10 # 52
    data_dir::String = "results/sensitivity_analysis"
end

# ----------------------------
# Sensitivity Analysis Function
# ----------------------------
function run_sensitivity_analysis(algorithms, config::SensitivityConfig, pomdp_config; log_space=true)

    mkpath(config.data_dir)

    # Initialize results dataframe
    sensitivity_results = DataFrame(
        algorithm=String[],
        log_space=Bool[],
        heuristic_threshold=Float64[],
        belief_threshold=Float64[],
        growth_rate=Float64[],
        lambda=Float64[],
        process_noise=Float64[],
        observation_noise=Float64[],
        avg_treatment_cost=Float64[],
        avg_sealice=Float64[],
        avg_reward=Float64[],
    )

    for algo in algorithms
        @info "Running sensitivity analysis for algorithm: $(algo.solver_name) in $(log_space ? "log" : "raw") space"
        
        for ht in config.heuristic_thresholds, bt in config.belief_thresholds, gr in config.growth_rates, λ in config.lambdas, pn in config.process_noises, on in config.observation_noises
            
            @info "Testing: ht=$ht, bt=$bt, gr=$gr, λ=$λ, pn=$pn, on=$on"
            
            # Update POMDP configuration
            pomdp_config = POMDPConfig(
                costOfTreatment=10.0,
                growthRate=gr,
                rho=0.7,
                discount_factor=0.95,
                log_space=log_space,
            )
            
            # Update algorithm configuration for Heuristic Policy
            algo_config = algo
            if algo.solver_name == "Heuristic_Policy"
                algo_config = Algorithm(
                    solver_name="Heuristic_Policy",
                    heuristic_threshold=ht,
                    heuristic_belief_threshold=bt
                )
            end

            # Update simulation configuration for noise parameters
            sim_config = Config(
                lambda_values=[λ],
                num_episodes=config.num_episodes,
                steps_per_episode=config.steps_per_episode,
                heuristic_threshold=ht,
                heuristic_belief_threshold=bt,
                process_noise=pn,
                observation_noise=on,
                data_dir=config.data_dir,
            )

            # Run simulation
            algo_results = test_optimizer(algo_config, sim_config, pomdp_config)

            # Extract results
            for row in eachrow(algo_results)
                push!(sensitivity_results, (
                    algo.solver_name,
                    log_space,
                    ht,
                    bt,
                    gr,
                    row.lambda,
                    pn,
                    on,
                    row.avg_treatment_cost,
                    row.avg_sealice,
                    row.avg_reward
                ))
            end
        end
    end

    # Save results
    results_file_path = joinpath(config.data_dir, "sensitivity_analysis_results_$(log_space ? true : false)_log_space.csv")
    CSV.write(results_file_path, sensitivity_results)
    @save joinpath(config.data_dir, "sensitivity_analysis_results_$(log_space ? true : false)_log_space.jld2") sensitivity_results

    return sensitivity_results

end

# ----------------------------
# Plotting Functions
# ----------------------------
function plot_sensitivity_results(config::SensitivityConfig; log_space::Bool)

    # Load results
    results_file_path = joinpath(config.data_dir, "sensitivity_analysis_results_$(log_space ? true : false)_log_space.jld2")
    if !isfile(results_file_path)
        @warn "Results file not found at $results_file_path. Run sensitivity analysis first."
        sensitivity_results = run_sensitivity_analysis(algorithms, config, pomdp_config, log_space=log_space)
    else
        @load results_file_path sensitivity_results
    end

    # Plot settings
    plotlyjs()

    # Plot sensitivity for each parameter
    parameters = [
        ("Heuristic Threshold", :heuristic_threshold),
        ("Belief Threshold", :belief_threshold),
        ("Growth Rate", :growth_rate),
        ("Lambda", :lambda),
        ("Process Noise", :process_noise),
        ("Observation Noise", :observation_noise)
    ]
    
    metrics = [
        ("Average Treatment Cost", :avg_treatment_cost),
        ("Average Sea Lice Level", :avg_sealice),
        ("Average Reward", :avg_reward)
    ]

    # Get unique algorithms and create colors
    algorithms = unique(sensitivity_results.algorithm)
    colors = [:steelblue, :coral, :forestgreen, :gold, :purple, :brown]
    algo_colors = Dict(algo => colors[i] for (i, algo) in enumerate(algorithms))

    # Create plots for each parameter
    for (i, (param_name, param_col)) in enumerate(parameters)
        
        # Create plot with three subplots (one for each metric)
        p = plot(
            layout=(1, 3),
            size=(1800, 500),
            title="$param_name Sensitivity Analysis ($(log_space ? "Log Space" : "Raw Space"))",
            titlefontsize=12
        )
        
        # Plot each metric as a subplot
        for (metric_idx, (metric_name, metric_col)) in enumerate(metrics)
            
            # Plot each algorithm on this subplot
            for algo in algorithms
                algo_results = sensitivity_results[sensitivity_results.algorithm .== algo, :]
                
                # Aggregate results by parameter value
                grouped = combine(
                    groupby(algo_results, Symbol(param_col)),
                    Symbol(metric_col) => mean => Symbol(metric_col),
                    Symbol(metric_col) => std => Symbol("$(metric_col)_std")
                )

                # Plot with same style as before, but add color and label
                plot!(
                    p,
                    grouped[!, Symbol(param_col)],
                    grouped[!, Symbol(metric_col)],
                    ribbon=grouped[!, Symbol("$(metric_col)_std")],
                    label=algo,
                    color=algo_colors[algo],
                    subplot=metric_idx,
                    xlabel=param_name,
                    ylabel=metric_name,
                    title=metric_name,
                    lw=2,
                    legend=:best,
                    legendfontsize=10
                )
            end
        end

        # Save plot
        plot_dir = joinpath(config.data_dir, "plots")
        mkpath(plot_dir)
        plot_file = joinpath(plot_dir, "$(param_col)_sensitivity_$(log_space ? "log_space" : "raw_space").png")
        savefig(p, plot_file)
        @info "Saved plot to $plot_file"
    end

    # Generate heatmap for heuristic_threshold vs belief_threshold
    p = plot(
        layout=(1, 3),
        size=(1800, 500),
        title="Heuristic Parameters Heatmap ($(log_space ? "Log Space" : "Raw Space"))",
        titlefontsize=12
    )
    
    for (metric_idx, (metric_name, metric_col)) in enumerate(metrics)
        for algo in algorithms
            algo_results = sensitivity_results[sensitivity_results.algorithm .== algo, :]
            heatmap!(
                p,
                config.heuristic_thresholds,
                config.belief_thresholds,
                [mean(algo_results[(algo_results.heuristic_threshold .== ht) .& (algo_results.belief_threshold .== bt), Symbol(metric_col)]) for ht in config.heuristic_thresholds, bt in config.belief_thresholds],
                subplot=metric_idx,
                xlabel="Heuristic Threshold",
                ylabel="Belief Threshold",
                title=metric_name,
                color=:viridis
            )
        end
    end
    
    plot_dir = joinpath(config.data_dir, "plots")
    mkpath(plot_dir)
    plot_file = joinpath(plot_dir, "heuristic_heatmap_$(log_space ? "log_space" : "raw_space").png")
    savefig(p, plot_file)
    @info "Saved heatmap to $plot_file"
end

# ----------------------------
# Main Function
# ----------------------------
function main_sensitivity()

    # Define algorithms to test
    algorithms = [
        Algorithm(solver_name="Heuristic_Policy"),
        Algorithm(solver=QMDPSolver(max_iterations=30), solver_name="QMDP_Policy")
    ]

    # Define configurations
    config = SensitivityConfig()
    pomdp_config = POMDPConfig(log_space=true)

    # Run sensitivity analysis for log-space and raw-space
    for log_space in [true, false]
        # sensitivity_results = run_sensitivity_analysis(algorithms, config, pomdp_config, log_space=log_space)
        plot_sensitivity_results(config; log_space=log_space)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main_sensitivity()
end
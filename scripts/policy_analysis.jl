#!/usr/bin/env julia

#=
policy_analysis.jl

Analyzes and generates plots comparing multiple policies for a single experiment.
Creates PLOS One style figures showing sea lice levels, treatment costs, and other
metrics over time.

Usage:
    julia --project scripts/policy_analysis.jl [--output-dir DIR]

Configuration:
    - EXPERIMENT_PATH: Path to the experiment directory to analyze
    - POLICIES_TO_PLOT: List of policy names to include in plots
    - DEFAULT_OUTPUT_DIR: Default output directory (inside experiment/final_plots)

Outputs:
    - Various PLOS One style plots in the output directory
=#

using AquaOpt
using DataFrames
using JLD2
using Printf

# Hardcoded experiment path and policies
const EXPERIMENT_PATH = "results/experiments/2025-11-19/2025-11-19T22:18:33.024_log_space_ukf_paper_north_[0.46, 0.12, 0.12, 0.18, 0.12]"
const POLICIES_TO_PLOT = ["Random_Policy", "Heuristic_Policy", "QMDP_Policy", "Native_SARSOP_Policy", "VI_Policy"]
const DEFAULT_OUTPUT_DIR = "final_results/policy_outputs"
const TIME_PLOT_LEGEND_GROUPS = [
    :sealice => ["sealice", "sea_lice"],
    :reward => ["reward", "reward_per_step"],
    :biomass => ["biomass", "biomass_loss"],
    :regulatory => ["regulatory", "penalty"],
    :fish_disease => ["fish_disease", "disease"],
    :treatment_cost => ["treatment_cost", "cost"],
    :treatment_probability => ["treatment_probability", "probability"],
    :episode_sealice => ["episode_sealice", "episode"],
]
const TIME_PLOT_LEGEND_ALIASES = Dict(
    alias => key for (key, names) in TIME_PLOT_LEGEND_GROUPS for alias in names
)
const TIME_PLOT_CANONICAL_NAMES = [string(first(group)) for group in TIME_PLOT_LEGEND_GROUPS]

default_time_plot_legends() = Set([:reward])

function available_time_plot_legend_text()
    return join(TIME_PLOT_CANONICAL_NAMES, ", ")
end

function parse_time_plot_legend_arg(value::String)
    normalized = lowercase(strip(value))
    if isempty(normalized)
        return Set{Symbol}()
    elseif normalized == "all"
        return Set(first.(TIME_PLOT_LEGEND_GROUPS))
    elseif normalized == "none"
        return Set{Symbol}()
    end

    selections = Set{Symbol}()
    for token in split(value, ",")
        trimmed = lowercase(strip(token))
        isempty(trimmed) && continue
        key = get(TIME_PLOT_LEGEND_ALIASES, trimmed, nothing)
        if isnothing(key)
            error("Unknown time plot legend \"$token\". Available: $(available_time_plot_legend_text()) or \"all\"/\"none\".")
        end
        push!(selections, key)
    end
    return selections
end

struct AnalyzeOptions
    experiment_path::String
    output_dir::String
    generate_plots::Bool
    policies_to_plot::Union{Nothing, Vector{String}}
    time_plot_legends::Set{Symbol}
end

function usage()
    println("Usage: julia --project scripts/policy_analysis.jl [--output-dir DIR] [--no-plots]")
    println("  --output-dir DIR  Directory where plots should be saved (default: $(DEFAULT_OUTPUT_DIR)).")
    println("  --no-plots        Skip generating plots.")
    println("  --time-plot-legends LIST  Comma-separated list of time plots whose legends should be shown.")
    println("                           Available: $(available_time_plot_legend_text()), 'all', or 'none'.")
    println("                           Default: reward.")
    println("  --help, -h        Show this help message")
end

function parse_cli_args(args::Vector{String})
    output_dir::Union{Nothing, String} = nothing
    generate_plots = true
    time_plot_legends = default_time_plot_legends()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--output-dir", "--output")
            i += 1
            if i > length(args)
                usage()
                error("Missing value for --output-dir")
            end
            output_dir = args[i]
        elseif arg == "--no-plots"
            generate_plots = false
        elseif arg == "--time-plot-legends"
            i += 1
            if i > length(args)
                usage()
                error("Missing value for --time-plot-legends")
            end
            time_plot_legends = parse_time_plot_legend_arg(args[i])
        elseif arg in ("-h", "--help")
            usage()
            exit()
        elseif startswith(arg, "-")
            usage()
            error("Unknown argument: $arg")
        else
            usage()
            error("Unexpected positional argument: $arg")
        end
        i += 1
    end

    exp_path = abspath(EXPERIMENT_PATH)
    resolved_output = isnothing(output_dir) ? DEFAULT_OUTPUT_DIR : abspath(output_dir)

    return AnalyzeOptions(exp_path, resolved_output, generate_plots, POLICIES_TO_PLOT, time_plot_legends)
end

function adjust_config_paths!(config, experiment_root::String)
    config.experiment_dir = experiment_root
    config.policies_dir = joinpath(experiment_root, "policies")
    config.simulations_dir = joinpath(experiment_root, "simulation_histories")
    config.results_dir = joinpath(experiment_root, "avg_results")
    config.figures_dir = joinpath(experiment_root, "figures")
    return config
end

function load_experiment_config(experiment_root::String)
    cfg_path = joinpath(experiment_root, "config", "experiment_config.jld2")
    isfile(cfg_path) || error("Could not find config file at $cfg_path")
    @load cfg_path config
    return adjust_config_paths!(config, experiment_root)
end

function ensure_dataframe(data)
    df = data isa DataFrame ? data : DataFrame(data)
    rename!(df, Symbol.(names(df)))
    return df
end

function load_parallel_data(experiment_root::String)
    data_path = joinpath(experiment_root, "simulation_histories", "all_policies_simulation_data.jld2")
    isfile(data_path) || error("Could not find simulation data at $data_path")
    @load data_path data
    return ensure_dataframe(data)
end

function summarize_experiment(config, parallel_data, algorithms;
        output_dir::Union{Nothing,String}=nothing,
        generate_plots::Bool=true,
        policies_to_plot=nothing,
        time_plot_legends=nothing)
    processed_data = extract_reward_metrics(parallel_data, config)
    println("Loaded $(nrow(processed_data)) episodes across $(length(unique(processed_data.policy))) policies.")

    mkpath(config.results_dir)
    # display_reward_metrics(processed_data, config, false)

    if generate_plots
        target_dir = isnothing(output_dir) ? config.figures_dir : output_dir
        mkpath(target_dir)
        temp_config = deepcopy(config)
        temp_config.figures_dir = target_dir
        plot_plos_one_plots(
            processed_data,
            temp_config,
            algorithms;
            policies_to_plot=policies_to_plot,
            time_plot_legends=time_plot_legends,
        )
        println("Plots saved to $(temp_config.figures_dir).")
    else
        println("Skipping plot generation (--no-plots flag).")
    end

    return processed_data
end

function main()
    opts = parse_cli_args(ARGS)
    config = load_experiment_config(opts.experiment_path)
    algorithms = define_algorithms(config)
    parallel_data = load_parallel_data(opts.experiment_path)
    summarize_experiment(
        config,
        parallel_data,
        algorithms;
        output_dir = opts.output_dir,
        generate_plots = opts.generate_plots,
        policies_to_plot = opts.policies_to_plot,
        time_plot_legends = opts.time_plot_legends,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

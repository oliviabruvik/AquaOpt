#!/usr/bin/env julia

#=
region_analysis.jl

Generates regional comparison plots for sea lice levels, treatment costs, and SARSOP
policy performance across West, North, and South regions.

Usage:
    julia --project scripts/region_analysis.jl [--output-dir DIR]

Outputs:
    - region_sealice_levels_over_time.pdf: Sea lice comparison across regions
    - region_treatment_cost_over_time.pdf/.tex: Treatment cost comparison
    - region_sarsop_sealice_stages.pdf: SARSOP stage breakdown
=#

using AquaOpt
using CSV
using DataFrames
using JLD2
using PGFPlotsX
using PGFPlotsX: Axis, GroupPlot, Options, Plot, @pgf
using Printf
using Statistics
using POMDPTools: state_hist, action_hist, reward_hist, observation_hist

const MANIFEST_PATH = "results/latest/experiment_manifest.txt"

function load_manifest(path::String)
    manifest = Dict{String,String}()
    for line in readlines(path)
        startswith(line, '#') && continue
        isempty(strip(line)) && continue
        parts = split(line, '\t')
        manifest[parts[1]] = parts[2]  # label => experiment_dir
    end
    return manifest
end

const DEFAULT_OUTPUT_DIR = "results/latest/region_outputs"
const REGION_TABLE_POLICIES = [
    (label = "Always Treat", csv_name = "AlwaysTreat_Policy"),
    (label = "Never Treat", csv_name = "NeverTreat_Policy"),
    (label = "Random", csv_name = "Random_Policy"),
    (label = "Heuristic", csv_name = "Heuristic_Policy"),
    (label = "QMDP", csv_name = "QMDP_Policy"),
    (label = "SARSOP", csv_name = "NUS_SARSOP_Policy"),
    (label = "VI", csv_name = "VI_Policy"),
]
const REGION_TABLE_METRICS = [
    (name = :reward, header = "Reward", mean_col = :mean_reward, ci_col = :ci_reward, higher_is_better = true),
    (name = :treatment_cost, header = "Treatment Cost (MNOK)", mean_col = :mean_treatment_cost, ci_col = :ci_treatment_cost, higher_is_better = false),
    (name = :penalties, header = "Reg.\\ Penalties", mean_col = :mean_reg_penalties, ci_col = :ci_reg_penalties, higher_is_better = false),
    (name = :lice, header = "Mean AF Lice/Fish", mean_col = :mean_sea_lice, ci_col = :ci_sea_lice, higher_is_better = false),
    (name = :biomass, header = "Biomass Loss (tons)", mean_col = :mean_lost_biomass, ci_col = :ci_lost_biomass, higher_is_better = false),
    (name = :disease, header = "Fish Disease", mean_col = :mean_fish_disease, ci_col = :ci_fish_disease, higher_is_better = false),
]
const REGION_TABLE_ORDER = ["North", "West", "South"]

struct RegionInput
    name::String
    experiment_path::String
end

struct RegionData
    name::String
    config::ExperimentConfig
    parallel_data::DataFrame
end

const REGION_ANALYSIS_CACHE_FILE = "region_analysis_cache.jld2"
const REGION_ANALYSIS_CACHE_VERSION = 2

function region_cache_path(region::RegionData)
    return joinpath(region.config.experiment_dir, REGION_ANALYSIS_CACHE_FILE)
end

function load_region_cache(region::RegionData)
    cache_path = region_cache_path(region)
    isfile(cache_path) || return Dict{Symbol, Any}()
    cache_version = nothing
    cache_data = Dict{Symbol, Any}()
    try
        @load cache_path cache_version cache_data
    catch err
        @warn "Unable to load cached region analysis stats; recomputing" region = region.name path = cache_path exception = (err, catch_backtrace())
        return Dict{Symbol, Any}()
    end
    if cache_version === nothing || cache_version != REGION_ANALYSIS_CACHE_VERSION || cache_data === nothing
        @warn "Cached region analysis stats are from an incompatible version; recomputing" region = region.name path = cache_path cache_version = cache_version
        return Dict{Symbol, Any}()
    end
    return cache_data
end

function save_region_cache(region::RegionData, cache_data::Dict{Symbol, Any})
    cache_path = region_cache_path(region)
    mkpath(dirname(cache_path))
    cache_version = REGION_ANALYSIS_CACHE_VERSION
    try
        @save cache_path cache_version cache_data
    catch err
        @warn "Failed to write region analysis cache (non-fatal)" region = region.name path = cache_path exception = (err, catch_backtrace())
    end
end

function usage()
    println("Usage: julia --project scripts/region_analysis.jl [--output-dir DIR]")
    println("  --output-dir, -o  Directory where plots should be saved (default: $(DEFAULT_OUTPUT_DIR))")
    println("  --help, -h        Show this help message")
    println("\nReads experiment paths from: $(MANIFEST_PATH)")
end

function parse_args(args)
    output_dir = DEFAULT_OUTPUT_DIR
    i = 1
    while i <= length(args)
        arg = strip(args[i])
        if isempty(arg)
            i += 1
            continue
        end
        if arg in ("-o", "--output-dir")
            i += 1
            i <= length(args) || error("--output-dir requires a value")
            output_dir = args[i]
        elseif arg in ("-h", "--help")
            usage()
            exit()
        else
            usage()
            error("Unknown argument: $arg")
        end
        i += 1
    end

    isfile(MANIFEST_PATH) || error("Manifest not found at $MANIFEST_PATH. Run run_experiments.jl first.")
    @info "Using manifest: $MANIFEST_PATH"
    manifest = load_manifest(MANIFEST_PATH)

    north = get(manifest, "baseline_norway_north", nothing)
    west = get(manifest, "dynamics_norway_west", nothing)
    south = get(manifest, "dynamics_norway_south", nothing)
    north === nothing && error("Manifest missing 'baseline_norway_north'")
    west === nothing && error("Manifest missing 'dynamics_norway_west'")
    south === nothing && error("Manifest missing 'dynamics_norway_south'")

    return (
        [RegionInput("North", north),
         RegionInput("West", west),
         RegionInput("South", south)],
        output_dir
    )
end

function adjust_config_paths(config, experiment_root::String)
    return ExperimentConfig(
        solver_config = config.solver_config,
        simulation_config = config.simulation_config,
        experiment_name = config.experiment_name,
        experiment_dir = experiment_root,
        policies_dir = joinpath(experiment_root, "policies"),
        simulations_dir = joinpath(experiment_root, "simulation_histories"),
        results_dir = joinpath(experiment_root, "avg_results"),
        figures_dir = joinpath(experiment_root, "figures"),
    )
end

function load_experiment_config(experiment_root::String)
    cfg_path = joinpath(experiment_root, "config", "experiment_config.jld2")
    isfile(cfg_path) || error("Could not find config file at $cfg_path")
    @load cfg_path config
    return adjust_config_paths(config, experiment_root)
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

function load_region(region::RegionInput)
    path = abspath(region.experiment_path)
    config = load_experiment_config(path)
    data = load_parallel_data(path)
    return RegionData(region.name, config, data)
end

function compute_sealice_stats(parallel_data, config)
    stats = Dict{String, Tuple{Vector{Int}, Vector{Float64}, Vector{Float64}, Vector{Float64}}}()
    styles = AquaOpt.PLOS_POLICY_STYLE_ORDERED
    time_steps = 1:config.simulation_config.steps_per_episode
    for (policy_name, _) in styles
        data_filtered = filter(row -> row.policy == policy_name, parallel_data)
        isempty(data_filtered) && continue
        seeds = unique(data_filtered.seed)
        isempty(seeds) && continue
        mean_vals = fill(NaN, length(time_steps))
        lower_vals = similar(mean_vals)
        upper_vals = similar(mean_vals)
        for (idx, t) in enumerate(time_steps)
            samples = Float64[]
            for seed in seeds
                seed_df = filter(row -> row.seed == seed, data_filtered)
                isempty(seed_df) && continue
                states = collect(state_hist(seed_df.history[1]))
                if t <= length(states)
                    push!(samples, states[t].SeaLiceLevel)
                end
            end
            if !isempty(samples)
                mean_level = mean(samples)
                std_level = length(samples) > 1 ? std(samples) : 0.0
                se_level = std_level / sqrt(length(samples))
                margin = 1.96 * se_level
                mean_vals[idx] = mean_level
                lower_vals[idx] = mean_level - margin
                upper_vals[idx] = mean_level + margin
            end
        end
        valid = .!isnan.(mean_vals) .&& .!isnan.(lower_vals) .&& .!isnan.(upper_vals)
        if any(valid)
            stats[policy_name] = (collect(time_steps[valid]),
                                  mean_vals[valid],
                                  lower_vals[valid],
                                  upper_vals[valid])
        end
    end
    return stats
end

function extract_metric_caches(data_filtered, seeds)
    caches = NamedTuple[]
    for seed in seeds
        seed_df = filter(row -> row.seed == seed, data_filtered)
        isempty(seed_df) && continue
        history = seed_df.history[1]
        states = collect(state_hist(history))
        actions = collect(action_hist(history))
        rewards = collect(reward_hist(history))
        push!(caches, (; states, actions, rewards))
    end
    return caches
end

function compute_treatment_cost_stats(parallel_data, config)
    stats = Dict{String, Tuple{Vector{Int}, Vector{Float64}, Vector{Float64}, Vector{Float64}}}()
    styles = AquaOpt.PLOS_POLICY_STYLE_ORDERED
    time_steps = 1:config.simulation_config.steps_per_episode
    for (policy_name, _) in styles
        data_filtered = filter(row -> row.policy == policy_name, parallel_data)
        isempty(data_filtered) && continue
        seeds = unique(data_filtered.seed)
        isempty(seeds) && continue
        caches = extract_metric_caches(data_filtered, seeds)
        isempty(caches) && continue
        means = fill(NaN, length(time_steps))
        lowers = similar(means)
        uppers = similar(means)
        for (idx, t) in enumerate(time_steps)
            values = Float64[]
            for cache in caches
                if t <= length(cache.actions)
                    push!(values, get_treatment_cost(cache.actions[t]))
                end
            end
            if !isempty(values)
                mean_val = mean(values)
                std_val = length(values) > 1 ? std(values) : 0.0
                se = std_val / sqrt(length(values))
                margin = 1.96 * se
                means[idx] = mean_val
                lowers[idx] = mean_val - margin
                uppers[idx] = mean_val + margin
            end
        end
        valid = .!isnan.(means) .&& .!isnan.(lowers) .&& .!isnan.(uppers)
        if any(valid)
            stats[policy_name] = (collect(time_steps[valid]),
                                  means[valid],
                                  lowers[valid],
                                  uppers[valid])
        end
    end
    return stats
end

function load_or_compute_region_stats(region::RegionData)
    cache_data = load_region_cache(region)
    dirty = false
    sealice_stats = get(cache_data, :sealice_stats, nothing)
    if sealice_stats === nothing
        @info "Computing sea lice stats" region = region.name
        sealice_stats = compute_sealice_stats(region.parallel_data, region.config)
        cache_data[:sealice_stats] = sealice_stats
        dirty = true
    else
        @info "Loaded cached sea lice stats" region = region.name
    end

    treatment_stats = get(cache_data, :treatment_stats, nothing)
    if treatment_stats === nothing
        @info "Computing treatment cost stats" region = region.name
        treatment_stats = compute_treatment_cost_stats(region.parallel_data, region.config)
        cache_data[:treatment_stats] = treatment_stats
        dirty = true
    else
        @info "Loaded cached treatment cost stats" region = region.name
    end
    return cache_data, dirty, sealice_stats, treatment_stats
end

function load_or_compute_sarsop_stats(region::RegionData, cache_data::Dict{Symbol, Any})
    stats = get(cache_data, :sarsop_stats, nothing)
    if stats !== nothing
        @info "Loaded cached SARSOP stats" region = region.name
        return stats, false
    end
    @info "Computing SARSOP stats" region = region.name
    stats = compute_sarsop_stage_stats(region)
    cache_data[:sarsop_stats] = stats
    return stats, true
end

function compute_sarsop_stage_stats(region::RegionData)
    policy = "NUS_SARSOP_Policy"
    data_policy = filter(row -> row.policy == policy, region.parallel_data)
    histories = collect(data_policy.history)
    steps = region.config.simulation_config.steps_per_episode
    stages = Dict(
        :adult => (Float64[], Float64[], Float64[]),
        :sessile => (Float64[], Float64[], Float64[]),
        :motile => (Float64[], Float64[], Float64[]),
        :predicted => (Float64[], Float64[], Float64[])
    )
    for t in 1:steps
        samples = Dict(
            :adult => Float64[],
            :sessile => Float64[],
            :motile => Float64[],
            :predicted => Float64[],
        )
        for episode in histories
            states = collect(state_hist(episode))
            observations = collect(observation_hist(episode))
            if t <= length(states)
                push!(samples[:adult], states[t].Adult)
                push!(samples[:sessile], states[t].Sessile)
                push!(samples[:motile], states[t].Motile)
            end
            if t <= length(observations)
                push!(samples[:predicted], observations[t].SeaLiceLevel)
            end
        end
        for (stage, data) in samples
            mean_vec, lower_vec, upper_vec = stages[stage]
            if !isempty(data)
                mean_val = mean(data)
                std_val = length(data) > 1 ? std(data) : 0.0
                se = std_val / sqrt(length(data))
                margin = 1.96 * se
                push!(mean_vec, mean_val)
                push!(lower_vec, mean_val - margin)
                push!(upper_vec, mean_val + margin)
            else
                push!(mean_vec, NaN)
                push!(lower_vec, NaN)
                push!(upper_vec, NaN)
            end
        end
    end
    return stages
end

function load_reward_metrics(region::RegionData)
    csv_path = joinpath(region.config.results_dir, "reward_metrics.csv")
    isfile(csv_path) || error("Could not find reward metrics CSV for $(region.name) at $csv_path")
    df = CSV.read(csv_path, DataFrame)
    rename!(df, Symbol.(names(df)))
    return df
end

function build_policy_row_map(df::DataFrame)
    mapping = Dict{String, DataFrameRow}()
    for row in eachrow(df)
        mapping[String(row.policy)] = row
    end
    return mapping
end

function compute_best_policy_sets(policy_rows::Dict{String, DataFrameRow})
    best_sets = Dict{Symbol, Set{String}}()
    for metric in REGION_TABLE_METRICS
        values = Float64[]
        ordered_policies = String[]
        for policy in REGION_TABLE_POLICIES
            csv_name = policy.csv_name
            haskey(policy_rows, csv_name) || continue
            push!(ordered_policies, csv_name)
            push!(values, Float64(policy_rows[csv_name][metric.mean_col]))
        end
        isempty(values) && continue
        target = metric.higher_is_better ? maximum(values) : minimum(values)
        winners = Set{String}()
        for (idx, csv_name) in enumerate(ordered_policies)
            if isapprox(values[idx], target; atol = 1e-6, rtol = 0.0)
                push!(winners, csv_name)
            end
        end
        best_sets[metric.name] = winners
    end
    return best_sets
end

function format_metric_entry(row::DataFrameRow, metric; highlight::Bool)
    mean_val = Float64(row[metric.mean_col])
    ci_val = Float64(row[metric.ci_col])
    text = @sprintf("%.2f \\pm %.2f", mean_val, ci_val)
    return highlight ? "\$\\mathBF{$text}\$" : "\$$text\$"
end

function region_table_block(region::RegionData)
    metrics_df = load_reward_metrics(region)
    policy_rows = build_policy_row_map(metrics_df)
    best_sets = compute_best_policy_sets(policy_rows)
    available_policies = [p for p in REGION_TABLE_POLICIES if haskey(policy_rows, p.csv_name)]
    isempty(available_policies) && return String[]

    lines = String[]
    multirow_count = length(available_policies)
    for (idx, policy) in enumerate(available_policies)
        csv_name = policy.csv_name
        row = policy_rows[csv_name]
        entries = String[]
        for metric in REGION_TABLE_METRICS
            best_policies = get(best_sets, metric.name, Set{String}())
            highlight = csv_name in best_policies
            push!(entries, format_metric_entry(row, metric; highlight))
        end
        prefix = idx == 1 ? "    \\multirow{$multirow_count}{*}{$(region.name)} &" : "      &"
        policy_label = rpad(policy.label, 9)
        entry_str = join(entries, " & ")
        push!(lines, "$(prefix) $(policy_label) & $(entry_str) \\\\")
    end
    return lines
end

function generate_region_table(regions::Vector{RegionData}, out_dir::String)
    output_path = joinpath(out_dir, "region_policy_comparison.tex")
    lines = String[
        "\\begin{table}[htbp!]",
        "\\centering",
        "\\begin{adjustwidth}{-2.25in}{0in}",
        "\\caption{Comparison of Policies Across North, West, and South of Norway (common reward weights = \$(0.46,0.12,0.12,0.18,0.12)\$)}",
        "\\label{tab:norway-methods-comparable}",
        "\\begin{threeparttable}",
        "  \\begin{adjustbox}{max width=\\linewidth}",
        "  \\begin{tabular}{@{}llcccccc@{}}",
        "    \\arrayrulecolor{black}",
        "    \\toprule",
        "    Region & Method & " * join([metric.header for metric in REGION_TABLE_METRICS], " & ") * " \\\\",
        "    \\midrule",
        "    \\arrayrulecolor{white}",
    ]
    region_lookup = Dict(region.name => region for region in regions)
    ordered_regions = RegionData[]
    seen_regions = Set{String}()
    for name in REGION_TABLE_ORDER
        if haskey(region_lookup, name)
            push!(ordered_regions, region_lookup[name])
            push!(seen_regions, name)
        end
    end
    for region in regions
        region.name in seen_regions && continue
        push!(ordered_regions, region)
    end

    region_sections = [(region, region_table_block(region)) for region in ordered_regions]
    region_sections = [(region, rows) for (region, rows) in region_sections if !isempty(rows)]
    for (idx, (_, rows)) in enumerate(region_sections)
        append!(lines, rows)
        if idx < length(region_sections)
            push!(lines, "    \\midrule")
        end
    end
    append!(lines, [
        "    \\arrayrulecolor{black}",
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\end{adjustbox}",
        "    \\begin{tablenotes}",
        "      \\item[*]{Mean \$\\pm\$ standard error over the seeds in the corresponding run. Bold values denote the best performance (highest reward or lowest cost/penalties/lice/biomass loss/fish disease) within each region. Runs used: North, West, and South correspond to the \\texttt{log\\_space\\_ukf\\_paper\\_{north,west,south}\\_[0.46,0.12,0.12,0.18,0.12]} chemical-change experiments.}",
        "    \\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{adjustwidth}",
        "\\end{table}",
    ])
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        write(io, join(lines, "\n"))
        write(io, "\n")
    end
    println("Region comparison table saved to $(abspath(output_path)).")
end

function time_ticks(config)
    return AquaOpt.plos_time_ticks(config)
end

function region_axis(region::RegionData, stats; ylabel::String, show_xlabel::Bool, show_legend::Bool)
    ticks, labels = time_ticks(region.config)
    option_pairs = [
        :width => "18cm",
        :height => "6cm",
        :title => region.name,
        :ylabel => ylabel,
        :xlabel_style => AquaOpt.PLOS_LABEL_STYLE,
        :ylabel_style => AquaOpt.PLOS_LABEL_STYLE,
        :tick_label_style => AquaOpt.PLOS_TICK_STYLE,
        :xmin => 0,
        :xmax => region.config.simulation_config.steps_per_episode,
        :ymin => 0,
        :ymax => 0.6,
        :xtick => ticks,
        :xticklabels => labels,
        "axis background/.style" => Options("fill" => "white"),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
    ]
    if show_xlabel
        push!(option_pairs, :xlabel => "Time of Year")
    end
    if show_legend
        push!(option_pairs, "legend style" => AquaOpt.plos_top_legend(columns=length(AquaOpt.PLOS_POLICY_STYLE_ORDERED)))
    else
        push!(option_pairs, :legend => false)
    end
    ax = Axis(Options(option_pairs...))
    for (policy_name, style) in AquaOpt.PLOS_POLICY_STYLE_ORDERED
        haskey(stats, policy_name) || continue
        times, mean_vals, lower_vals, upper_vals = stats[policy_name]
        mean_coords = join(["($(times[j]), $(mean_vals[j]))" for j in eachindex(times)], " ")
        upper_coords = join(["($(times[j]), $(upper_vals[j]))" for j in eachindex(times)], " ")
        lower_coords = join(["($(times[j]), $(lower_vals[j]))" for j in eachindex(times)], " ")
        safe_name = replace(policy_name, r"[^A-Za-z0-9]" => "")
        push!(ax, @pgf("\\addplot[name path=upper$(safe_name), draw=none, forget plot] coordinates {$(upper_coords)};"))
        push!(ax, @pgf("\\addplot[name path=lower$(safe_name), draw=none, forget plot] coordinates {$(lower_coords)};"))
        push!(ax, @pgf("\\addplot[forget plot, fill=$(style.fill), fill opacity=$(AquaOpt.PLOS_FILL_OPACITY)] fill between[of=upper$(safe_name) and lower$(safe_name)];"))
        push!(ax, @pgf("\\addplot[color=$(style.line), mark=none, line width=1.2pt] coordinates {$(mean_coords)};"))
        show_legend && push!(ax, @pgf("\\addlegendentry{$(style.label)}"))
    end
    push!(ax, @pgf("\\addplot[black!70, densely dashed, line width=1pt] coordinates {(0,0.5) ($(region.config.simulation_config.steps_per_episode),0.5)};"))
    show_legend && push!(ax, @pgf("\\addlegendentry{Reg. limit (0.5)}"))
    return ax
end

function treatment_cost_axis(region::RegionData, stats; show_xlabel::Bool, show_legend::Bool)
    ticks, labels = time_ticks(region.config)
    option_pairs = [
        :width => "15cm",
        :height => "4.5cm",
        :title => region.name,
        :ylabel => "Treatment Cost per Step",
        :xlabel_style => AquaOpt.PLOS_LABEL_STYLE,
        :ylabel_style => AquaOpt.PLOS_LABEL_STYLE,
        :tick_label_style => AquaOpt.PLOS_TICK_STYLE,
        :xmin => 0,
        :xmax => region.config.simulation_config.steps_per_episode,
        :ymin => 0,
        :xtick => ticks,
        :xticklabels => labels,
        "axis background/.style" => Options("fill" => "white"),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
    ]
    if show_xlabel
        push!(option_pairs, :xlabel => "Time of Year")
    end
    if show_legend
        push!(option_pairs, "legend style" => AquaOpt.plos_top_legend(columns=length(AquaOpt.PLOS_POLICY_STYLE_ORDERED)))
    else
        push!(option_pairs, :legend => false)
    end
    ax = Axis(Options(option_pairs...))
    for (policy_name, style) in AquaOpt.PLOS_POLICY_STYLE_ORDERED
        haskey(stats, policy_name) || continue
        times, mean_vals, lower_vals, upper_vals = stats[policy_name]
        mean_coords = join(["($(times[j]), $(mean_vals[j]))" for j in eachindex(times)], " ")
        upper_coords = join(["($(times[j]), $(upper_vals[j]))" for j in eachindex(times)], " ")
        lower_coords = join(["($(times[j]), $(lower_vals[j]))" for j in eachindex(times)], " ")
        safe_name = replace(policy_name, r"[^A-Za-z0-9]" => "")
        push!(ax, @pgf("\\addplot[name path=upper$(safe_name), draw=none, forget plot] coordinates {$(upper_coords)};"))
        push!(ax, @pgf("\\addplot[name path=lower$(safe_name), draw=none, forget plot] coordinates {$(lower_coords)};"))
        push!(ax, @pgf("\\addplot[forget plot, fill=$(style.fill), fill opacity=$(AquaOpt.PLOS_FILL_OPACITY)] fill between[of=upper$(safe_name) and lower$(safe_name)];"))
        push!(ax, @pgf("\\addplot[color=$(style.line), mark=none, line width=1.2pt] coordinates {$(mean_coords)};"))
        show_legend && push!(ax, @pgf("\\addlegendentry{$(style.label)}"))
    end
    return ax
end

function sarsop_axis(region::RegionData, stats; ylabel::String, show_xlabel::Bool, show_legend::Bool)
    ticks, labels = time_ticks(region.config)
    option_pairs = [
        :width => "15cm",
        :height => "4.5cm",
        :title => region.name,
        :ylabel => ylabel,
        :xlabel_style => AquaOpt.PLOS_LABEL_STYLE,
        :ylabel_style => AquaOpt.PLOS_LABEL_STYLE,
        :tick_label_style => AquaOpt.PLOS_TICK_STYLE,
        :xmin => 0,
        :xmax => region.config.simulation_config.steps_per_episode,
        :ymin => 0,
        :xtick => ticks,
        :xticklabels => labels,
        "axis background/.style" => Options("fill" => "white"),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
    ]
    if show_xlabel
        push!(option_pairs, :xlabel => "Time of Year")
    end
    legend_entries = [
        (:adult, "Adult (true)"),
        (:sessile, "Sessile"),
        (:motile, "Motile"),
        (:predicted, "Belief (predicted)")
    ]
    if show_legend
        push!(option_pairs, "legend style" => AquaOpt.plos_top_legend(columns=length(legend_entries)))
    else
        push!(option_pairs, :legend => false)
    end
    ax = Axis(Options(option_pairs...))
    colors = Dict(
        :adult => "blue!80!black",
        :sessile => "purple!70!black",
        :motile => "teal!70!black",
        :predicted => "black!70"
    )
    fill_colors = Dict(
        :adult => "blue!25!white",
        :sessile => "purple!20!white",
        :motile => "teal!20!white",
        :predicted => "black!10"
    )
    for (key, label) in legend_entries
        mean_vals, lower_vals, upper_vals = stats[key]
        times = collect(1:length(mean_vals))
        valid = .!isnan.(mean_vals) .&& .!isnan.(lower_vals) .&& .!isnan.(upper_vals)
        any(valid) || continue
        t = times[valid]
        μ = mean_vals[valid]
        lo = lower_vals[valid]
        hi = upper_vals[valid]
        mean_coords = join(["($(t[i]), $(μ[i]))" for i in eachindex(t)], " ")
        upper_coords = join(["($(t[i]), $(hi[i]))" for i in eachindex(t)], " ")
        lower_coords = join(["($(t[i]), $(lo[i]))" for i in eachindex(t)], " ")
        safe_name = replace(String(label), r"[^A-Za-z0-9]" => "")
        push!(ax, @pgf("\\addplot[name path=upper$(safe_name), draw=none, forget plot] coordinates {$(upper_coords)};"))
        push!(ax, @pgf("\\addplot[name path=lower$(safe_name), draw=none, forget plot] coordinates {$(lower_coords)};"))
        push!(
            ax,
            @pgf("\\addplot[forget plot, fill=$(fill_colors[key]), fill opacity=$(AquaOpt.PLOS_FILL_OPACITY)] fill between[of=upper$(safe_name) and lower$(safe_name)];")
        )
        push!(ax, @pgf("\\addplot[color=$(colors[key]), mark=none, line width=1.2pt] coordinates {$(mean_coords)};"))
        show_legend && push!(ax, @pgf("\\addlegendentry{$label}"))
    end
    push!(ax, @pgf("\\addplot[black!70, densely dashed, line width=1pt] coordinates {(0,0.5) ($(region.config.simulation_config.steps_per_episode),0.5)};"))
    show_legend && push!(ax, @pgf("\\addlegendentry{Reg. limit (0.5)}"))
    return ax
end

function build_group_plot(axes::Vector{Axis}; vertical_sep::String="12pt")
    gp = GroupPlot(Options(
        "group style" => Options(
            "group size" => "1 by $(length(axes))",
            "vertical sep" => vertical_sep
        ),
        :width => "18cm"
    ))
    for ax in axes
        push!(gp, ax)
    end
    return gp
end

function save_output(gp, out_pdf::String; save_tex::Bool=false)
    mkpath(dirname(out_pdf))
    PGFPlotsX.save(out_pdf, gp)
    if save_tex
        PGFPlotsX.save(replace(out_pdf, ".pdf" => ".tex"), gp; include_preamble=false)
    end
end

function main()
    @info "Parsing args"
    regions_input, out_dir = parse_args(ARGS)

    @info "Loading regions"
    regions = [load_region(r) for r in regions_input]

    cache_data = Vector{Dict{Symbol, Any}}(undef, length(regions))
    cache_dirty = fill(false, length(regions))
    sealice_stats = Vector{Any}(undef, length(regions))
    treatment_stats = Vector{Any}(undef, length(regions))
    for (idx, region) in enumerate(regions)

        @info "Loading or computing stats for $region"
        cache, dirty, sealice, treatment = load_or_compute_region_stats(region)
        cache_data[idx] = cache
        cache_dirty[idx] = dirty
        sealice_stats[idx] = sealice
        treatment_stats[idx] = treatment
    end

    sarsop_stats = Vector{Any}(undef, length(regions))
    for (idx, region) in enumerate(regions)
        stats, dirty = load_or_compute_sarsop_stats(region, cache_data[idx])
        sarsop_stats[idx] = stats
        cache_dirty[idx] = cache_dirty[idx] || dirty
    end

    for (idx, region) in enumerate(regions)
        cache_dirty[idx] && save_region_cache(region, cache_data[idx])
    end

    axes_sealice = Axis[]
    axes_cost = Axis[]
    axes_sarsop = Axis[]

    for (idx, region) in enumerate(regions)
        push!(axes_sealice, region_axis(region, sealice_stats[idx];
            ylabel = idx == 1 ? "Adult Female Sea Lice per Fish" : "",
            show_xlabel = idx == length(regions),
            show_legend = idx == 1))

        push!(axes_cost, treatment_cost_axis(region, treatment_stats[idx];
            show_xlabel = idx == length(regions),
            show_legend = idx == 1))

        push!(axes_sarsop, sarsop_axis(region, sarsop_stats[idx];
            ylabel = idx == 1 ? "Avg. Lice per Fish" : "",
            show_xlabel = idx == length(regions),
            show_legend = idx == 1))
    end

    mkpath(out_dir)
    save_output(build_group_plot(axes_sealice), joinpath(out_dir, "region_sealice_levels_over_time.pdf"))
    save_output(build_group_plot(axes_cost; vertical_sep="40pt"), joinpath(out_dir, "region_treatment_cost_over_time.pdf"), save_tex=true)
    save_output(build_group_plot(axes_sarsop), joinpath(out_dir, "region_sarsop_sealice_stages.pdf"))
    generate_region_table(regions, out_dir)

    println("Region plots saved under $(abspath(out_dir)).")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

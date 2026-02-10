#!/usr/bin/env julia

#=
region_analysis.jl

Generates regional comparison plots for sea lice levels and treatment costs
across West, North, and South regions, plus a policy comparison table.

Usage:
    julia --project scripts/region_analysis.jl [--output-dir DIR]

Outputs:
    - region_sealice_levels_over_time.pdf: Sea lice comparison across regions
    - region_treatment_cost_over_time.pdf/.tex: Treatment cost comparison
    - region_policy_comparison.tex: LaTeX table comparing policies across regions
=#

using AquaOpt
using CSV
using DataFrames
using JLD2
using PGFPlotsX
using PGFPlotsX: Axis, GroupPlot, Options, Plot, @pgf
using Printf
using Statistics

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

# ──────────────────────────────────────────────────
# Table generation
# ──────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────
# Plot axis builders (using _add_metric_lines! from PlosOnePlots)
# ──────────────────────────────────────────────────

function region_sealice_axis(region::RegionData; show_xlabel::Bool, show_legend::Bool)
    ticks, labels = AquaOpt.plos_time_ticks(region.config)
    axis_opts = Any[
        :title => region.name,
        :title_style => AquaOpt.PLOS_TITLE_STYLE,
        :ylabel => "Adult Female Lice\\\\per Fish",
        :ylabel_style => "{" * AquaOpt.PLOS_LABEL_STYLE * ", align=center}",
        :tick_label_style => AquaOpt.PLOS_TICK_STYLE,
        :xmin => 1,
        :xmax => region.config.simulation_config.steps_per_episode,
        :ymin => 0,
        :ymax => 0.6,
        :xtick => ticks,
        :xticklabels => show_xlabel ? labels : String[],
        "axis background/.style" => Options("fill" => "white"),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
        "scaled y ticks" => "false",
        "yticklabel style" => "{/pgf/number format/fixed, /pgf/number format/precision=2}",
    ]
    if show_xlabel
        push!(axis_opts, :xlabel => "Time of Year")
        push!(axis_opts, :xlabel_style => AquaOpt.PLOS_LABEL_STYLE)
    end
    if show_legend
        push!(axis_opts, "legend style" => AquaOpt.plos_top_legend(columns=5))
    end
    ax = @pgf Axis(Options(axis_opts...))

    compute = (cache, t, cfg) -> t <= length(cache.states) ? cache.states[t].SeaLiceLevel : nothing
    AquaOpt._add_metric_lines!(ax, region.parallel_data, region.config, compute;
        show_legend=show_legend, ymin=0.0, ymax=0.6)

    AquaOpt._add_reg_limit!(ax, region.config.simulation_config.steps_per_episode, 0.5;
        use_legend=false)

    return ax
end

function region_treatment_cost_axis(region::RegionData; show_xlabel::Bool, show_legend::Bool)
    ticks, labels = AquaOpt.plos_time_ticks(region.config)
    axis_opts = Any[
        :title => region.name,
        :title_style => AquaOpt.PLOS_TITLE_STYLE,
        :ylabel => "Treatment Cost\\\\per Step",
        :ylabel_style => "{" * AquaOpt.PLOS_LABEL_STYLE * ", align=center}",
        :tick_label_style => AquaOpt.PLOS_TICK_STYLE,
        :xmin => 1,
        :xmax => region.config.simulation_config.steps_per_episode,
        :ymin => 0,
        :xtick => ticks,
        :xticklabels => show_xlabel ? labels : String[],
        "axis background/.style" => Options("fill" => "white"),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
        "scaled y ticks" => "false",
        "yticklabel style" => "{/pgf/number format/fixed, /pgf/number format/precision=2}",
    ]
    if show_xlabel
        push!(axis_opts, :xlabel => "Time of Year")
        push!(axis_opts, :xlabel_style => AquaOpt.PLOS_LABEL_STYLE)
    end
    if show_legend
        push!(axis_opts, "legend style" => AquaOpt.plos_top_legend(columns=5))
    end
    ax = @pgf Axis(Options(axis_opts...))

    AquaOpt._add_metric_lines!(ax, region.parallel_data, region.config,
        AquaOpt.treatment_cost_step_value;
        show_legend=show_legend, ymin=0.0)

    return ax
end

function build_group_plot(axes::Vector{Axis}; vertical_sep::String="16pt")
    gp = GroupPlot(Options(
        "group style" => Options(
            "group size" => "1 by $(length(axes))",
            "vertical sep" => vertical_sep,
        ),
        :width => "18cm",
        :height => "6cm",
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

# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

function main()
    @info "Parsing args"
    regions_input, out_dir = parse_args(ARGS)

    @info "Loading regions"
    regions = [load_region(r) for r in regions_input]

    axes_sealice = Axis[]
    axes_cost = Axis[]

    for (idx, region) in enumerate(regions)
        @info "Building plots for $(region.name)"
        push!(axes_sealice, region_sealice_axis(region;
            show_xlabel = idx == length(regions),
            show_legend = idx == 1))

        push!(axes_cost, region_treatment_cost_axis(region;
            show_xlabel = idx == length(regions),
            show_legend = idx == 1))
    end

    mkpath(out_dir)
    save_output(build_group_plot(axes_sealice; vertical_sep="40pt"), joinpath(out_dir, "region_sealice_levels_over_time.pdf"))
    save_output(build_group_plot(axes_cost; vertical_sep="40pt"), joinpath(out_dir, "region_treatment_cost_over_time.pdf"), save_tex=true)
    generate_region_table(regions, out_dir)

    println("Region plots saved under $(abspath(out_dir)).")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

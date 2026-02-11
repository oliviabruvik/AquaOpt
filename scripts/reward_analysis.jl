#!/usr/bin/env julia

#=
aggregate_lambda_summaries.jl

Aggregates treatment summaries and SARSOP dominant action heatmaps across multiple
experiments with different lambda combinations. Generates:
  1. A LaTeX table comparing average treatment counts per policy across experiments
  2. Side-by-side heatmaps showing SARSOP policy actions vs. temperature and sea lice levels

Usage:
    julia --project scripts/aggregate_lambda_summaries.jl

Configuration:
    - EXPERIMENT_FOLDERS: Paths to experiments to compare
    - TABLE_OUTPUT_PATH: Where to save the treatment summary table
    - FIGURE_OUTPUT_PATH: Where to save the dominant action figure

Outputs:
    - Quick_Access/policy_treatment_summary.tex: LaTeX table
    - Quick_Access/policy_dominant_actions.tex: LaTeX figure (GroupPlot)
    - Quick_Access/policy_dominant_actions.pdf: PDF figure
=#

isnothing(Base.active_project()) && @warn "No active Julia project detected. Run this script with `julia --project=.` to ensure dependencies are available."

using AquaOpt
using CSV
using DataFrames
using JLD2
using PGFPlotsX
using PGFPlotsX: @pgf, Axis, Plot, Coordinates, GroupPlot, Options
using POMDPs: action, states
using Printf
using Statistics

const MANIFEST_PATH = "results/latest/experiment_manifest.txt"

function load_manifest(path::String)
    manifest = Dict{String,String}()
    for line in readlines(path)
        startswith(line, '#') && continue
        isempty(strip(line)) && continue
        parts = split(line, '\t')
        manifest[parts[1]] = parts[2]
    end
    return manifest
end

const _MANIFEST = if isfile(MANIFEST_PATH)
    @info "Using manifest: $MANIFEST_PATH"
    load_manifest(MANIFEST_PATH)
else
    error("Manifest not found at $MANIFEST_PATH. Run run_experiments.jl first.")
end

const EXPERIMENT_FOLDERS = Dict(
    "Scotland" => _MANIFEST["regulation_scotland_north"],
    "Chile" => _MANIFEST["regulation_chile_north"],
    "Southern Norway" => _MANIFEST["dynamics_norway_south"],
    "Northern Norway" => _MANIFEST["baseline_norway_north"],
)

const LAMBDA_FOLDERS = Dict(
    "Balanced" => _MANIFEST["baseline_norway_north"],
    "Cost-Saving" => _MANIFEST["lambda_cost_norway_north"],
    "Welfare" => _MANIFEST["lambda_welfare_norway_north"],
)

const TABLE_OUTPUT_PATH = "results/latest/reward_outputs/policy_treatment_summary.tex"
const FIGURE_OUTPUT_PATH = "results/latest/reward_outputs/policy_dominant_actions.tex"
const LAMBDA_TABLE_OUTPUT_PATH = "results/latest/reward_outputs/lambda_parameters.tex"
const LAMBDA_TREATMENT_TABLE_OUTPUT_PATH = "results/latest/reward_outputs/lambda_treatment_summary.tex"

const TREATMENT_COLUMNS = ["NoTreatment", "ChemicalTreatment", "MechanicalTreatment", "ThermalTreatment"]
const TREATMENT_LABELS = Dict(
    "NoTreatment" => "No Tx",
    "MechanicalTreatment" => "Mechanical",
    "ChemicalTreatment" => "Chemical",
    "ThermalTreatment" => "Thermal",
)
const TREATMENT_ACTIONS = Dict(
    "NoTreatment" => NoTreatment,
    "MechanicalTreatment" => MechanicalTreatment,
    "ChemicalTreatment" => ChemicalTreatment,
    "ThermalTreatment" => ThermalTreatment,
)

const POLICY_ORDER = [
    "Heuristic_Policy",
    "QMDP_Policy",
    "Native_SARSOP_Policy",
    "VI_Policy",
]

const POLICY_LABELS = Dict(
    "NeverTreat_Policy" => "Never Treat",
    "Random_Policy" => "Random",
    "Heuristic_Policy" => "Heuristic",
    "AlwaysTreat_Policy" => "Always Treat",
    "VI_Policy" => "VI",
    "QMDP_Policy" => "QMDP",
    "Native_SARSOP_Policy" => "SARSOP",
)

const LAMBDA_COMPONENTS = [
    (label=raw"$\lambda_{\text{trt}}$", idx=1),
    (label=raw"$\lambda_{\text{reg}}$", idx=2),
    (label=raw"$\lambda_{\text{bio}}$", idx=3),
    (label=raw"$\lambda_{\text{fd}}$", idx=4),
    (label=raw"$\lambda_{\text{lice}}$", idx=5),
]
const LAMBDA_REGION_ORDER = ["Northern Norway", "Southern Norway", "Scotland", "Chile"]
const LAMBDA_SCENARIO_ORDER = ["Balanced", "Cost-Saving", "Welfare"]

const TreatmentStats = Dict{String, Dict{String, Union{Missing, Float64}}}

struct ExperimentSummary
    name::String
    label::String
    config::ExperimentConfig
    treatment::DataFrame
    treatment_std::TreatmentStats
end

function adjust_config_paths(config::ExperimentConfig, experiment_root::String)
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

function extract_lambda_label(experiment_root::String)
    base = basename(experiment_root)
    if (m = match(r"\[(.*)\]", base)) !== nothing
        return raw"\(\lambda = [" * m.captures[1] * raw"]\)"
    else
        return base
    end
end

function load_experiment_config(experiment_root::String)
    cfg_path = joinpath(experiment_root, "config", "experiment_config.jld2")
    isfile(cfg_path) || error("Could not find config file at $cfg_path")
    @load cfg_path config
    return adjust_config_paths(config, experiment_root)
end

function load_treatment_data(experiment_root::String)
    csv_path = joinpath(experiment_root, "avg_results", "treatment_data.csv")
    isfile(csv_path) || error("Missing treatment summary at $csv_path")
    df = CSV.read(csv_path, DataFrame)
    rename!(df, Symbol.(names(df)))
    return df
end

function compute_treatment_std(config::ExperimentConfig)
    data_path = joinpath(config.simulations_dir, "all_policies_simulation_data.jld2")
    if !isfile(data_path)
        @warn "Simulation data not found at $data_path; cannot compute standard deviations."
        return TreatmentStats()
    end

    @load data_path data
    processed = AquaOpt.extract_reward_metrics(data, config)
    if !(:treatments in propertynames(processed))
        @warn "Processed data is missing treatment counts; cannot compute standard deviations."
        return TreatmentStats()
    end

    grouped = groupby(processed, :policy)
    stats = TreatmentStats()
    for grp in grouped
        policy = grp.policy[1]
        action_stats = Dict{String, Union{Missing, Float64}}()
        for col in TREATMENT_COLUMNS
            action_obj = TREATMENT_ACTIONS[col]
            counts = [get(row.treatments, action_obj, 0) for row in eachrow(grp)]
            action_stats[col] = isempty(counts) ? missing : (length(counts) == 1 ? 0.0 : std(counts))
        end
        stats[policy] = action_stats
    end
    return stats
end

function load_experiment(experiment_root::String, label::String="")
    abs_root = abspath(experiment_root)
    isdir(abs_root) || error("Experiment folder does not exist: $abs_root")
    config = load_experiment_config(abs_root)
    treatment = load_treatment_data(abs_root)
    if isempty(label)
        label = extract_lambda_label(abs_root)
    end
    treatment_std = compute_treatment_std(config)
    return ExperimentSummary(basename(abs_root), label, config, treatment, treatment_std)
end

function format_value(mean_value, std_value=missing)
    if ismissing(mean_value) || (mean_value isa Float64 && isnan(mean_value))
        return "--"
    else
        mean_str = @sprintf("%.2f", Float64(mean_value))
        if ismissing(std_value)
            return mean_str
        end
        std_num = Float64(std_value)
        if isnan(std_num)
            return mean_str
        end
        std_str = @sprintf("%.2f", std_num)
        return string("\\(", mean_str, " \\pm ", std_str, "\\)")
    end
end

function format_lambda_value(value)
    if value === missing || isnan(value)
        return "--"
    else
        return @sprintf("%.2f", Float64(value))
    end
end

function fetch_policy_value(df::DataFrame, policy::String, column::String)
    idx = findfirst(==(policy), df.policy)
    if idx === nothing
        return missing
    else
        return df[idx, Symbol(column)]
    end
end

function fetch_policy_std(stats::TreatmentStats, policy::String, column::String)
    policy_stats = get(stats, policy, nothing)
    if policy_stats === nothing
        return missing
    else
        return get(policy_stats, column, missing)
    end
end

function build_table(entries::Vector{ExperimentSummary})
    isempty(entries) && error("No experiments provided.")
    col_spec = "ll" * join(fill("c", length(POLICY_ORDER)), "")
    lines = String[]
    push!(lines, "\\footnotesize")
    push!(lines, "\\begin{tabular}{$col_spec}")
    push!(lines, "\\toprule")

    header_row = ["Location", "Treatment"]
    append!(header_row, [get(POLICY_LABELS, policy, policy) for policy in POLICY_ORDER])
    push!(lines, join(header_row, " & ") * " \\\\")
    push!(lines, "\\midrule")

    for (entry_idx, entry) in enumerate(entries)
        for (col_idx, col) in enumerate(TREATMENT_COLUMNS)
            row = String[]
            push!(row, col_idx == 1 ? entry.label : "")
            push!(row, TREATMENT_LABELS[col])
            for policy in POLICY_ORDER
                mean_value = fetch_policy_value(entry.treatment, policy, col)
                std_value = fetch_policy_std(entry.treatment_std, policy, col)
                push!(row, format_value(mean_value, std_value))
            end
            push!(lines, join(row, " & ") * " \\\\")
        end
        if entry_idx != length(entries)
            push!(lines, "\\addlinespace")
        end
    end

    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    return join(lines, "\n")
end

function save_table(entries::Vector{ExperimentSummary}; output_path::String=TABLE_OUTPUT_PATH)
    table_tex = build_table(entries)
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        write(io, table_tex)
    end
    return output_path
end

function build_lambda_table(entries::Dict{String, ExperimentSummary};
        row_order::Vector{String}=LAMBDA_REGION_ORDER,
        caption::String="", label::String="",  # unused, wrappers in paper
        row_header::String="Region")
    # Column keys (no underscores for pgfplotstable inline header)
    col_keys = ["scenario", "ltrt", "lreg", "lbio", "lfd", "llice"]
    col_displays = [row_header,
        raw"$\lambda_{\text{trt}}$", raw"$\lambda_{\text{reg}}$",
        raw"$\lambda_{\text{bio}}$", raw"$\lambda_{\text{fd}}$",
        raw"$\lambda_{\text{lice}}$"]

    lines = String[]
    push!(lines, "\\pgfplotstabletypeset[")
    push!(lines, "    col sep=&,")
    push!(lines, "    row sep=\\\\,")
    push!(lines, "    columns={$(join(col_keys, ", "))},")
    push!(lines, "    columns/scenario/.style={string type, column type=l, column name={$(col_displays[1])}},")
    for (i, key) in enumerate(col_keys[2:end])
        push!(lines, "    columns/$key/.style={string type, column type=c, column name={$(col_displays[i+1])}},")
    end
    push!(lines, "    every head row/.style={before row=\\toprule, after row=\\midrule},")
    push!(lines, "    every last row/.style={after row=\\bottomrule},")
    push!(lines, "]{")

    # Header row
    push!(lines, "    " * join(col_keys, " & ") * " \\\\")

    # Data rows
    for key in row_order
        entry = get(entries, key, nothing)
        if entry === nothing
            push!(lines, "    $key & " * join(fill("--", 5), " & ") * " \\\\")
        else
            lambdas = entry.config.solver_config.reward_lambdas
            vals = [comp.idx <= length(lambdas) ? format_lambda_value(lambdas[comp.idx]) : "--"
                    for comp in LAMBDA_COMPONENTS]
            push!(lines, "    $key & " * join(vals, " & ") * " \\\\")
        end
    end

    push!(lines, "}")
    return join(lines, "\n")
end

function save_lambda_table(entries::Dict{String, ExperimentSummary};
        output_path::String=LAMBDA_TABLE_OUTPUT_PATH, kwargs...)
    lambda_tex = build_lambda_table(entries; kwargs...)
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        write(io, lambda_tex)
    end
    return output_path
end

function build_lambda_treatment_table(entries::Dict{String, ExperimentSummary};
        scenario_order::Vector{String}=LAMBDA_SCENARIO_ORDER)
    ordered = [entries[k] for k in scenario_order if haskey(entries, k)]
    isempty(ordered) && error("No lambda experiments provided.")

    col_spec = "ll" * join(fill("c", length(POLICY_ORDER)), "")
    lines = String[]
    push!(lines, "\\footnotesize")
    push!(lines, "\\begin{tabular}{$col_spec}")
    push!(lines, "\\toprule")

    header_row = ["Configuration", "Treatment"]
    append!(header_row, [get(POLICY_LABELS, policy, policy) for policy in POLICY_ORDER])
    push!(lines, join(header_row, " & ") * " \\\\")
    push!(lines, "\\midrule")

    for (entry_idx, entry) in enumerate(ordered)
        for (col_idx, col) in enumerate(TREATMENT_COLUMNS)
            row = String[]
            push!(row, col_idx == 1 ? entry.label : "")
            push!(row, TREATMENT_LABELS[col])
            for policy in POLICY_ORDER
                mean_value = fetch_policy_value(entry.treatment, policy, col)
                std_value = fetch_policy_std(entry.treatment_std, policy, col)
                push!(row, format_value(mean_value, std_value))
            end
            push!(lines, join(row, " & ") * " \\\\")
        end
        if entry_idx != length(ordered)
            push!(lines, "\\addlinespace")
        end
    end

    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    return join(lines, "\n")
end

function save_lambda_treatment_table(entries::Dict{String, ExperimentSummary};
        output_path::String=LAMBDA_TREATMENT_TABLE_OUTPUT_PATH, kwargs...)
    tex = build_lambda_treatment_table(entries; kwargs...)
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        write(io, tex)
    end
    return output_path
end

function build_dominant_action_axis(config::ExperimentConfig;
        include_legend::Bool=false,
        include_axis_labels::Bool=true,
        axis_width::String="5.6cm",
        axis_height::String="4.8cm")
    policy_path = joinpath(config.policies_dir, "policies_pomdp_mdp.jld2")
    isfile(policy_path) || error("SARSOP policy not found at $policy_path")
    @load policy_path all_policies
    policy_bundle = all_policies["Native_SARSOP_Policy"]
    policy = policy_bundle.policy
    pomdp = policy_bundle.pomdp

    temp_range = 8.0:0.5:24.0
    sealice_range = 0.0:0.01:1.0
    fixed_sessile = 0.25
    fixed_motile = 0.25

    action_coords = Dict(action => Vector{Tuple{Float64, Float64}}() for (action, _) in AquaOpt.PLOS_ACTION_STYLE_ORDERED)

    state_space = collect(states(pomdp))
    n_states = length(state_space)

    for sealice_level in sealice_range
        for temp in temp_range
            pred_adult, pred_motile, pred_sessile = predict_next_abundances(
                sealice_level,
                fixed_motile,
                fixed_sessile,
                temp,
                config.solver_config.location,
                config.solver_config.reproduction_rate
            )

            if pomdp isa AquaOpt.SeaLiceLogPOMDP
                pred_adult = log(max(pred_adult, 1e-6))
            end

            belief = zeros(Float64, n_states)
            distances = [abs(s.SeaLiceLevel - pred_adult) for s in state_space]
            closest_idx = argmin(distances)
            belief[closest_idx] = 1.0

            chosen_action = try
                action(policy, belief)
            catch e
                @warn "Falling back to NoTreatment for temp=$temp, sealice=$sealice_level" exception=e
                AquaOpt.NoTreatment
            end
            push!(action_coords[chosen_action], (temp, sealice_level))
        end
    end

    # Use larger fonts so text stays readable after includegraphics scaling
    label_style = "color=black, font=\\normalsize"
    tick_style  = "color=black, font=\\normalsize"
    title_style = "color=black, font=\\normalsize\\bfseries"

    opts = Options(
        :xmin => first(temp_range),
        :xmax => last(temp_range),
        :ymin => first(sealice_range),
        :ymax => last(sealice_range),
        :width => axis_width,
        :height => axis_height,
        :title_style => title_style,
        :tick_label_style => tick_style,
        "axis background/.style" => Options("fill" => "white"),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.3",
    )

    if include_axis_labels
        opts[:xlabel] = "Sea Temperature (°C)"
        opts[:ylabel] = "Avg. Adult Female Sea Lice per Fish"
        opts[:xlabel_style] = label_style
        opts[:ylabel_style] = label_style
    end

    if include_legend
        opts["legend style"] = AquaOpt.plos_top_legend(columns=2)
    end

    ax = @pgf Axis(opts)
    for (act, style) in AquaOpt.PLOS_ACTION_STYLE_ORDERED
        coords = get(action_coords, act, Tuple{Float64, Float64}[])
        if !isempty(coords)
            push!(ax,
                Plot(
                    Options(
                        :only_marks => nothing,
                        :mark => style.marker,
                        :mark_size => "2pt",
                        :color => style.color,
                        "mark options" => style.mark_opts,
                    ),
                    Coordinates(coords)
                )
            )
            if include_legend
                push!(ax, @pgf("\\addlegendentry{$(style.label)}"))
            end
        elseif include_legend
            # Add legend-only entry for actions with no data using \addlegendimage
            legend_img = "\\addlegendimage{only marks, mark=$(style.marker), mark size=2pt, color=$(style.color), mark options=$(style.mark_opts)}"
            push!(ax, legend_img)
            push!(ax, "\\addlegendentry{$(style.label)}")
        end
    end

    return ax
end

function save_combined_dominant_plot(entries::Vector{ExperimentSummary};
        output_path::String=FIGURE_OUTPUT_PATH)
    axes = Vector{Axis}()
    for (idx, entry) in enumerate(entries)
        # Put legend in the middle subplot and position it above all plots
        include_legend = (idx == 2)  # Middle subplot
        ax = build_dominant_action_axis(entry.config; include_legend=include_legend, include_axis_labels=false)
        ax.options["title"] = entry.label

        # Position legend above the middle plot, centered over all three plots
        if include_legend
            ax.options["legend style"] = Options(
                "fill" => "white",
                "draw" => "black!40",
                "text" => "black",
                "font" => "\\normalsize",
                "at" => "{(0.5,1.25)}",
                "anchor" => "south",
                "row sep" => "1pt",
                "column sep" => "0.5cm",
                "legend columns" => "4",
            )
        end

        push!(axes, ax)
    end

    label_style = "color=black, font=\\normalsize"
    group_opts = Options(
        "group style" => "{group size=$(length(axes)) by 1, horizontal sep=1.1cm, x descriptions at=edge bottom, y descriptions at=edge left}",
        :xlabel => "Sea Temperature (°C)",
        :ylabel => "Avg. Adult Female Sea Lice per Fish",
        :xlabel_style => label_style,
        :ylabel_style => label_style,
    )
    plot_obj = @pgf GroupPlot(group_opts, axes...)
    mkpath(dirname(output_path))
    PGFPlotsX.save(output_path, plot_obj)

    # Also save PDF version
    pdf_path = replace(output_path, ".tex" => ".pdf")
    PGFPlotsX.save(pdf_path, plot_obj, include_preamble=false)

    return output_path
end

function save_quad_dominant_plot(entries_dict::Dict{String, ExperimentSummary};
        output_path::String=replace(FIGURE_OUTPUT_PATH, ".tex" => "_quad.tex"))

    # Define the order: Northern Norway, Southern Norway (top row), Scotland, Chile (bottom row)
    plot_order = ["Northern Norway", "Southern Norway", "Scotland", "Chile"]

    axes = Vector{Axis}()
    for (idx, region_name) in enumerate(plot_order)
        if !haskey(entries_dict, region_name)
            @warn "Region $region_name not found in experiment data, skipping"
            continue
        end

        entry = entries_dict[region_name]

        # Put legend above the top-left plot (NorthernNorway)
        include_legend = (idx == 1)
        ax = build_dominant_action_axis(entry.config; include_legend=include_legend, include_axis_labels=false)
        ax.options["title"] = entry.label

        # Position legend above the first plot
        if include_legend
            ax.options["legend style"] = Options(
                "fill" => "white",
                "draw" => "black!40",
                "text" => "black",
                "font" => "\\normalsize",
                "at" => "{(1.0,1.25)}",  # Position above first subplot
                "anchor" => "south",
                "row sep" => "1pt",
                "column sep" => "0.5cm",
                "legend columns" => "4",
            )
        end

        label_style = "color=black, font=\\normalsize"

        # Add x-label only to bottom row plots (idx 3, 4)
        if idx >= 3
            ax.options["xlabel"] = "Sea Temperature (°C)"
            ax.options["xlabel style"] = label_style
        end

        # Add y-label only to left column plots (idx 1, 3)
        if idx == 1 || idx == 3
            ax.options["ylabel"] = "Avg. AF Sea Lice per Fish"
            ax.options["ylabel style"] = label_style
        end

        push!(axes, ax)
    end

    # Create 2x2 grid layout - don't add xlabel/ylabel here since we added them to individual axes
    group_opts = Options(
        "group style" => "{group size=2 by 2, horizontal sep=1.1cm, vertical sep=1.2cm}",
    )
    plot_obj = @pgf GroupPlot(group_opts, axes...)
    mkpath(dirname(output_path))
    PGFPlotsX.save(output_path, plot_obj)

    # Also save PDF version
    pdf_path = replace(output_path, ".tex" => ".pdf")
    PGFPlotsX.save(pdf_path, plot_obj, include_preamble=false)

    return output_path
end

function main()
    if length(EXPERIMENT_FOLDERS) < 3
        @warn "Expected at least three experiment folders; found $(length(EXPERIMENT_FOLDERS))."
    end

    # Load all experiments into a vector and dict
    entries = ExperimentSummary[]
    entries_dict = Dict{String, ExperimentSummary}()

    for (label, folder) in EXPERIMENT_FOLDERS
        entry = load_experiment(folder, label)
        push!(entries, entry)
        entries_dict[label] = entry
    end

    # Define order for 3-plot figure: Southern Norway, Scotland, Chile
    three_plot_order = ["Southern Norway", "Scotland", "Chile"]
    three_plot_entries = [entries_dict[name] for name in three_plot_order if haskey(entries_dict, name)]

    # Generate table (using the 3 selected entries)
    table_path = save_table(three_plot_entries)

    # Build lambda scenario entries for the lambda table
    lambda_entries = Dict{String, ExperimentSummary}()
    for (scenario_label, folder) in LAMBDA_FOLDERS
        lambda_entries[scenario_label] = load_experiment(folder, scenario_label)
    end
    lambda_table_path = save_lambda_table(lambda_entries;
        row_order=LAMBDA_SCENARIO_ORDER,
        caption="Reward weight configurations for lambda sensitivity analysis.",
        label="tab:lambda_params",
        row_header="Scenario")
    lambda_treatment_path = save_lambda_treatment_table(lambda_entries;
        scenario_order=LAMBDA_SCENARIO_ORDER)

    # Generate original 1-row plot with Southern Norway, Scotland, Chile
    figure_path = save_combined_dominant_plot(three_plot_entries)

    # Generate new 2x2 quad plot if we have all 4 regions
    if length(entries_dict) >= 4
        quad_path = save_quad_dominant_plot(entries_dict)
        println("Wrote quad dominant action figure (.tex) to $(abspath(quad_path)).")
    end

    println("Wrote treatment summary table to $(abspath(table_path)).")
    println("Wrote lambda parameter table to $(abspath(lambda_table_path)).")
    println("Wrote lambda treatment summary to $(abspath(lambda_treatment_path)).")
    println("Wrote dominant action figure (.tex) to $(abspath(figure_path)).")

    # Generate lambda decision maps (week × lice heatmaps side by side)
    lambda_order = ["Balanced", "Cost-Saving", "Welfare"]
    lambda_configs = [load_experiment_config(LAMBDA_FOLDERS[k]) for k in lambda_order]
    output_dir = "results/latest/reward_outputs"
    @info "Generating lambda decision maps..."
    plos_one_lambda_decision_maps(lambda_configs, lambda_order, "Native_SARSOP_Policy";
        output_dir=output_dir)
    println("Wrote lambda decision maps to $(abspath(joinpath(output_dir, "lambda_decision_maps.pdf"))).")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

#!/usr/bin/env julia

isnothing(Base.active_project()) && @warn "No active Julia project detected. Run this script with `julia --project=.` to ensure dependencies are available."

using AquaOpt
using CSV
using DataFrames
using JLD2
using PGFPlotsX
using PGFPlotsX: @pgf, Axis, Plot, Coordinates, GroupPlot, Options
using POMDPs: action, states
using Printf

const EXPERIMENT_FOLDERS = [
    "results/experiments/2025-11-18/2025-11-18T22:40:00.463_log_space_ukf_paper_north_[0.46, 0.12, 0.12, 0.18, 0.12]",
    "results/experiments/2025-11-18/2025-11-18T21:28:47.647_log_space_ukf_paper_north_[0.4, 0.2, 0.1, 0.0, 0.1]",
    "results/experiments/2025-11-18/2025-11-18T22:08:47.305_log_space_ukf_paper_north_[0.4, 0.1, 0.1, 0.5, 0.2]",
]

const TARGET_LAMBDA = 0.6
const TABLE_OUTPUT_PATH = "Quick_Access/lambda_treatment_summary.tex"
const FIGURE_OUTPUT_PATH = "Quick_Access/lambda_dominant_actions.tex"

const TREATMENT_COLUMNS = ["NoTreatment", "MechanicalTreatment", "ChemicalTreatment", "ThermalTreatment"]
const TREATMENT_LABELS = Dict(
    "NoTreatment" => "No Tx",
    "MechanicalTreatment" => "Mechanical",
    "ChemicalTreatment" => "Chemical",
    "ThermalTreatment" => "Thermal",
)

const POLICY_ORDER = [
    "NeverTreat_Policy",
    "Random_Policy",
    "Heuristic_Policy",
    "AlwaysTreat_Policy",
    "VI_Policy",
    "QMDP_Policy",
    "NUS_SARSOP_Policy",
]

const POLICY_LABELS = Dict(
    "NeverTreat_Policy" => "Never Treat",
    "Random_Policy" => "Random",
    "Heuristic_Policy" => "Heuristic",
    "AlwaysTreat_Policy" => "Always Treat",
    "VI_Policy" => "VI",
    "QMDP_Policy" => "QMDP",
    "NUS_SARSOP_Policy" => "SARSOP",
)

struct ExperimentSummary
    name::String
    label::String
    config::ExperimentConfig
    treatment::DataFrame
end

function adjust_config_paths!(config::ExperimentConfig, experiment_root::String)
    config.experiment_dir = experiment_root
    config.policies_dir = joinpath(experiment_root, "policies")
    config.simulations_dir = joinpath(experiment_root, "simulation_histories")
    config.results_dir = joinpath(experiment_root, "avg_results")
    config.figures_dir = joinpath(experiment_root, "figures")
    return config
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
    return adjust_config_paths!(config, experiment_root)
end

function load_treatment_data(experiment_root::String; λ::Float64=TARGET_LAMBDA)
    csv_path = joinpath(experiment_root, "avg_results", "treatment_data_lambda_$(λ).csv")
    isfile(csv_path) || error("Missing treatment summary at $csv_path")
    df = CSV.read(csv_path, DataFrame)
    rename!(df, Symbol.(names(df)))
    return df
end

function load_experiment(experiment_root::String)
    abs_root = abspath(experiment_root)
    isdir(abs_root) || error("Experiment folder does not exist: $abs_root")
    config = load_experiment_config(abs_root)
    treatment = load_treatment_data(abs_root)
    label = extract_lambda_label(abs_root)
    return ExperimentSummary(basename(abs_root), label, config, treatment)
end

function format_value(value)
    if value === missing || isnan(value)
        return "--"
    else
        return @sprintf("%.3f", value)
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

function build_table(entries::Vector{ExperimentSummary})
    isempty(entries) && error("No experiments provided.")
    col_spec = "l" * join(fill("cccc", length(entries)), "")
    lines = String[]
    push!(lines, "\\begin{table}[htbp]")
    push!(lines, "\\centering")
    push!(lines, "\\footnotesize")
    push!(lines, "\\begin{tabular}{$col_spec}")
    push!(lines, "\\toprule")

    first_row = ["Policy"]
    for entry in entries
        push!(first_row, "\\multicolumn{4}{c}{" * entry.label * "}")
    end
    push!(lines, join(first_row, " & ") * " \\\\")

    cmidrules = String[]
    for (i, _) in enumerate(entries)
        start_col = 2 + 4 * (i - 1)
        stop_col = start_col + 3
        push!(cmidrules, "\\cmidrule(lr){$(start_col)-$(stop_col)}")
    end
    push!(lines, join(cmidrules, " "))

    header_row = ["Policy"]
    for _ in entries
        append!(header_row, [TREATMENT_LABELS[col] for col in TREATMENT_COLUMNS])
    end
    push!(lines, join(header_row, " & ") * " \\\\")
    push!(lines, "\\midrule")

    for policy in POLICY_ORDER
        row = [get(POLICY_LABELS, policy, policy)]
        for entry in entries
            for col in TREATMENT_COLUMNS
                value = fetch_policy_value(entry.treatment, policy, col)
                push!(row, format_value(value))
            end
        end
        push!(lines, join(row, " & ") * " \\\\")
    end

    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\caption{Average number of treatments per policy for each \\(\\lambda\\) combination.}")
    push!(lines, "\\label{tab:lambda_treatment_summary}")
    push!(lines, "\\end{table}")
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

function build_dominant_action_axis(config::ExperimentConfig;
        λ::Float64=TARGET_LAMBDA,
        include_legend::Bool=false,
        axis_width::String="5.6cm",
        axis_height::String="4.8cm")
    policy_path = joinpath(config.policies_dir, "NUS_SARSOP_Policy", "policy_pomdp_mdp_$(λ)_lambda.jld2")
    isfile(policy_path) || error("SARSOP policy not found at $policy_path")
    @load policy_path policy pomdp mdp

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

    opts = Options(
        :xlabel => "Sea Temperature (°C)",
        :ylabel => "Avg. Adult Female Sea Lice per Fish",
        :xmin => first(temp_range),
        :xmax => last(temp_range),
        :ymin => first(sealice_range),
        :ymax => last(sealice_range),
        :width => axis_width,
        :height => axis_height,
        :title_style => AquaOpt.PLOS_TITLE_STYLE,
        :xlabel_style => AquaOpt.PLOS_LABEL_STYLE,
        :ylabel_style => AquaOpt.PLOS_LABEL_STYLE,
        :tick_label_style => AquaOpt.PLOS_TICK_STYLE,
        "axis background/.style" => Options("fill" => "white"),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.3",
    )
    if include_legend
        opts["legend style"] = AquaOpt.plos_top_legend(columns=2)
    end

    ax = @pgf Axis(opts)
    for (act, style) in AquaOpt.PLOS_ACTION_STYLE_ORDERED
        coords = get(action_coords, act, Tuple{Float64, Float64}[])
        isempty(coords) && continue
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
    end

    return ax
end

function save_combined_dominant_plot(entries::Vector{ExperimentSummary};
        λ::Float64=TARGET_LAMBDA,
        output_path::String=FIGURE_OUTPUT_PATH)
    axes = Vector{Axis}()
    for (idx, entry) in enumerate(entries)
        include_legend = idx == 1
        ax = build_dominant_action_axis(entry.config; λ=λ, include_legend=include_legend)
        ax.options["title"] = entry.label
        push!(axes, ax)
    end

    group_opts = Options(
        "group style" => "{group size=$(length(axes)) by 1, horizontal sep=1.1cm}",
    )
    plot_obj = @pgf GroupPlot(group_opts, axes...)
    mkpath(dirname(output_path))
    PGFPlotsX.save(output_path, plot_obj)
    return output_path
end

function main()
    if length(EXPERIMENT_FOLDERS) != 3
        @warn "Expected three experiment folders; found $(length(EXPERIMENT_FOLDERS))."
    end
    entries = ExperimentSummary[]
    for folder in EXPERIMENT_FOLDERS
        push!(entries, load_experiment(folder))
    end

    table_path = save_table(entries)
    figure_path = save_combined_dominant_plot(entries)

    println("Wrote LaTeX table to $(abspath(table_path)).")
    println("Wrote dominant action figure (.tex) to $(abspath(figure_path)).")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

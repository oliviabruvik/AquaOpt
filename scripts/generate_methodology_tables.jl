#!/usr/bin/env julia

isnothing(Base.active_project()) && @warn "No active Julia project detected. Run this script with `julia --project=.` so dependencies resolve correctly."

using AquaOpt
using JLD2
using Printf: @sprintf
using CSV
using DataFrames

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))

const EXPERIMENT_PATH = joinpath(
    REPO_ROOT,
    "results",
    "experiments",
    "2025-11-19",
    "2025-11-19T22:18:33.024_log_space_ukf_paper_north_[0.46, 0.12, 0.12, 0.18, 0.12]"
)

const OUTPUT_DIR = joinpath(REPO_ROOT, "final_results", "methodology_tables")
const DEFAULT_SIM_POMDP = AquaOpt.SeaLiceSimPOMDP()
const DEFAULT_INITIAL_FISH = 200_000
const SOURCE_ALDRIN = "Aldrin et al. (2023)"
const SOURCE_MODEL = "Model assumption"
const SOURCE_STIGE = "Stige et al. (2025)"
const TREATMENT_ACTIONS = (NoTreatment, MechanicalTreatment, ThermalTreatment, ChemicalTreatment)
const TREATMENT_SYMBOLS = (raw"$a_1$", raw"$a_2$", raw"$a_3$", raw"$a_4$")
const POLICY_ORDER_SUMMARY = [
    "Heuristic_Policy",
    "QMDP_Policy",
    "NUS_SARSOP_Policy",
    "VI_Policy",
]

function load_experiment_config()
    cfg_path = joinpath(EXPERIMENT_PATH, "config", "experiment_config.jld2")
    isfile(cfg_path) || error("Could not find config file at $(abspath(cfg_path))")
    @load cfg_path config
    return config
end

# --- Formatting helpers -------------------------------------------------------

function strip_trailing_zeros(s::String)
    if occursin('.', s)
        stripped = replace(s, r"0+$" => "")
        stripped = replace(stripped, r"\.$" => "")
        return stripped == "" ? "0" : stripped
    else
        return s
    end
end

format_float(value::Real; digits::Int=3) =
    strip_trailing_zeros(@sprintf("%.*f", digits, Float64(value)))

format_number(value::Integer; digits::Int=0) = digits == 0 ? string(value) : strip_trailing_zeros(@sprintf("%.*f", digits, Float64(value)))
format_number(value::Real; digits::Int=3) = format_float(value; digits=digits)

format_bool(flag::Bool) = flag ? "\\texttt{true}" : "\\texttt{false}"

function format_text(value::String)
    escaped = replace(value, "_" => "\\_")
    return "\\texttt{$escaped}"
end

function latex_array(values::AbstractVector{<:Real}; digits::Int=2)
    inner = join((format_number(v; digits=digits) for v in values), ", ")
    return raw"$[" * inner * raw"]$"
end

function latex_pm(mean_value::Real, sd_value::Real; digits::Int=3, sd_digits::Int=3)
    mean_str = format_number(mean_value; digits=digits)
    sd_str = format_number(sd_value; digits=sd_digits)
    return raw"$" * mean_str * raw"\pm " * sd_str * raw"$"
end

function latex_seconds(value::Real)
    seconds = value isa Integer ? string(value) : format_number(value; digits=2)
    return "\\SI{$seconds}{\\second}"
end

function latex_math(value::Real; digits::Int=3)
    return raw"$" * format_number(value; digits=value isa Integer ? 0 : digits) * raw"$"
end

format_fixed(value::Real; digits::Int=2) = @sprintf("%.*f", digits, Float64(value))
latex_fixed(value::Real; digits::Int=2) = raw"$" * format_fixed(value; digits=digits) * raw"$"

function latex_percent(value::Real; digits::Int=0)
    return latex_math(value * 100; digits=digits)
end

function latex_interval(bounds::Tuple{<:Real, <:Real}; digits::Int=2)
    lo_digits = bounds[1] isa Integer ? 0 : digits
    hi_digits = bounds[2] isa Integer ? 0 : digits
    lo = format_number(bounds[1]; digits=lo_digits)
    hi = format_number(bounds[2]; digits=hi_digits)
    return raw"$(" * lo * ", " * hi * raw")$"
end

# --- Table builder ------------------------------------------------------------

function latex_table(rows::Vector{Tuple{String, String}}; caption::String, label::String, column_names::Tuple{String, String}=("Parameter", "Value"), column_spec::String="@{} l l")
    isempty(rows) && error("Cannot build a LaTeX table with zero rows.")
    lines = String[]
    push!(lines, "\\begin{table}[htbp]")
    push!(lines, "    \\centering")
    push!(lines, "    \\caption{$caption}")
    push!(lines, "    \\label{$label}")
    push!(lines, "    \\begin{tabular}{$column_spec}")
    push!(lines, "        \\toprule")
    push!(lines, "        $(column_names[1]) & $(column_names[2])\\\\")
    push!(lines, "        \\midrule")
    for (name, value) in rows
        push!(lines, "        $name & $value\\\\")
    end
    push!(lines, "        \\bottomrule")
    push!(lines, "    \\end{tabular}")
    push!(lines, "\\end{table}")
    return join(lines, "\n")
end

function write_table(rows::Vector{Tuple{String, String}}; filename::String, caption::String, label::String)
    mkpath(OUTPUT_DIR)
    table_tex = latex_table(rows; caption=caption, label=label)
    output_path = joinpath(OUTPUT_DIR, filename)
    open(output_path, "w") do io
        write(io, table_tex)
    end
    return output_path
end

function write_lambda_table(headers::Vector{String}, rows::Vector{Vector{String}}; filename::String, caption::String, label::String)
    mkpath(OUTPUT_DIR)
    lines = String[]
    push!(lines, "\\begin{table}[htbp]")
    push!(lines, "    \\centering")
    push!(lines, "    \\caption{$caption}")
    push!(lines, "    \\label{$label}")
    col_spec = "@{} " * join(fill("c", length(headers)), " ") * " @{}"
    push!(lines, "    \\begin{tabular}{" * col_spec * "}")
    push!(lines, "        \\toprule")
    push!(lines, "        " * join(headers, " & ") * " \\\\")
    push!(lines, "        \\midrule")
    for row in rows
        push!(lines, "        " * join(row, " & ") * " \\\\")
    end
    push!(lines, "        \\bottomrule")
    push!(lines, "    \\end{tabular}")
    push!(lines, "\\end{table}")
    output_path = joinpath(OUTPUT_DIR, filename)
    open(output_path, "w") do io
        write(io, join(lines, "\n"))
    end
    return output_path
end

function write_location_table(rows; filename::String, caption::String, label::String)
    mkpath(OUTPUT_DIR)
    lines = String[]
    push!(lines, "\\begin{table}[htbp!]")
    push!(lines, "\\centering")
    push!(lines, "    \\caption{$caption}")
    push!(lines, "    \\label{$label}")
    push!(lines, "    \\begin{tabular}{@{}lccc l@{}}")
    push!(lines, "    \\toprule")
    push!(lines, "    Parameter & North & West & South & Description \\\\")
    push!(lines, "    \\midrule")
    for row in rows
        push!(lines, "    $(row.parameter) & $(row.north) & $(row.west) & $(row.south) & $(row.description) \\\\")
    end
    push!(lines, "    \\bottomrule")
    push!(lines, "    \\end{tabular}")
    push!(lines, "\\end{table}")
    output_path = joinpath(OUTPUT_DIR, filename)
    open(output_path, "w") do io
        write(io, join(lines, "\n"))
    end
    return output_path
end

function build_fish_population_table(rows; caption::String, label::String)
    isempty(rows) && error("Cannot build an empty fish population table.")
    lines = String[]
    push!(lines, "\\begin{table}[htbp!]")
    push!(lines, "\\centering")
    push!(lines, "\\begin{adjustwidth}{-2.25in}{0in}")
    push!(lines, "    \\caption{$caption}")
    push!(lines, "    \\label{$label}")
    push!(lines, "    \\begin{threeparttable}")
    push!(lines, "    \\begin{adjustbox}{max width=\\linewidth}")
    push!(lines, "    \\begin{tabular}{@{}lrrl@{}}")
    push!(lines, "    \\arrayrulecolor{black}")
    push!(lines, "    \\toprule")
    push!(lines, "    Parameter & Value & Description & Source \\\\")
    push!(lines, "    \\midrule")
    push!(lines, "    \\arrayrulecolor{white}")
    for row in rows
        push!(lines, "    $(row.parameter) & $(row.value) & $(row.description) & $(row.source) \\\\")
    end
    push!(lines, "    \\arrayrulecolor{black}")
    push!(lines, "    \\bottomrule")
    push!(lines, "    \\end{tabular}")
    push!(lines, "    \\end{adjustbox}")
    push!(lines, "    \\end{threeparttable}")
    push!(lines, "\\end{adjustwidth}")
    push!(lines, "\\end{table}")
    return join(lines, "\n")
end

function write_fish_population_table(rows; filename::String, caption::String, label::String)
    mkpath(OUTPUT_DIR)
    output_path = joinpath(OUTPUT_DIR, filename)
    table_tex = build_fish_population_table(rows; caption=caption, label=label)
    open(output_path, "w") do io
        write(io, table_tex)
    end
    return output_path
end

function build_sectioned_table(sections; caption::String, label::String)
    lines = String[]
    push!(lines, "\\begin{table}[htbp!]")
    push!(lines, "\\centering")
    push!(lines, "\\begin{adjustwidth}{-2.25in}{0in}")
    push!(lines, "    \\caption{$caption}")
    push!(lines, "    \\label{$label}")
    push!(lines, "    \\begin{threeparttable}")
    push!(lines, "    \\begin{adjustbox}{max width=\\linewidth}")
    push!(lines, "    \\begin{tabular}{@{}lrrrl@{}}")
    push!(lines, "    \\arrayrulecolor{black}")
    push!(lines, "    \\toprule")
    push!(lines, "    & Parameter & Value & Description & Source \\\\")
    push!(lines, "    \\midrule")
    push!(lines, "    \\arrayrulecolor{white}")
    for (idx, (label_text, entries)) in enumerate(sections)
        multirow = "\\multirow{$(length(entries))}{*}{$(label_text)}"
        for (j, entry) in enumerate(entries)
            category_cell = j == 1 ? multirow : ""
            push!(lines, "    $category_cell & $(entry.parameter) & $(entry.value) & $(entry.description) & $(entry.source) \\\\")
        end
        idx < length(sections) && push!(lines, "    \\midrule")
    end
    push!(lines, "    \\arrayrulecolor{black}")
    push!(lines, "    \\bottomrule")
    push!(lines, "    \\end{tabular}")
    push!(lines, "    \\end{adjustbox}")
    push!(lines, "    \\end{threeparttable}")
    push!(lines, "\\end{adjustwidth}")
    push!(lines, "\\end{table}")
    return join(lines, "\n")
end

function write_sectioned_table(sections; filename::String, caption::String, label::String)
    mkpath(OUTPUT_DIR)
    output_path = joinpath(OUTPUT_DIR, filename)
    table_tex = build_sectioned_table(sections; caption=caption, label=label)
    open(output_path, "w") do io
        write(io, table_tex)
    end
    return output_path
end

function build_treatment_table(rows; caption::String, label::String)
    lines = String[]
    push!(lines, "\\begin{table}[htbp!]")
    push!(lines, "  \\centering")
    push!(lines, "  \\caption{$caption}")
    push!(lines, "  \\label{$label}")
    push!(lines, "  \\begin{tabular}{@{} l c c c c l @{} }")
    push!(lines, "    \\toprule")
    push!(lines, "    Treatment & None & Mechanical & Thermal & Chemical & Source \\\\")
    push!(lines, "    \\midrule")
    for row in rows
        push!(lines, "    $(row.label) & $(row.none) & $(row.mechanical) & $(row.thermal) & $(row.chemical) & $(row.source) \\\\")
    end
    push!(lines, "    \\bottomrule")
    push!(lines, "  \\end{tabular}")
    push!(lines, "\\end{table}")
    return join(lines, "\n")
end

function write_treatment_table(rows; filename::String, caption::String, label::String)
    mkpath(OUTPUT_DIR)
    output_path = joinpath(OUTPUT_DIR, filename)
    table_tex = build_treatment_table(rows; caption=caption, label=label)
    open(output_path, "w") do io
        write(io, table_tex)
    end
    return output_path
end

# --- Row builders -------------------------------------------------------------

function solver_model_rows(config::ExperimentConfig)
    sc = config.solver_config
    return [
        ("Deployment region", format_text(sc.location)),
        ("Discount factor", format_number(sc.discount_factor; digits=2)),
        ("Regulation limit", format_number(sc.regulation_limit; digits=2)),
        ("Reproduction rate", format_number(sc.reproduction_rate; digits=2)),
        ("State discretization step", format_number(sc.discretization_step; digits=2)),
        ("Raw-space sampling sd", format_number(sc.adult_sd; digits=2)),
        ("Solver reward weights \\(\\lambda\\)", latex_array(sc.reward_lambdas)),
        ("Log-space solver", format_bool(sc.log_space)),
    ]
end

function solver_runtime_rows(config::ExperimentConfig)
    sc = config.solver_config
    return [
        ("Max SARSOP runtime", latex_seconds(sc.sarsop_max_time)),
        ("VI iterations", format_number(sc.VI_max_iterations)),
        ("QMDP iterations", format_number(sc.QMDP_max_iterations)),
        ("Full observability solver", format_bool(sc.full_observability_solver)),
        ("Heuristic threshold", format_number(sc.heuristic_threshold; digits=2)),
        ("Mechanical belief threshold", format_number(sc.heuristic_belief_threshold_mechanical; digits=2)),
        ("Chemical belief threshold", format_number(sc.heuristic_belief_threshold_chemical; digits=2)),
        ("Thermal belief threshold", format_number(sc.heuristic_belief_threshold_thermal; digits=2)),
        ("Heuristic \\(\\rho\\)", format_number(sc.heuristic_rho; digits=2)),
    ]
end

function simulation_rows(config::ExperimentConfig)
    sim = config.simulation_config
    return [
        ("Episodes per policy", format_number(sim.num_episodes)),
        ("Steps per episode", format_number(sim.steps_per_episode)),
        ("High-fidelity simulator", format_bool(sim.high_fidelity_sim)),
        ("EKF belief filter", format_bool(sim.ekf_filter)),
        ("Step-through mode", format_bool(sim.step_through)),
        (raw"Adult lice mean $\\pm$ sd", latex_pm(sim.adult_mean, sim.adult_sd)),
        (raw"Motile lice mean $\\pm$ sd", latex_pm(sim.motile_mean, sim.motile_sd)),
        (raw"Sessile lice mean $\\pm$ sd", latex_pm(sim.sessile_mean, sim.sessile_sd)),
        ("Temperature noise sd", format_number(sim.temp_sd; digits=2)),
        ("Samples per observation", format_number(sim.n_sample)),
        (raw"$\\rho_{\\text{adult}}$ detection probability", format_number(sim.ρ_adult; digits=3)),
        (raw"$\\rho_{\\text{motile}}$ detection probability", format_number(sim.ρ_motile; digits=3)),
        (raw"$\\rho_{\\text{sessile}}$ detection probability", format_number(sim.ρ_sessile; digits=3)),
        ("Under-reporting enabled", format_bool(sim.use_underreport)),
        (raw"$\\beta_0$ bias term", format_number(sim.beta0_Scount_f; digits=3)),
        (raw"$\\beta_1$ slope term", format_number(sim.beta1_Scount; digits=3)),
        ("Mean fish weight (kg)", format_number(sim.mean_fish_weight_kg; digits=2)),
        (raw"Initial biomass $W_0$", format_number(sim.W0; digits=2)),
        ("Simulation reward weights \\(\\lambda\\)", latex_array(sim.sim_reward_lambdas)),
    ]
end

function fish_population_rows(config::ExperimentConfig)
    sim = DEFAULT_SIM_POMDP
    initial_weight = config.simulation_config.W0
    return [
        (parameter=raw"Asymptotic weight ($w_{\max}$)", value=latex_math(sim.w_max; digits=1), description="Maximum harvest weight (kg)", source=SOURCE_MODEL),
        (parameter=raw"Growth rate ($k_{\text{growth}}$)", value=latex_math(sim.k_growth; digits=2), description="Weekly von Bertalanffy growth rate", source=SOURCE_ALDRIN),
        (parameter=raw"Temperature sensitivity ($\alpha_T$)", value=latex_math(sim.temp_sensitivity; digits=2), description="Temperature effect on growth (°C)", source=SOURCE_ALDRIN),
        (parameter="Natural mortality rate", value=latex_math(sim.nat_mort_rate; digits=4), description="Weekly natural mortality fraction", source=SOURCE_ALDRIN),
        (parameter="Treatment mortality bump", value=latex_math(sim.trt_mort_bump; digits=3), description="Extra mortality applied in treatment weeks", source=SOURCE_ALDRIN),
        (parameter="Production start week", value=latex_math(sim.production_start_week; digits=0), description="Week of production start", source=SOURCE_MODEL),
        (parameter="Weight bounds", value=latex_interval(sim.weight_bounds; digits=1), description="Fish weight range (\\si{\\kilogram})", source=SOURCE_MODEL),
        (parameter="Number of fish bounds", value=latex_interval(sim.number_of_fish_bounds; digits=0), description="Fish count range", source=SOURCE_MODEL),
        (parameter="Initial number of fish", value=latex_math(DEFAULT_INITIAL_FISH; digits=0), description="Fish count at production start", source=SOURCE_MODEL),
        (parameter="Initial fish weight", value=latex_math(initial_weight; digits=2), description="Average weight at production start (\\si{\\kilogram})", source=SOURCE_MODEL),
    ]
end

function sea_lice_biology_sections(config::ExperimentConfig)
    params = get_location_params(config.solver_config.location)
    survival_entries = [
        (parameter=raw"Sessile survival ($s_1$)", value=latex_math(params.s1_sessile; digits=2), description="Weekly sessile survival rate", source=SOURCE_STIGE),
        (parameter=raw"Sessile-to-motile scaling ($s_2$)", value=latex_math(params.s2_scaling; digits=1), description="Sessile to motile transition scaling", source=SOURCE_STIGE),
        (parameter=raw"Motile survival ($s_3$)", value=latex_math(params.s3_motile; digits=2), description="Weekly motile survival rate", source=SOURCE_STIGE),
        (parameter=raw"Adult survival ($s_4$)", value=latex_math(params.s4_adult; digits=2), description="Weekly adult survival rate", source=SOURCE_STIGE),
    ]
    development_entries = [
        (parameter=raw"$d_1$ intercept ($c_1$)", value=latex_math(params.d1_intercept; digits=1), description="Sessile-to-motile development intercept", source=SOURCE_STIGE),
        (parameter=raw"$d_1$ temperature coefficient ($t_1$)", value=latex_math(params.d1_temp_coef; digits=2), description="Temperature effect on sessile-to-motile", source=SOURCE_STIGE),
        (parameter=raw"$d_2$ intercept ($c_2$)", value=latex_math(params.d2_intercept; digits=1), description="Motile-to-adult development intercept", source=SOURCE_STIGE),
        (parameter=raw"$d_2$ temperature coefficient ($t_2$)", value=latex_math(params.d2_temp_coef; digits=3), description="Temperature effect on motile-to-adult", source=SOURCE_STIGE),
        (parameter=raw"Temperature reference ($r$)", value=latex_math(9.0; digits=1), description="Reference temperature (\\si{\\celsius})", source=SOURCE_STIGE),
    ]
    return [
        ("\\textbf{Survival Probabilities}", survival_entries),
        ("\\textbf{Development Rate Parameters}", development_entries),
    ]
end

function observation_model_sections(config::ExperimentConfig)
    sim = config.simulation_config
    nb_entries = [
        (parameter=raw"Number of fish sampled ($n$)", value=latex_math(sim.n_sample; digits=0), description="Fish counted per monitoring event", source=SOURCE_ALDRIN),
        (parameter=raw"Adult aggregation parameter ($\rho_{\text{A}}$)", value=latex_math(sim.ρ_adult; digits=3), description="Over-dispersion for adult lice counts", source=SOURCE_ALDRIN),
        (parameter=raw"Motile aggregation parameter ($\rho_{\text{M}}$)", value=latex_math(sim.ρ_motile; digits=3), description="Over-dispersion for motile lice counts", source=SOURCE_ALDRIN),
        (parameter=raw"Sessile aggregation parameter ($\rho_{\text{S}}$)", value=latex_math(sim.ρ_sessile; digits=3), description="Over-dispersion for sessile lice counts", source=SOURCE_ALDRIN),
    ]
    undercount_entries = [
        (parameter=raw"Farm intercept ($\beta_0$)", value=latex_math(sim.beta0_Scount_f; digits=3), description="Logistic function intercept", source=SOURCE_ALDRIN),
        (parameter=raw"Weight slope ($\beta_1$)", value=latex_math(sim.beta1_Scount; digits=3), description="Weight-dependent correction factor", source=SOURCE_ALDRIN),
        (parameter=raw"Weight centering ($W_0$)", value=latex_math(sim.W0; digits=2), description="Reference weight (\\si{\\kilogram})", source=SOURCE_ALDRIN),
        (parameter=raw"Mean fish weight ($W_{\text{mean}}$)", value=latex_math(sim.mean_fish_weight_kg; digits=2), description="Average fish weight (\\si{\\kilogram})", source=SOURCE_ALDRIN),
    ]
    gaussian_entries = [
        (parameter=raw"Temperature noise ($\sigma_T$)", value=latex_math(sim.temp_sd; digits=2), description="Temperature measurement std dev (\\si{\\celsius})", source=SOURCE_MODEL),
        (parameter=raw"Fish count noise ($\sigma_N$)", value=latex_math(DEFAULT_SIM_POMDP.number_of_fish_sd; digits=1), description="Fish count measurement std dev", source=SOURCE_MODEL),
        (parameter=raw"Weight noise ($\sigma_W$)", value=latex_math(DEFAULT_SIM_POMDP.weight_sd; digits=2), description="Weight measurement std dev (\\si{\\kilogram})", source=SOURCE_MODEL),
    ]
    return [
        ("\\textbf{Negative Binomial Distribution Parameters}", nb_entries),
        ("\\textbf{Under-counting Correction Parameters}", undercount_entries),
        ("\\textbf{Gaussian Noise Parameters}", gaussian_entries),
    ]
end

const LOCATION_ORDER = ["north", "west", "south"]

function location_value(params::Dict{String, LocationParams}, loc::String, field::Symbol)
    value = getfield(params[loc], field)
    formatted = @sprintf("%0.2f", Float64(value))
    return raw"$" * formatted * raw"$"
end

function location_row(params::Dict{String, LocationParams}, label::String, field::Symbol, description::String)
    return (
        parameter=label,
        north=location_value(params, "north", field),
        west=location_value(params, "west", field),
        south=location_value(params, "south", field),
        description=description,
    )
end

function location_dynamics_rows()
    params = Dict(loc => get_location_params(loc) for loc in LOCATION_ORDER)
    rows = [
        (
            parameter=raw"Mean temperature ($T_{\text{mean}}$)",
            north=location_value(params, "north", :T_mean),
            west=location_value(params, "west", :T_mean),
            south=location_value(params, "south", :T_mean),
            description="Average annual sea temperature (\\si{\\celsius})",
        ),
        (
            parameter=raw"Temperature amplitude ($T_{\text{amp}}$)",
            north=location_value(params, "north", :T_amp),
            west=location_value(params, "west", :T_amp),
            south=location_value(params, "south", :T_amp),
            description="Seasonal temperature swing (\\si{\\celsius})",
        ),
        (
            parameter="Peak temperature week",
            north=location_value(params, "north", :peak_week),
            west=location_value(params, "west", :peak_week),
            south=location_value(params, "south", :peak_week),
            description="Week of maximum temperature",
        ),
        location_row(params, raw"$d_1$ intercept", :d1_intercept, "Sessile→motile development intercept"),
        location_row(params, raw"$d_1$ temperature coefficient", :d1_temp_coef, "Temperature effect on sessile→motile"),
        location_row(params, raw"$d_2$ intercept", :d2_intercept, "Motile→adult development intercept"),
        location_row(params, raw"$d_2$ temperature coefficient", :d2_temp_coef, "Temperature effect on motile→adult"),
        location_row(params, raw"Sessile survival ($s_1$)", :s1_sessile, "Weekly sessile survival probability"),
        location_row(params, raw"Sessile→motile scaling ($s_2$)", :s2_scaling, "Scaling from sessile to motile stage"),
        location_row(params, raw"Motile survival ($s_3$)", :s3_motile, "Weekly motile survival probability"),
        location_row(params, raw"Adult survival ($s_4$)", :s4_adult, "Weekly adult survival probability"),
        location_row(params, "External larval influx", :external_influx, "Weekly external sessile influx"),
    ]
    return rows
end

function region_environment_rows()
    params = Dict(loc => get_location_params(loc) for loc in LOCATION_ORDER)
    labels = Dict("north" => "North", "west" => "West", "south" => "South")
    rows = Vector{NamedTuple{(:region, :t_mean, :t_amp, :peak_week, :external_influx), NTuple{5, String}}}()
    for loc in LOCATION_ORDER
        par = params[loc]
        push!(rows, (
            region = labels[loc],
            t_mean = latex_math(par.T_mean; digits=1),
            t_amp = latex_math(par.T_amp; digits=1),
            peak_week = latex_math(par.peak_week; digits=0),
            external_influx = latex_math(par.external_influx; digits=2) * raw"\tnote{a}"
        ))
    end
    return rows
end

function write_region_environment_table(rows; filename::String, caption::String, label::String)
    mkpath(OUTPUT_DIR)
    lines = String[]
    push!(lines, "\\begin{table}[htbp!]")
    push!(lines, "\\centering")
    push!(lines, "    \\caption{$caption}")
    push!(lines, "    \\label{$label}")
    push!(lines, "    \\begin{threeparttable}")
    push!(lines, "    \\begin{tabular}{@{}lcccc@{}}")
    push!(lines, "    \\toprule")
    push!(lines, "    Region & \\(T_{\\text{mean}}\\) (\\(^{\\circ}\\)C) & \\(T_{\\text{amp}}\\) (\\(^{\\circ}\\)C) & \\(W_{\\text{peak}}\\) (week) & \\(\\lambda_{\\text{ext}}\\) \\\\")
    push!(lines, "    \\midrule")
    for row in rows
        push!(lines, "    $(row.region) & $(row.t_mean) & $(row.t_amp) & $(row.peak_week) & $(row.external_influx) \\\\")
    end
    push!(lines, "    \\bottomrule")
    push!(lines, "    \\end{tabular}")
    push!(lines, "    \\begin{tablenotes}")
    push!(lines, "    \\item[a] Model assumptions calibrated to reflect warmer-region larval pressure.")
    push!(lines, "    \\end{tablenotes}")
    push!(lines, "    \\end{threeparttable}")
    push!(lines, "\\end{table}")
    output_path = joinpath(OUTPUT_DIR, filename)
    open(output_path, "w") do io
        write(io, join(lines, "\n"))
    end
    return output_path
end

function noise_parameter_rows(config::ExperimentConfig)
    sim = config.simulation_config
    sim_defaults = DEFAULT_SIM_POMDP
    rows = [
        (symbol=raw"$\sigma_A$", value=latex_fixed(sim.adult_sd; digits=2), description="Adult lice process noise"),
        (symbol=raw"$\sigma_A^{\mathrm{obs}}$", value=latex_fixed(sim.adult_sd; digits=2), description="Adult lice observation noise (Kalman filter)"),
        (symbol=raw"$\sigma_M$", value=latex_fixed(sim.motile_sd; digits=2), description="Motile lice process noise"),
        (symbol=raw"$\sigma_M^{\mathrm{obs}}$", value=latex_fixed(sim.motile_sd; digits=2), description="Motile lice observation noise (Kalman filter)"),
        (symbol=raw"$\sigma_S$", value=latex_fixed(sim.sessile_sd; digits=2), description="Sessile lice process noise"),
        (symbol=raw"$\sigma_S^{\mathrm{obs}}$", value=latex_fixed(sim.sessile_sd; digits=2), description="Sessile lice observation noise (Kalman filter)"),
        (symbol=raw"$\sigma_T$", value=latex_fixed(sim.temp_sd; digits=2), description="Temperature process noise"),
        (symbol=raw"$\sigma_T^{\mathrm{obs}}$", value=latex_fixed(sim.temp_sd; digits=2), description="Temperature observation noise"),
        (symbol=raw"$\sigma_W$", value=latex_fixed(sim_defaults.weight_sd; digits=2), description="Fish weight observation noise"),
        (symbol=raw"$\sigma_N$", value=latex_fixed(sim_defaults.number_of_fish_sd; digits=2), description="Fish count observation noise"),
    ]
    return rows
end

function write_noise_table(rows; filename::String, caption::String, label::String)
    mkpath(OUTPUT_DIR)
    lines = String[]
    push!(lines, "\\begin{table}[htbp!]")
    push!(lines, "\\centering")
    push!(lines, "    \\caption{$caption}")
    push!(lines, "    \\label{$label}")
    push!(lines, "    \\begin{threeparttable}")
    push!(lines, "    \\begin{tabular}{@{}lcc@{}}")
    push!(lines, "    \\toprule")
    push!(lines, "    Noise term & Value & Description \\\\")
    push!(lines, "    \\midrule")
    for row in rows
        push!(lines, "    $(row.symbol) & $(row.value) & $(row.description) \\\\")
    end
    push!(lines, "    \\bottomrule")
    push!(lines, "    \\end{tabular}")
    push!(lines, "    \\end{threeparttable}")
    push!(lines, "\\end{table}")
    output_path = joinpath(OUTPUT_DIR, filename)
    open(output_path, "w") do io
        write(io, join(lines, "\n"))
    end
    return output_path
end

function load_reward_metrics(config::ExperimentConfig)
    csv_path = joinpath(config.results_dir, "reward_metrics.csv")
    isfile(csv_path) || error("Missing reward metrics at $(csv_path)")
    df = CSV.read(csv_path, DataFrame)
    return df
end

function format_metric(value::Real)
    return latex_math(value; digits=2)
end

function policy_metrics_rows(df::DataFrame)
    rows = Vector{Vector{String}}()
    for policy in POLICY_ORDER_SUMMARY
        idx = findfirst(==(policy), df.policy)
        idx === nothing && continue
        row = [
            format_text(replace(policy, "_" => "\\_")),
            format_metric(df[idx, :mean_reward]),
            format_metric(df[idx, :mean_sea_lice]),
            format_metric(df[idx, :mean_reg_penalties]),
            format_metric(df[idx, :mean_lost_biomass]),
        ]
        push!(rows, row)
    end
    isempty(rows) && @warn "No policies found for summary table."
    return rows
end

function solver_lambda_rows(config::ExperimentConfig)
    lambdas = config.solver_config.reward_lambdas
    length(lambdas) == 5 || error("Expected 5 solver reward weights, found $(length(lambdas))")
    formatted = [latex_math(val; digits=2) for val in lambdas]
    return [format_text(config.solver_config.location); formatted]
end

function treatment_table_rows()
    configs = map(act -> AquaOpt.get_action_config(act), TREATMENT_ACTIONS)
    row(label::String, values, source::String) = begin
        vals = collect(values)
        @assert length(vals) == 4
        (label=label, none=vals[1], mechanical=vals[2], thermal=vals[3], chemical=vals[4], source=source)
    end

    format_cost(value) = latex_math(value; digits=isapprox(value, round(value); atol=1e-8) ? 0 : 2)
    cost_values = map(cfg -> format_cost(cfg.cost), configs)
    adult_eff = map(cfg -> latex_percent(cfg.adult_reduction; digits=0), configs)
    motile_eff = map(cfg -> latex_percent(cfg.motile_reduction; digits=0), configs)
    sessile_eff = map(cfg -> latex_percent(cfg.sessile_reduction; digits=0), configs)
    fish_disease = map(cfg -> latex_math(cfg.fish_disease; digits=isapprox(cfg.fish_disease, round(cfg.fish_disease); atol=1e-8) ? 0 : 1), configs)
    mortality_rates = map(cfg -> latex_percent(cfg.mortality_rate; digits=0), configs)

    return [
        (label="Action", none=TREATMENT_SYMBOLS[1], mechanical=TREATMENT_SYMBOLS[2], thermal=TREATMENT_SYMBOLS[3], chemical=TREATMENT_SYMBOLS[4], source="N/A"),
        row("Cost (MNOK)", cost_values, SOURCE_MODEL),
        row("Effectiveness, AF (\\%)", adult_eff, SOURCE_ALDRIN),
        row("Effectiveness, MO (\\%)", motile_eff, SOURCE_ALDRIN),
        row("Effectiveness, S (\\%)", sessile_eff, SOURCE_ALDRIN),
        row("Fish disease rate (\\%)", fish_disease, SOURCE_MODEL),
        row("Fish mortality rate (\\%)", mortality_rates, SOURCE_MODEL),
    ]
end

# --- Main ---------------------------------------------------------------------

function main()
    config = load_experiment_config()
    outputs = String[]
    push!(outputs, write_table(solver_model_rows(config);
        filename="solver_parameters.tex",
        caption="Solver implementation parameters inferred from the hardcoded experiment configuration.",
        label="tab:solver_parameters"))
    push!(outputs, write_table(solver_runtime_rows(config);
        filename="solver_runtime.tex",
        caption="Solver runtime controls and heuristic thresholds.",
        label="tab:solver_runtime"))
    push!(outputs, write_table(simulation_rows(config);
        filename="simulation_parameters.tex",
        caption="Simulation and observation parameters used to evaluate learned policies.",
        label="tab:simulation_parameters"))
    push!(outputs, write_fish_population_table(fish_population_rows(config);
        filename="fish_population_parameters.tex",
        caption="Fish population and growth model parameters.",
        label="tab:fish_population_params"))
    push!(outputs, write_sectioned_table(sea_lice_biology_sections(config);
        filename="sea_lice_biology_parameters.tex",
        caption="Sea lice biology parameters from Stige et al. (2025).",
        label="tab:sea_lice_biology_params"))
    push!(outputs, write_sectioned_table(observation_model_sections(config);
        filename="observation_parameters.tex",
        caption="Observation model parameters for sea lice monitoring and environmental measurements.",
        label="tab:observation_params"))
    push!(outputs, write_location_table(location_dynamics_rows();
        filename="regional_dynamics.tex",
        caption="Regional temperature and development dynamics for Norwegian sites.",
        label="tab:regional_dynamics"))
    push!(outputs, write_region_environment_table(region_environment_rows();
        filename="regional_environment_summary.tex",
        caption="Region-specific environmental parameters.",
        label="tab:region_environment_params"))
    push!(outputs, write_noise_table(noise_parameter_rows(config);
        filename="noise_parameters.tex",
        caption="Process and observation noise parameters used in solvers and simulator.",
        label="tab:noise_params"))
    push!(outputs, write_lambda_table(
        ["Region", raw"$\lambda_{trt}$", raw"$\lambda_{reg}$", raw"$\lambda_{bio}$", raw"$\lambda_{fd}$", raw"$\lambda_{lice}$"],
        [solver_lambda_rows(config)];
        filename="reward_lambda_parameters.tex",
        caption="Regional reward weights for solver optimization.",
        label="tab:reward_lambdas"))
    reward_df = load_reward_metrics(config)
    push!(outputs, write_lambda_table(
        ["Policy", "Mean reward", "Adult lice", "Regulatory penalties", "Lost biomass (1000 kg)"],
        policy_metrics_rows(reward_df);
        filename="policy_evaluation_summary.tex",
        caption="Policy evaluation summary.",
        label="tab:policy_summary"))
    push!(outputs, write_treatment_table(treatment_table_rows();
        filename="treatment_parameters.tex",
        caption="Treatment costs, effectiveness, and fish impact.",
        label="tab:actions"))

    println("Wrote $(length(outputs)) methodology tables:")
    foreach(path -> println("  -> ", path), outputs)
end

main()

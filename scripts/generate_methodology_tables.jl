#!/usr/bin/env julia

#=
generate_methodology_tables.jl

Generates methodology parameter tables for the paper. Each table produces a single
.tex file with inline data using \pgfplotstabletypeset (col sep=&, row sep=\\).
No external CSV files needed — data is embedded directly in the TeX snippet.

Usage:
    julia --project scripts/generate_methodology_tables.jl

Prerequisites:
    Run run_experiments.jl first to create results/latest/

Outputs:
    results/latest/methodology_tables/*.tex

LaTeX preamble requirements:
    \usepackage{pgfplotstable}
    \usepackage{booktabs}
    \usepackage{threeparttable}  % only if tables have notes
    \pgfplotsset{compat=1.18}
=#

isnothing(Base.active_project()) && @warn "No active Julia project detected. Run with `julia --project=.`"

using AquaOpt
using JLD2
using CSV
using DataFrames
using Printf: @sprintf

# --- Manifest-based experiment loading (same pattern as policy_analysis.jl) ---

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

const EXPERIMENT_PATH = if isfile(MANIFEST_PATH)
    @info "Using manifest: $MANIFEST_PATH"
    load_manifest(MANIFEST_PATH)["baseline_norway_north"]
else
    error("Manifest not found at $MANIFEST_PATH. Run run_experiments.jl first.")
end

const OUTPUT_DIR = "results/latest/methodology_tables"

# --- Constants ----------------------------------------------------------------

const POLICY_ORDER_SUMMARY = [
    "Random_Policy",
    "Heuristic_Policy",
    "QMDP_Policy",
    "Native_SARSOP_Policy",
    "VI_Policy",
]

const POLICY_ORDER_FINANCIAL = [
    "Random_Policy",
    "NeverTreat_Policy",
    "AlwaysTreat_Policy",
    "Heuristic_Policy",
    "QMDP_Policy",
    "VI_Policy",
    "Native_SARSOP_Policy",
]

const POLICY_DISPLAY_NAMES = Dict(
    "Random_Policy" => "Random",
    "NeverTreat_Policy" => "Never Treat",
    "AlwaysTreat_Policy" => "Always Treat",
    "Heuristic_Policy" => "Heuristic",
    "QMDP_Policy" => "QMDP",
    "VI_Policy" => "VI",
    "Native_SARSOP_Policy" => "SARSOP",
)

const SOURCE_ALDRIN = "Aldrin et al. (2023)"
const SOURCE_MODEL = "Model assumption"
const SOURCE_STIGE = "Stige et al. (2025)"
const LOCATION_ORDER = ["north", "west", "south"]
const TREATMENT_ACTIONS = (NoTreatment, MechanicalTreatment, ThermalTreatment, ChemicalTreatment)
const DEFAULT_INITIAL_FISH = 200_000

# --- Config loading -----------------------------------------------------------

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

# --- Formatting helpers -------------------------------------------------------

fmt(v::Real; digits::Int=3) = @sprintf("%.*f", digits, Float64(v))
fmt(v::Integer; digits::Int=0) = digits == 0 ? string(v) : @sprintf("%.*f", digits, Float64(v))
fmt_array(v::AbstractVector{<:Real}; digits::Int=2) =
    "[" * join((fmt(x; digits=digits) for x in v), ", ") * "]"
fmt_pct(v::Real; digits::Int=1) = fmt(v * 100; digits=digits)
fmt_mean_ci(m_val::Real, c::Real; digits::Int=2) =
    @sprintf("%.*f \$\\pm\$ %.*f", digits, Float64(m_val), digits, Float64(c))
bold_tex(s::String) = "\\textbf{" * s * "}"

# TeX math wrapper: m(raw"\sigma_T") → "$\sigma_T$"
m(s::String) = "\$" * s * "\$"

# --- Inline pgfplotstabletypeset .tex builder ---------------------------------

"""
    write_table(df; tex_filename, caption, label, column_specs, [note])

Write a .tex file with inline `\\pgfplotstabletypeset` data.
Uses `col sep=&, row sep=\\\\` so cell values can safely contain commas,
LaTeX math, `\\%`, etc.

`column_specs`: Vector of `(col_key, display_name, pgf_style)` tuples.
  - `col_key`: column identifier (no underscores — used in inline header row)
  - `display_name`: LaTeX column header (can include math)
  - `pgf_style`: pgfplotstable column style, e.g. `"string type"`
"""
function write_table(df::DataFrame;
    tex_filename::String,
    column_specs::Vector{<:Tuple{String,String,String}},
    note::String="",
    # Legacy kwargs kept for backward compatibility but no longer used:
    caption::String="", label::String="", wide::Bool=true,  # unused
)
    mkpath(OUTPUT_DIR)

    col_keys = [spec[1] for spec in column_specs]

    lines = String[]
    push!(lines, "% Auto-generated by generate_methodology_tables.jl")
    !isempty(note) && push!(lines, "\\begin{threeparttable}")
    push!(lines, "\\pgfplotstabletypeset[")
    push!(lines, "    col sep=&,")
    push!(lines, "    row sep=\\\\,")
    push!(lines, "    columns={$(join(col_keys, ", "))},")
    push!(lines, "    every column/.style={column type=l},")
    for (col_key, display_name, style) in column_specs
        push!(lines, "    columns/$col_key/.style={$style, column name={$display_name}},")
    end
    push!(lines, "    every head row/.style={before row=\\toprule, after row=\\midrule},")
    push!(lines, "    every last row/.style={after row=\\bottomrule},")
    push!(lines, "]{")
    # Header row
    push!(lines, "    " * join(col_keys, " & ") * " \\\\")
    # Data rows
    for row in eachrow(df)
        vals = [string(row[Symbol(col)]) for col in col_keys]
        push!(lines, "    " * join(vals, " & ") * " \\\\")
    end
    push!(lines, "}")
    if !isempty(note)
        push!(lines, "\\begin{tablenotes}")
        push!(lines, "    \\small")
        push!(lines, "    \\item $note")
        push!(lines, "\\end{tablenotes}")
        push!(lines, "\\end{threeparttable}")
    end

    tex_path = joinpath(OUTPUT_DIR, tex_filename)
    open(tex_path, "w") do io
        write(io, join(lines, "\n") * "\n")
    end

    @info "  $tex_filename"
    return tex_path
end

# --- Table generators ---------------------------------------------------------

function generate_solver_parameters(config::ExperimentConfig)
    sc = config.solver_config
    df = DataFrame(
        parameter = [
            "Deployment region",
            "Discount factor",
            "Regulation limit (default)",
            "Season regulation limits",
            "Reproduction rate",
            "Discretization step",
            "Sampling noise sd",
            "Reward weights ($(m(raw"\lambda")))",
            "Log-space solver",
            "Salmon price (MNOK/tonne)",
            "Regulatory violation cost (MNOK)",
            "Welfare cost (MNOK)",
            "Chronic lice cost (MNOK)",
        ],
        value = [
            sc.location,
            fmt(sc.discount_factor; digits=2),
            fmt(sc.regulation_limit; digits=2),
            fmt_array(sc.season_regulation_limits),
            fmt(sc.reproduction_rate; digits=2),
            fmt(sc.discretization_step; digits=2),
            fmt(sc.adult_sd; digits=2),
            fmt_array(sc.reward_lambdas),
            string(sc.log_space),
            fmt(sc.salmon_price_MNOK_per_tonne; digits=3),
            fmt(sc.regulatory_violation_cost_MNOK; digits=1),
            fmt(sc.welfare_cost_MNOK; digits=1),
            fmt(sc.chronic_lice_cost_MNOK; digits=1),
        ],
    )
    write_table(df;
        tex_filename="solver_parameters.tex",
        caption="Solver POMDP structure and financial parameters.",
        label="tab:solver_parameters",
        column_specs=[
            ("parameter", "Parameter", "string type"),
            ("value", "Value", "string type"),
        ],
    )
end

function generate_solver_runtime(config::ExperimentConfig)
    sc = config.solver_config
    df = DataFrame(
        parameter = [
            "Max SARSOP runtime (s)",
            "VI iterations",
            "QMDP iterations",
            "Full observability solver",
            "Heuristic threshold",
            "Mechanical belief threshold",
            "Chemical belief threshold",
            "Thermal belief threshold",
            "Heuristic $(m(raw"\rho"))",
        ],
        value = [
            fmt(sc.sarsop_max_time; digits=0),
            string(sc.VI_max_iterations),
            string(sc.QMDP_max_iterations),
            string(sc.full_observability_solver),
            fmt(sc.heuristic_threshold; digits=2),
            fmt(sc.heuristic_belief_threshold_mechanical; digits=2),
            fmt(sc.heuristic_belief_threshold_chemical; digits=2),
            fmt(sc.heuristic_belief_threshold_thermal; digits=2),
            fmt(sc.heuristic_rho; digits=2),
        ],
    )
    write_table(df;
        tex_filename="solver_runtime.tex",
        caption="Solver runtime controls and heuristic thresholds.",
        label="tab:solver_runtime",
        column_specs=[
            ("parameter", "Parameter", "string type"),
            ("value", "Value", "string type"),
        ],
    )
end

function generate_fish_population(config::ExperimentConfig)
    sim_pomdp = AquaOpt.SeaLiceSimPOMDP(location=config.solver_config.location)
    initial_weight = config.simulation_config.W0
    wb = sim_pomdp.weight_bounds
    fb = sim_pomdp.number_of_fish_bounds
    df = DataFrame(
        parameter = [
            "Asymptotic weight ($(m("w_{\\max}")))",
            "Growth rate ($(m("k_g")))",
            "Temperature sensitivity ($(m(raw"\alpha_T")))",
            "Natural mortality rate",
            "Treatment mortality bump",
            "Production start week",
            "Weight bounds (kg)",
            "Fish count bounds",
            "Initial fish count",
            "Initial fish weight (kg)",
        ],
        value = [
            fmt(sim_pomdp.w_max; digits=1),
            fmt(sim_pomdp.k_growth; digits=2),
            fmt(sim_pomdp.temp_sensitivity; digits=2),
            fmt(sim_pomdp.nat_mort_rate; digits=4),
            fmt(sim_pomdp.trt_mort_bump; digits=3),
            string(sim_pomdp.production_start_week),
            "($(fmt(wb[1]; digits=1)), $(fmt(wb[2]; digits=1)))",
            "($(fmt(fb[1]; digits=0)), $(fmt(fb[2]; digits=0)))",
            string(DEFAULT_INITIAL_FISH),
            fmt(initial_weight; digits=2),
        ],
        description = [
            "Maximum harvest weight (kg)",
            "Weekly von Bertalanffy growth rate",
            "Temperature effect on growth",
            "Weekly natural mortality fraction",
            "Extra mortality in treatment weeks",
            "Week of production start",
            "Fish weight range (kg)",
            "Fish count range",
            "Fish count at production start",
            "Average weight at production start (kg)",
        ],
        source = [
            SOURCE_MODEL, SOURCE_ALDRIN, SOURCE_ALDRIN, SOURCE_ALDRIN,
            SOURCE_ALDRIN, SOURCE_MODEL, SOURCE_MODEL, SOURCE_MODEL,
            SOURCE_MODEL, SOURCE_MODEL,
        ],
    )
    write_table(df;
        tex_filename="fish_population_parameters.tex",
        caption="Fish population and growth model parameters.",
        label="tab:fish_population_params",
        column_specs=[
            ("parameter", "Parameter", "string type"),
            ("value", "Value", "string type"),
            ("description", "Description", "string type"),
            ("source", "Source", "string type"),
        ],
    )
end

function generate_sea_lice_biology(config::ExperimentConfig)
    params = get_location_params(config.solver_config.location)
    df = DataFrame(
        section = [
            "Survival", "Survival", "Survival", "Survival",
            "Development", "Development", "Development", "Development", "Development",
        ],
        parameter = [
            "Sessile survival ($(m("s_1")))",
            "Sessile-to-motile scaling ($(m("s_2")))",
            "Motile survival ($(m("s_3")))",
            "Adult survival ($(m("s_4")))",
            "$(m("d_1")) intercept ($(m("c_1")))",
            "$(m("d_1")) temperature coefficient ($(m("t_1")))",
            "$(m("d_2")) intercept ($(m("c_2")))",
            "$(m("d_2")) temperature coefficient ($(m("t_2")))",
            "Temperature reference ($(m("r")))",
        ],
        value = [
            fmt(params.s1_sessile; digits=2),
            fmt(params.s2_scaling; digits=1),
            fmt(params.s3_motile; digits=2),
            fmt(params.s4_adult; digits=2),
            fmt(params.d1_intercept; digits=1),
            fmt(params.d1_temp_coef; digits=2),
            fmt(params.d2_intercept; digits=1),
            fmt(params.d2_temp_coef; digits=3),
            fmt(9.0; digits=1),
        ],
        description = [
            "Weekly sessile survival rate",
            "Sessile to motile transition scaling",
            "Weekly motile survival rate",
            "Weekly adult survival rate",
            "Sessile-to-motile development intercept",
            "Temperature effect on sessile-to-motile",
            "Motile-to-adult development intercept",
            "Temperature effect on motile-to-adult",
            "Reference temperature (\\si{\\celsius})",
        ],
        source = fill(SOURCE_STIGE, 9),
    )
    write_table(df;
        tex_filename="sea_lice_biology_parameters.tex",
        caption="Sea lice biology parameters from Stige et al.\\ (2025).",
        label="tab:sea_lice_biology_params",
        column_specs=[
            ("section", "Category", "string type"),
            ("parameter", "Parameter", "string type"),
            ("value", "Value", "string type"),
            ("description", "Description", "string type"),
            ("source", "Source", "string type"),
        ],
    )
end

function generate_observation_parameters(config::ExperimentConfig)
    sim = config.simulation_config
    sim_pomdp = AquaOpt.SeaLiceSimPOMDP(location=config.solver_config.location)
    df = DataFrame(
        section = [
            "Negative Binomial", "Negative Binomial", "Negative Binomial", "Negative Binomial",
            "Under-counting", "Under-counting", "Under-counting", "Under-counting",
            "Gaussian Noise", "Gaussian Noise", "Gaussian Noise",
        ],
        parameter = [
            "Fish sampled ($(m("n")))",
            "$(m(raw"\rho_A"))",
            "$(m(raw"\rho_M"))",
            "$(m(raw"\rho_S"))",
            "Farm intercept ($(m(raw"\beta_0")))",
            "Weight slope ($(m(raw"\beta_1")))",
            "Weight centering ($(m("W_0")))",
            "Mean fish weight ($(m("W_{\\text{mean}}")))",
            "Temperature noise ($(m(raw"\sigma_T")))",
            "Fish count noise ($(m(raw"\sigma_N")))",
            "Weight noise ($(m(raw"\sigma_W")))",
        ],
        value = [
            string(sim.n_sample),
            fmt(sim.ρ_adult; digits=3),
            fmt(sim.ρ_motile; digits=3),
            fmt(sim.ρ_sessile; digits=3),
            fmt(sim.beta0_Scount_f; digits=3),
            fmt(sim.beta1_Scount; digits=3),
            fmt(sim.W0; digits=2),
            fmt(sim.mean_fish_weight_kg; digits=2),
            fmt(sim.temp_sd; digits=2),
            fmt(sim_pomdp.number_of_fish_sd; digits=1),
            fmt(sim_pomdp.weight_sd; digits=2),
        ],
        description = [
            "Fish counted per monitoring event",
            "Over-dispersion for adult lice counts",
            "Over-dispersion for motile lice counts",
            "Over-dispersion for sessile lice counts",
            "Logistic function intercept",
            "Weight-dependent correction factor",
            "Reference weight (kg)",
            "Average fish weight (kg)",
            "Temperature measurement std dev (\\si{\\celsius})",
            "Fish count measurement std dev",
            "Weight measurement std dev (kg)",
        ],
        source = [
            SOURCE_ALDRIN, SOURCE_ALDRIN, SOURCE_ALDRIN, SOURCE_ALDRIN,
            SOURCE_ALDRIN, SOURCE_ALDRIN, SOURCE_ALDRIN, SOURCE_ALDRIN,
            SOURCE_MODEL, SOURCE_MODEL, SOURCE_MODEL,
        ],
    )
    write_table(df;
        tex_filename="observation_parameters.tex",
        caption="Observation model parameters for sea lice monitoring and environmental measurements.",
        label="tab:observation_params",
        column_specs=[
            ("section", "Category", "string type"),
            ("parameter", "Parameter", "string type"),
            ("value", "Value", "string type"),
            ("description", "Description", "string type"),
            ("source", "Source", "string type"),
        ],
    )
end

function generate_regional_dynamics()
    params = Dict(loc => get_location_params(loc) for loc in LOCATION_ORDER)
    # Only include parameters that vary across regions
    fields = [
        ("Mean temperature ($(m("T_{\\text{mean}}")))",     :T_mean,          2, "Average annual sea temperature (\\si{\\celsius})"),
        ("External larval influx",                          :external_influx, 2, "Weekly external sessile influx"),
    ]
    df = DataFrame(
        parameter   = [f[1] for f in fields],
        north       = [fmt(Float64(getfield(params["north"], f[2])); digits=f[3]) for f in fields],
        west        = [fmt(Float64(getfield(params["west"],  f[2])); digits=f[3]) for f in fields],
        south       = [fmt(Float64(getfield(params["south"], f[2])); digits=f[3]) for f in fields],
        description = [f[4] for f in fields],
    )
    write_table(df;
        tex_filename="regional_dynamics.tex",
        caption="Regional temperature and development dynamics for Norwegian sites.",
        label="tab:regional_dynamics",
        column_specs=[
            ("parameter",   "Parameter",   "string type"),
            ("north",       "North",       "string type"),
            ("west",        "West",        "string type"),
            ("south",       "South",       "string type"),
            ("description", "Description", "string type"),
        ],
    )
end

function generate_regional_environment()
    params = Dict(loc => get_location_params(loc) for loc in LOCATION_ORDER)
    labels = Dict("north" => "North", "west" => "West", "south" => "South")
    df = DataFrame(
        region  = [labels[loc] for loc in LOCATION_ORDER],
        tmean   = [fmt(params[loc].T_mean; digits=1) for loc in LOCATION_ORDER],
        tamp    = [fmt(params[loc].T_amp; digits=1) for loc in LOCATION_ORDER],
        peakwk  = [string(params[loc].peak_week) for loc in LOCATION_ORDER],
        extinfl = [fmt(params[loc].external_influx; digits=2) for loc in LOCATION_ORDER],
    )
    write_table(df;
        tex_filename="regional_environment_summary.tex",
        caption="Region-specific environmental parameters.",
        label="tab:region_environment_params",
        column_specs=[
            ("region",  "Region",                                                       "string type"),
            ("tmean",   "\$T_{\\text{mean}}\$ (\\si{\\celsius})",                       "string type"),
            ("tamp",    "\$T_{\\text{amp}}\$ (\\si{\\celsius})",                        "string type"),
            ("peakwk",  "\$W_{\\text{peak}}\$ (week)",                                  "string type"),
            ("extinfl", "\$\\lambda_{\\text{ext}}\$",                                   "string type"),
        ],
        note="External influx values are model assumptions calibrated to reflect warmer-region larval pressure.",
    )
end

function generate_noise_parameters(config::ExperimentConfig)
    sim = config.simulation_config
    sim_pomdp = AquaOpt.SeaLiceSimPOMDP(location=config.solver_config.location)
    df = DataFrame(
        noiseterm = [
            "$(m(raw"\sigma_A")) (measurement)",
            "$(m(raw"\sigma_{A,\text{obs}}")) (biological)",
            "$(m(raw"\sigma_M")) (measurement)",
            "$(m(raw"\sigma_{M,\text{obs}}")) (biological)",
            "$(m(raw"\sigma_S")) (measurement)",
            "$(m(raw"\sigma_{S,\text{obs}}")) (biological)",
            m(raw"\sigma_T"),
            m(raw"\sigma_W"),
            m(raw"\sigma_N"),
        ],
        value = [
            fmt(sim.adult_sd; digits=2),
            fmt(sim.adult_obs_sd; digits=3),
            fmt(sim.motile_sd; digits=2),
            fmt(sim.motile_obs_sd; digits=3),
            fmt(sim.sessile_sd; digits=2),
            fmt(sim.sessile_obs_sd; digits=3),
            fmt(sim.temp_sd; digits=2),
            fmt(sim_pomdp.weight_sd; digits=2),
            fmt(sim_pomdp.number_of_fish_sd; digits=2),
        ],
        description = [
            "Adult lice measurement noise (NB sampling)",
            "Adult lice biological variability (Kalman filter)",
            "Motile lice measurement noise (NB sampling)",
            "Motile lice biological variability (Kalman filter)",
            "Sessile lice measurement noise (NB sampling)",
            "Sessile lice biological variability (Kalman filter)",
            "Temperature noise",
            "Fish weight observation noise",
            "Fish count observation noise",
        ],
    )
    write_table(df;
        tex_filename="noise_parameters.tex",
        caption="Process and observation noise parameters used in solvers and simulator.",
        label="tab:noise_params",
        column_specs=[
            ("noiseterm",   "Noise Term",   "string type"),
            ("value",       "Value",        "string type"),
            ("description", "Description",  "string type"),
        ],
        wide=false,
    )
end

function generate_reward_lambdas(config::ExperimentConfig)
    lambdas = config.solver_config.reward_lambdas
    length(lambdas) == 5 || error("Expected 5 solver reward weights, found $(length(lambdas))")
    df = DataFrame(
        region = [config.solver_config.location],
        ltrt   = [fmt(lambdas[1]; digits=2)],
        lreg   = [fmt(lambdas[2]; digits=2)],
        lbio   = [fmt(lambdas[3]; digits=2)],
        lfd    = [fmt(lambdas[4]; digits=2)],
        llice  = [fmt(lambdas[5]; digits=2)],
    )
    write_table(df;
        tex_filename="reward_lambda_parameters.tex",
        caption="Reward weights for solver optimization.",
        label="tab:reward_lambdas",
        column_specs=[
            ("region", "Region",                          "string type"),
            ("ltrt",   "\$\\lambda_{\\text{trt}}\$",     "string type"),
            ("lreg",   "\$\\lambda_{\\text{reg}}\$",     "string type"),
            ("lbio",   "\$\\lambda_{\\text{bio}}\$",     "string type"),
            ("lfd",    "\$\\lambda_{\\text{fd}}\$",      "string type"),
            ("llice",  "\$\\lambda_{\\text{lice}}\$",    "string type"),
        ],
    )
end

function generate_policy_summary(config::ExperimentConfig)
    csv_path = joinpath(config.results_dir, "reward_metrics.csv")
    isfile(csv_path) || error("Missing reward metrics at $csv_path")
    metrics = CSV.read(csv_path, DataFrame)

    rows = NamedTuple[]
    for policy in POLICY_ORDER_SUMMARY
        idx = findfirst(==(policy), metrics.policy)
        idx === nothing && continue
        push!(rows, (
            policy      = get(POLICY_DISPLAY_NAMES, policy, policy),
            meanreward  = fmt(metrics[idx, :mean_reward]; digits=2),
            meanlice    = fmt(metrics[idx, :mean_sea_lice]; digits=2),
            meanreg     = fmt(metrics[idx, :mean_reg_penalties]; digits=2),
            meanbio     = fmt(metrics[idx, :mean_lost_biomass]; digits=2),
        ))
    end
    isempty(rows) && @warn "No policies found for summary table."
    df = DataFrame(rows)
    write_table(df;
        tex_filename="policy_evaluation_summary.tex",
        caption="Policy evaluation summary.",
        label="tab:policy_summary",
        column_specs=[
            ("policy",     "Policy",                    "string type"),
            ("meanreward", "Mean Reward",               "string type"),
            ("meanlice",   "Adult Lice",                "string type"),
            ("meanreg",    "Reg.\\ Penalties",          "string type"),
            ("meanbio",    "Lost Biomass (1000 kg)",  "string type"),
        ],
    )
end

function generate_treatment_parameters()
    cfgs = map(act -> AquaOpt.get_action_config(act), TREATMENT_ACTIONS)
    df = DataFrame(
        property = [
            "Cost (MNOK)",
            "Adult reduction (\\%)",
            "Motile reduction (\\%)",
            "Sessile reduction (\\%)",
            "Fish disease score",
            "Fish mortality (\\%)",
        ],
        none = [
            fmt(cfgs[1].cost; digits=1),       fmt_pct(cfgs[1].adult_reduction),
            fmt_pct(cfgs[1].motile_reduction),  fmt_pct(cfgs[1].sessile_reduction),
            fmt(cfgs[1].fish_disease; digits=2), fmt_pct(cfgs[1].mortality_rate),
        ],
        mechanical = [
            fmt(cfgs[2].cost; digits=1),       fmt_pct(cfgs[2].adult_reduction),
            fmt_pct(cfgs[2].motile_reduction),  fmt_pct(cfgs[2].sessile_reduction),
            fmt(cfgs[2].fish_disease; digits=2), fmt_pct(cfgs[2].mortality_rate),
        ],
        thermal = [
            fmt(cfgs[3].cost; digits=1),       fmt_pct(cfgs[3].adult_reduction),
            fmt_pct(cfgs[3].motile_reduction),  fmt_pct(cfgs[3].sessile_reduction),
            fmt(cfgs[3].fish_disease; digits=2), fmt_pct(cfgs[3].mortality_rate),
        ],
        chemical = [
            fmt(cfgs[4].cost; digits=1),       fmt_pct(cfgs[4].adult_reduction),
            fmt_pct(cfgs[4].motile_reduction),  fmt_pct(cfgs[4].sessile_reduction),
            fmt(cfgs[4].fish_disease; digits=2), fmt_pct(cfgs[4].mortality_rate),
        ],
        source = [
            SOURCE_MODEL, SOURCE_ALDRIN, SOURCE_ALDRIN, SOURCE_ALDRIN,
            SOURCE_MODEL, SOURCE_MODEL,
        ],
    )
    write_table(df;
        tex_filename="treatment_parameters.tex",
        caption="Treatment costs, effectiveness, and fish impact.",
        label="tab:actions",
        column_specs=[
            ("property",   "Property",                     "string type"),
            ("none",       "None (\$a_1\$)",               "string type"),
            ("mechanical", "Mechanical (\$a_2\$)",         "string type"),
            ("thermal",    "Thermal (\$a_3\$)",            "string type"),
            ("chemical",   "Chemical (\$a_4\$)",           "string type"),
            ("source",     "Source",                        "string type"),
        ],
    )
end

# --- NEW: Country-specific regulatory comparison ------------------------------

function generate_country_regulatory()
    df = DataFrame(
        parameter = [
            "Season regulation limits (lice/fish)",
            "Regulatory violation cost (MNOK)",
            "Salmon price (MNOK/tonne)",
        ],
        norway = [
            "[0.2, 0.5, 0.5, 0.5]",
            "10.0",
            "0.07",
        ],
        scotland = [
            "[1.0, 2.0, 2.0, 2.0]",
            "3.0",
            "0.075",
        ],
        chile = [
            "[3.0, 3.0, 3.0, 3.0]",
            "5.0",
            "0.05",
        ],
    )
    write_table(df;
        tex_filename="country_regulatory.tex",
        caption="Country-specific regulatory framework parameters.",
        label="tab:country_regulatory",
        column_specs=[
            ("parameter", "Parameter", "string type"),
            ("norway",    "Norway",    "string type"),
            ("scotland",  "Scotland",  "string type"),
            ("chile",     "Chile",     "string type"),
        ],
    )
end

# --- NEW: Reward function components ------------------------------------------

function generate_reward_components()
    df = DataFrame(
        component = [
            "Treatment cost",
            "Regulatory penalty",
            "Biomass loss",
            "Fish health",
            "Sea lice burden",
        ],
        description = [
            "Direct operational cost of treatment",
            "Cost triggered by exceeding seasonal lice limits",
            "Value of fish biomass lost to mortality and growth reduction",
            "Welfare cost from treatment stress on fish",
            "Chronic production loss from sustained lice burden",
        ],
        defaultval = [
            "1.5 / 2.5 / 4.0",
            "10.0",
            "0.07",
            "1.0",
            "0.5",
        ],
        unit = [
            "MNOK per treatment",
            "MNOK per violation",
            "MNOK per tonne lost",
            "MNOK per stress-unit",
            "MNOK per burden-unit",
        ],
        source = [
            SOURCE_MODEL,
            SOURCE_MODEL,
            "Spot price",
            SOURCE_MODEL,
            SOURCE_MODEL,
        ],
    )
    write_table(df;
        tex_filename="reward_components.tex",
        caption="Reward function components (all denominated in MNOK).",
        label="tab:reward_components",
        column_specs=[
            ("component",   "Component",     "string type"),
            ("description", "Description",   "string type, column type={p{6cm}}"),
            ("defaultval",  "Default Value", "string type"),
            ("unit",        "Unit",          "string type"),
            ("source",      "Source",        "string type"),
        ],
    )
end

# --- Financial results summary ------------------------------------------------

function generate_financial_results_summary(config::ExperimentConfig)
    csv_path = joinpath(config.results_dir, "financial_summary.csv")
    isfile(csv_path) || error("Missing financial summary at $csv_path. " *
        "Ensure high_fidelity_sim=true and run_experiments.jl completed.")
    fin = CSV.read(csv_path, DataFrame)

    # (mean_col, ci_col, is_cost) — is_cost=true means lower is better
    metric_specs = [
        (:mean_harvest_rev,  :ci_harvest_rev,  false),
        (:mean_trt_cost,     :ci_trt_cost,     true),
        (:mean_reg_cost,     :ci_reg_cost,     true),
        (:mean_bio_cost,     :ci_bio_cost,     true),
        (:mean_welfare_cost, :ci_welfare_cost, true),
        (:mean_lice_cost,    :ci_lice_cost,    true),
        (:mean_net_profit,   :ci_net_profit,   false),
    ]

    # Find best policy index per metric
    best_idx = Dict{Symbol,Int}()
    for (mcol, _, is_cost) in metric_specs
        vals = fin[!, mcol]
        best_idx[mcol] = is_cost ? argmin(vals) : argmax(vals)
    end

    rows = NamedTuple[]
    for policy in POLICY_ORDER_FINANCIAL
        idx = findfirst(==(policy), fin.policy)
        idx === nothing && continue

        display_name = get(POLICY_DISPLAY_NAMES, policy, policy)

        function fmt_metric(mcol, ccol)
            s = fmt_mean_ci(fin[idx, mcol], fin[idx, ccol]; digits=2)
            return idx == best_idx[mcol] ? bold_tex(s) : s
        end

        push!(rows, (
            policy      = display_name,
            harvestrev  = fmt_metric(:mean_harvest_rev, :ci_harvest_rev),
            trtcost     = fmt_metric(:mean_trt_cost, :ci_trt_cost),
            regcost     = fmt_metric(:mean_reg_cost, :ci_reg_cost),
            biocost     = fmt_metric(:mean_bio_cost, :ci_bio_cost),
            welfarecost = fmt_metric(:mean_welfare_cost, :ci_welfare_cost),
            licecost    = fmt_metric(:mean_lice_cost, :ci_lice_cost),
            netprofit   = fmt_metric(:mean_net_profit, :ci_net_profit),
        ))
    end
    isempty(rows) && @warn "No policies found in financial summary."
    df = DataFrame(rows)

    write_table(df;
        tex_filename="financial_results_summary.tex",
        caption="Production cycle financial summary per policy (all values in MNOK). " *
                "Values shown as mean \$\\pm\$ 95\\% CI. Best per column in bold.",
        label="tab:financial_results_summary",
        column_specs=[
            ("policy",      "Policy",        "string type"),
            ("harvestrev",  "Harvest Rev.",   "string type"),
            ("trtcost",     "Treatment",      "string type"),
            ("regcost",     "Regulatory",     "string type"),
            ("biocost",     "Biomass Loss",   "string type"),
            ("welfarecost", "Welfare",        "string type"),
            ("licecost",    "Lice Burden",    "string type"),
            ("netprofit",   "Net Profit",     "string type"),
        ],
        note="All financial metrics in MNOK (million Norwegian kroner). " *
             "Costs are per production cycle ($(config.simulation_config.steps_per_episode) weeks). " *
             "Bold indicates best-performing policy per column.",
    )
end

# --- Main ---------------------------------------------------------------------

function main()
    config = load_experiment_config(EXPERIMENT_PATH)
    println("Generating methodology tables from: $EXPERIMENT_PATH")
    println("Output directory: $OUTPUT_DIR\n")

    generate_solver_parameters(config)
    generate_solver_runtime(config)
    generate_fish_population(config)
    generate_sea_lice_biology(config)
    generate_observation_parameters(config)
    generate_regional_dynamics()
    generate_regional_environment()
    generate_noise_parameters(config)
    generate_reward_lambdas(config)
    generate_policy_summary(config)
    generate_treatment_parameters()
    generate_country_regulatory()
    generate_reward_components()
    generate_financial_results_summary(config)

    println("\nDone! Generated 14 tables (.tex) in $OUTPUT_DIR")
    println("\nLaTeX preamble requirements:")
    println("    \\usepackage{pgfplotstable}")
    println("    \\usepackage{booktabs}")
    println("    \\usepackage{threeparttable}  % if using tables with notes")
    println("    \\pgfplotsset{compat=1.18}")
end

main()

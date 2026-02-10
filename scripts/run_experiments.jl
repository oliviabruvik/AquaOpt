#!/usr/bin/env julia

#=
run_experiments.jl

Runs three experiment groups + ablations (9 runs total):
  0. Baseline:             balanced lambdas, norway, north (shared by all 3 groups)
  1. Regulation analysis:  + scotland, chile (north)
  2. Dynamics analysis:    + west, south (norway)
  3. Lambda analysis:      + cost, welfare lambdas (norway, north)
  4. Ablations:            raw-space UKF + log-space EKF (norway, north)

Outputs a manifest file and creates a symlink at results/latest/ for use
by analysis scripts (policy_analysis.jl, region_analysis.jl, etc.).

Usage:
    julia --project scripts/run_experiments.jl <mode>
    julia --project scripts/run_experiments.jl debug
    julia --project scripts/run_experiments.jl paper
=#

using AquaOpt
using Dates

function run_experiments(mode)

    plot_flag = true

    # ── Create timestamped run directory ──
    run_dir = joinpath("results", "run_$(Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"))_$(mode)")
    mkpath(run_dir)
    @info "Run directory: $run_dir"

    # ── Initialize manifest file ──
    manifest_path = joinpath(run_dir, "experiment_manifest.txt")
    open(manifest_path, "w") do f
        println(f, "# Experiment manifest — $(Dates.now())")
        println(f, "# Mode: $mode")
        println(f, "# Run directory: $run_dir")
        println(f, "#")
        println(f, "# label\texperiment_dir\tresults_dir\tlocation\tlambdas\treg_limits")
    end

    function append_manifest(label, cfg)
        sc = cfg.solver_config
        open(manifest_path, "a") do f
            println(f, join([label, cfg.experiment_dir, cfg.results_dir,
                sc.location, string(sc.reward_lambdas),
                string(sc.season_regulation_limits)], '\t'))
        end
        @info "Manifest updated: $label → $(cfg.experiment_dir)"
    end

    # Country-specific regulatory frameworks
    # Each defines regulation limits, violation costs, and salmon prices
    # that reflect real-world management philosophies.
    # Lambdas are kept uniform — policy differences arise from the regulatory parameters.
    country_configs = Dict(
        "norway" => (
            season_regulation_limits = [0.2, 0.5, 0.5, 0.5],   # Strict: 0.2 spring (smolt), 0.5 otherwise
            regulatory_violation_cost_MNOK = 10.0,               # Severe: mandatory treatment orders + production caps
            salmon_price_MNOK_per_tonne = 0.07,                  # ~70 NOK/kg spot price
        ),
        "scotland" => (
            season_regulation_limits = [1.0, 2.0, 2.0, 2.0],   # CoGP: reporting at 0.5, intervention at 2.0
            regulatory_violation_cost_MNOK = 3.0,                # Graduated: warnings before penalties
            salmon_price_MNOK_per_tonne = 0.075,                 # ~£6/kg ≈ 75 NOK/kg
        ),
        "chile" => (
            season_regulation_limits = [3.0, 3.0, 3.0, 3.0],   # SERNAPESCA: ~3 gravid females, no seasonal variation
            regulatory_violation_cost_MNOK = 5.0,                # Moderate enforcement
            salmon_price_MNOK_per_tonne = 0.05,                  # ~$5/kg ≈ 50 NOK/kg
        ),
    )

    balanced_lambdas = [0.2, 0.2, 0.2, 0.2, 0.2]
    norway = country_configs["norway"]

    # ── Baseline: balanced lambdas, Norway, north ──
    # Shared across regulation (norway), dynamics (north), and lambda (balanced) analyses
    @info "Running baseline: balanced lambdas, norway, north"
    cfg = AquaOpt.main(
        log_space=true,
        experiment_name="baseline_norway_north",
        mode=mode,
        location="north",
        ekf_filter=false,
        plot=plot_flag,
        reward_lambdas=balanced_lambdas,
        sim_reward_lambdas=balanced_lambdas,
        season_regulation_limits=norway.season_regulation_limits,
        regulatory_violation_cost_MNOK=norway.regulatory_violation_cost_MNOK,
        salmon_price_MNOK_per_tonne=norway.salmon_price_MNOK_per_tonne,
    )
    append_manifest("baseline_norway_north", cfg)

    # ── 1. Regulation analysis: scotland + chile (norway = baseline) ──
    for country in ["scotland", "chile"]
        cc = country_configs[country]
        label = "regulation_$(country)_north"
        @info "Running regulation analysis: country=$country, location=north"
        cfg = AquaOpt.main(
            log_space=true,
            experiment_name=label,
            mode=mode,
            location="north",
            ekf_filter=false,
            plot=plot_flag,
            reward_lambdas=balanced_lambdas,
            sim_reward_lambdas=balanced_lambdas,
            season_regulation_limits=cc.season_regulation_limits,
            regulatory_violation_cost_MNOK=cc.regulatory_violation_cost_MNOK,
            salmon_price_MNOK_per_tonne=cc.salmon_price_MNOK_per_tonne,
        )
        append_manifest(label, cfg)
    end

    # ── 2. Dynamics analysis: west + south (north = baseline) ──
    for location in ["west", "south"]
        label = "dynamics_norway_$(location)"
        @info "Running dynamics analysis: location=$location, country=norway"
        cfg = AquaOpt.main(
            log_space=true,
            experiment_name=label,
            mode=mode,
            location=location,
            ekf_filter=false,
            plot=plot_flag,
            reward_lambdas=balanced_lambdas,
            sim_reward_lambdas=balanced_lambdas,
            season_regulation_limits=norway.season_regulation_limits,
            regulatory_violation_cost_MNOK=norway.regulatory_violation_cost_MNOK,
            salmon_price_MNOK_per_tonne=norway.salmon_price_MNOK_per_tonne,
        )
        append_manifest(label, cfg)
    end

    # ── 3. Lambda analysis: cost + welfare (balanced = baseline) ──
    lambda_scenarios = Dict(
        "cost"     => [0.35, 0.1, 0.3, 0.1, 0.15],
        "welfare"  => [0.1, 0.15, 0.1, 0.3, 0.35],
    )
    for (scenario_name, scenario_lambdas) in lambda_scenarios
        label = "lambda_$(scenario_name)_norway_north"
        @info "Running lambda analysis: scenario=$scenario_name, location=north"
        cfg = AquaOpt.main(
            log_space=true,
            experiment_name=label,
            mode=mode,
            location="north",
            ekf_filter=false,
            plot=plot_flag,
            reward_lambdas=scenario_lambdas,
            sim_reward_lambdas=scenario_lambdas,
            season_regulation_limits=norway.season_regulation_limits,
            regulatory_violation_cost_MNOK=norway.regulatory_violation_cost_MNOK,
            salmon_price_MNOK_per_tonne=norway.salmon_price_MNOK_per_tonne,
        )
        append_manifest(label, cfg)
    end

    # ── Ablations: Raw space and EKF (Norway balanced, north only) ──
    cfg = AquaOpt.main(log_space=false, experiment_name="ablation_raw_space_ukf", mode=mode, location="north", ekf_filter=false, plot=plot_flag,
        reward_lambdas=balanced_lambdas, sim_reward_lambdas=balanced_lambdas,
        season_regulation_limits=norway.season_regulation_limits,
        regulatory_violation_cost_MNOK=norway.regulatory_violation_cost_MNOK,
        salmon_price_MNOK_per_tonne=norway.salmon_price_MNOK_per_tonne)
    append_manifest("ablation_raw_space_ukf", cfg)

    cfg = AquaOpt.main(log_space=true, experiment_name="ablation_log_space_ekf", mode=mode, location="north", ekf_filter=true, plot=plot_flag,
        reward_lambdas=balanced_lambdas, sim_reward_lambdas=balanced_lambdas,
        season_regulation_limits=norway.season_regulation_limits,
        regulatory_violation_cost_MNOK=norway.regulatory_violation_cost_MNOK,
        salmon_price_MNOK_per_tonne=norway.salmon_price_MNOK_per_tonne)
    append_manifest("ablation_log_space_ekf", cfg)

    # ── Create/update symlink results/latest -> this run ──
    latest_link = joinpath("results", "latest")
    islink(latest_link) && rm(latest_link)
    isdir(latest_link) && error("results/latest exists as a real directory, not a symlink. Remove it manually.")
    symlink(abspath(run_dir), latest_link)

    @info "All experiments complete. Manifest at $manifest_path"
    @info "Symlink updated: results/latest -> $run_dir"
    return
end

# CLI entry point
if length(ARGS) < 1
    println("Usage: julia --project scripts/run_experiments.jl <mode>")
    println("  mode: 'debug' or 'paper'")
    exit(1)
end

run_experiments(ARGS[1])

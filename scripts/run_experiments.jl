#!/usr/bin/env julia

#=
run_experiments.jl

Runs three experiment groups + ablations (9 runs total):
  0. Baseline:             balanced lambdas, norway, north (shared by all 3 groups)
  1. Regulation analysis:  + scotland, chile (north)
  2. Dynamics analysis:    + west, south (norway)
  3. Lambda analysis:      + cost, welfare lambdas (norway, north)
  4. Ablations:            raw-space UKF + log-space EKF (norway, north)

Usage:
    julia --project scripts/run_experiments.jl <mode>
    julia --project scripts/run_experiments.jl debug
    julia --project scripts/run_experiments.jl paper
=#

using AquaOpt

function run_experiments(mode)

    plot_flag = true

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
    AquaOpt.main(
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

    # ── 1. Regulation analysis: scotland + chile (norway = baseline) ──
    for country in ["scotland", "chile"]
        cfg = country_configs[country]
        @info "Running regulation analysis: country=$country, location=north"
        AquaOpt.main(
            log_space=true,
            experiment_name="regulation_$(country)_north",
            mode=mode,
            location="north",
            ekf_filter=false,
            plot=plot_flag,
            reward_lambdas=balanced_lambdas,
            sim_reward_lambdas=balanced_lambdas,
            season_regulation_limits=cfg.season_regulation_limits,
            regulatory_violation_cost_MNOK=cfg.regulatory_violation_cost_MNOK,
            salmon_price_MNOK_per_tonne=cfg.salmon_price_MNOK_per_tonne,
        )
    end

    # ── 2. Dynamics analysis: west + south (north = baseline) ──
    for location in ["west", "south"]
        @info "Running dynamics analysis: location=$location, country=norway"
        AquaOpt.main(
            log_space=true,
            experiment_name="dynamics_norway_$(location)",
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
    end

    # ── 3. Lambda analysis: cost + welfare (balanced = baseline) ──
    lambda_scenarios = Dict(
        "cost"     => [0.35, 0.1, 0.3, 0.1, 0.15],
        "welfare"  => [0.1, 0.15, 0.1, 0.3, 0.35],
    )
    for (scenario_name, scenario_lambdas) in lambda_scenarios
        @info "Running lambda analysis: scenario=$scenario_name, location=north"
        AquaOpt.main(
            log_space=true,
            experiment_name="lambda_$(scenario_name)_norway_north",
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
    end

    # ── Ablations: Raw space and EKF (Norway balanced, north only) ──
    AquaOpt.main(log_space=false, experiment_name="ablation_raw_space_ukf", mode=mode, location="north", ekf_filter=false, plot=plot_flag,
        reward_lambdas=balanced_lambdas, sim_reward_lambdas=balanced_lambdas,
        season_regulation_limits=norway.season_regulation_limits,
        regulatory_violation_cost_MNOK=norway.regulatory_violation_cost_MNOK,
        salmon_price_MNOK_per_tonne=norway.salmon_price_MNOK_per_tonne)
    AquaOpt.main(log_space=true, experiment_name="ablation_log_space_ekf", mode=mode, location="north", ekf_filter=true, plot=plot_flag,
        reward_lambdas=balanced_lambdas, sim_reward_lambdas=balanced_lambdas,
        season_regulation_limits=norway.season_regulation_limits,
        regulatory_violation_cost_MNOK=norway.regulatory_violation_cost_MNOK,
        salmon_price_MNOK_per_tonne=norway.salmon_price_MNOK_per_tonne)

    return
end

# CLI entry point
if length(ARGS) < 1
    println("Usage: julia --project scripts/run_experiments.jl <mode>")
    println("  mode: 'debug' or 'paper'")
    exit(1)
end

run_experiments(ARGS[1])

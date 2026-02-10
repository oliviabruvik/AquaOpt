#!/usr/bin/env julia

#=
heuristic_sweep.jl

Grid search over heuristic policy parameters to find optimal thresholds.
Runs the heuristic through the simulation pipeline for each parameter combination
and records the mean reward.

Usage:
    julia --project scripts/heuristic_sweep.jl <location> [<mode>]
    julia --project scripts/heuristic_sweep.jl south
    julia --project scripts/heuristic_sweep.jl north debug
=#

using AquaOpt
using Statistics
using Dates
using Printf
using CSV
using DataFrames

function optimize_heuristic(location::String, mode::String="debug";
    reward_lambdas::Vector{Float64}=[0.2, 0.2, 0.2, 0.2, 0.2])

    # Parameter grids
    threshold_grid = [0.3, 0.4, 0.5, 0.6, 0.7]
    thermal_grid   = [0.3, 0.4, 0.5, 0.6, 0.7]
    chemical_grid  = [0.2, 0.3, 0.4, 0.5]
    mechanical_grid = [0.1, 0.2, 0.3, 0.4]

    results = DataFrame(
        heuristic_threshold=Float64[],
        belief_thermal=Float64[],
        belief_chemical=Float64[],
        belief_mechanical=Float64[],
        mean_reward=Float64[],
        ci_reward=Float64[],
        mean_reg_penalties=Float64[],
        mean_treatment_cost=Float64[],
    )

    n_combos = 0
    for ht in threshold_grid, bt in thermal_grid, bc in chemical_grid, bm in mechanical_grid
        # Enforce cascade ordering: thermal > mechanical > chemical > 0
        bt > bm > bc || continue
        n_combos += 1
    end
    @info "Total valid parameter combinations: $n_combos"

    combo_idx = 0
    best_reward = -Inf
    best_params = nothing

    for ht in threshold_grid
        for bt in thermal_grid
            for bm in mechanical_grid
                for bc in chemical_grid
                    # Enforce cascade ordering: thermal > mechanical > chemical > 0
                    bt > bm > bc || continue
                    combo_idx += 1

                    # Create config with these heuristic parameters
                    config = AquaOpt.setup_experiment_configs(
                        "heuristic_opt_$(location)",
                        true,   # log_space
                        false,  # ekf_filter (UKF)
                        mode,
                        location;
                        reward_lambdas=reward_lambdas,
                        sim_reward_lambdas=reward_lambdas,
                    )

                    # Override heuristic parameters
                    sc = config.solver_config
                    solver_cfg = SolverConfig(
                        log_space=sc.log_space, reward_lambdas=sc.reward_lambdas,
                        sarsop_max_time=sc.sarsop_max_time, VI_max_iterations=sc.VI_max_iterations,
                        QMDP_max_iterations=sc.QMDP_max_iterations, discount_factor=sc.discount_factor,
                        discretization_step=sc.discretization_step, location=sc.location,
                        adult_sd=sc.adult_sd, regulation_limit=sc.regulation_limit,
                        season_regulation_limits=sc.season_regulation_limits,
                        full_observability_solver=sc.full_observability_solver,
                        reproduction_rate=sc.reproduction_rate,
                        salmon_price_MNOK_per_tonne=sc.salmon_price_MNOK_per_tonne,
                        regulatory_violation_cost_MNOK=sc.regulatory_violation_cost_MNOK,
                        welfare_cost_MNOK=sc.welfare_cost_MNOK,
                        chronic_lice_cost_MNOK=sc.chronic_lice_cost_MNOK,
                        heuristic_threshold=ht,
                        heuristic_belief_threshold_thermal=bt,
                        heuristic_belief_threshold_chemical=bc,
                        heuristic_belief_threshold_mechanical=bm,
                    )

                    # Only run the heuristic
                    algo = Algorithm(solver_name="Heuristic_Policy", solver_config=solver_cfg)
                    algorithms = [algo]

                    # Solve (trivial for heuristic) and simulate
                    pomdp, mdp = AquaOpt.create_pomdp_mdp(config)
                    policy = AquaOpt.generate_policy(algo, pomdp, mdp)
                    all_policies = Dict("Heuristic_Policy" => (policy=policy, pomdp=pomdp, mdp=mdp))

                    parallel_data, sim_pomdp = AquaOpt.simulate_all_policies(algorithms, config, all_policies)
                    processed = AquaOpt.extract_reward_metrics(parallel_data, config, sim_pomdp)

                    # Extract metrics
                    mean_rwd = mean(processed.mean_rewards_across_sims)
                    ci_rwd = 1.96 * std(processed.mean_rewards_across_sims) / sqrt(nrow(processed))
                    mean_reg = mean(processed.num_regulatory_penalties)
                    mean_cost = mean(processed.treatment_cost)

                    push!(results, (ht, bt, bc, bm, mean_rwd, ci_rwd, mean_reg, mean_cost))

                    if mean_rwd > best_reward
                        best_reward = mean_rwd
                        best_params = (ht=ht, bt=bt, bc=bc, bm=bm)
                    end

                    @info @sprintf("[%d/%d] ht=%.1f bt=%.2f bm=%.2f bc=%.2f → reward=%.3f%s",
                        combo_idx, n_combos, ht, bt, bm, bc, mean_rwd,
                        mean_rwd == best_reward ? " ★ BEST" : "")
                end
            end
        end
    end

    # Sort by mean reward
    sort!(results, :mean_reward, rev=true)

    # Save results
    outdir = joinpath("results", "heuristic_optimization")
    mkpath(outdir)
    outfile = joinpath(outdir, "grid_search_$(location).csv")
    CSV.write(outfile, results)

    # Print top 10
    println("\n" * "="^80)
    println("Top 10 parameter combinations for location=$location:")
    println("="^80)
    println(first(results, 10))
    println("\nBest parameters: $best_params → reward=$best_reward")
    println("Results saved to: $outfile")

    return results, best_params
end

# CLI entry point
location = length(ARGS) >= 1 ? ARGS[1] : "south"
mode = length(ARGS) >= 2 ? ARGS[2] : "debug"

optimize_heuristic(location, mode)

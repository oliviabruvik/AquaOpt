using GaussianFilters
using POMDPs
using POMDPModels
using POMDPTools
using POMDPXFiles
using DataFrames
using JLD2
using Plots
using Distributions
using Parameters
using Statistics
using Printf
using CSV

# ----------------------------
# Mean and confidence interval function
# ----------------------------
function mean_and_ci(x)
    m = mean(x)
    ci = 1.96 * std(x) / sqrt(length(x))  # 95% confidence interval
    return (mean = m, ci = ci)
end

function _expected_biomass_shortfall(sim_params::SeaLiceSimPOMDP, s, sp)
    ideal_survival_rate = 1 - sim_params.nat_mort_rate
    expected_fish = max(s.NumberOfFish * ideal_survival_rate, 0.0)
    k0_base = sim_params.k_growth * (1.0 + sim_params.temp_sensitivity * (s.Temperature - 10.0))
    ideal_k0 = max(k0_base, 0.0)
    expected_weight = s.AvgFishWeight + ideal_k0 * (sim_params.w_max - s.AvgFishWeight)
    expected_weight = clamp(expected_weight, sim_params.weight_bounds...)
    expected_biomass = biomass_tons(expected_weight, expected_fish)
    next_biomass = biomass_tons(sp)
    return max(expected_biomass - next_biomass, 0.0)
end

# ----------------------------
# Display the mean and confidence interval for each policy
# ----------------------------
function display_rewards_across_policies(parallel_data, config)

    data_grouped_by_policy = groupby(parallel_data, :policy)
    result = combine(data_grouped_by_policy, :reward => mean_and_ci => AsTable)

    # Order by mean reward
    result = sort(result, :mean, rev=true)
    println(result)
end

# ----------------------------
# Extract the number of treatments, regulatory penalties, lost biomass, and fish disease for each policy from the histories and add as columns to the parallel_data dataframe
# Returns a new DataFrame without modifying the input
# ----------------------------
function extract_reward_metrics(data, config, sim_pomdp=nothing)

    # Create a copy of the data to avoid mutating the input
    processed_data = copy(data)
    high_fidelity = config.simulation_config.high_fidelity_sim

    # Add new columns to the DataFrame copy
    processed_data.mean_rewards_across_sims = zeros(Float64, nrow(processed_data))
    processed_data.treatment_cost = zeros(Float64, nrow(processed_data))
    processed_data.treatments = Vector{Dict{Action, Int}}(undef, nrow(processed_data))
    processed_data.num_regulatory_penalties = zeros(Float64, nrow(processed_data))

    if high_fidelity
        processed_data.fish_disease = zeros(Float64, nrow(processed_data))
        processed_data.lost_biomass_1000kg = zeros(Float64, nrow(processed_data))
        processed_data.mean_adult_sea_lice_level = zeros(Float64, nrow(processed_data))
        # Financial columns (all MNOK)
        processed_data.harvest_revenue_MNOK = zeros(Float64, nrow(processed_data))
        processed_data.total_treatment_cost_MNOK = zeros(Float64, nrow(processed_data))
        processed_data.total_regulatory_cost_MNOK = zeros(Float64, nrow(processed_data))
        processed_data.total_biomass_loss_MNOK = zeros(Float64, nrow(processed_data))
        processed_data.total_welfare_cost_MNOK = zeros(Float64, nrow(processed_data))
        processed_data.total_lice_cost_MNOK = zeros(Float64, nrow(processed_data))
        processed_data.net_profit_MNOK = zeros(Float64, nrow(processed_data))
        # Use provided sim_pomdp or create one with config params
        sim_params = sim_pomdp !== nothing ? sim_pomdp : create_sim_pomdp(config)
    end

    for (i, row) in enumerate(eachrow(processed_data))

        # Get the history
        h = row.history
        states = collect(h[:s])
        actions = collect(h[:a])
        rewards = collect(h[:r])

        # Get distribution of treatments
        treatments = Dict{Action, Int}()
        for a in actions
            treatments[a] = get(treatments, a, 0) + 1
        end

        # Add to dataframe copy
        processed_data.treatment_cost[i] = sum(get_treatment_cost(a) for a in actions)
        processed_data.treatments[i] = treatments
        processed_data.mean_rewards_across_sims[i] = mean(rewards)

        if high_fidelity
            # Regulatory penalty based on observations (sampled counts) with season-dependent limits
            observations = collect(h[:o])
            processed_data.num_regulatory_penalties[i] = sum(
                begin
                    season = week_to_season(s.AnnualWeek)
                    reg_limit = sim_params.season_regulation_limits[season]
                    o.Adult > reg_limit ? 1.0 : 0.0
                end
                for (s, o) in zip(states, observations)
            )
            processed_data.mean_adult_sea_lice_level[i] = mean(s.Adult for s in states)
            processed_data.fish_disease[i] = sum(get_fish_disease(a) + 100.0 * s.SeaLiceLevel for (s, a) in zip(states, actions))

            lost_biomass_1000kg = 0.0
            if length(states) > 1
                n_pairs = min(length(states) - 1, length(actions))
                for t in 1:n_pairs
                    lost_biomass_1000kg += _expected_biomass_shortfall(sim_params, states[t], states[t+1])
                end
            end
            processed_data.lost_biomass_1000kg[i] = lost_biomass_1000kg

            # === Financial summary (all in MNOK) ===
            salmon_price = sim_params.salmon_price_MNOK_per_tonne

            # Harvest revenue: final biomass × salmon price
            sp_states = collect(h[:sp])
            final_state = last(sp_states)
            processed_data.harvest_revenue_MNOK[i] = biomass_tons(final_state) * salmon_price

            # Treatment costs (already in MNOK)
            processed_data.total_treatment_cost_MNOK[i] = processed_data.treatment_cost[i]

            # Regulatory violation costs
            processed_data.total_regulatory_cost_MNOK[i] = processed_data.num_regulatory_penalties[i] * sim_params.regulatory_violation_cost_MNOK

            # Biomass loss in MNOK
            processed_data.total_biomass_loss_MNOK[i] = lost_biomass_1000kg * salmon_price

            # Welfare costs (with cooldown stress multiplier)
            total_welfare = 0.0
            for t in eachindex(actions)
                base_stress = get_fish_disease(actions[t])
                cooldown = states[t].Cooldown
                stress_mult = cooldown == 1 ? sim_params.cooldown_stress_multiplier : 1.0
                total_welfare += base_stress * stress_mult * sim_params.welfare_cost_MNOK
            end
            processed_data.total_welfare_cost_MNOK[i] = total_welfare

            # Sea lice burden costs
            total_lice = 0.0
            for s in states
                adult = s.Adult
                total_lice += adult * (1.0 + 0.2 * max(0, adult - 0.5)) * sim_params.chronic_lice_cost_MNOK
            end
            processed_data.total_lice_cost_MNOK[i] = total_lice

            # Net profit = harvest revenue - all costs
            processed_data.net_profit_MNOK[i] = processed_data.harvest_revenue_MNOK[i] - (
                processed_data.total_treatment_cost_MNOK[i] +
                processed_data.total_regulatory_cost_MNOK[i] +
                processed_data.total_biomass_loss_MNOK[i] +
                processed_data.total_welfare_cost_MNOK[i] +
                processed_data.total_lice_cost_MNOK[i]
            )
        else
            # Season-dependent regulation for solver POMDPs
            processed_data.num_regulatory_penalties[i] = sum(
                begin
                    reg_limit = config.solver_config.season_regulation_limits[s.Season]
                    s.SeaLiceLevel > reg_limit ? 1.0 : 0.0
                end
                for s in states
            )
        end
    end

    # Save processed data
    mkpath(config.results_dir)
    @save joinpath(config.results_dir, "processed_data.jld2") processed_data

    return processed_data

end

# ----------------------------
# Display the mean and confidence interval for each policy
# ----------------------------
function display_reward_metrics(parallel_data, config, display_ci=false, print_sd=false)

    data_grouped_by_policy = groupby(parallel_data, :policy)
    col_syms = Set(Symbol.(names(parallel_data)))
    has_treatments = :treatments in col_syms

    function push_ci!(agg_pairs, col, mean_sym, ci_sym; digits=2, f=identity)
        push!(agg_pairs, col => (x -> round(mean_and_ci(f(x)).mean, digits=digits)) => mean_sym)
        push!(agg_pairs, col => (x -> round(mean_and_ci(f(x)).ci, digits=digits)) => ci_sym)
    end

    # Build aggregation pairs — always include CI
    agg_pairs = Any[]
    push_ci!(agg_pairs, :reward, :mean_reward, :ci_reward)
    push_ci!(agg_pairs, :mean_rewards_across_sims, :mean_sim_reward, :ci_sim_reward)
    push_ci!(agg_pairs, :treatment_cost, :mean_treatment_cost, :ci_treatment_cost)
    push_ci!(agg_pairs, :num_regulatory_penalties, :mean_reg_penalties, :ci_reg_penalties)

    # High-fidelity-only metrics
    for (col, base) in [
        (:mean_adult_sea_lice_level, :sea_lice),
        (:lost_biomass_1000kg, :lost_biomass),
        (:fish_disease, :fish_disease),
    ]
        col in col_syms && push_ci!(agg_pairs, col, Symbol("mean_", base), Symbol("ci_", base))
    end

    if has_treatments
        for (action, base) in [
            (NoTreatment, :no_treatment),
            (MechanicalTreatment, :mechanical),
            (ChemicalTreatment, :chemical),
            (ThermalTreatment, :thermal),
        ]
            f = x -> [get(t, action, 0) for t in x]
            push_ci!(agg_pairs, :treatments, Symbol("mean_", base), Symbol("ci_", base); f=f)
        end
    end

    result = combine(data_grouped_by_policy, agg_pairs...)
    result = sort(result, :mean_reward, rev=true)

    # Display result — optionally hide CI columns
    if display_ci
        println(result)
    else
        ci_cols = filter(n -> startswith(String(n), "ci_"), names(result))
        println(select(result, Not(ci_cols)))
    end

    # Format mean±CI helper
    fmt_ci(m, c, digits) = @sprintf("%.*f±%.*f", digits, m, digits, c)

    # Print formatted summary table with ± CI
    if print_sd
        has_sea_lice = :mean_sea_lice in names(result)
        has_fish_disease = :mean_fish_disease in names(result)

        println("\n" * "="^80)
        summary_cols = [
            (:mean_reward, :ci_reward, "Mean Reward", 3),
            (:mean_treatment_cost, :ci_treatment_cost, "Treatment Cost", 3),
            (:mean_reg_penalties, :ci_reg_penalties, "Reg. Penalties", 3),
        ]
        has_sea_lice && push!(summary_cols, (:mean_sea_lice, :ci_sea_lice, "Sea Lice Level", 3))
        has_fish_disease && push!(summary_cols, (:mean_fish_disease, :ci_fish_disease, "Fish Disease", 1))

        header = ["Policy"; [c[3] for c in summary_cols]]
        println(join([@sprintf("%-20s", header[1]); [@sprintf("%14s", h) for h in header[2:end]]], ""))
        println("-"^(20 + 14 * length(summary_cols)))

        for row in eachrow(result)
            parts = [@sprintf("%-20s", row.policy)]
            for (mcol, ccol, _, digits) in summary_cols
                push!(parts, @sprintf("%14s", fmt_ci(row[mcol], row[ccol], digits)))
            end
            println(join(parts, ""))
        end

        if has_treatments
            println("\nTreatment Distribution:")
            println("-"^(20 + 12 * 4))
            println(@sprintf("%-20s %12s %12s %12s %12s", "Policy", "No Treatment", "Mechanical", "Chemical", "Thermal"))
            println("-"^(20 + 12 * 4))

            for row in eachrow(result)
                println(@sprintf("%-20s %12s %12s %12s %12s",
                            row.policy,
                            fmt_ci(row.mean_no_treatment, row.ci_no_treatment, 1),
                            fmt_ci(row.mean_mechanical, row.ci_mechanical, 1),
                            fmt_ci(row.mean_chemical, row.ci_chemical, 1),
                            fmt_ci(row.mean_thermal, row.ci_thermal, 1)))
            end
        end
    end

    println("\n")

    # Ensure results directory exists before saving any CSVs
    mkpath(config.results_dir)

    # --- Financial summary (high-fidelity only) ---
    if :net_profit_MNOK in col_syms
        # Already aggregated above; build financial-specific aggregation
        fin_pairs = Any[]
        for (col, sym) in [
            (:harvest_revenue_MNOK, :harvest_rev),
            (:total_treatment_cost_MNOK, :trt_cost),
            (:total_regulatory_cost_MNOK, :reg_cost),
            (:total_biomass_loss_MNOK, :bio_cost),
            (:total_welfare_cost_MNOK, :welfare_cost),
            (:total_lice_cost_MNOK, :lice_cost),
            (:net_profit_MNOK, :net_profit),
        ]
            push!(fin_pairs, col => (x -> round(mean_and_ci(x).mean, digits=2)) => Symbol("mean_", sym))
            push!(fin_pairs, col => (x -> round(mean_and_ci(x).ci, digits=2)) => Symbol("ci_", sym))
        end
        fin_result = combine(data_grouped_by_policy, fin_pairs...)
        fin_result = sort(fin_result, :mean_net_profit, rev=true)

        println("="^110)
        println("PRODUCTION CYCLE FINANCIAL SUMMARY (MNOK)")
        println("="^110)
        header = @sprintf("%-20s %12s %12s %12s %12s %12s %12s %14s",
            "Policy", "Harvest Rev", "Trt Cost", "Reg Cost", "Bio Loss", "Welfare", "Lice Cost", "Net Profit")
        println(header)
        println("-"^110)

        for row in eachrow(fin_result)
            println(@sprintf("%-20s %12s %12s %12s %12s %12s %12s %14s",
                row.policy,
                fmt_ci(row.mean_harvest_rev, row.ci_harvest_rev, 2),
                fmt_ci(row.mean_trt_cost, row.ci_trt_cost, 2),
                fmt_ci(row.mean_reg_cost, row.ci_reg_cost, 2),
                fmt_ci(row.mean_bio_cost, row.ci_bio_cost, 2),
                fmt_ci(row.mean_welfare_cost, row.ci_welfare_cost, 2),
                fmt_ci(row.mean_lice_cost, row.ci_lice_cost, 2),
                fmt_ci(row.mean_net_profit, row.ci_net_profit, 2)))
        end
        println("="^110)
        println("\n")

        # Save financial summary to CSV
        CSV.write(joinpath(config.results_dir, "financial_summary.csv"), fin_result)
    end

    # Save results to csv
    CSV.write(joinpath(config.results_dir, "reward_metrics.csv"), result)

    # Save treatment data to csv (used by reward_analysis.jl and aggregate_lambda_summaries.jl)
    if has_treatments
        treatment_df = select(result, :policy,
            :mean_no_treatment => :NoTreatment,
            :mean_mechanical => :MechanicalTreatment,
            :mean_chemical => :ChemicalTreatment,
            :mean_thermal => :ThermalTreatment,
        )
        CSV.write(joinpath(config.results_dir, "treatment_data.csv"), treatment_df)
    end

end

# ----------------------------
# Print all histories to a text file
# Data stores the histories in a dataframe with the following columns:
# reward, n_steps, history, policy, seed
# Creates a simulation_steps folder with a text file for each episode.
# The text file contains the history for each step in the episode.
# The text file is named seed_<seed>_simulation_history.txt
# ----------------------------
function print_histories(data, config)

    # Get all unique policies
    policies = unique(data.policy)

    for policy in policies

        # Get histories for this policy
        data_policy = filter(row -> row.policy == policy, data)

        # Create policy folder
        mkpath(joinpath(config.simulations_dir, policy))

        # Get all unique seeds
        seeds = unique(data_policy.seed)

        for seed in seeds

            # Get histories for this seed
            data_seed = filter(row -> row.seed == seed, data_policy)
            h = data_seed.history[1]
            episode = 1

            filename = "seed_$(seed)_simulation_history.txt"
            filepath = joinpath(config.simulations_dir, policy, filename)

                # Create file if it doesn't exist
                if !isfile(filepath)
                    open(filepath, "w") do file
                        println(file, "Simulation History")
                        println(file, "--------------------------------")
                    end
                end

                for (s, a, r, o, b, bp, sp) in eachstep(h, "(s, a, r, o, b, bp, sp)")

                    # State
                    s_adult = round(s.Adult, digits=2)
                    s_motile = round(s.Motile, digits=2)
                    s_sessile = round(s.Sessile, digits=2)
                    s_temp = round(s.Temperature, digits=2)
                    s_pred = round(s.SeaLiceLevel, digits=2)

                    # Observation
                    o_adult = round(o.Adult, digits=2)
                    o_motile = round(o.Motile, digits=2)
                    o_sessile = round(o.Sessile, digits=2)
                    o_temp = round(o.Temperature, digits=2)
                    o_pred = round(o.SeaLiceLevel, digits=2)

                    # New state
                    sp_adult = round(sp.Adult, digits=2)
                    sp_motile = round(sp.Motile, digits=2)
                    sp_sessile = round(sp.Sessile, digits=2)
                    sp_temp = round(sp.Temperature, digits=2)
                    sp_pred = round(sp.SeaLiceLevel, digits=2)

                    # Belief
                    b_adult = round(b.μ[BELIEF_IDX_ADULT], digits=2)
                    b_motile = round(b.μ[BELIEF_IDX_MOTILE], digits=2)
                    b_sessile = round(b.μ[BELIEF_IDX_SESSILE], digits=2)
                    b_temp = round(b.μ[BELIEF_IDX_TEMPERATURE], digits=2)

                    # New belief
                    bp_adult = round(bp.μ[BELIEF_IDX_ADULT], digits=2)
                    bp_motile = round(bp.μ[BELIEF_IDX_MOTILE], digits=2)
                    bp_sessile = round(bp.μ[BELIEF_IDX_SESSILE], digits=2)
                    bp_temp = round(bp.μ[BELIEF_IDX_TEMPERATURE], digits=2)

                    # Create table with state, observation, belief, and new state information
                    # Save to file with filepath
                    open(filepath, "a") do file
                        println(file, "--------------------------------")
                        println(file, "\nEpisode $episode:")
                        println(file, "   Took action: $a")
                        println(file, "   Received reward: $(round(r, digits=2))")
                        println(file, "┌─────────┬──────────┬─────────────┬──────────┬─────────────┬─────────────┐")
                        println(file, "│ Variable│   State  │ Observation │  Belief  │  New Belief │  New State  │")
                        println(file, "├─────────┼──────────┼─────────────┼──────────┼─────────────┼─────────────┤")
                        println(file, "│ Adult   │ $(s_adult)    │ $(o_adult)       │ $(b_adult)    │ $(bp_adult)       │ $(sp_adult)       │")
                        println(file, "│ Motile  │ $(s_motile)    │ $(o_motile)       │ $(b_motile)    │ $(bp_motile)       │ $(sp_motile)       │")
                        println(file, "│ Sessile │ $(s_sessile)    │ $(o_sessile)       │ $(b_sessile)    │ $(bp_sessile)       │ $(sp_sessile)       │")
                        println(file, "│ Pred    │ $(s_pred)    │ $(o_pred)       │ $(b_temp)    │ $(bp_temp)       │ $(sp_pred)       │")
                        println(file, "└─────────┴──────────┴─────────────┴──────────┴─────────────┴─────────────┘")
                    end

                    episode += 1
            end
        end
    end
end

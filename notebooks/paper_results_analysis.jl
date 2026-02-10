### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 20527f16-ba0c-11f0-b1c2-e3cc93d0c374
# ╠═╡ show_logs = false
begin
    using Pkg #, Revise
	import Revise
    Pkg.activate("/Users/oliviabeyerbruvik/Desktop/AquaOpt")
    Pkg.instantiate()
	using AquaOpt

	# we need this to fix
	# ```
	# type Main.AquaOpt.ExperimentConfig does not exist in workspace
	# ```
	Core.eval(Main, :(using AquaOpt))
end

# ╔═╡ f9df9c0d-eeda-44a3-9958-a127e0253478
begin
    using DataFrames
    using StatsPlots   # for @df
	using Statistics
	using Formatting
	using PrettyTables
	using Printf
	using Dates
	using Random
	using POMDPTools
	using CSV
	using JLD2
end

# ╔═╡ cc902235-e492-46cb-a279-b69b1230b93d
md"""
# Notebook set-up
"""

# ╔═╡ 8ff0af96-63c9-43ed-9999-00dbc0bc2112
Random.seed!(42)

# ╔═╡ 2dc2285b-673b-4aec-a65c-def6ebb65f0e
pwd()

# ╔═╡ 722456d5-123e-4d2d-95ad-faa8b547f6a8
cd("..")

# ╔═╡ d2aa9519-8f39-4c8f-80b2-398d0a5bbbfd
md"""
# Experiment
"""

# ╔═╡ 34f769c5-2a81-43cf-ae03-0380e3cc6896
begin
	experiment_path = "results/experiments/2026-02-09/2026-02-09T16:42:22.732_baseline_norway_north_debug_north_[0.2, 0.2, 0.2, 0.2, 0.2]"
	config_path = joinpath(experiment_path, "config/experiment_config.jld2")
	sim_data_path = joinpath(experiment_path, "simulation_histories/all_policies_simulation_data.jld2")
end

# ╔═╡ f14b476c-c54a-4a3d-b9cc-733353870342
function read_in_reward_metrics(experiment_path)
    ## Read in results
    reward_path = joinpath(
		experiment_path, "avg_results/reward_metrics.csv"
	)
    reward_metrics = CSV.read(reward_path, DataFrame)

	sort!(reward_metrics, :mean_reward, rev=true)

    metric_pairs = [
        (:mean_reward, :ci_reward),
        (:mean_sim_reward, :ci_sim_reward),
        (:mean_treatment_cost, :ci_treatment_cost),
        (:mean_reg_penalties, :ci_reg_penalties),
        (:mean_sea_lice, :ci_sea_lice),
        (:mean_lost_biomass, :ci_lost_biomass),
        (:mean_fish_disease, :ci_fish_disease),
    ]

    # store numeric values inside NamedTuples
    for (m, ci) in metric_pairs
        reward_metrics[!, m] = map((x,y) -> (mean=x, ci=y),
                                   reward_metrics[!, m],
                                   reward_metrics[!, ci])
        select!(reward_metrics, Not(ci))
    end

    # ---- Identify Best Values ----
    # highest reward
    best_reward = maximum(row.mean_reward.mean for row in eachrow(reward_metrics))

    # lowest minimization metrics
    best_cost      = minimum(row.mean_treatment_cost.mean for row in eachrow(reward_metrics))
    best_penalty   = minimum(row.mean_reg_penalties.mean for row in eachrow(reward_metrics))
    best_lice      = minimum(row.mean_sea_lice.mean for row in eachrow(reward_metrics))
    best_biomass   = minimum(row.mean_lost_biomass.mean for row in eachrow(reward_metrics))
    best_disease   = minimum(row.mean_fish_disease.mean for row in eachrow(reward_metrics))

    best_map = Dict(
        :mean_reward => best_reward,
        :mean_treatment_cost => best_cost,
        :mean_reg_penalties => best_penalty,
        :mean_sea_lice => best_lice,
        :mean_lost_biomass => best_biomass,
        :mean_fish_disease => best_disease
    )

    # ---- Pretty-Printing with Bold Best Values ----
    formatter = (row, col, val) -> begin
        if !(val isa NamedTuple)   # skip non-metric columns
            return val
        end
        metric = names(reward_metrics)[col]
        is_best = val.mean == best_map[metric]
        txt = @sprintf("%.2f (%.2f)", val.mean, val.ci)
        return is_best ? "\\textbf{$txt}" : txt
    end

 #    pretty_table(
	#     reward_metrics;
	#     backend = :latex,
	#     formatters = [formatter],   # <-- ADD BRACKETS
	#     alignment=:l
	# )

    reward_metrics
end

# ╔═╡ 45e548e4-85ba-4840-892b-5c205876c964
begin

	# Read in config
	config = JLD2.load(config_path)["config"]

	# Read in pomdps
	@load joinpath(
		experiment_path, "policies", "policies_pomdp_mdp.jld2") all_policies
	policy_bundle = all_policies["Native_SARSOP_Policy"]
	policy = policy_bundle.policy
	pomdp = policy_bundle.pomdp
	mdp = policy_bundle.mdp

	# Read in parallel data
	parallel_data = JLD2.load(sim_data_path)["data"]
	algorithms = define_algorithms(config)

end

# ╔═╡ e69d2cb4-feed-401e-b1aa-a7de6fdd40c7
processed_data = extract_reward_metrics(parallel_data, config)

# ╔═╡ 7c78c1d1-f5a1-405a-b678-f4ba1bd0113c
plot_kalman_filter_trajectory_with_uncertainty(processed_data, "Native_SARSOP_Policy", config)

# ╔═╡ 102a2969-4626-4461-96db-0539f03b3186
plot_kalman_filter_belief_trajectory_two_panel(processed_data, "Native_SARSOP_Policy", config)

# ╔═╡ e0423dbc-5d25-4c0b-8d19-4b18ef5c56c2
plos_one_sarsop_dominant_action(processed_data, config)

# ╔═╡ f35c8ed7-8fbf-473d-adc7-7e3041597257
md"""
# Paper Plots
"""

# ╔═╡ 7843da50-8d48-42a2-b032-7d057ba0f393
md"""
## Fig 1
"""

# ╔═╡ 695ff0c4-2660-4e2c-98a0-04fb09b44de0
plos_one_algo_sealice_levels_over_time(parallel_data, config, "NUS_SARSOP_Policy")

# ╔═╡ e15561c6-b046-4fe0-a2e9-fc7972b40878
md"""
## Fig 2
"""

# ╔═╡ 58ac5036-29f7-4778-a57f-0cc88eda3177
plos_one_plot_kalman_filter_belief_trajectory(processed_data, "NUS_SARSOP_Policy", config)

# ╔═╡ 2b69136a-37c2-4654-b38f-737b06147dc6
md"""
## Fig 3
"""

# ╔═╡ 74cf901d-813c-4837-bd74-546c2d04b927
plos_one_reward_over_time(processed_data, config)

# ╔═╡ e1f79b0c-7d20-4b03-aa57-726482977032
plos_one_sealice_levels_over_time(processed_data, config)

# ╔═╡ a38e480c-e2f1-46bd-8065-aa5c8359269a
plos_one_combined_treatment_probability_over_time(processed_data, config)

# ╔═╡ a7c85a15-4cbb-4297-9e51-da8888546529
plos_one_biomass_loss_over_time(processed_data, config)

# ╔═╡ 9e938b25-ccc0-48b8-96c5-c8397de01cce
md"""
## Fig 4
"""

# ╔═╡ 348cad19-1ffb-4d12-96e7-c950ef261389
plos_one_sarsop_dominant_action(parallel_data, config)

# ╔═╡ a519ab83-4cb4-452e-bf24-de5b73653681
plos_one_treatment_distribution_comparison(parallel_data, config)

# ╔═╡ Cell order:
# ╟─cc902235-e492-46cb-a279-b69b1230b93d
# ╠═20527f16-ba0c-11f0-b1c2-e3cc93d0c374
# ╠═f9df9c0d-eeda-44a3-9958-a127e0253478
# ╠═8ff0af96-63c9-43ed-9999-00dbc0bc2112
# ╠═2dc2285b-673b-4aec-a65c-def6ebb65f0e
# ╠═722456d5-123e-4d2d-95ad-faa8b547f6a8
# ╟─d2aa9519-8f39-4c8f-80b2-398d0a5bbbfd
# ╠═34f769c5-2a81-43cf-ae03-0380e3cc6896
# ╟─f14b476c-c54a-4a3d-b9cc-733353870342
# ╠═45e548e4-85ba-4840-892b-5c205876c964
# ╠═e69d2cb4-feed-401e-b1aa-a7de6fdd40c7
# ╠═7c78c1d1-f5a1-405a-b678-f4ba1bd0113c
# ╠═102a2969-4626-4461-96db-0539f03b3186
# ╠═e0423dbc-5d25-4c0b-8d19-4b18ef5c56c2
# ╟─f35c8ed7-8fbf-473d-adc7-7e3041597257
# ╠═7843da50-8d48-42a2-b032-7d057ba0f393
# ╠═695ff0c4-2660-4e2c-98a0-04fb09b44de0
# ╠═e15561c6-b046-4fe0-a2e9-fc7972b40878
# ╠═58ac5036-29f7-4778-a57f-0cc88eda3177
# ╠═2b69136a-37c2-4654-b38f-737b06147dc6
# ╠═74cf901d-813c-4837-bd74-546c2d04b927
# ╠═e1f79b0c-7d20-4b03-aa57-726482977032
# ╠═a38e480c-e2f1-46bd-8065-aa5c8359269a
# ╠═a7c85a15-4cbb-4297-9e51-da8888546529
# ╟─9e938b25-ccc0-48b8-96c5-c8397de01cce
# ╠═348cad19-1ffb-4d12-96e7-c950ef261389
# ╠═a519ab83-4cb4-452e-bf24-de5b73653681

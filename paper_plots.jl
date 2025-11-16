### A Pluto.jl notebook ###
# v0.20.20

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
end

# ╔═╡ ae1013de-f22e-42d4-90e0-9edf6859711c
# ╠═╡ show_logs = false
begin
	Pkg.add("Formatting")
	Pkg.add("PrettyTables")
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
end

# ╔═╡ 8ff0af96-63c9-43ed-9999-00dbc0bc2112
Random.seed!(42)

# ╔═╡ a96b743f-c18a-4fff-8222-4f83f3152e7d
begin
	# Test that we can access exported types and functions
	@info "Testing AquaOpt imports..."

	# Available actions
	actions = [NoTreatment, Treatment, ThermalTreatment]
	@info "Available actions: $actions"

	# Get treatment costs
	for action in actions
		cost = get_treatment_cost(action)
		@info "Cost of $action: $cost"
	end

	md"""
	## Available Functions and Types

	The AquaOpt module exports:
	- **Actions**: `NoTreatment`, `Treatment`, `ThermalTreatment`
	- **Config Types**: `ExperimentConfig`, `HeuristicConfig`, `Algorithm`
	- **Main Functions**: `main()`, `run_experiments()`, `setup_experiment_configs()`, `define_algorithms()`
	"""
end

# ╔═╡ 76d768c5-f387-4dda-af1f-c3ad8df923bc
begin
	# Create a custom experiment configuration
	my_config = ExperimentConfig(
		num_episodes = 100,
		steps_per_episode = 52,
		log_space = true,
		ekf_filter = true,
		experiment_name = "pluto_experiment",
		verbose = true,
		step_through = false,
		reward_lambdas = [0.7, 0.2, 0.1, 0.1, 0.8],
		sarsop_max_time = 5.0,
		VI_max_iterations = 10,
		QMDP_max_iterations = 10,
		location = "north",
	)

	md"""
	## Custom Experiment Configuration

	Created a configuration with:
	- **Episodes**: $(my_config.num_episodes)
	- **Steps per episode**: $(my_config.steps_per_episode)
	- **Location**: $(my_config.location)
	- **Log space**: $(my_config.log_space)
	"""
end

# ╔═╡ e2f5b90b-acbd-4741-929e-42609f6f8b53
begin
	# Example: Run a simple experiment
	# Uncomment to run (this will take some time)

	# main(
	#     first_step_flag="solve",
	#     log_space=true,
	#     experiment_name="pluto_test",
	#     mode="debug",
	#     ekf_filter=true
	# )

	md"""
	## Running Experiments

	To run an experiment, uncomment the code above or call:
	```julia
	main(
		first_step_flag="solve",
		log_space=true,
		experiment_name="pluto_test",
		mode="debug",
		ekf_filter=true
	)
	```

	Available modes:
	- `"debug"`: Fast mode with 10 episodes, 20 steps
	- `"light"`: 1000 episodes, 104 steps
	- `"paper"`: Full paper configuration
	- `"VIdebug"`: Debug Value Iteration
	"""
end

# ╔═╡ b351fa87-17d3-4dd2-a2de-05f3a8314d98
md"""
# Load Configs
"""

# ╔═╡ 8e3f0645-0f96-4913-baa5-b2950a18c455
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
begin
	using JLD2

	# Load parallel data
	parallel_data_path = "results/experiments/2025-11-06/2025-11-06T10:30:10.403_exp_mode_debug/simulation_histories/all_policies_simulation_data.jld2"
	saved_parallel_data = load(parallel_data_path)

	# Load config
	config_path = "results/experiments/2025-11-06/2025-11-06T10:30:10.403_exp_mode_debug//config/experiment_config.jld2"
	saved_config = load(config_path)
end
  ╠═╡ =#

# ╔═╡ d2aa9519-8f39-4c8f-80b2-398d0a5bbbfd
md"""
# Run experiment
"""

# ╔═╡ 34f769c5-2a81-43cf-ae03-0380e3cc6896
md"""
### Set up configuration
"""

# ╔═╡ 4e19dbd4-dd55-4e72-9d9e-fca1af3fe520
begin
	# Parameters
	experiment_name = "exp"
	log_space = true
	ekf_filter = false
	mode = "debug"

	# Solver parameters
	discount_factor=0.95
	sarsop_max_time=5.0
	VI_max_iterations=10
	QMDP_max_iterations=10

	# Simulation
	num_episodes=10
	steps_per_episode=20

	# Solver parameters
	

	# Algorithms
	lambda_values = [0.6] # collect(0.0:0.2:1.0)
	# [treatment, regulatory, biomass, health, sea lice]
    reward_lambdas=[0.7, 0.2, 0.1, 0.1, 0.2]
	# for high-fidelity sim
    sim_reward_lambdas = [0.7, 0.2, 0.1, 0.9, 2.0]

	# Location
	location = "south" # "north", "west", or "south"

	# Heuristic parameters
    heuristic_threshold::Float64 = 0.5  # In absolute space
    heuristic_belief_threshold_mechanical::Float64 = 0.3
    heuristic_belief_threshold_thermal::Float64 = 0.4
    heuristic_rho::Float64 = 0.8
end

# ╔═╡ f14b476c-c54a-4a3d-b9cc-733353870342
md"""
### Define config, algorithms, POMDPs, solve
"""

# ╔═╡ c8496cff-6478-4ace-a026-54c7cb6ddb40
# ╠═╡ show_logs = false
begin
	Revise.revise()

	# Define experiment configuration
    exp_name = string(
		Dates.today(), "/", Dates.now(), "_", experiment_name, "_mode_", mode
	)

	solver_cfg = SolverConfig(
		log_space=log_space,
		reward_lambdas=reward_lambdas,
		sarsop_max_time=sarsop_max_time,
		VI_max_iterations=VI_max_iterations,
		QMDP_max_iterations=QMDP_max_iterations,
		discount_factor = discount_factor,
		location = location, # "north", "west", or "south"
	)
	
	sim_cfg = SimulationConfig(
		num_episodes=num_episodes,
		steps_per_episode=steps_per_episode,
		ekf_filter=ekf_filter,
	)
	
	config = ExperimentConfig(
		solver_config=solver_cfg,
		simulation_config=sim_cfg,
		experiment_name=exp_name,
    )

	heuristic_config = HeuristicConfig(
        raw_space_threshold=config.solver_config.heuristic_threshold,
		belief_threshold_mechanical=config.solver_config.heuristic_belief_threshold_mechanical,
		belief_threshold_thermal=config.solver_config.heuristic_belief_threshold_thermal,
        rho=config.solver_config.heuristic_rho
    )

	# Define algorithms
	algorithms = define_algorithms(config, heuristic_config)
	
end

# ╔═╡ 2781158c-835c-472b-b913-64e582bfd184
md"""
### Solve policies
"""

# ╔═╡ 024e3947-6d7a-4ee7-b7ff-a10c2e633072
begin
	Revise.revise()

	@info "Solving policies"
	algo_to_policy_pomdp_mdp = Dict()

	# Generate policies and POMDPs
    for algo in algorithms
        algo_to_policy_pomdp_mdp[algo] = generate_mdp_pomdp_policies(algo, config)
    end
	
end

# ╔═╡ d2d1d642-9abb-4e70-8d17-ac5318dd12e7
# ╠═╡ show_logs = false
begin
	Revise.revise()

	@info "Simulating policies"
    parallel_data = simulate_all_policies(algorithms, config)

    # If we are simulating on high fidelity model, we want 
	# to evaluate the simulation results
    if config.simulation_config.high_fidelity_sim
        for algo in algorithms
            histories = extract_simulation_histories(config, algo, parallel_data)
            evaluate_simulation_results(config, algo, histories)
        end
    else
        print_reward_metrics_for_vi_policy(parallel_data, config)
        exit()
    end

	# Extract reward metrics
    processed_data = extract_reward_metrics(parallel_data, config)

    # Display reward metrics
    display_reward_metrics(processed_data, config, false)
end

# ╔═╡ 46b264d9-1339-48fd-bc9a-0fb733269f3f
md"""
### Inspect results

#### Processed data
Results come from `parallel_data = simulate_all_policies(algorithms, config)` together with `extract_reward_metrics()`

Inside `simulate_all_policies` function:

	1. For each policy:
	2. Run N episodes in parallel using Distributed or Threads.
	3. Convert all raw results into rows.
	4. vcat them into a single DataFrame.



processed_data returns (#policies × #episodes) rows   ×   (≈14 metrics) columns

| Column                      | Type                                  | Meaning                                                    |
| --------------------------- | ------------------------------------- | ---------------------------------------------------------- |
| `episode_number`            | `Int64`                               | Episode index (1..N episodes)                              |
| `history`                   | `@NamedTuple` (EvaluationState, etc.) | Raw simulation trajectory, including states, actions, etc. |
| `lambda`                    | `Float64`                             | Weight on reward components used in that experiment        |
| `n_steps`                   | `Int64`                               | Episode length                                             |
| `policy`                    | `String`                              | Policy name (e.g., `"VI_Policy"`)                          |
| `reward`                    | `Float64`                             | Total episode reward                                       |
| `seed`                      | `Int64`                               | RNG seed for reproducibility                               |
| `mean_rewards_across_sims`  | `Float64`                             | Average reward at that step across parallel runs           |
| `treatment_cost`            | `Float64`                             | Total cost from treatments used                            |
| `treatments`                | `Dict{Action,Int}`                    | Count of each action taken                                 |
| `num_regulatory_penalties`  | `Float64`                             | Penalty cost from regulatory violations                    |
| `fish_disease`              | `Float64`                             | Final disease metric                                       |
| `lost_biomass_1000kg`       | `Float64`                             | Biomass lost in tons                                       |
| `mean_adult_sea_lice_level` | `Float64`                             | Final lice population                                      |


#### History field

This is rich structured data:

	@NamedTuple{
	    s::EvaluationState,
	    a::Action,
	    r::Float64,
	    ...
	}


Each episode has a vector of them. We can access it like:
	
	first(processed_data.history).s   # first state in episode 1


If we want full trajectories, use:
	
	histories = extract_simulation_histories(config, algo, processed_data)

"""

# ╔═╡ c16a12a0-8aab-472d-b10a-b806fd6fe5d2
md"""
### Results table 6
"""

# ╔═╡ 3bb2144b-5383-4df1-9652-53db1500b003
# ╠═╡ show_logs = false
begin
    # ========= Helpers =========
    mean_std(x) = @sprintf("%.2f ± %.2f", mean(x), std(x))
    get_mean(s) = parse(Float64, first(split(s)))

    desired_order = [
        "NUS_SARSOP_Policy",
        "QMDP_Policy",
        "VI_Policy",
        "Heuristic_Policy",
        "Random_Policy",
        "AlwaysTreat_Policy",
        "NeverTreat_Policy"
    ]
	
	metrics = Dict(
		:reward                    => "Expected Reward",
		:treatment_cost => "Total Treatment Cost (MNOK)",
		:num_regulatory_penalties  => "Number of Regulatory Penalties",
		:mean_adult_sea_lice_level => "Mean Adult Female Lice per Fish",
		:lost_biomass_1000kg       => "Mean Biomass Loss (tons)",
		:fish_disease              => "Fish Disease"
	)

    # ========= Build table =========
    row_dicts = []

    for (metric, rowname) in metrics
        row = Dict{Symbol,String}()
        row[:Metric] = rowname

        values = Dict{String,String}()
        means  = Dict{String,Float64}()

        for subdf in groupby(processed_data, :policy)
            p = first(subdf.policy)
            vals = skipmissing(subdf[!, metric])

            if isempty(vals)
                values[p] = "—"
            else
                formatted = mean_std(vals)
                values[p] = formatted
                means[p]  = get_mean(formatted)
            end
        end

        # === Bold best in row ===
        if !isempty(means)
            best_policy =
                metric == :reward ? argmax(means) : argmin(means)

            values[best_policy] =
                "\\textbf{" * values[best_policy] * "}"
        end

        for p in desired_order
            row[Symbol(p)] = get(values, p, "—")
        end

        push!(row_dicts, row)
    end

    tab_df = DataFrame(row_dicts)

    # ========= LATEX OUTPUT (simple, version-proof) =========
    io = IOBuffer()
	pretty_table(io, tab_df; backend=:latex)
	latex_tabular = String(take!(io))

    println("==== COPY INTO OVERLEAF (inside table environment) ====\n")
    println(latex_tabular)

	tab_df
end

# ╔═╡ df311c71-8c93-4b9c-9eea-acabb23fdce8
# ╠═╡ show_logs = false
begin
    policies = unique(processed_data.policy)

    rows = DataFrame(Metric = String[])

    for pol in policies
        push!(rows, (; Metric = String(pol)))
    end

    # === Compute raw means & stds ===
    for metric in keys(metrics)
        means = combine(groupby(processed_data, :policy), metric => mean => :μ)
        stds  = combine(groupby(processed_data, :policy), metric => std  => :σ)

        col_mean = Dict(means.policy .=> means.μ)
        col_std  = Dict(stds.policy .=> stds.σ)

        μ_col = [col_mean[p] for p in policies]
        σ_col = [col_std[p]  for p in policies]

        insertcols!(rows, Symbol("$(metric)_mean") => μ_col)
        insertcols!(rows, Symbol("$(metric)_std")  => σ_col)
    end

    # === MOVE reward_mean LEFT ===
    # 1. Copy the columns we want at the front
    front_cols = [:Metric, :reward_mean, :reward_std]

    # 2. Append all remaining columns
    remaining = Symbol.(filter(c -> !(Symbol(c) in front_cols), names(rows)))
    rows = rows[:, vcat(front_cols, remaining)]

    # === SORT BY decreasing reward ===
    sort!(rows, :reward_mean, rev = true)

    println("\n========= DEBUG TABLE 6 (sorted by reward) =========\n")
    show(rows, allcols=true, allrows=true)

    rows
end

# ╔═╡ 40ef8ed3-c843-4921-b7d0-3c25bd952b93
md"""
# Plot results
"""

# ╔═╡ 35ffc7cd-fc24-4e39-9a59-eb77121a6762
begin
	Revise.revise()
	plos_one_plot_kalman_filter_belief_trajectory(processed_data, "NUS_SARSOP_Policy", config, 0.6)
end

# ╔═╡ 2296ace3-61cb-4fdb-a4a8-d1528f34033d
begin
	Revise.revise()
	plot_kalman_filter_trajectory_with_uncertainty(processed_data, "NUS_SARSOP_Policy", config, 0.6)
end

# ╔═╡ b7dc8ea5-babc-45b0-bb95-b54e5cb52a55
begin
	Revise.revise()
	plot_kalman_filter_belief_trajectory_two_panel(processed_data, "NUS_SARSOP_Policy", config, 0.6)
end

# ╔═╡ 22abef54-3797-44db-955a-59875ca715dc
begin
	Revise.revise()
	plos_one_sealice_levels_over_time(processed_data, config)
end

# ╔═╡ 9fd7e94c-b94e-47d9-9c26-f3ac56261db9
begin
	Revise.revise()
    plos_one_combined_treatment_probability_over_time(processed_data, config)
end

# ╔═╡ 8862002c-4688-457d-9ae1-cc75d836077a
begin
	Revise.revise()
    plos_one_sarsop_dominant_action(processed_data, config, 0.6)
end

# ╔═╡ 7bf9ac1f-6718-4c66-aca4-114b4e59642a
begin
	Revise.revise()
    plos_one_algo_sealice_levels_over_time(config, "NUS_SARSOP_Policy", 0.6)
end

# ╔═╡ Cell order:
# ╠═20527f16-ba0c-11f0-b1c2-e3cc93d0c374
# ╠═ae1013de-f22e-42d4-90e0-9edf6859711c
# ╠═f9df9c0d-eeda-44a3-9958-a127e0253478
# ╠═8ff0af96-63c9-43ed-9999-00dbc0bc2112
# ╟─a96b743f-c18a-4fff-8222-4f83f3152e7d
# ╟─76d768c5-f387-4dda-af1f-c3ad8df923bc
# ╟─e2f5b90b-acbd-4741-929e-42609f6f8b53
# ╟─b351fa87-17d3-4dd2-a2de-05f3a8314d98
# ╠═8e3f0645-0f96-4913-baa5-b2950a18c455
# ╟─d2aa9519-8f39-4c8f-80b2-398d0a5bbbfd
# ╟─34f769c5-2a81-43cf-ae03-0380e3cc6896
# ╠═4e19dbd4-dd55-4e72-9d9e-fca1af3fe520
# ╟─f14b476c-c54a-4a3d-b9cc-733353870342
# ╟─c8496cff-6478-4ace-a026-54c7cb6ddb40
# ╟─2781158c-835c-472b-b913-64e582bfd184
# ╠═024e3947-6d7a-4ee7-b7ff-a10c2e633072
# ╠═d2d1d642-9abb-4e70-8d17-ac5318dd12e7
# ╟─46b264d9-1339-48fd-bc9a-0fb733269f3f
# ╟─c16a12a0-8aab-472d-b10a-b806fd6fe5d2
# ╟─3bb2144b-5383-4df1-9652-53db1500b003
# ╟─df311c71-8c93-4b9c-9eea-acabb23fdce8
# ╟─40ef8ed3-c843-4921-b7d0-3c25bd952b93
# ╟─35ffc7cd-fc24-4e39-9a59-eb77121a6762
# ╟─2296ace3-61cb-4fdb-a4a8-d1528f34033d
# ╟─b7dc8ea5-babc-45b0-bb95-b54e5cb52a55
# ╟─22abef54-3797-44db-955a-59875ca715dc
# ╟─9fd7e94c-b94e-47d9-9c26-f3ac56261db9
# ╟─8862002c-4688-457d-9ae1-cc75d836077a
# ╟─7bf9ac1f-6718-4c66-aca4-114b4e59642a

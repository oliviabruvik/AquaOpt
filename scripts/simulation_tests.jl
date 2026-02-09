# Activate the project environment
using Pkg
Pkg.activate(".")

using AquaOpt
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


experiment_paths = [
	"results/experiments/2025-11-17/2025-11-17T14:21:28.204_log_space_ekf_paper_north_[0.7, 2.0, 0.1, 0.1, 0.8]",
]

# ----------------------------
# Load config from past run
# ----------------------------
function read_in_config(experiment_path)
    config_path = joinpath(experiment_path, "config/experiment_config.jld2")
    @info "Loading config from: $config_path"
    data = JLD2.load(config_path)
    config = data["config"]  # Extract the "config" key
    return config
end

# ----------------------------
# Load policy, pomdp, and mdp for a given algorithm
# ----------------------------
function load_policy_pomdp_mdp(experiment_path, solver_name)
    policy_path = joinpath(experiment_path, "policies", "policies_pomdp_mdp.jld2")

    if !isfile(policy_path)
        error("Policy file not found at $policy_path")
    end

    @info "Loading policies from: $policy_path"
    data = JLD2.load(policy_path)
    policy_bundle = data["all_policies"][solver_name]
    return policy_bundle.policy, policy_bundle.pomdp, policy_bundle.mdp
end

# ----------------------------
# Main execution
# ----------------------------
@info "Loading configuration from trained experiment"
training_config = read_in_config(experiment_paths[1])

# ----------------------------
# Create NEW simulation config (this is what you want to change!)
# ----------------------------
@info "Creating new simulation configuration"

new_sim_reward_lambdas = [1.0, 0.1, 0.3, 0.3, 2.0]  # [treatment, regulatory, biomass, health, sea_lice]
new_sim_config = SimulationConfig(
    num_episodes=1000,
    steps_per_episode=104,
    ekf_filter=true,
    high_fidelity_sim=true,
    n_sample=100,
    adult_mean=training_config.simulation_config.adult_mean,
    motile_mean=training_config.simulation_config.motile_mean,
    sessile_mean=training_config.simulation_config.sessile_mean,
    adult_sd=training_config.simulation_config.adult_sd,
    motile_sd=training_config.simulation_config.motile_sd,
    sessile_sd=training_config.simulation_config.sessile_sd,
    temp_sd=training_config.simulation_config.temp_sd,
    sim_reward_lambdas=new_sim_reward_lambdas,
)

new_experiment_name = "resim_$(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS.sss"))"
solver_reward_lambdas = training_config.solver_config.reward_lambdas

config = setup_experiment_configs(
    new_experiment_name,
    training_config.solver_config.log_space,
    training_config.simulation_config.ekf_filter,
    "paper",
    training_config.solver_config.location;
    reward_lambdas=solver_reward_lambdas,
    sim_reward_lambdas=new_sim_reward_lambdas,
    solver_reproduction_rate=training_config.solver_config.reproduction_rate,
    solver_regulation_limit=training_config.solver_config.regulation_limit,
)

# Overwrite with training solver config and custom simulation settings
config.solver_config = training_config.solver_config
config.simulation_config = new_sim_config
config.policies_dir = training_config.policies_dir
config.simulations_dir = joinpath("results", "experiments", new_experiment_name, "simulation_histories")
config.results_dir = joinpath("results", "experiments", new_experiment_name, "avg_results")
config.figures_dir = joinpath("results", "experiments", new_experiment_name, "figures")
config.experiment_dir = joinpath("results", "experiments", new_experiment_name)

# Define algorithms (same as original run)
@info "Defining algorithms"
algorithms = define_algorithms(config)

# Simulate policies with NEW simulation config but OLD trained policies
@info "Simulating trained policies with NEW simulation configuration"
@info "  - Loading policies from: $(config.policies_dir)"
@info "  - Using simulation reward lambdas: $(config.simulation_config.sim_reward_lambdas)"
@info "  - Running $(config.simulation_config.num_episodes) episodes"

policies_path = joinpath(config.policies_dir, "policies_pomdp_mdp.jld2")
all_policies = JLD2.load(policies_path)["all_policies"]
parallel_data = simulate_all_policies(algorithms, config, all_policies)

# Evaluate simulation results
processed_data = extract_reward_metrics(parallel_data, config)
display_reward_metrics(processed_data, config, false)

plot_plos_one_plots(parallel_data, config)

@info "Simulation complete!"

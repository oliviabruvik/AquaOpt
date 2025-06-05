include("kalmanFilter.jl")
include("SeaLicePOMDP.jl")
include("SeaLiceLogPOMDP.jl")
include("SimulationPOMDP.jl")
include("SimulationLogPOMDP.jl")

using GaussianFilters
using POMDPs
using POMDPModels
using POMDPTools
using DataFrames
using JLD2
using Plots
using Distributions
using Parameters

# TODO: sweep of different threshold parameters - sensitivity analysis
# randomness - add stochasticity to the model
# if above threshold, choose treatment with probability rho
# TODO: never treat - model, nothing policy + check growth rate

# CONSTANTS
const process_noise = 1.0
const observation_noise = 1.0

# ----------------------------
# Configuration struct
# ----------------------------
@with_kw struct Config
    lambda_values::Vector{Float64} = collect(0.0:0.05:1.0)
    num_episodes::Int = 10
    steps_per_episode::Int = 20
    heuristic_threshold::Float64 = 5.0  # In absolute space
    heuristic_belief_threshold::Float64 = 0.5
    policies_dir::String = joinpath("results", "policies")
    figures_dir::String = joinpath("results", "figures")
    data_dir::String = joinpath("results", "data")
    ekf_filter::Bool = true
end

# ----------------------------
# Algorithm struct
# ----------------------------
@with_kw struct Algorithm{S<:Union{Solver, Nothing}}
    solver::S = nothing # TODO: set to heuristic solver
    convert_to_mdp::Bool = false
    solver_name::String = "Heuristic_Policy"
    heuristic_threshold::Union{Float64, Nothing} = nothing # set to heuristic threshold
    heuristic_belief_threshold::Union{Float64, Nothing} = nothing
end

# ----------------------------
# POMDP config struct
# ----------------------------
@with_kw struct POMDPConfig
    costOfTreatment::Float64 = 10.0
    growthRate::Float64 = 1.2
    rho::Float64 = 0.7
    discount_factor::Float64 = 0.95
    log_space::Bool = true
end

# ----------------------------
# Policy Saving & Loading
# ----------------------------
function save_policy(policy, pomdp, mdp, solver_name, lambda, config)
    mkpath(config.policies_dir)
    save(joinpath(config.policies_dir, "$(solver_name)/$(lambda)_lambda_policy.jld2"), "policy", policy)
    save(joinpath(config.policies_dir, "$(solver_name)/$(lambda)_lambda_pomdp.jld2"), "pomdp", pomdp)
    save(joinpath(config.policies_dir, "$(solver_name)/$(lambda)_lambda_mdp.jld2"), "mdp", mdp)
end

function load_policy(solver_name, lambda, config)
    policy = load(joinpath(config.policies_dir, "$(solver_name)/$(lambda)_lambda_policy.jld2"), "policy")
    pomdp = load(joinpath(config.policies_dir, "$(solver_name)/$(lambda)_lambda_pomdp.jld2"), "pomdp")
    mdp = load(joinpath(config.policies_dir, "$(solver_name)/$(lambda)_lambda_mdp.jld2"), "mdp")
    return (policy, pomdp, mdp)
end

# ----------------------------
# Policy Generation
# ----------------------------
function generate_policy(algorithm, λ, pomdp_config)

    if pomdp_config.log_space
        pomdp = SeaLiceLogMDP(lambda=λ, costOfTreatment=pomdp_config.costOfTreatment, growthRate=pomdp_config.growthRate, rho=pomdp_config.rho, discount_factor=pomdp_config.discount_factor)
    else
        pomdp = SeaLiceMDP(lambda=λ, costOfTreatment=pomdp_config.costOfTreatment, growthRate=pomdp_config.growthRate, rho=pomdp_config.rho, discount_factor=pomdp_config.discount_factor)
    end
    mdp = UnderlyingMDP(pomdp)

    policy = if algorithm.solver_name == "Heuristic_Policy"
        threshold = pomdp_config.log_space ? log(algorithm.heuristic_threshold) : algorithm.heuristic_threshold
        HeuristicPolicy(pomdp, threshold, algorithm.heuristic_belief_threshold)
    elseif algorithm.convert_to_mdp
       solve(algorithm.solver, mdp)
    else
        solve(algorithm.solver, pomdp)
    end
    
    return (policy, pomdp, mdp)
end

# ----------------------------
# Simulation & Evaluation
# ----------------------------
function run_simulation(policy, mdp, pomdp, config, algorithm)

    # Store all histories
    belief_hists = []
    r_total_hists = []
    action_hists = []
    state_hists = []
    measurement_hists = []
    reward_hists = []

    # Create simulator POMDP based on whether we're in log space
    sim_pomdp = if typeof(pomdp) <: SeaLiceLogMDP
        SeaLiceLogSimMDP(
            lambda=pomdp.lambda,
            costOfTreatment=pomdp.costOfTreatment,
            growthRate=pomdp.growthRate,
            rho=pomdp.rho,
            discount_factor=pomdp.discount_factor
        )
    else
        SeaLiceSimMDP(
            lambda=pomdp.lambda,
            costOfTreatment=pomdp.costOfTreatment,
            growthRate=pomdp.growthRate,
            rho=pomdp.rho,
            discount_factor=pomdp.discount_factor
        )
    end

    # Create simulator
    sim = RolloutSimulator(max_steps=config.steps_per_episode)
    updaterStruct = KFUpdater(sim_pomdp, process_noise=process_noise, observation_noise=observation_noise)
    updater = config.ekf_filter ? updaterStruct.ekf : updaterStruct.ukf

    # Run simulation for each episode
    for _ in 1:config.num_episodes
        
        # Get initial state
        s = rand(initialstate(sim_pomdp))

        # Get initial belief from initial mean and sampling sd
        initial_belief = if typeof(sim_pomdp) <: SeaLiceLogSimMDP
            Normal(sim_pomdp.log_lice_initial_mean, sim_pomdp.sampling_sd)
        else
            Normal(sim_pomdp.sea_lice_initial_mean, sim_pomdp.sampling_sd)
        end

        r_total, action_hist, state_hist, measurement_hist, reward_hist, belief_hist = simulate_helper(sim, sim_pomdp, policy, updater, initial_belief, s)

        push!(r_total_hists, r_total)
        push!(action_hists, action_hist)
        push!(state_hists, state_hist)
        push!(measurement_hists, measurement_hist)
        push!(reward_hists, reward_hist)
        push!(belief_hists, belief_hist)
    end

    # Return averages
    return r_total_hists, action_hists, state_hists, measurement_hists, reward_hists, belief_hists
end

function calculate_averages(config, pomdp, action_hists, state_hists, reward_hists)

    total_steps = config.num_episodes * config.steps_per_episode
    total_cost, total_sealice, total_reward = 0.0, 0.0, 0.0

    # TODO: run for one episode to error check
    for i in 1:config.num_episodes
        total_cost += sum(a == Treatment for a in action_hists[i]) * pomdp.costOfTreatment
        # Handle both regular and log space states
        total_sealice += if typeof(state_hists[i][1]) <: SeaLiceLogState
            sum(exp(s.SeaLiceLevel) for s in state_hists[i])
        else
            sum(s.SeaLiceLevel for s in state_hists[i])
        end
        total_reward += sum(reward_hists[i])
    end

    return total_reward / total_steps, total_cost / total_steps, total_sealice / total_steps
end

# ----------------------------
# Simulation Helper Function
# ----------------------------
function simulate_helper(sim::RolloutSimulator, sim_pomdp::POMDP, policy::Policy, updater::Any, initial_belief, s)
    
    # Store histories
    action_hist = []
    state_hist = []
    measurement_hist = []
    reward_hist = []
    belief_hist = []
    disc = 1.0
    r_total = 0.0

    b = initialize_belief(updater, initial_belief)

    step = 1

    while disc > sim.eps && !isterminal(sim_pomdp, s) && step <= sim.max_steps

        # Calculate b as beliefvec from normal distribution
        norm_distr = Normal(b.μ[1], b.Σ[1,1])

        # Generate a belief vector from the normal distribution for POMDP policies
        if typeof(policy) <: ValueIterationPolicy
            a = action(policy, s)
        else
            # For POMDP policies, use the belief state
            state_space = states(policy.pomdp)
            bvec = [pdf(norm_distr, s.SeaLiceLevel) for s in state_space]
            bvec = normalize(bvec, 1)
            a = action(policy, bvec)
        end

        sp, o, r = @gen(:sp,:o,:r)(sim_pomdp, s, a, sim.rng)

        r_total += disc * r

        s = sp

        b = runKalmanFilter(updater, b, a, o)

        disc *= discount(sim_pomdp)
        step += 1

        # Update histories
        push!(action_hist, a)
        push!(state_hist, s)
        push!(measurement_hist, o)
        push!(reward_hist, r)
        push!(belief_hist, b)
    end

    return r_total, action_hist, state_hist, measurement_hist, reward_hist, belief_hist
end


# ----------------------------
# Heuristic Policy
# ----------------------------
struct HeuristicPolicy{P<:POMDP} <: Policy
    pomdp::P
    threshold::Float64
    belief_threshold::Float64
end

# Heuristic action
function POMDPs.action(policy::HeuristicPolicy, b)
    if heuristicChooseAction(policy, b, true)
        return Treatment
    else
        return rand((Treatment, NoTreatment))
    end
end

# TODO: plot heuristic bvec with imageMap

# Function to decide whether we choose the action or randomize
function heuristicChooseAction(policy::HeuristicPolicy, b, use_cdf=true)

    # Convert belief vector to a probability distribution
    state_space = states(policy.pomdp)

    if use_cdf
        # Method 1: Calculate probability of being above threshold
        # TODO: cumsum(b[0:discretizer.index(policy.threshold)])
        prob_above_threshold = sum(b[i] for (i, s) in enumerate(state_space) if s.SeaLiceLevel > policy.threshold)
        return prob_above_threshold > policy.belief_threshold
    else
        # Method 2: Use mode of belief vector
        mode_sealice_level_index = argmax(b)
        mode_sealice_level = state_space[mode_sealice_level_index]
        return mode_sealice_level.SeaLiceLevel > policy.threshold
    end
end

function POMDPs.updater(policy::HeuristicPolicy)
    return DiscreteUpdater(policy.pomdp)
end

# ----------------------------
# Optimizer Wrapper
# ----------------------------
function test_optimizer(algorithm, config, pomdp_config)

    results = DataFrame(
        lambda=Float64[],
        avg_treatment_cost=Float64[],
        avg_sealice=Float64[],
        action_hists=Vector{Any}[],
        belief_hists=Vector{Any}[]
    )

    # Generate policies for each lambda
    for λ in config.lambda_values

        # Generate policy
        policy, pomdp, mdp = generate_policy(algorithm, λ, pomdp_config)

        # Run simulation to calculate average cost and average sea lice level
        r_total_hists, action_hists, state_hists, measurement_hists, reward_hists, belief_hists = run_simulation(policy, mdp, pomdp, config, algorithm)
        avg_reward, avg_cost, avg_sealice = calculate_averages(config, pomdp, action_hists, state_hists, reward_hists)

        # Calculate the average reward, cost, and sea lice level
        push!(results, (λ, avg_cost, avg_sealice, action_hists, belief_hists))
    end

    # Plot results
    results_plot = plot_mdp_results(results, algorithm.solver_name)
    belief_plot = plot_policy_belief_levels(results, algorithm.solver_name)
    
    # Save results
    mkpath(joinpath(config.figures_dir, algorithm.solver_name))
    mkpath(joinpath(config.data_dir, algorithm.solver_name))
    @save joinpath(config.data_dir, "$(algorithm.solver_name)/results_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.jld2") results
    savefig(results_plot, joinpath(config.figures_dir, "$(algorithm.solver_name)/results_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    savefig(belief_plot, joinpath(config.figures_dir, "$(algorithm.solver_name)/beliefs_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    return results
end
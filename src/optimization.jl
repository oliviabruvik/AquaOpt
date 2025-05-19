include("kalmanFilter.jl")
include("SeaLicePOMDP.jl")
include("SimulationPOMDP.jl")

using GaussianFilters
using POMDPs
using POMDPModels
using POMDPTools
using DataFrames
using JLD2
using Plots

# ----------------------------
# Algorithm struct
# ----------------------------
struct Algorithm{S<:Union{Solver, Nothing}}
    solver::S
    convert_to_mdp::Bool
    solver_name::String
    heuristic_threshold::Union{Float64, Nothing}
    heuristic_belief_threshold::Union{Float64, Nothing}
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
function generate_policy(algorithm, λ)
    pomdp = SeaLiceMDP(lambda=λ)
    mdp = UnderlyingMDP(pomdp)

    policy = if algorithm.solver_name == "Heuristic_Policy"
        HeuristicPolicy(pomdp, algorithm.heuristic_threshold, algorithm.heuristic_belief_threshold)
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

    # Create simulator POMDP
    sim_pomdp = SeaLiceSimMDP(
        lambda=pomdp.lambda,
        costOfTreatment=pomdp.costOfTreatment,
        growthRate=pomdp.growthRate,
        rho=pomdp.rho,
        discount_factor=pomdp.discount_factor
    )

    # Create simulator
    sim = RolloutSimulator(max_steps=config.steps_per_episode)
    updaterStruct = KFUpdater(sim_pomdp, process_noise=STD_DEV, observation_noise=STD_DEV)
    updater = config.ekf_filter ? updaterStruct.ekf : updaterStruct.ukf

    # Run simulation for each episode
    for _ in 1:config.num_episodes
        
        # Get initial state
        s = rand(initialstate(sim_pomdp))

        # TODO: Not needed for kalman filter / fix arguments
        initial_belief = Normal(0.5, 1.0)

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

    for i in 1:config.num_episodes
        total_cost += sum(a == Treatment for a in action_hists[i]) * pomdp.costOfTreatment
        total_sealice += sum(s.SeaLiceLevel for s in state_hists[i])
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
            bvec = bvec ./ sum(bvec)
            a = action(policy, bvec)
        end

        sp, o, r = @gen(:sp,:o,:r)(sim_pomdp, s, a, sim.rng)

        r_total += disc*r

        s = sp

        bp = runKalmanFilter(updater, b, a, o)
        b = bp

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
    return heuristicChooseAction(policy, b, true) ? Treatment : NoTreatment
end

# Function to decide whether we choose the action or randomize
function heuristicChooseAction(policy::HeuristicPolicy, b, use_cdf=true)
    # Convert belief vector to a probability distribution
    state_space = states(policy.pomdp)

    if use_cdf
        # Method 1: Calculate probability of being above threshold
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
function test_optimizer(algorithm, config)

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
        policy, pomdp, mdp = generate_policy(algorithm, λ)
        save_policy(policy, pomdp, mdp, algorithm.solver_name, λ, config)

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
end
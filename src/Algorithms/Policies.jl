using POMDPs
using POMDPModels
using POMDPTools
using Distributions
using Parameters
using GaussianFilters
using POMDPs
using POMDPModels
using POMDPTools
using DataFrames
using JLD2
using Plots
using Distributions
using Parameters

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
        HeuristicPolicy(pomdp, threshold, algorithm.heuristic_belief_threshold, algorithm.heuristic_rho)
    elseif algorithm.solver_name == "Random_Policy"
        RandomPolicy(pomdp)
    elseif algorithm.solver_name == "NoTreatment_Policy"
        NoTreatmentPolicy(pomdp)
    elseif algorithm.convert_to_mdp
       solve(algorithm.solver, mdp)
    else
        solve(algorithm.solver, pomdp)
    end
    
    return (policy, pomdp, mdp)
end

# ----------------------------
# No Treatment Policy
# ----------------------------
struct NoTreatmentPolicy{P<:POMDP} <: Policy
    pomdp::P
end

# No Treatment action
function POMDPs.action(policy::NoTreatmentPolicy, b)
    return NoTreatment
end

function POMDPs.updater(policy::NoTreatmentPolicy)
    return DiscreteUpdater(policy.pomdp)
end

# ----------------------------
# Random Policy
# ----------------------------
struct RandomPolicy{P<:POMDP} <: Policy
    pomdp::P
end

# Random action
function POMDPs.action(policy::RandomPolicy, b)
    return rand((Treatment, NoTreatment))
end

function POMDPs.updater(policy::RandomPolicy)
    return DiscreteUpdater(policy.pomdp)
end

# ----------------------------
# Heuristic Policy
# ----------------------------
struct HeuristicPolicy{P<:POMDP} <: Policy
    pomdp::P
    threshold::Float64
    belief_threshold::Float64
    heuristic_rho::Float64
end

# Heuristic action
function POMDPs.action(policy::HeuristicPolicy, b)
    if heuristicChooseAction(policy, b, true)
        # Choose Treatment with probability heuristic_rho, otherwise NoTreatment
        return rand() < policy.heuristic_rho ? Treatment : NoTreatment
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
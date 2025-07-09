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
function generate_policy(algorithm, pomdp, mdp)

    # Heuristic Policy
    if algorithm.solver_name == "Heuristic_Policy"
        return HeuristicPolicy(pomdp, algorithm.heuristic_config)

    # Random policy
    elseif algorithm.solver_name == "Random_Policy"
        return RandomPolicy(pomdp)

    # No Treatment policy
    elseif algorithm.solver_name == "NoTreatment_Policy"
        return NoTreatmentPolicy(pomdp)

    # Value Iteration policy
    elseif algorithm.solver isa ValueIterationSolver
        return solve(algorithm.solver, mdp)

    # SARSOP and QMDP policies
    else
        return solve(algorithm.solver, pomdp)
    end
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
    heuristic_config::HeuristicConfig
end

# Heuristic action
function POMDPs.action(policy::HeuristicPolicy, b)
    if heuristicChooseAction(policy, b, true)
        # Choose Treatment with probability heuristic_rho, otherwise NoTreatment
        return rand() < policy.heuristic_config.rho ? Treatment : NoTreatment
    else
        return rand((Treatment, NoTreatment))
    end
end

# TODO: plot heuristic bvec with imageMap

# Function to decide whether we choose the action or randomize
function heuristicChooseAction(policy::HeuristicPolicy, b, use_cdf=true)

    # Convert belief vector to a probability distribution
    state_space = states(policy.pomdp)

    # Convert the threshold in log space if needed
    if policy.pomdp isa SeaLiceLogMDP
        threshold = log(policy.heuristic_config.raw_space_threshold)
    else
        threshold = policy.heuristic_config.raw_space_threshold
    end

    if use_cdf
        # Method 1: Calculate probability of being above threshold
        prob_above_threshold = sum(b[i] for (i, s) in enumerate(state_space) if s.SeaLiceLevel > threshold)
        return prob_above_threshold > policy.heuristic_config.belief_threshold
    else
        # Method 2: Use mode of belief vector
        mode_sealice_level_index = argmax(b)
        mode_sealice_level = state_space[mode_sealice_level_index]
        return mode_sealice_level.SeaLiceLevel > threshold
    end
end

function POMDPs.updater(policy::HeuristicPolicy)
    return DiscreteUpdater(policy.pomdp)
end
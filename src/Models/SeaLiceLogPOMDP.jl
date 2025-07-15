using DataFrames
import Distributions: Normal, Uniform, SkewNormal
using JLD2

using POMDPs
using QuickPOMDPs
using POMDPTools
using POMDPModels
using QMDP
using DiscreteValueIteration
using POMDPLinter
using Distributions
using Parameters
using Discretizers

include("../Utils/Utils.jl")

# -------------------------
# State, Observation, Action
# -------------------------
"State representing the sea lice level in log space."
struct SeaLiceLogState
	SeaLiceLevel::Float64
end

"Observation representing an observed sea lice level in log space."
struct SeaLiceLogObservation
	SeaLiceLevel::Float64
end

"Available actions: NoTreatment or Treatment."
@enum Action NoTreatment Treatment

# -------------------------
# SeaLiceLogMDP Definition
# -------------------------
"Sea lice MDP with growth dynamics and treatment effects in log space."
@with_kw struct SeaLiceLogMDP <: POMDP{SeaLiceLogState, Action, SeaLiceLogObservation}
	lambda::Float64 = 0.5
	costOfTreatment::Float64 = 10.0
	growthRate::Float64 = 1.2
	rho::Float64 = 0.7
    discount_factor::Float64 = 0.95
    skew::Bool = false
    min_lice_level::Float64 = 1e-3 # 1e-3 is the minimum sea lice level
    max_lice_level::Float64 = 30 # 10.0 # 10.0 is the maximum sea lice level
    min_log_initial_level::Float64 = log(1e-3)
    max_log_initial_level::Float64 = log(0.25)
    sea_lice_initial_mean::Float64 = log(0.125)
    sampling_sd::Float64 = abs(log(0.25))
    catdisc::CategoricalDiscretizer = CategoricalDiscretizer([NoTreatment, Treatment])

    # Log space
    min_log_lice_level::Float64 = log(min_lice_level)
    max_log_lice_level::Float64 = log(max_lice_level)
    log_discretization_step::Float64 = 0.005  # Reduced from 0.01 for finer granularity
    initial_range::Vector{Float64} = collect(range(min_log_initial_level, stop=max_log_initial_level, step=log_discretization_step))
    log_sea_lice_range::Vector{Float64} = collect(range(min_log_lice_level, stop=max_log_lice_level, step=log_discretization_step))
end

# -------------------------
# POMDPs.jl Interface
# -------------------------
POMDPs.states(mdp::SeaLiceLogMDP) = [SeaLiceLogState(i) for i in mdp.log_sea_lice_range]
POMDPs.actions(mdp::SeaLiceLogMDP) = [NoTreatment, Treatment]
POMDPs.observations(mdp::SeaLiceLogMDP) = [SeaLiceLogObservation(i) for i in mdp.log_sea_lice_range]
POMDPs.discount(mdp::SeaLiceLogMDP) = mdp.discount_factor
POMDPs.isterminal(mdp::SeaLiceLogMDP, s::SeaLiceLogState) = false
POMDPs.actionindex(mdp::SeaLiceLogMDP, a::Action) = encode(mdp.catdisc, a)

# -------------------------
# State and Observation Index
# -------------------------
function POMDPs.stateindex(mdp::SeaLiceLogMDP, s::SeaLiceLogState)
    closest_idx = argmin(abs.(mdp.log_sea_lice_range .- s.SeaLiceLevel))
    return closest_idx
end

function POMDPs.obsindex(mdp::SeaLiceLogMDP, o::SeaLiceLogObservation)
    closest_idx = argmin(abs.(mdp.log_sea_lice_range .- o.SeaLiceLevel))
    return closest_idx
end

# -------------------------
# Conversion Utilities
# -------------------------
# Required by LocalApproximationValueIteration
function POMDPs.convert_s(::Type{Vector{Float64}}, s::SeaLiceLogState, mdp::SeaLiceLogMDP)
    return [s.SeaLiceLevel]
end

function POMDPs.convert_s(::Type{SeaLiceLogState}, v::Vector{Float64}, mdp::SeaLiceLogMDP)
    return SeaLiceLogState(v[1])
end

# -------------------------
# Transition, Observation, Reward, Initial State
# -------------------------
function POMDPs.transition(mdp::SeaLiceLogMDP, s::SeaLiceLogState, a::Action)

    # Calculate the mean of the transition distribution
    μ = log(1 - (a == Treatment ? mdp.rho : 0.0)) + mdp.growthRate + s.SeaLiceLevel

    # Clamp the mean to the range of the sea lice range
    # TODO: consider the correctness of this
    μ = clamp(μ, mdp.min_log_lice_level, mdp.max_log_lice_level)

    # Get the distribution
    dist = Normal(μ, mdp.sampling_sd)

    # Get the states
    states = POMDPs.states(mdp)

    # Calculate the probs using the cdf
    probs = discretize_distribution(dist, states, mdp.skew)

    return SparseCat(states, probs)
end


function POMDPs.observation(mdp::SeaLiceLogMDP, a::Action, s::SeaLiceLogState)

    # Get the distribution
    dist = Normal(s.SeaLiceLevel, mdp.sampling_sd)

    # Get the observations
    observations = POMDPs.observations(mdp)

    # Calculate the probs using the cdf
    probs = discretize_distribution(dist, observations, mdp.skew)

    return SparseCat(observations, probs)

end

function POMDPs.reward(mdp::SeaLiceLogMDP, s::SeaLiceLogState, a::Action)
    # Convert log lice level back to actual lice level for penalty calculation
    lice_level = exp(s.SeaLiceLevel)
    if lice_level > 0.5
        lice_penalty = 1000.0
    else
        lice_penalty = mdp.lambda * lice_level
    end 
    treatment_penalty = a == Treatment ? (1 - mdp.lambda) * mdp.costOfTreatment : 0.0
    return -(lice_penalty + treatment_penalty)
end

function POMDPs.initialstate(mdp::SeaLiceLogMDP)
    states = [SeaLiceLogState(i) for i in mdp.initial_range]
    return SparseCat(states, fill(1/length(states), length(states)))
end
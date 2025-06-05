using DataFrames
import Distributions: Normal, Uniform
using JLD2

using POMDPs
using QuickPOMDPs
using POMDPTools
using POMDPModels
using QMDP
using NativeSARSOP
using DiscreteValueIteration
using POMDPLinter
using Distributions
using Parameters
using Discretizers

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
    min_lice_level::Float64 = 1e-3 # 1e-3 is the minimum sea lice level
    max_lice_level::Float64 = 10.0 # 10.0 is the maximum sea lice level
    discretization_step::Float64 = 0.1
    sea_lice_range::Vector{Float64} = collect(min_lice_level:discretization_step:(max_lice_level + discretization_step))
    initial_range::Vector{Float64} = collect(min_lice_level:discretization_step:(max_lice_level + discretization_step))
    sampling_sd::Float64 = 1.0
    catdisc::CategoricalDiscretizer = CategoricalDiscretizer([NoTreatment, Treatment])

    # Log space
    min_log_lice_level::Float64 = log(min_lice_level)
    max_log_lice_level::Float64 = log(max_lice_level)
    log_discretization_step::Float64 = 0.1
    log_sea_lice_range::Vector{Float64} = collect(range(min_log_lice_level, stop=max_log_lice_level, step=log_discretization_step))
    log_initial_range::Vector{Float64} = collect(range(min_log_lice_level, stop=max_log_lice_level, step=log_discretization_step))
    log_state_dict::Dict{Float64, Int} = Dict(level => idx for (idx, level) in enumerate(log_sea_lice_range))
    log_obs_dict::Dict{Float64, Int} = Dict(level => idx for (idx, level) in enumerate(log_sea_lice_range))
end

# -------------------------
# Discretized Normal Sampling Utility
# -------------------------
"Returns a 5-point approximation of a normal distribution."
function discretized_normal_points(mean::Float64, mdp::SeaLiceLogMDP)
    # Calculate the points
    points = mean .+ mdp.sampling_sd .* [-2, -1, 0, 1, 2]

    # Ensure points are within the range of the sea lice range
    points = clamp.(points, mdp.min_log_lice_level, mdp.max_log_lice_level)

    # Calculate and normalize the probabilities
    probs = pdf.(Normal(mean, mdp.sampling_sd), points)
    probs = normalize(probs, 1)

    return points, probs
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

function POMDPs.stateindex(mdp::SeaLiceLogMDP, s::SeaLiceLogState)
    ks = collect(keys(mdp.log_state_dict))
    closest_key = ks[argmin(abs.(ks .- s.SeaLiceLevel))]
    return mdp.log_state_dict[closest_key]
end

function POMDPs.obsindex(mdp::SeaLiceLogMDP, o::SeaLiceLogObservation)
    ks = collect(keys(mdp.log_obs_dict))
    closest_key = ks[argmin(abs.(ks .- o.SeaLiceLevel))]
    return mdp.log_obs_dict[closest_key]
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
    μ = log(1 - (a == Treatment ? mdp.rho : 0.0)) + mdp.growthRate + s.SeaLiceLevel
    pts, probs = discretized_normal_points(μ, mdp)
    states = [SeaLiceLogState(x) for x in pts]
    return SparseCat(states, probs)
end

function POMDPs.observation(mdp::SeaLiceLogMDP, a::Action, s::SeaLiceLogState)
    pts, probs = discretized_normal_points(s.SeaLiceLevel, mdp)
    obs = [SeaLiceLogObservation(x) for x in pts]
    return SparseCat(obs, probs)
end

function POMDPs.reward(mdp::SeaLiceLogMDP, s::SeaLiceLogState, a::Action)
    # Convert log lice level back to actual lice level for penalty calculation
    lice_level = exp(s.SeaLiceLevel)
    lice_penalty = mdp.lambda * lice_level
    treatment_penalty = a == Treatment ? (1 - mdp.lambda) * mdp.costOfTreatment : 0.0
    return -(lice_penalty + treatment_penalty)
end

function POMDPs.initialstate(mdp::SeaLiceLogMDP)
    states = [SeaLiceLogState(i) for i in mdp.log_initial_range]
    return SparseCat(states, fill(1/length(states), length(states)))
end
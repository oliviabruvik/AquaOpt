using DataFrames
import Distributions: Normal, Uniform
using JLD2

using POMDPs
using QuickPOMDPs
using POMDPTools
using POMDPModels
using QMDP
# using NativeSARSOP
using DiscreteValueIteration
using POMDPLinter
using Distributions
using Parameters
using Random

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
# SeaLiceLogSimMDP Definition
# -------------------------
"Sea lice MDP with growth dynamics and treatment effects in log space."
@with_kw struct SeaLiceLogSimMDP <: POMDP{SeaLiceLogState, Action, SeaLiceLogObservation}
	lambda::Float64 = 0.5
	costOfTreatment::Float64 = 10.0
	growthRate::Float64 = 1.2
	rho::Float64 = 0.7
    skew::Bool = false
    discount_factor::Float64 = 0.95
    min_lice_level::Float64 = 1e-3
    max_lice_level::Float64 = 30 # 10.0
    log_lice_bounds::Tuple{Float64, Float64} = (log(min_lice_level), log(max_lice_level))
    initial_bounds::Tuple{Float64, Float64} = (log(1e-3), log(0.25))
    log_lice_initial_mean::Float64 = log(0.125)
    sampling_sd::Float64 = abs(log(0.25))
    rng::AbstractRNG = Random.GLOBAL_RNG
    normal_dist::Distribution = Normal(0, sampling_sd)
    skew_normal_dist::Distribution = SkewNormal(0, sampling_sd, 2.0)
end

# -------------------------
# POMDPs.jl Interface
# -------------------------
POMDPs.actions(mdp::SeaLiceLogSimMDP) = [NoTreatment, Treatment]
POMDPs.discount(mdp::SeaLiceLogSimMDP) = mdp.discount_factor
POMDPs.isterminal(mdp::SeaLiceLogSimMDP, s::SeaLiceLogState) = false

function POMDPs.transition(pomdp::SeaLiceLogSimMDP, s::SeaLiceLogState, a::Action)
    ImplicitDistribution(pomdp, s, a) do pomdp, s, a, rng
        μ = log(1 - (a == Treatment ? pomdp.rho : 0.0)) + pomdp.growthRate + s.SeaLiceLevel
        next_state = μ + rand(rng, pomdp.normal_dist)
        return SeaLiceLogState(clamp(next_state, pomdp.log_lice_bounds...))
    end
end

function POMDPs.observation(pomdp::SeaLiceLogSimMDP, a::Action, s::SeaLiceLogState)
    ImplicitDistribution(pomdp, s, a) do pomdp, s, a, rng
        dist = pomdp.skew ? pomdp.skew_normal_dist : pomdp.normal_dist
        next_state = s.SeaLiceLevel + rand(rng, dist)
        return SeaLiceLogObservation(clamp(next_state, pomdp.log_lice_bounds...))
    end
end

function POMDPs.reward(pomdp::SeaLiceLogSimMDP, s::SeaLiceLogState, a::Action)
    # Convert log lice level back to actual lice level for penalty calculation
    lice_level = exp(s.SeaLiceLevel)
    if lice_level > 0.5
        lice_penalty = 1000.0
    else
        lice_penalty = pomdp.lambda * lice_level
    end
    treatment_penalty = a == Treatment ? (1 - pomdp.lambda) * pomdp.costOfTreatment : 0.0
    return - (lice_penalty + treatment_penalty)
end

function POMDPs.initialstate(pomdp::SeaLiceLogSimMDP)
    ImplicitDistribution(pomdp) do pomdp, rng
        next_state = pomdp.log_lice_initial_mean + rand(rng, pomdp.normal_dist)
        return SeaLiceLogState(clamp(next_state, pomdp.initial_bounds...))
    end
end
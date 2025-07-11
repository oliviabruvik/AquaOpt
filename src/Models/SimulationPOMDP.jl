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
using Random

# -------------------------
# State, Observation, Action
# -------------------------
"State representing the sea lice level."
struct SeaLiceState
	SeaLiceLevel::Float64
end

"Observation representing an observed sea lice level."
struct SeaLiceObservation
	SeaLiceLevel::Float64
end

"Available actions: NoTreatment or Treatment."
@enum Action NoTreatment Treatment

# -------------------------
# SeaLiceMDP Definition
# -------------------------
"Sea lice MDP with growth dynamics and treatment effects."
@with_kw struct SeaLiceSimMDP <: POMDP{SeaLiceState, Action, SeaLiceObservation}
	lambda::Float64 = 0.5
	costOfTreatment::Float64 = 10.0
	growthRate::Float64 = 1.2
	rho::Float64 = 0.7
    discount_factor::Float64 = 0.95
    sea_lice_bounds::Tuple{Float64, Float64} = (0.0, 10.0)
    initial_bounds::Tuple{Float64, Float64} = (0.0, 1.0)
    sea_lice_initial_mean::Float64 = 1.0
    sampling_sd::Float64 = 0.5
    rng::AbstractRNG = Random.GLOBAL_RNG
    normal_dist::Distribution = Normal(0, sampling_sd)
end

# -------------------------
# POMDPs.jl Interface
# -------------------------
POMDPs.actions(mdp::SeaLiceSimMDP) = [NoTreatment, Treatment]
POMDPs.discount(mdp::SeaLiceSimMDP) = mdp.discount_factor
POMDPs.isterminal(mdp::SeaLiceSimMDP, s::SeaLiceState) = false

function POMDPs.transition(pomdp::SeaLiceSimMDP, s::SeaLiceState, a::Action)
    ImplicitDistribution(pomdp, s, a) do pomdp, s, a, rng
        μ = (1 - (a == Treatment ? pomdp.rho : 0.0)) * exp(pomdp.growthRate) * s.SeaLiceLevel
        next_state = μ + rand(rng, pomdp.normal_dist)
        return SeaLiceState(clamp(next_state, pomdp.sea_lice_bounds...))
    end
end

function POMDPs.observation(pomdp::SeaLiceSimMDP, a::Action, s::SeaLiceState)
    ImplicitDistribution(pomdp, s, a) do pomdp, s, a, rng
        next_state = s.SeaLiceLevel + rand(rng, pomdp.normal_dist)
        return SeaLiceObservation(clamp(next_state, pomdp.sea_lice_bounds...))
    end
end

function POMDPs.reward(pomdp::SeaLiceSimMDP, s::SeaLiceState, a::Action)
    lice_penalty = pomdp.lambda * s.SeaLiceLevel
    treatment_penalty = a == Treatment ? (1 - pomdp.lambda) * pomdp.costOfTreatment : 0.0
    return - (lice_penalty + treatment_penalty)
end

function POMDPs.initialstate(pomdp::SeaLiceSimMDP)
    ImplicitDistribution(pomdp) do pomdp, rng
        next_state = pomdp.sea_lice_initial_mean + rand(rng, pomdp.normal_dist)
        return SeaLiceState(clamp(next_state, pomdp.initial_bounds...))
    end
end
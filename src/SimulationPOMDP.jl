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

# -------------------------
# Constants
# -------------------------
const SEA_LICE_BOUNDS = (0.0, 10.0)
const INITIAL_BOUNDS = (0.0, 1.0)
const SEA_LICE_INITIAL_MEAN = 1.0
const STD_DEV = 1.0

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
struct SeaLiceSimMDP <: POMDP{SeaLiceState, Action, SeaLiceObservation}
	lambda::Float64
	costOfTreatment::Float64
	growthRate::Float64
	rho::Float64
    discount_factor::Float64
end

"Constructor with default parameters."
function SeaLiceSimMDP(; lambda=0.5, costOfTreatment=10.0, growthRate=1.2, rho=0.7, discount_factor=0.95)
    SeaLiceSimMDP(lambda, costOfTreatment, growthRate, rho, discount_factor)
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
        point = rand(rng, Normal(μ, STD_DEV))
        return SeaLiceState(clamp(point, SEA_LICE_BOUNDS...))
    end
end

function POMDPs.observation(pomdp::SeaLiceSimMDP, a::Action, s::SeaLiceState)
    ImplicitDistribution(pomdp, s, a) do pomdp, s, a, rng
        point = rand(rng, Normal(s.SeaLiceLevel, STD_DEV))
        return SeaLiceObservation(clamp(point, SEA_LICE_BOUNDS...))
    end
end

function POMDPs.reward(pomdp::SeaLiceSimMDP, s::SeaLiceState, a::Action)
    lice_penalty = pomdp.lambda * s.SeaLiceLevel
    treatment_penalty = a == Treatment ? (1 - pomdp.lambda) * pomdp.costOfTreatment : 0.0
    return - (lice_penalty + treatment_penalty)
end

function POMDPs.initialstate(pomdp::SeaLiceSimMDP)
    ImplicitDistribution(pomdp) do pomdp, rng
        # TODO: Change from hardcoded mean
        point = rand(rng, Normal(SEA_LICE_INITIAL_MEAN, STD_DEV))
        return SeaLiceState(clamp(point, INITIAL_BOUNDS...))
    end
end
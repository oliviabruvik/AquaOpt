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
const SEA_LICE_RANGE = 0.0:0.1:10.0
const INITIAL_RANGE = 0.0:0.1:1.0
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
struct SeaLiceMDP <: POMDP{SeaLiceState, Action, SeaLiceObservation}
	lambda::Float64
	costOfTreatment::Float64
	growthRate::Float64
	rho::Float64
    discount_factor::Float64
end

"Constructor with default parameters."
function SeaLiceMDP(; lambda=0.5, costOfTreatment=10.0, growthRate=1.2, rho=0.7, discount_factor=0.95)
    SeaLiceMDP(lambda, costOfTreatment, growthRate, rho, discount_factor)
end

# -------------------------
# Discretized Normal Sampling Utility
# -------------------------
"Returns a 5-point approximation of a normal distribution."
function discretized_normal_points(mean::Float64; std_dev=1.0)
    points = mean .+ std_dev .* [-2, -1, 0, 1, 2]
    probs = exp.(-((points .- mean).^2) ./ (2 * std_dev^2))
    probs ./= sum(probs)
    return points, probs
end

# -------------------------
# POMDPs.jl Interface
# -------------------------
POMDPs.states(mdp::SeaLiceMDP) = [SeaLiceState(round(i, digits=1)) for i in SEA_LICE_RANGE]
POMDPs.actions(mdp::SeaLiceMDP) = [NoTreatment, Treatment]
POMDPs.observations(mdp::SeaLiceMDP) = [SeaLiceObservation(round(i, digits=1)) for i in SEA_LICE_RANGE]
POMDPs.discount(mdp::SeaLiceMDP) = mdp.discount_factor
POMDPs.isterminal(mdp::SeaLiceMDP, s::SeaLiceState) = false
POMDPs.stateindex(mdp::SeaLiceMDP, s::SeaLiceState) = clamp(round(Int, s.SeaLiceLevel * 10) + 1, 1, 101)
POMDPs.actionindex(mdp::SeaLiceMDP, a::Action) = clamp(Int(a) + 1, 1, 2)
POMDPs.obsindex(mdp::SeaLiceMDP, o::SeaLiceObservation) = clamp(round(Int, o.SeaLiceLevel * 10) + 1, 1, 101)

# Required by LocalApproximationValueIteration
function POMDPs.convert_s(::Type{Vector{Float64}}, s::SeaLiceState, mdp::SeaLiceMDP)
    return [s.SeaLiceLevel]
end

function POMDPs.convert_s(::Type{SeaLiceState}, v::Vector{Float64}, mdp::SeaLiceMDP)
    return SeaLiceState(round(clamp(v[1], 0.0, 10.0), digits=1))
end

function POMDPs.transition(pomdp::SeaLiceMDP, s::SeaLiceState, a::Action)
    ImplicitDistribution(pomdp, s, a) do pomdp, s, a, rng
        μ = (1 - (a == Treatment ? pomdp.rho : 0.0)) * exp(pomdp.growthRate) * s.SeaLiceLevel
        point = rand(rng, Normal(μ, STD_DEV))
        return SeaLiceState(round(clamp(point, 0.0, 10.0), digits=1))
    end
end

function POMDPs.observation(pomdp::SeaLiceMDP, a::Action, s::SeaLiceState)
    ImplicitDistribution(pomdp, s, a) do pomdp, s, a, rng
        μ = (1 - (a == Treatment ? pomdp.rho : 0.0)) * exp(pomdp.growthRate) * s.SeaLiceLevel
        point = rand(rng, MvNormal(μ, STD_DEV))
        return SeaLiceObservation(round(clamp(point, 0.0, 10.0), digits=1))
    end
end

function POMDPs.reward(pomdp::SeaLiceMDP, s::SeaLiceState, a::Action)
    lice_penalty = pomdp.lambda * s.SeaLiceLevel
    treatment_penalty = a == Treatment ? (1 - pomdp.lambda) * pomdp.costOfTreatment : 0.0
    return - (lice_penalty + treatment_penalty)
end

function POMDPs.initialstate(pomdp::SeaLiceMDP)
    ImplicitDistribution(pomdp) do pomdp, rng
        point = rand(rng, MvNormal(0.0, STD_DEV))
        return SeaLiceState(round(clamp(point, 0.0, 10.0), digits=1))
    end
end

# # Support for ImplicitDistribution
# function POMDPs.support(d::ImplicitDistribution{<:Any, Tuple{SeaLiceMDP, SeaLiceState, Action}})
#     # Return all possible states for transitions
#     return [SeaLiceState(round(i, digits=1)) for i in SEA_LICE_RANGE]
# end

# function POMDPs.support(d::ImplicitDistribution{<:Any, Tuple{SeaLiceMDP}})
#     # Return all possible initial states
#     return [SeaLiceState(round(i, digits=1)) for i in SEA_LICE_RANGE]
# end

# function POMDPs.support(d::ImplicitDistribution{<:Any, Tuple{SeaLiceMDP, Action, SeaLiceState}})
#     # Return all possible observations
#     return [SeaLiceObservation(round(i, digits=1)) for i in SEA_LICE_RANGE]
# end

# # Add pdf for ImplicitDistribution
# function POMDPs.pdf(d::ImplicitDistribution{<:Any, Tuple{SeaLiceMDP, SeaLiceState, Action}}, s::SeaLiceState)
#     mdp, prev_s, a = d.args
#     μ = (1 - (a == Treatment ? mdp.rho : 0.0)) * exp(mdp.growthRate) * prev_s.SeaLiceLevel
#     return pdf(Normal(μ, STD_DEV), s.SeaLiceLevel)
# end

# function POMDPs.pdf(d::ImplicitDistribution{<:Any, Tuple{SeaLiceMDP}}, s::SeaLiceState)
#     return pdf(Normal(0.0, STD_DEV), s.SeaLiceLevel)
# end

# function POMDPs.pdf(d::ImplicitDistribution{<:Any, Tuple{SeaLiceMDP, Action, SeaLiceState}}, o::SeaLiceObservation)
#     mdp, a, s = d.args
#     μ = s.SeaLiceLevel
#     return pdf(Normal(μ, STD_DEV), o.SeaLiceLevel)
# end
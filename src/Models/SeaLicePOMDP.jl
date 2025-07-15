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
using Discretizers

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
@with_kw struct SeaLiceMDP <: POMDP{SeaLiceState, Action, SeaLiceObservation}
	lambda::Float64 = 0.5
	costOfTreatment::Float64 = 10.0
	growthRate::Float64 = 1.2
	rho::Float64 = 0.7
    discount_factor::Float64 = 0.95
    skew::Bool = false
    min_lice_level::Float64 = 0.0
    max_lice_level::Float64 = 30.0
    min_initial_level::Float64 = 0.0
    max_initial_level::Float64 = 0.25
    discretization_step::Float64 = 0.1
    num_points::Int = Int((max_lice_level - min_lice_level) / discretization_step) + 1
    sea_lice_range::Vector{Float64} = collect(min_lice_level:discretization_step:max_lice_level)
    initial_range::Vector{Float64} = collect(min_initial_level:discretization_step:max_initial_level)
    sampling_sd::Float64 = 2 # TODO: debug sampling sd and its significance
    lindisc::LinearDiscretizer = LinearDiscretizer(collect(min_lice_level:discretization_step:(max_lice_level+discretization_step)))
    catdisc::CategoricalDiscretizer = CategoricalDiscretizer([NoTreatment, Treatment])
end

# -------------------------
# Discretized Normal Sampling Utility
# -------------------------
"Returns a transition distribution that ensures all states are reachable."
function discretized_normal_points(mean::Float64, mdp::SeaLiceMDP, skew::Bool=false)

    if skew
        skew_factor = 2.0
        dist = SkewNormal(mean, mdp.sampling_sd, skew_factor)
    else
        dist = Normal(mean, mdp.sampling_sd)
    end

    # Calculate the points
    points = mean .+ mdp.sampling_sd .* [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3]
    if skew
        points = points .* (1 + 0.3 * skew_factor)
    end
    
    # Ensure points are within the range of the sea lice range
    points = clamp.(points, mdp.min_lice_level, mdp.max_lice_level)

    # Calculate and normalize the probabilities
    probs = pdf.(dist, points)
    probs = normalize(probs, 1)
    
    # Ensure we have at least one transition
    if length(points) == 0 || sum(probs) == 0
        # Fallback to mean state
        points = [mean]
        probs = [1.0]
    end

    return points, probs
end

# -------------------------
# POMDPs.jl Interface
# -------------------------
POMDPs.states(mdp::SeaLiceMDP) = [SeaLiceState(round(i, digits=1)) for i in mdp.sea_lice_range]
POMDPs.actions(mdp::SeaLiceMDP) = [NoTreatment, Treatment]
POMDPs.observations(mdp::SeaLiceMDP) = [SeaLiceObservation(round(i, digits=1)) for i in mdp.sea_lice_range]
POMDPs.discount(mdp::SeaLiceMDP) = mdp.discount_factor
POMDPs.isterminal(mdp::SeaLiceMDP, s::SeaLiceState) = false
POMDPs.stateindex(mdp::SeaLiceMDP, s::SeaLiceState) = encode(mdp.lindisc, s.SeaLiceLevel)
POMDPs.actionindex(mdp::SeaLiceMDP, a::Action) = encode(mdp.catdisc, a)
POMDPs.obsindex(mdp::SeaLiceMDP, o::SeaLiceObservation) = encode(mdp.lindisc, o.SeaLiceLevel)

# -------------------------
# Conversion Utilities
# -------------------------
# Required by LocalApproximationValueIteration
function POMDPs.convert_s(::Type{Vector{Float64}}, s::SeaLiceState, mdp::SeaLiceMDP)
    return [s.SeaLiceLevel]
end

function POMDPs.convert_s(::Type{SeaLiceState}, v::Vector{Float64}, mdp::SeaLiceMDP)
    return SeaLiceState(v[1])
end


# -------------------------
# Transition, Observation, Reward, Initial State
# -------------------------
function POMDPs.transition(mdp::SeaLiceMDP, s::SeaLiceState, a::Action)

    # Calculate the mean of the transition distribution
    μ = (1 - (a == Treatment ? mdp.rho : 0.0)) * exp(mdp.growthRate) * s.SeaLiceLevel

    # Clamp the mean to the range of the sea lice range
    # TODO: consider the correctness of this
    μ = clamp(μ, mdp.min_lice_level, mdp.max_lice_level)

    # Get the distribution
    dist = Normal(μ, mdp.sampling_sd)

    # Get the states
    states = POMDPs.states(mdp)

    # Calculate the probs using the cdf
    probs = zeros(length(states))
    past_cdf = 0.0
    for (i, s) in enumerate(states)
        curr_cdf = cdf(dist, s.SeaLiceLevel)
        probs[i] = curr_cdf - past_cdf
        past_cdf = curr_cdf
    end

    # Normalize the probs
    probs = normalize(probs, 1)

    # Check that the probs sum to 1.0
    try
        @assert sum(probs) ≈ 1.0
    catch
        println("probs: $probs")
        println("sum(probs): $(sum(probs))")
        error("Probs do not sum to 1.0")
    end
    return SparseCat(states, probs)
end

function POMDPs.observation(mdp::SeaLiceMDP, a::Action, s::SeaLiceState)

    # Get the distribution
    dist = Normal(s.SeaLiceLevel, mdp.sampling_sd)

    # Get the observations
    observations = POMDPs.observations(mdp)

    # Calculate the probs using the cdf
    probs = zeros(length(observations))
    past_cdf = 0.0
    for (i, o) in enumerate(observations)
        curr_cdf = cdf(dist, o.SeaLiceLevel)
        probs[i] = curr_cdf - past_cdf
        past_cdf = curr_cdf
    end

    if sum(probs) < 0.01
        println("probs: $probs")
        println("sum(probs): $(sum(probs))")
        println("observations: $observations \n")
        error("Probs do not sum to 1.0")
    end

    # Normalize the probs
    probs = normalize(probs, 1)

    # Check that the probs sum to 1.0
    try
        @assert sum(probs) ≈ 1.0
    catch
        println("probs: $probs")
        println("sum(probs): $(sum(probs))")
        error("Probs do not sum to 1.0")
    end

    return SparseCat(observations, probs)

end

function POMDPs.reward(mdp::SeaLiceMDP, s::SeaLiceState, a::Action)
    # if s.SeaLiceLevel > 0.5
    #    lice_penalty = 1000.0
    # else
    #    lice_penalty = mdp.lambda * s.SeaLiceLevel
    # end
    lice_penalty = mdp.lambda * s.SeaLiceLevel
    treatment_penalty = a == Treatment ? (1 - mdp.lambda) * mdp.costOfTreatment : 0.0
    return - (lice_penalty + treatment_penalty)
end

function POMDPs.initialstate(mdp::SeaLiceMDP)
    states = [SeaLiceState(round(i, digits=1)) for i in mdp.initial_range]
    return SparseCat(states, fill(1/length(states), length(states)))
end
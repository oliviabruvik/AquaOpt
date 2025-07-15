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
# Discretized Normal Sampling Utility
# -------------------------
"Returns a transition distribution that ensures all states are reachable."
function discretized_normal_points(mean::Float64, mdp::SeaLiceLogMDP, skew::Bool=false)

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
    points = clamp.(points, mdp.min_log_lice_level, mdp.max_log_lice_level)

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
    μ = log(1 - (a == Treatment ? mdp.rho : 0.0)) + mdp.growthRate + s.SeaLiceLevel
    pts, probs = discretized_normal_points(μ, mdp, false)
    
    # Map points to nearest valid states in the discretized state space
    valid_states = []
    valid_probs = []
    
    for (pt, prob) in zip(pts, probs)
        # Find the closest state in the discretized state space
        closest_state = mdp.log_sea_lice_range[argmin(abs.(mdp.log_sea_lice_range .- pt))]
        state = SeaLiceLogState(closest_state)
        
        # Check if this state is already in our list
        state_idx = findfirst(s -> s.SeaLiceLevel == closest_state, valid_states)
        if state_idx === nothing
            push!(valid_states, state)
            push!(valid_probs, prob)
        else
            valid_probs[state_idx] += prob
        end
    end
    
    # Ensure we have at least one valid transition
    if isempty(valid_states)
        println("No valid states")
        # Fallback to current state
        valid_states = [s]
        valid_probs = [1.0]
    end
    
    # Normalize probabilities
    valid_probs = normalize(valid_probs, 1)

    # Check that the probabilities sum to 1.0
    try
        @assert sum(valid_probs) ≈ 1.0
    catch
        println(valid_probs)
        println(sum(valid_probs))
        error("Probs do not sum to 1.0")
    end
    
    return SparseCat(valid_states, valid_probs)
end

function POMDPs.observation(mdp::SeaLiceLogMDP, a::Action, s::SeaLiceLogState)
    pts, probs = discretized_normal_points(s.SeaLiceLevel, mdp, mdp.skew)
    
    # Map points to nearest valid observations in the discretized observation space
    valid_obs = []
    valid_probs = []
    
    for (pt, prob) in zip(pts, probs)
        # Find the closest observation in the discretized observation space
        closest_obs = mdp.log_sea_lice_range[argmin(abs.(mdp.log_sea_lice_range .- pt))]
        obs = SeaLiceLogObservation(closest_obs)
        
        # Check if this observation is already in our list
        obs_idx = findfirst(o -> o.SeaLiceLevel == closest_obs, valid_obs)
        if obs_idx === nothing
            push!(valid_obs, obs)
            push!(valid_probs, prob)
        else
            valid_probs[obs_idx] += prob
        end
    end
    
    # Ensure we have at least one valid observation
    if isempty(valid_obs)
        # Fallback to current state observation
        valid_obs = [SeaLiceLogObservation(s.SeaLiceLevel)]
        valid_probs = [1.0]
    end
    
    # Normalize probabilities
    valid_probs = normalize(valid_probs, 1)
    
    return SparseCat(valid_obs, valid_probs)
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
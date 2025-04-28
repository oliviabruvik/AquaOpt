module OptimizeMDP

using DataFrames
using DiscreteValueIteration
import Distributions: Normal, Uniform
using JLD2
using POMDPLinter
using POMDPModels
using POMDPs
using POMDPTools
using QMDP
using QuickPOMDPs
using SARSOP

# # Define the state
struct SeaLiceState
	SeaLiceLevel::Float64
end

# Define the observation
struct SeaLiceObservation
	SeaLiceLevel::Float64
end

# Define the action
@enum Action NoTreatment Treatment

# Define the MDP
struct SeaLiceMDP <: POMDP{SeaLiceState, Action, SeaLiceObservation}
	lambda::Float64
	costOfTreatment::Float64
	growthRate::Float64
	rho::Float64
    discount_factor::Float64
end

# Constructor
function SeaLiceMDP(;
    lambda::Float64 = 0.5,
    costOfTreatment::Float64 = 10.0,
    growthRate::Float64 = 1.2,
    rho::Float64 = 0.7,
    discount_factor::Float64 = 0.95
)
    return SeaLiceMDP(
        lambda,
        costOfTreatment,
        growthRate,
        rho,
        discount_factor
    )
end

# POMDP interface functions
POMDPs.actions(mdp::SeaLiceMDP) = [NoTreatment, Treatment]
POMDPs.states(mdp::SeaLiceMDP) = [SeaLiceState(round(i, digits=1)) for i in 0:0.1:10]

# stateindex and actionindex functions
function POMDPs.stateindex(mdp::SeaLiceMDP, s::SeaLiceState)
    return clamp(round(Int, s.SeaLiceLevel * 10) + 1, 1, 101)
end

# Convert action to index (1 for NoTreatment, 2 for Treatment)
function POMDPs.actionindex(mdp::SeaLiceMDP, a::Action)
    return Int(a) + 1
end

function POMDPs.transition(mdp::SeaLiceMDP, s::SeaLiceState, a::Action)
    growth_rate = mdp.growthRate
    rho = a == Treatment ? mdp.rho : 0.0
    next_sea_lice_mean = (1-rho) * exp(growth_rate) * s.SeaLiceLevel
    
    std_dev = 1.0
    points = [
        next_sea_lice_mean - 2*std_dev,
        next_sea_lice_mean - std_dev,
        next_sea_lice_mean,
        next_sea_lice_mean + std_dev,
        next_sea_lice_mean + 2*std_dev
    ]
    
    # Calculate probabilities using normal distribution
    probs = [exp(-(x - next_sea_lice_mean)^2 / (2*std_dev^2)) for x in points]
    
    # Normalize
    probs = probs / sum(probs)
    
    # Create states and clamp/round values
    states = [SeaLiceState(round(clamp(x, 0.0, 10.0), digits=1)) for x in points]
    
    return SparseCat(states, probs)
end

function POMDPs.reward(mdp::SeaLiceMDP, s::SeaLiceState, a::Action)
    if a == Treatment
        return - (mdp.lambda * s.SeaLiceLevel + (1 - mdp.lambda) * mdp.costOfTreatment)
    else
        return - (mdp.lambda * s.SeaLiceLevel)
    end
end

POMDPs.discount(mdp::SeaLiceMDP) = mdp.discount_factor

function POMDPs.initialstate(mdp::SeaLiceMDP)
    # Create a uniform distribution over initial states from 0.0 to 1.0
    states = [SeaLiceState(round(i, digits=1)) for i in 0:0.1:1.0]
    probs = ones(length(states)) / length(states)
    return SparseCat(states, probs)
end

function POMDPs.observation(mdp::SeaLiceMDP, a::Action, s::SeaLiceState)
    std_dev = 1.0
    points = [
        s.SeaLiceLevel - 2*std_dev,
        s.SeaLiceLevel - std_dev,
        s.SeaLiceLevel,
        s.SeaLiceLevel + std_dev,
        s.SeaLiceLevel + 2*std_dev
    ]
    
    # Calculate probabilities using normal distribution
    probs = [exp(-(x - s.SeaLiceLevel)^2 / (2*std_dev^2)) for x in points]
    
    # Normalize
    probs = probs / sum(probs)
    
    # Create observations and clamp/round values
    observations = [SeaLiceObservation(round(clamp(x, 0.0, 10.0), digits=1)) for x in points]
    
    return SparseCat(observations, probs)
end

POMDPs.isterminal(mdp::SeaLiceMDP, s::SeaLiceState) = false

function find_policies_across_lambdas(lambda_values; solver)
    policies = Dict{Float64, Tuple{Policy, SeaLiceMDP, MDP}}()
    for λ in lambda_values
        pomdp = SeaLiceMDP(lambda=λ)
        mdp = UnderlyingMDP(pomdp)
        policy = solve(solver, mdp)
        policies[λ] = (policy, pomdp, mdp)

        # save policy
        save("results/policies/sea_lice_mdp_policy_$(λ).jld2", "policy", policy)
    end
    return policies
end

function evaluate_policy(policies_dict; episodes=100, steps_per_episode=50)
    results = DataFrame(lambda=Float64[], avg_treatment_cost=Float64[], avg_sealice=Float64[])
    
    for (λ, (policy, pomdp, mdp)) in pairs(policies_dict)
        
        # optimize mdp
        avg_cost, avg_sealice = run_simulation(policy, mdp, pomdp, episodes, steps_per_episode)

        # add results to dataframe
        push!(results, (λ, avg_cost, avg_sealice))

    end

    rename!(results, [:lambda, :avg_treatment_cost, :avg_sealice])
    return results
end

function create_heuristic_policy_dict(lambda_values)
    policies = Dict{Float64, Tuple{Policy, SeaLiceMDP, MDP}}()

    for λ in lambda_values
        pomdp = SeaLiceMDP(lambda=λ)
        mdp = UnderlyingMDP(pomdp)
        policy = HeuristicPolicy(mdp)
        policies[λ] = (policy, pomdp, mdp)
    end
    return policies
end

function run_simulation(policy, mdp, pomdp, episodes=100, steps_per_episode=50)
    
    total_cost = 0.0
    total_sealice = 0.0
    total_steps = episodes * steps_per_episode

    for _ in 1:episodes
        s = rand(initialstate(mdp))
        for _ in 1:steps_per_episode
            a = action(policy, s)
            total_cost += (a == Treatment ? pomdp.costOfTreatment : 0.0)
            total_sealice += s.SeaLiceLevel
            s = rand(transition(pomdp, s, a))
        end
    end

    avg_cost = total_cost / total_steps
    avg_sealice = total_sealice / total_steps

    return avg_cost, avg_sealice
end

# Add heuristic policy
struct HeuristicPolicy{P<:MDP} <: Policy
    mdp::P
end

function POMDPs.action(policy::HeuristicPolicy, s::SeaLiceState)
    if s.SeaLiceLevel > 0.5
        return Treatment
    else
        return NoTreatment
    end
end


end
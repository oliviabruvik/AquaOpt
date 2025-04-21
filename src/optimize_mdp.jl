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

# Define the state
struct SeaLiceState
	SeaLiceLevel::Float64
end

# Define the action
@enum Action NoTreatment Treatment

# Define the MDP
struct SeaLiceMDP <: MDP{SeaLiceState, Action}
	lambda::Float64
	costOfTreatment::Float64
	growthRate::Float64
	rho::Float64
    discount_factor::Float64
end

# Constructor
function SeaLiceMDP(;
    lambda::Float64 = 0.5,
    costOfTreatment::Float64 = 1.0,
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

# MDP interface functions
POMDPs.actions(mdp::SeaLiceMDP) = [NoTreatment, Treatment]
POMDPs.states(mdp::SeaLiceMDP) = [SeaLiceState(round(i, digits=1)) for i in 0:0.1:10]

# stateindex and actionindex functions
function POMDPs.stateindex(mdp::SeaLiceMDP, s::SeaLiceState)
    return clamp(round(Int, s.SeaLiceLevel * 10) + 1, 1, 101)
end

function POMDPs.actionindex(mdp::SeaLiceMDP, a::Action)
    # Convert action to index (1 for NoTreatment, 2 for Treatment)
    return Int(a) + 1
end

function POMDPs.transition(mdp::SeaLiceMDP, s::SeaLiceState, a::Action)
    growth_rate = mdp.growthRate
    rho = a == Treatment ? mdp.rho : 0.0
    next_sea_lice_mean = (1-rho) * exp(growth_rate) * s.SeaLiceLevel
    
    # Create a discrete distribution around the mean
    # Sample 5 points around the mean with decreasing probabilities
    std_dev = 0.2
    points = [
        next_sea_lice_mean - 2*std_dev,
        next_sea_lice_mean - std_dev,
        next_sea_lice_mean,
        next_sea_lice_mean + std_dev,
        next_sea_lice_mean + 2*std_dev
    ]
    
    # Calculate probabilities using normal distribution
    probs = [exp(-(x - next_sea_lice_mean)^2 / (2*std_dev^2)) for x in points]
    probs = probs / sum(probs)  # Normalize
    
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

POMDPs.isterminal(mdp::SeaLiceMDP, s::SeaLiceState) = false

function mdp_optimize(df::DataFrame)
    # Create the MDP
    mdp = SeaLiceMDP()

    s = SeaLiceState(1.0)
    a = Treatment

    # Generate some samples to test
    for _ in 1:5
        next_state = rand(transition(mdp, s, a))
        @show next_state.SeaLiceLevel
    end
    
    # Create the solver
    solver = ValueIterationSolver(max_iterations=30)
    # @show_requirements POMDPs.solve(solver, mdp)
    
    # Solve for the policy
    policy = solve(solver, mdp)
    
    # Save the policy
    mkpath("results/policies")
    save("results/policies/sea_lice_mdp_policy.jld2", "policy", policy)

    return policy
end

end
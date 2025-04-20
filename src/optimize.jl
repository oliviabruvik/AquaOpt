module Optimize

using DataFrames
using DiscreteValueIteration
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
POMDPs.states(mdp::SeaLiceMDP) = [SeaLiceState(round(i, digits=1)) for i in 0:0.1:10] # Reduced max state to 10

# stateindex and actionindex functions
function POMDPs.stateindex(mdp::SeaLiceMDP, s::SeaLiceState)
    # Convert sea lice level to index (0-10)
    if s.SeaLiceLevel < 0
        return 1
    elseif s.SeaLiceLevel > 10
        return 101 # 101 indices for 0:0.1:10
    else
        return Int(round(s.SeaLiceLevel * 10)) + 1
    end
end

function POMDPs.actionindex(mdp::SeaLiceMDP, a::Action)
    # Convert action to index (1 for NoTreatment, 2 for Treatment)
    return Int(a) + 1
end

function POMDPs.transition(mdp::SeaLiceMDP, s::SeaLiceState, a::Action)
    growth_rate = mdp.growthRate
    rho = a == Treatment ? mdp.rho : 0.0
    next_sea_lice_level = round((1-rho) * exp(growth_rate) * s.SeaLiceLevel, digits=1)
    next_state = SeaLiceState(next_sea_lice_level)
    return Deterministic(next_state)
end

function POMDPs.reward(mdp::SeaLiceMDP, s::SeaLiceState, a::Action)
    if a == Treatment
        return mdp.lambda * s.SeaLiceLevel + (1 - mdp.lambda) * mdp.costOfTreatment
    else
        return mdp.lambda * s.SeaLiceLevel
    end
end

POMDPs.discount(mdp::SeaLiceMDP) = mdp.discount_factor
POMDPs.initialstate(mdp::SeaLiceMDP) = Deterministic(SeaLiceState(1.0))
POMDPs.isterminal(mdp::SeaLiceMDP, s::SeaLiceState) = false

function mdp_optimize(df::DataFrame)
    # Create the MDP
    mdp = SeaLiceMDP()
    
    # Create the solver
    solver = ValueIterationSolver(max_iterations=30)
    # @show_requirements POMDPs.solve(solver, mdp)
    
    # Solve for the policy
    policy = solve(solver, mdp)
    
    # Save the policy
    mkpath("results/policies")
    save("results/policies/sea_lice_policy.jld2", "policy", policy)

    # Plot the policy
    # plot(policy, mdp)
    # savefig("results/figures/sea_lice_policy.png")
    
    return policy
end

end
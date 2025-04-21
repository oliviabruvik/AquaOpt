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
    return ImplicitDistribution() do rng
        growth_rate = mdp.growthRate
        rho = a == Treatment ? mdp.rho : 0.0
        next_sea_lice_mean = (1-rho) * exp(growth_rate) * s.SeaLiceLevel

        # Sample the next sea lice level with 0.2 std
        next_sea_lice_level = rand(rng, Normal(next_sea_lice_mean, 0.2))

        # Clamp the next sea lice level to the range 0-10
        next_sea_lice_level = clamp(next_sea_lice_level, 0.0, 10.0)

        # Round the next sea lice level to 1 decimal place
        next_sea_lice_level = round(next_sea_lice_level, digits=1)

        return SeaLiceState(next_sea_lice_level)
    end
end

function POMDPs.reward(mdp::SeaLiceMDP, s::SeaLiceState, a::Action)
    if a == Treatment
        return - (mdp.lambda * s.SeaLiceLevel + (1 - mdp.lambda) * mdp.costOfTreatment)
    else
        return - (mdp.lambda * s.SeaLiceLevel)
    end
end

POMDPs.discount(mdp::SeaLiceMDP) = mdp.discount_factor
POMDPs.initialstate(mdp::SeaLiceMDP) = ImplicitDistribution() do rng
    initial_lice = round(rand(rng, Uniform(0.0, 1.0)), digits=1)
    return SeaLiceState(initial_lice)
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
    save("results/policies/sea_lice_policy.jld2", "policy", policy)

    # Plot the policy
    # plot(policy, mdp)
    # savefig("results/figures/sea_lice_policy.png")
    
    return policy
end

end
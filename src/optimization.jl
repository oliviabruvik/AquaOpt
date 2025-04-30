include("optimize_mdp.jl")

using POMDPs
using POMDPModels
using POMDPTools
using DataFrames
using JLD2
using Plots

# ----------------------------
# Policy Generation
# ----------------------------
"Train a policy for each lambda using the given solver."
function find_policies_across_lambdas(lambda_values; solver, convert_to_mdp=false)
    policies = Dict{Float64, Tuple{Policy, SeaLiceMDP, MDP}}()

    for λ in lambda_values
        pomdp = SeaLiceMDP(lambda=λ)
        mdp = UnderlyingMDP(pomdp)
        policy = convert_to_mdp ? solve(solver, mdp) : solve(solver, pomdp)
        
        policies[λ] = (policy, pomdp, mdp)
        save("results/policies/sea_lice_mdp_policy_$(λ).jld2", "policy", policy)
    end

    return policies
end

"Create a heuristic policy for each lambda."
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

# ----------------------------
# Simulation & Evaluation
# ----------------------------
function run_simulation(policy, mdp, pomdp, episodes=100, steps_per_episode=50)
    
    total_cost, total_sealice = 0.0, 0.0
    total_steps = episodes * steps_per_episode

    for _ in 1:episodes
        s = rand(initialstate(mdp))
        b = Deterministic(s)
        
        for _ in 1:steps_per_episode
            a = policy isa AlphaVectorPolicy ? action(policy, b) : action(policy, s)
            total_cost += (a == Treatment ? pomdp.costOfTreatment : 0.0)
            total_sealice += s.SeaLiceLevel
            
            # Transition to next state
            s = rand(transition(pomdp, s, a))
            
            # Update belief
            if policy isa AlphaVectorPolicy
                o = rand(observation(pomdp, a, s))
                b = update_belief(b, a, o, pomdp)
            end
        end
    end

    return total_cost / total_steps, total_sealice / total_steps
end


"Very simple belief updater (placeholder for future particle filter)."
function update_belief(b, a, o, pomdp)
    # TODO: Update belief based on observation (avoided now because of complexity)
    # TODO: Review discretization of state space (now 0.1)
    # TODO: Implement particle filter for belief update?
    return Deterministic(SeaLiceState(o.SeaLiceLevel))
end

# Calculate average cost and average sea lice level for each lambda
function calculate_avg_rewards(policies_dict; episodes=100, steps_per_episode=50)
    results = DataFrame(lambda=Float64[], avg_treatment_cost=Float64[], avg_sealice=Float64[])
    
    for (λ, (policy, pomdp, mdp)) in pairs(policies_dict)
        avg_cost, avg_sealice = run_simulation(policy, mdp, pomdp, episodes, steps_per_episode)
        push!(results, (λ, avg_cost, avg_sealice))
    end

    rename!(results, [:lambda, :avg_treatment_cost, :avg_sealice])
    return results
end

# ----------------------------
# Heuristic Policy
# ----------------------------
struct HeuristicPolicy{P<:MDP} <: Policy
    mdp::P
end

# Heuristic action
function POMDPs.action(policy::HeuristicPolicy, s::SeaLiceState)
    return s.SeaLiceLevel > 0.5 ? Treatment : NoTreatment
end

# ----------------------------
# Optimizer Wrapper
# ----------------------------
function test_optimizer(lambda_values, solver; episodes=100, steps_per_episode=50, convert_to_mdp=false, plot_name="MDP Policy")
    if solver isa Nothing
        policies_dict = create_heuristic_policy_dict(lambda_values)
    else
        policies_dict = find_policies_across_lambdas(lambda_values, solver=solver, convert_to_mdp=convert_to_mdp)
    end
    results = calculate_avg_rewards(policies_dict, episodes=episodes, steps_per_episode=steps_per_episode)

    # Save results
    @save "results/data/$(plot_name)_$(episodes)_$(steps_per_episode).jld2" results
    
    # Plot results
    results_plot = plot_mdp_results(results, plot_name)
    savefig(results_plot, "results/figures/$(plot_name)_$(episodes)_$(steps_per_episode).png")
end
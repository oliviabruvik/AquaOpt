include("SeaLicePOMDP.jl")

using POMDPs
using POMDPModels
using POMDPTools
using DataFrames
using JLD2
using Plots

# ----------------------------
# Algorithm struct
# ----------------------------
struct Algorithm{S<:Solver}
    solver::S
    convert_to_mdp::Bool
    solver_name::String
end

# ----------------------------
# Policy Generation
# ----------------------------
"Train a policy for each lambda using the given solver."
function find_policies_across_lambdas(config; algorithm)
    policies = Dict{Float64, Tuple{Policy, SeaLiceMDP, MDP}}()

    for λ in config.lambda_values
        pomdp = SeaLiceMDP(lambda=λ)
        mdp = UnderlyingMDP(pomdp)
        policy = algorithm.convert_to_mdp ? solve(algorithm.solver, mdp) : solve(algorithm.solver, pomdp)
        
        policies[λ] = (policy, pomdp, mdp)

        # Save the policy, pomdp, and mdp to a file
        save("results/policies/$(algorithm.solver_name)_sea_lice_mdp_policy_$(λ).jld2", "policy", policy)
        save("results/policies/$(algorithm.solver_name)_sea_lice_pomdp_$(λ).jld2", "pomdp", pomdp)
        save("results/policies/$(algorithm.solver_name)_sea_lice_mdp_$(λ).jld2", "mdp", mdp)
    end

    return policies
end

"Create a heuristic policy for each lambda."
function create_heuristic_policy_dict(config)
    policies = Dict{Float64, Tuple{Policy, SeaLiceMDP, MDP}}()

    for λ in config.lambda_values
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

            # TODO: save the state and action to a file
            # TODO: use discrete updator
            
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

"Calculate average cost and average sea lice level for each lambda."
function calculate_avg_rewards(algorithm, config)

    results = DataFrame(lambda=Float64[], avg_treatment_cost=Float64[], avg_sealice=Float64[])
    episodes = config.num_episodes
    steps_per_episode = config.steps_per_episode
    lambda_values = config.lambda_values
    
    for λ in lambda_values
        policy = load("results/policies/$(algorithm.solver_name)_sea_lice_mdp_policy_$(λ).jld2", "policy")
        pomdp = load("results/policies/$(algorithm.solver_name)_sea_lice_pomdp_$(λ).jld2", "pomdp")
        mdp = load("results/policies/$(algorithm.solver_name)_sea_lice_mdp_$(λ).jld2", "mdp")

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
    threshold::Float64
end

# Heuristic action
function POMDPs.action(policy::HeuristicPolicy, s::SeaLiceState)
    return s.SeaLiceLevel > policy.threshold ? Treatment : NoTreatment
end

# ----------------------------
# Optimizer Wrapper
# ----------------------------
function test_optimizer(algorithm, config)

    if algorithm.solver isa Nothing # No special cases - 
        policies_dict = create_heuristic_policy_dict(config)
    else
        policies_dict = find_policies_across_lambdas(config, algorithm=algorithm)
    end
    results = calculate_avg_rewards(algorithm, config)

    # Save results
    @save "results/data/$(algorithm.solver_name)_$(config.num_episodes)_$(config.steps_per_episode).jld2" results
    
    # Plot results
    results_plot = plot_mdp_results(results, algorithm.solver_name)
    savefig(results_plot, "results/figures/$(algorithm.solver_name)_$(config.num_episodes)_$(config.steps_per_episode).png")
end
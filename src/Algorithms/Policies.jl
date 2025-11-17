
using POMDPs
using POMDPModels
using POMDPTools
using Distributions
using Parameters
using GaussianFilters
using DataFrames
using JLD2
using Plots

# ----------------------------
# Create POMDP and MDP for a given lambda
# ----------------------------
function create_pomdp_mdp(λ, config)

    # Create directory for POMDP and MDP
    pomdp_mdp_dir = joinpath(config.experiment_dir, "pomdp_mdp")
    mkpath(pomdp_mdp_dir)

    if config.solver_config.log_space
        pomdp = SeaLiceLogPOMDP(
            lambda=λ,
            reward_lambdas=config.solver_config.reward_lambdas,
            costOfTreatment=config.solver_config.costOfTreatment,
            growthRate=config.solver_config.growthRate,
            discount_factor=config.solver_config.discount_factor,
            discretization_step=config.solver_config.discretization_step,
            adult_sd=abs(log(config.solver_config.raw_space_sampling_sd)),
            regulation_limit=config.solver_config.regulation_limit,
            full_observability_solver=config.solver_config.full_observability_solver,
        )
    else
        pomdp = SeaLicePOMDP(
            lambda=λ,
            reward_lambdas=config.solver_config.reward_lambdas,
            costOfTreatment=config.solver_config.costOfTreatment,
            growthRate=config.solver_config.growthRate,
            discount_factor=config.solver_config.discount_factor,
            discretization_step=config.solver_config.discretization_step,
            adult_sd=config.solver_config.raw_space_sampling_sd,
            regulation_limit=config.solver_config.regulation_limit,
            full_observability_solver=config.solver_config.full_observability_solver,
        )
    end

    @info "Created POMDP with reward lambdas: $(pomdp.reward_lambdas)"

    mdp = UnderlyingMDP(pomdp)

    # Save POMDP and MDP to file
    pomdp_mdp_filename = "pomdp_mdp_$(λ)_lambda"
    pomdp_mdp_file_path = joinpath(pomdp_mdp_dir, "$(pomdp_mdp_filename).jld2")
    @save pomdp_mdp_file_path pomdp mdp
    # @info "Saved POMDP and MDP to file $(pomdp_mdp_file_path)"

    # Save POMDP as POMDPX file for NUS SARSOP
    pomdpx_file_path = joinpath(pomdp_mdp_dir, "pomdp.pomdpx")
    pomdpx = POMDPXFile(pomdpx_file_path)
    # POMDPXFiles.write(pomdp, pomdpx)
    # @info "Saved POMDP as POMDPX file $(pomdpx_file_path)"

    return pomdp, mdp
end

# ----------------------------
# Generate MDP and POMDP policies
# ----------------------------
function generate_mdp_pomdp_policies(algorithm, config)

    policies_dir = joinpath(config.policies_dir, "$(algorithm.solver_name)")
    mkpath(policies_dir)

    # Generate policies for each lambda
    for λ in config.lambda_values

        # Generate POMDP and MDP
        pomdp, mdp = create_pomdp_mdp(λ, config)

        # Generate policy
        policy = generate_policy(algorithm, pomdp, mdp)

        # Save policy, pomdp, and mdp to file
        policy_pomdp_mdp_filename = "policy_pomdp_mdp_$(λ)_lambda"
        @save joinpath(policies_dir, "$(policy_pomdp_mdp_filename).jld2") policy pomdp mdp
        #@info "Saved policy, pomdp, and mdp to file $(joinpath(policies_dir, "$(policy_pomdp_mdp_filename).jld2"))"

        return policy, pomdp, mdp
    end
end

# ----------------------------
# Policy Generation
# ----------------------------
function generate_policy(algorithm, pomdp, mdp)

    # Heuristic Policy
    if algorithm.solver_name == "Heuristic_Policy"
        return HeuristicPolicy(pomdp, algorithm.heuristic_config)

    # Random policy
    elseif algorithm.solver_name == "Random_Policy"
        return RandomPolicy(pomdp)

    # No Treatment policy
    elseif algorithm.solver_name == "NeverTreat_Policy"
        return NeverTreatPolicy(pomdp)

    # Always Treat policy
    elseif algorithm.solver_name == "AlwaysTreat_Policy"
        return AlwaysTreatPolicy(pomdp)

    # Value Iteration policy
    elseif algorithm.solver isa ValueIterationSolver
        return solve(algorithm.solver, mdp)

    # SARSOP and QMDP policies
    else
        return solve(algorithm.solver, pomdp)
    end
end

# ----------------------------
# Never Treat Policy
# ----------------------------
struct NeverTreatPolicy{P<:POMDP} <: Policy
    pomdp::P
end

# Never Treat action
function POMDPs.action(policy::NeverTreatPolicy, b)
    return NoTreatment
end

function POMDPs.updater(policy::NeverTreatPolicy)
    return DiscreteUpdater(policy.pomdp)
end

# ----------------------------
# Always Treat Policy
# ----------------------------
struct AlwaysTreatPolicy{P<:POMDP} <: Policy
    pomdp::P
end

# Always Treat action
function POMDPs.action(policy::AlwaysTreatPolicy, b)
    return Treatment
end

function POMDPs.updater(policy::AlwaysTreatPolicy)
    return DiscreteUpdater(policy.pomdp)
end

# ----------------------------
# Random Policy
# ----------------------------
struct RandomPolicy{P<:POMDP} <: Policy
    pomdp::P
end

# Random action
function POMDPs.action(policy::RandomPolicy, b)
    return rand((Treatment, NoTreatment))
end

function POMDPs.updater(policy::RandomPolicy)
    return DiscreteUpdater(policy.pomdp)
end

# ----------------------------
# Heuristic Policy
# ----------------------------
struct HeuristicPolicy{P<:POMDP} <: Policy
    pomdp::P
    heuristic_config::HeuristicConfig
end

# Heuristic action
# TODO: add some stochasticity
function POMDPs.action(policy::HeuristicPolicy, b)

    # Get the probability of the current sea lice level being above the threshold
    state_space = states(policy.pomdp)
    threshold = policy.heuristic_config.raw_space_threshold
    if policy.pomdp isa SeaLiceLogPOMDP
        threshold = log(threshold)
    end
    prob_above_threshold = sum(b[i] for (i, s) in enumerate(state_space) if s.SeaLiceLevel > threshold)

    # If the probability of the current sea lice level being above the threshold is greater than the thermal threshold, choose ThermalTreatment
    if prob_above_threshold > policy.heuristic_config.belief_threshold_thermal
        return ThermalTreatment
    # If the probability of the current sea lice level being above the threshold is greater than the mechanical threshold, choose Treatment
    elseif prob_above_threshold > policy.heuristic_config.belief_threshold_mechanical
        return Treatment
    # Otherwise, choose NoTreatment
    else
        return NoTreatment
    end
end

# Function to decide whether we choose the action or randomize
function heuristicChooseAction(policy::HeuristicPolicy, b, use_cdf=true)

    # Convert belief vector to a probability distribution
    state_space = states(policy.pomdp)

    # Convert the threshold in log space if needed
    if policy.pomdp isa SeaLiceLogPOMDP
        threshold = log(policy.heuristic_config.raw_space_threshold)
    else
        threshold = policy.heuristic_config.raw_space_threshold
    end

    if use_cdf
        # Method 1: Calculate probability of being above threshold
        prob_above_threshold = sum(b[i] for (i, s) in enumerate(state_space) if s.SeaLiceLevel > threshold)
        return prob_above_threshold > policy.heuristic_config.belief_threshold
    else
        # Method 2: Use mode of belief vector
        mode_sealice_level_index = argmax(b)
        mode_sealice_level = state_space[mode_sealice_level_index]
        return mode_sealice_level.SeaLiceLevel > threshold
    end
end

function POMDPs.updater(policy::HeuristicPolicy)
    return DiscreteUpdater(policy.pomdp)
end


# ----------------------------
# Adaptor Policy
# ----------------------------
struct AdaptorPolicy <: Policy
    lofi_policy::Policy
    pomdp::POMDP
    location::String
end

# Adaptor action
function POMDPs.action(policy::AdaptorPolicy, b)

    # Predict the next state
    pred_adult, pred_motile, pred_sessile = predict_next_abundances(b.μ[1][1], b.μ[3][1], b.μ[2][1], b.μ[4][1], policy.location)
    adult_sd = sqrt(b.Σ[1,1])

    # Clamp predictions to be positive
    pred_adult = max(pred_adult, 1e-3)

    if policy.pomdp isa SeaLiceLogPOMDP
        pred_adult = log(pred_adult)
        adult_sd = abs(log(adult_sd))
    end

    # Get next action from policy
    # TODO: write wrapper around ValueIterationPolicy action function that takes a belief vector and converts it to a state
    if policy.lofi_policy isa ValueIterationPolicy
        closest_idx = argmin(abs.(policy.pomdp.sea_lice_range .- pred_adult))
        pred_adult_state = policy.pomdp.sea_lice_range[closest_idx]
        if policy.pomdp isa SeaLiceLogPOMDP
            pred_adult_state = SeaLiceLogState(pred_adult_state)
        else
            pred_adult_state = SeaLiceState(pred_adult_state)
        end
        @assert pred_adult_state.SeaLiceLevel - pred_adult < policy.pomdp.discretization_step
        @assert pred_adult_state in states(policy.pomdp)
        return action(policy.lofi_policy, pred_adult_state)
    end

    # Discretize alpha vectors (representation of utility over belief states per action)
    state_space = states(policy.lofi_policy.pomdp)
    bvec = discretize_distribution(Normal(pred_adult, adult_sd), state_space)
    return action(policy.lofi_policy, bvec)
end

# ----------------------------
# LOFI Adaptor Policy
# ----------------------------
struct LOFIAdaptorPolicy{LP <: Policy, P <: POMDP} <: Policy
    lofi_policy::LP
    pomdp::P
end

function POMDPs.action(policy::LOFIAdaptorPolicy{<:ValueIterationPolicy}, b)
    mode_idx = argmax(b.b)
    all_states = states(policy.pomdp)
    state_with_highest_probability = all_states[mode_idx]

    # Assertions
    @assert state_with_highest_probability in states(policy.pomdp)
    @assert length(b.b) == length(states(policy.pomdp))
    
    # Get next action from policy
    return action(policy.lofi_policy, state_with_highest_probability)
end

# Adaptor action
function POMDPs.action(policy::LOFIAdaptorPolicy, b)
    # Discretize alpha vectors (representation of utility over belief states per action)
    return action(policy.lofi_policy, b)
end

# ----------------------------
# Full Observability Adaptor Policy
# ----------------------------
struct FullObservabilityAdaptorPolicy <: Policy
    lofi_policy::Policy
    pomdp::POMDP
    mdp::MDP
    location::String
end

# Adaptor action
function POMDPs.action(policy::FullObservabilityAdaptorPolicy, s)

    # Predict the next state
    pred_adult, pred_motile, pred_sessile = predict_next_abundances(s.sessile, s.motile, s.adult, s.temp, policy.location)

    # Clamp predictions to be positive
    pred_adult = max(pred_adult, 1e-3)

    if policy.pomdp isa SeaLiceLogPOMDP
        pred_adult = log(pred_adult)
    end

    # Get next action from policy
    # TODO: write wrapper around ValueIterationPolicy action function that takes a belief vector and converts it to a state
    if policy.lofi_policy isa ValueIterationPolicy
        closest_idx = argmin(abs.(policy.pomdp.sea_lice_range .- pred_adult))
        pred_adult_state = policy.pomdp.sea_lice_range[closest_idx]
        if policy.pomdp isa SeaLiceLogPOMDP
            pred_adult_state = SeaLiceLogState(pred_adult_state)
        else
            pred_adult_state = SeaLiceState(pred_adult_state)
        end
        @assert pred_adult_state.SeaLiceLevel - pred_adult < policy.pomdp.discretization_step
        @assert pred_adult_state in states(policy.pomdp)
        return action(policy.lofi_policy, pred_adult_state)
    end

    # Discretize alpha vectors (representation of utility over belief states per action)
    state_space = states(policy.lofi_policy.pomdp)
    bvec = discretize_distribution(Normal(pred_adult, adult_sd), state_space)
    return action(policy.lofi_policy, bvec)
end

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
# Create POMDP and MDP
# ----------------------------
function create_pomdp_mdp(config)

    sim_cfg = config.simulation_config
    adult_mean = max(sim_cfg.adult_mean, 1e-6)
    motile_ratio = sim_cfg.motile_mean / adult_mean
    sessile_ratio = sim_cfg.sessile_mean / adult_mean
    base_temperature = get_location_params(config.solver_config.location).T_mean

    if config.solver_config.log_space
        pomdp = SeaLiceLogPOMDP(
            reward_lambdas=config.solver_config.reward_lambdas,
            discount_factor=config.solver_config.discount_factor,
            discretization_step=config.solver_config.discretization_step,
            adult_sd=config.solver_config.adult_sd,
            regulation_limit=config.solver_config.regulation_limit,
            full_observability_solver=config.solver_config.full_observability_solver,
            location=config.solver_config.location,
            reproduction_rate=config.solver_config.reproduction_rate,
            motile_ratio=motile_ratio,
            sessile_ratio=sessile_ratio,
            base_temperature=base_temperature,
        )
    else
        pomdp = SeaLicePOMDP(
            reward_lambdas=config.solver_config.reward_lambdas,
            discount_factor=config.solver_config.discount_factor,
            discretization_step=config.solver_config.discretization_step,
            adult_sd=config.solver_config.adult_sd,
            regulation_limit=config.solver_config.regulation_limit,
            full_observability_solver=config.solver_config.full_observability_solver,
            location=config.solver_config.location,
            reproduction_rate=config.solver_config.reproduction_rate,
            motile_ratio=motile_ratio,
            sessile_ratio=sessile_ratio,
            base_temperature=base_temperature,
        )
    end

    mdp = UnderlyingMDP(pomdp)
    return pomdp, mdp
end

# ----------------------------
# Generate MDP and POMDP policies
# ----------------------------
function solve_policies(algorithms, config)

    pomdp, mdp = create_pomdp_mdp(config)

    all_policies = Dict{String, NamedTuple}()

    for algo in algorithms
        policy = generate_policy(algo, pomdp, mdp)
        all_policies[algo.solver_name] = (policy=policy, pomdp=pomdp, mdp=mdp)
    end

    # Save all_policies, pomdp, and mdp to file
    policies_dir = joinpath(config.policies_dir)
    mkpath(policies_dir)
    @save joinpath(policies_dir, "policies_pomdp_mdp.jld2") all_policies pomdp mdp

    return all_policies
end

# ----------------------------
# Policy Generation
# ----------------------------
function generate_policy(algorithm, pomdp, mdp)

    # Heuristic Policy
    if algorithm.solver_name == "Heuristic_Policy"
        return HeuristicPolicy(pomdp, algorithm.solver_config)

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
        if algorithm.solver isa SARSOP.SARSOPSolver
            mkpath(dirname(algorithm.solver.policy_filename))
            mkpath(dirname(algorithm.solver.pomdp_filename))
        end
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
    return MechanicalTreatment
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
    return rand((NoTreatment, MechanicalTreatment, ChemicalTreatment, ThermalTreatment))
end

function POMDPs.updater(policy::RandomPolicy)
    return DiscreteUpdater(policy.pomdp)
end

# ----------------------------
# Heuristic Policy
# ----------------------------
struct HeuristicPolicy{P<:POMDP} <: Policy
    pomdp::P
    solver_config::SolverConfig
end

# Heuristic action
# TODO: add some stochasticity
function POMDPs.action(policy::HeuristicPolicy, b)

    @assert policy.solver_config.heuristic_belief_threshold_thermal >
            policy.solver_config.heuristic_belief_threshold_mechanical >
            policy.solver_config.heuristic_belief_threshold_chemical "Heuristic thresholds must be strictly descending: thermal > mechanical > chemical"

    # Get the probability of the current sea lice level being above the threshold
    state_space = states(policy.pomdp)
    threshold = policy.solver_config.heuristic_threshold
    if policy.pomdp isa SeaLiceLogPOMDP
        threshold = log(threshold)
    end
    prob_above_threshold = sum(b[i] for (i, s) in enumerate(state_space) if s.SeaLiceLevel > threshold)

    # If the probability of the current sea lice level being above the threshold is greater than the thermal threshold, choose ThermalTreatment
    if prob_above_threshold > policy.solver_config.heuristic_belief_threshold_thermal
        return ThermalTreatment
    # If the probability of the current sea lice level being above the threshold is greater than the mechanical threshold, choose MechanicalTreatment
    elseif prob_above_threshold > policy.solver_config.heuristic_belief_threshold_mechanical
        return MechanicalTreatment
    # Chemical treatment for lower-level infestations
    elseif prob_above_threshold > policy.solver_config.heuristic_belief_threshold_chemical
        return ChemicalTreatment
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
        threshold = log(policy.solver_config.heuristic_threshold)
    else
        threshold = policy.solver_config.heuristic_threshold
    end

    if use_cdf
        # Method 1: Calculate probability of being above threshold
        prob_above_threshold = sum(b[i] for (i, s) in enumerate(state_space) if s.SeaLiceLevel > threshold)
        return prob_above_threshold > policy.solver_config.heuristic_belief_threshold_mechanical
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
    reproduction_rate::Float64
end

# Adaptor action
function POMDPs.action(policy::AdaptorPolicy, b)

    # Predict the next state (always in raw space from the biological model)
    # Belief means are ordered [Adult, Motile, Sessile, Temperature]
    pred_adult, pred_motile, pred_sessile = predict_next_abundances(
        b.μ[BELIEF_IDX_ADULT][1], b.μ[BELIEF_IDX_MOTILE][1], b.μ[BELIEF_IDX_SESSILE][1], b.μ[BELIEF_IDX_TEMPERATURE][1],
        policy.location, policy.reproduction_rate)
    adult_variance = b.Σ[1,1]  # Variance in raw space

    # Clamp predictions to be positive
    pred_adult = max(pred_adult, 1e-3)

    # Convert to log space if needed for the policy
    if policy.pomdp isa SeaLiceLogPOMDP

        # Calculate CV^2: Coefficient of Variation squared
        # IMPORTANT: Floor the denominator to prevent CV^2 explosion when pred_adult → 0
        # Using floor of 0.05^2 = 0.0025 caps maximum CV^2 when mean is very small
        # This prevents the log-space uncertainty from becoming unreasonably large
        # With typical adult_variance ≈ 0.0013, this caps CV^2 at ~0.52, giving σ_log ≈ 0.65
        pred_adult_squared_floored = max(pred_adult^2, 0.0025)

        # Ensure non-negative variance (numerical stability)
        cv_squared = max(adult_variance, 0.0) / pred_adult_squared_floored

        # Lognormal Formula for Log-Space SD (σ_log)
        # σ_log = sqrt(ln(1 + CV^2))
        # Ensure log argument is positive (cv_squared >= 0, so 1 + cv_squared >= 1.0)
        adult_sd_log = sqrt(log(1.0 + cv_squared))

        # Use log of median (not mean) for point prediction
        # Median of log-normal: log(μ) (no bias correction)
        # This ensures better alignment with discretized states
        pred_adult_log = log(pred_adult)

        # Update the variables
        pred_adult = pred_adult_log
        adult_sd = adult_sd_log
    else
        # Ensure non-negative variance (numerical stability)
        # Kalman filter covariance can become slightly negative due to floating-point errors
        adult_sd = sqrt(max(adult_variance, 0.0))
    end

    # Discretize the belief over the state space
    state_space = states(policy.pomdp)
    bvec = discretize_distribution(Normal(pred_adult, adult_sd), state_space)

    # For VI: compute belief-weighted Q-values (same principle as QMDP)
    if policy.lofi_policy isa ValueIterationPolicy
        best_action = NoTreatment
        best_value = -Inf
        for a in actions(policy.pomdp)
            q_val = sum(bvec[i] * value(policy.lofi_policy, s, a)
                        for (i, s) in enumerate(state_space))
            if q_val > best_value
                best_value = q_val
                best_action = a
            end
        end
        return best_action
    end

    # For QMDP/SARSOP: use alpha vector dot product with belief
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
    all_states = states(policy.pomdp)
    @assert length(b.b) == length(all_states)

    # Belief-weighted Q-values (same principle as QMDP)
    best_action = NoTreatment
    best_value = -Inf
    for a in actions(policy.pomdp)
        q_val = sum(b.b[i] * value(policy.lofi_policy, s, a)
                    for (i, s) in enumerate(all_states))
        if q_val > best_value
            best_value = q_val
            best_action = a
        end
    end
    return best_action
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
    reproduction_rate::Float64
end

# Adaptor action
function POMDPs.action(policy::FullObservabilityAdaptorPolicy, s)

    # Predict the next state
    pred_adult, pred_motile, pred_sessile = predict_next_abundances(s.Adult, s.Motile, s.Sessile, s.Temperature, policy.location, policy.reproduction_rate)

    # Clamp predictions to be positive
    pred_adult = max(pred_adult, 1e-3)

    if policy.pomdp isa SeaLiceLogPOMDP
        pred_adult = log(pred_adult)
    end

    # Get next action from policy (point estimate b/c full observability)
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

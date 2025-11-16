
using DataFrames
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
using Random

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

# Action enum is imported from SharedTypes.jl

# -------------------------
# SeaLiceLogPOMDP Definition
# -------------------------
"Sea lice POMDP with growth dynamics and treatment effects in log space."
@with_kw struct SeaLiceLogPOMDP <: POMDP{SeaLiceLogState, Action, SeaLiceLogObservation}

    # Parameters
	lambda::Float64 = 0.5
    reward_lambdas::Vector{Float64} = [0.5, 0.5, 0.0, 0.0, 0.0] # [treatment, regulatory, biomass, health, sea lice]
	costOfTreatment::Float64 = 10.0
	growthRate::Float64 = 1.2
    discount_factor::Float64 = 0.95
    full_observability_solver::Bool = false

    # Regulation parameters
    regulation_limit::Float64 = 0.5

    # Parameters from Aldrin et al. 2023
    n_sample::Int = 20                 # number of fish counted (ntc)
    ρ_adult::Float64 = 0.175            # aggregation/over-dispersion "ρ" (adult default)
    use_underreport::Bool = false      # toggle logistic under-count correction
    beta0_Scount_f::Float64 = -1.535   # farm intercept for under-count (if used)
    beta1_Scount::Float64 = 0.039      # weight slope for under-count (if used)
    mean_fish_weight_kg::Float64 = 1.5 # mean fish weight
    W0::Float64 = 0.1                  # weight centering (kg)

    # Count bounds
    sea_lice_bounds::Tuple{Float64, Float64} = (log(1e-3), log(5.0)) # (-3, 0.70)
    initial_bounds::Tuple{Float64, Float64} = (log(1e-3), log(0.25))
    initial_mean::Float64 = log(0.13)

    # Sampling parameters
    adult_sd::Float64 = abs(log(0.1))
    rng::AbstractRNG = Random.GLOBAL_RNG
    
    # Log space discretization
    discretization_step::Float64 = 0.1 # 0.01
    initial_range::Vector{Float64} = collect(initial_bounds[1]:discretization_step:initial_bounds[2])
    sea_lice_range::Vector{Float64} = collect(sea_lice_bounds[1]:discretization_step:sea_lice_bounds[2])
    lindisc::LinearDiscretizer = LinearDiscretizer(collect(sea_lice_bounds[1]:discretization_step:(sea_lice_bounds[2]+discretization_step)))
    catdisc::CategoricalDiscretizer = CategoricalDiscretizer([NoTreatment, Treatment, ThermalTreatment])
end

# -------------------------
# POMDPs.jl Interface
# -------------------------
POMDPs.states(pomdp::SeaLiceLogPOMDP) = [SeaLiceLogState(i) for i in pomdp.sea_lice_range]
POMDPs.actions(pomdp::SeaLiceLogPOMDP) = [NoTreatment, Treatment, ThermalTreatment]
POMDPs.observations(pomdp::SeaLiceLogPOMDP) = [SeaLiceLogObservation(i) for i in pomdp.sea_lice_range]
POMDPs.discount(pomdp::SeaLiceLogPOMDP) = pomdp.discount_factor
POMDPs.isterminal(pomdp::SeaLiceLogPOMDP, s::SeaLiceLogState) = false
POMDPs.actionindex(pomdp::SeaLiceLogPOMDP, a::Action) = encode(pomdp.catdisc, a)

# -------------------------
# State and Observation Index
# -------------------------
function POMDPs.stateindex(pomdp::SeaLiceLogPOMDP, s::SeaLiceLogState)
    return encode(pomdp.lindisc, s.SeaLiceLevel)
end

function POMDPs.obsindex(pomdp::SeaLiceLogPOMDP, o::SeaLiceLogObservation)
    return encode(pomdp.lindisc, o.SeaLiceLevel)
end

# -------------------------
# Transition, Observation, Reward, Initial State
# -------------------------
function POMDPs.transition(pomdp::SeaLiceLogPOMDP, s::SeaLiceLogState, a::Action)

    # Apply treatment
    rf_a, rf_m, rf_s = get_treatment_effectiveness(a)
    adult = s.SeaLiceLevel + log(1 - rf_a)

    # Calculate the mean of the transition distribution
    μ = pomdp.growthRate + adult

    # Clamp the mean to the range of the sea lice range
    μ = clamp(μ, pomdp.sea_lice_bounds...)

    # Get the distribution
    dist = truncated(Normal(μ, pomdp.adult_sd), pomdp.sea_lice_bounds...)

    # Get the states
    states = POMDPs.states(pomdp)

    # Calculate the probs using the cdf
    probs = discretize_distribution(dist, states)

    return SparseCat(states, probs)
end

# -------------------------
# Observation function: the current count of adults is measured with a negative binomial
# distribution.
# -------------------------
function POMDPs.observation(pomdp::SeaLiceLogPOMDP, a::Action, sp::SeaLiceLogState)

    # Observation grid
    observations = POMDPs.observations(pomdp)

    # If full observability solver, we return the exact state as the observation
    # to mimic full observability.
    if pomdp.full_observability_solver
        closest_idx = encode(pomdp.lindisc, sp.SeaLiceLevel)
        probs = zeros(length(observations))
        probs[closest_idx] = 1.0
        return SparseCat(observations, probs)
    end

    # (Optional) under-counting correction like p^Scount_ftc
    # paper uses (W - 0.1) with W in kg
    p_scount = if pomdp.use_underreport
        η = pomdp.beta0_Scount_f + pomdp.beta1_Scount*(pomdp.mean_fish_weight_kg - pomdp.W0)
        logistic(η)
    else
        1.0
    end

    # NB parameters on TOTAL counts over n_sample fish
    # mean of total counts = n_sample * p_scount * (true lice per fish)
    raw_adult = exp(sp.SeaLiceLevel)
    μ_total_adult = max(1e-12, pomdp.n_sample * p_scount * raw_adult)
    k = max(1e-9, pomdp.n_sample * pomdp.ρ_adult)  # NB "size" scales with n_sample
    r, p = nb_params_from_mean_k(μ_total_adult, k)
    nb = NegativeBinomial(r, p)

    # Build CDF at each grid value (threshold on total counts)
    cdfs = similar(observations, Float64)
    @inbounds for (i, o) in enumerate(observations)
        adult_threshold = clamp(o.SeaLiceLevel, pomdp.sea_lice_bounds...)
        raw_adult_threshold = exp(adult_threshold)
        total_adult_threshold = max(0, floor(Int, pomdp.n_sample * raw_adult_threshold))
        ci = cdf(nb, total_adult_threshold)
        if !isfinite(ci)
            @warn "NaN in observation distribution (SeaLiceLogPOMDP)"
            ci = 0.0
        end
        cdfs[i] = ci
    end

    # Ensure the last bin captures any tail mass beyond max grid value
    cdfs[end] = 1.0

    # Convert CDFs to bin probabilities
    probs = similar(cdfs)
    past = 0.0
    @inbounds for i in eachindex(cdfs)
        ci = cdfs[i]
        # guard against NaNs or non-monotonic numeric issues
        if !isfinite(ci) || ci < past
            @warn "NaN or non-monotonic in observation distribution (SeaLiceLogPOMDP)"
            ci = past
        end
        probs[i] = max(ci - past, 0.0)
        past = ci
    end

    # Normalize (and fallback if degenerate)
    if sum(probs) <= 0 || !isfinite(sum(probs))
        # Put all mass on the nearest bin to the mean of the NB distribution
        @warn "Degenerate observation distribution (SeaLiceLogPOMDP)"
        μ_adult = log(max(μ_total_adult / pomdp.n_sample, 1e-12))
        idx = findmin(abs.([o.SeaLiceLevel - μ_adult for o in observations]))[2]
        probs .= 0.0
        probs[idx] = 1.0
    else
        probs = normalize(probs, 1)
    end

    return SparseCat(observations, probs)
end

# -------------------------
# Reward
# -------------------------
function POMDPs.reward(pomdp::SeaLiceLogPOMDP, s::SeaLiceLogState, a::Action)

    # Get reward lambdas (previous: [0.8, 0.1, 0.0, 0.3, 0.1])
    λ_trt, λ_reg, λ_bio, λ_health, λ_sea_lice = pomdp.reward_lambdas

    # Treatment cost
    treatment_cost = get_treatment_cost(a)

    # Regulatory penalty
    regulatory_penalty = get_regulatory_penalty(a) * (exp(s.SeaLiceLevel) > pomdp.regulation_limit ? 1.0 : 0.0)

    # Lost biomass (not used in this simplified model)
    lost_biomass = 0

    # Fish disease penalty
    fish_disease = get_fish_disease(a)

    # Sea lice level
    sea_lice_level = exp(s.SeaLiceLevel)

    return - (λ_trt * treatment_cost + λ_reg * regulatory_penalty + λ_bio * lost_biomass + λ_health * fish_disease + λ_sea_lice * sea_lice_level)
end

# -------------------------
# Initial state
# -------------------------
function POMDPs.initialstate(pomdp::SeaLiceLogPOMDP)

    # Get the distribution
    dist = truncated(Normal(pomdp.initial_mean, pomdp.adult_sd), pomdp.sea_lice_bounds...)

    # Get the states
    states = POMDPs.states(pomdp)

    # Calculate the probs using the cdf
    probs = discretize_distribution(dist, states)

    return SparseCat(states, probs)
end
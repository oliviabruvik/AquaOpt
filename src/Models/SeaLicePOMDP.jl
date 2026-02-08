using DataFrames
import Distributions: Normal, Uniform
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
"State representing the predicted sea lice level the following week."
struct SeaLiceState
	SeaLiceLevel::Float64
end

"Observation representing an observed sea lice level the following week."
struct SeaLiceObservation
	SeaLiceLevel::Float64
end

# Action enum is imported from SharedTypes.jl

# -------------------------
# SeaLicePOMDP Definition
# -------------------------
"Sea lice POMDP with growth dynamics and treatment effects."
@with_kw struct SeaLicePOMDP <: POMDP{SeaLiceState, Action, SeaLiceObservation}

    # Parameters
	lambda::Float64 = 0.5
    reward_lambdas::Vector{Float64} = [0.5, 0.5, 0.0, 0.0, 0.0] # [treatment, regulatory, biomass, health, sea_lice]
	costOfTreatment::Float64 = 10.0
	growthRate::Float64 = 0.3
    discount_factor::Float64 = 0.95
    full_observability_solver::Bool = false
    location::String = "north"
    reproduction_rate::Float64 = 2.0
    motile_ratio::Float64 = 1.0
    sessile_ratio::Float64 = 1.0
    base_temperature::Float64 = 10.0

    # Regulation parameters
    regulation_limit::Float64 = 0.5

    # Parameters from Aldrin et al. 2023
    n_sample::Int = 20                 # number of fish counted (ntc)
    rho_nb::Float64 = 0.175            # aggregation/over-dispersion "ρ" (adult default)
    use_underreport::Bool = false      # toggle logistic under-count correction
    beta0_Scount_f::Float64 = -1.535   # farm intercept for under-count (if used)
    beta1_Scount::Float64 = 0.039      # weight slope for under-count (if used)
    mean_fish_weight_kg::Float64 = 1.5 # mean fish weight
    W0::Float64 = 0.1                  # weight centering (kg)

    # Count bounds
    sea_lice_bounds::Tuple{Float64, Float64} = (0.0, 10.0)
    initial_bounds::Tuple{Float64, Float64} = (0.0, 0.25)
    initial_mean::Float64 = 0.13
    
    # Sampling parameters
    adult_sd::Float64 = 0.1
    rng::AbstractRNG = Random.GLOBAL_RNG

    # Discretization
    discretization_step::Float64 = 0.05 # 0.01
    sea_lice_range::Vector{Float64} = collect(sea_lice_bounds[1]:discretization_step:sea_lice_bounds[2])
    initial_range::Vector{Float64} = collect(initial_bounds[1]:discretization_step:initial_bounds[2])
    lindisc::LinearDiscretizer = LinearDiscretizer(collect(sea_lice_bounds[1]:discretization_step:(sea_lice_bounds[2]+discretization_step)))
    catdisc::CategoricalDiscretizer = CategoricalDiscretizer([NoTreatment, MechanicalTreatment, ChemicalTreatment, ThermalTreatment])
end

# -------------------------
# POMDPs.jl Interface
# -------------------------
POMDPs.states(pomdp::SeaLicePOMDP) = [SeaLiceState(i) for i in pomdp.sea_lice_range]
POMDPs.actions(pomdp::SeaLicePOMDP) = [NoTreatment, MechanicalTreatment, ChemicalTreatment, ThermalTreatment]
POMDPs.observations(pomdp::SeaLicePOMDP) = [SeaLiceObservation(i) for i in pomdp.sea_lice_range]
POMDPs.discount(pomdp::SeaLicePOMDP) = pomdp.discount_factor
POMDPs.isterminal(pomdp::SeaLicePOMDP, s::SeaLiceState) = false
POMDPs.stateindex(pomdp::SeaLicePOMDP, s::SeaLiceState) = encode(pomdp.lindisc, s.SeaLiceLevel)
POMDPs.actionindex(pomdp::SeaLicePOMDP, a::Action) = encode(pomdp.catdisc, a)
POMDPs.obsindex(pomdp::SeaLicePOMDP, o::SeaLiceObservation) = encode(pomdp.lindisc, o.SeaLiceLevel)

# -------------------------
# Transition
# -------------------------
function POMDPs.transition(pomdp::SeaLicePOMDP, s::SeaLiceState, a::Action)

    # Apply treatment in raw space
    rf_a, rf_m, rf_s = get_treatment_effectiveness(a)
    adult_raw = max(s.SeaLiceLevel * (1 - rf_a), 0.0)
    motile_raw = max(adult_raw * pomdp.motile_ratio * (1 - rf_m), 0.0)
    sessile_raw = max(adult_raw * pomdp.sessile_ratio * (1 - rf_s), 0.0)

    # Predict next adult level using the biological drift
    pred_adult_raw, _, _ = predict_next_abundances(
        adult_raw,
        motile_raw,
        sessile_raw,
        pomdp.base_temperature,
        pomdp.location,
        pomdp.reproduction_rate,
    )

    # Clamp the mean to the range of the sea lice range
    μ = clamp(pred_adult_raw, pomdp.sea_lice_bounds...)

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
function POMDPs.observation(pomdp::SeaLicePOMDP, a::Action, sp::SeaLiceState)

    # Observation grid
    observations = POMDPs.observations(pomdp)

    # If full observability solver, we return the exact state as the observation
    # to mimic full observability.
    if pomdp.full_observability_solver
        # Set the probs to 1 for the state we are at and 0 for the rest
        observations = POMDPs.observations(pomdp)
        probs = zeros(length(observations))
        probs[obsindex(pomdp, SeaLiceObservation(sp.SeaLiceLevel))] = 1.0
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
    μ_total_adult = max(1e-12, pomdp.n_sample * p_scount * sp.SeaLiceLevel)
    k = max(1e-9, pomdp.n_sample * pomdp.rho_nb)  # NB "size" scales with n_sample
    r, p = nb_params_from_mean_k(μ_total_adult, k)
    nb = NegativeBinomial(r, p)

    # Build CDF at each grid value (threshold on total counts)
    cdfs = similar(observations, Float64)
    @inbounds for (i, o) in enumerate(observations)
        adult_threshold = clamp(o.SeaLiceLevel, pomdp.sea_lice_bounds...)
        total_adult_threshold = max(0, floor(Int, pomdp.n_sample * adult_threshold))
        ci = cdf(nb, total_adult_threshold)
        if !isfinite(ci)
            @warn "NaN in observation distribution (SeaLicePOMDP)"
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
            @warn "NaN or non-monotonic in observation distribution (SeaLicePOMDP)"
            ci = past
        end
        probs[i] = max(ci - past, 0.0)
        past = ci
    end

    # Normalize (and fallback if degenerate)
    if sum(probs) <= 0 || !isfinite(sum(probs))
        # Put all mass on the nearest bin to the mean of the NB distribution
        @warn "Degenerate observation distribution (SeaLicePOMDP)"
        idx = findmin(abs.([o.SeaLiceLevel - μ_total_adult / pomdp.n_sample for o in observations]))[2]
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
function POMDPs.reward(pomdp::SeaLicePOMDP, s::SeaLiceState, a::Action)

    λ_trt, λ_reg, λ_bio, λ_health, λ_sea_lice = pomdp.reward_lambdas

    # SeaLicePOMDP stores state in natural space (not log space)
    adult_level = s.SeaLiceLevel

    # === 1. DIRECT TREATMENT COSTS ===
    treatment_cost = get_treatment_cost(a)

    # === 2. REGULATORY PENALTY (exponential above limit) ===
    # Reflects escalating consequences: fines, production caps, license restrictions
    if adult_level > pomdp.regulation_limit
        regulatory_penalty = 100.0 # * ((excess_ratio - 1.0)^2) * excess_ratio
    else
        regulatory_penalty = 0.0
    end

    # === 3. BIOMASS LOSS ===
    # 3a. Mortality loss (acute) - approximation for typical farm
    # Typical: 200k fish at 2kg avg weight
    mortality_biomass_loss = get_treatment_mortality_rate(a) * 400.0

    # 3b. Growth reduction from sea lice (chronic)
    # Research shows 3-16% biomass growth lost per cycle above ~0.5 lice/fish
    # Using 10% per week at high infestation as conservative estimate
    if adult_level > 0.5
        lice_severity = min((adult_level - 0.5) / 1.5, 1.0)  # 0 at 0.5, 1.0 at 2.0+
        # Typical farm mid-cycle biomass ~400 tonnes
        growth_biomass_loss = 400.0 * 0.10 * lice_severity
    else
        growth_biomass_loss = 0.0
    end

    total_biomass_loss = mortality_biomass_loss + growth_biomass_loss

    # === 4. FISH HEALTH (treatment side effects only) ===
    # Stress, injuries, disease susceptibility from treatments
    # Does NOT include sea lice damage (that's in sea_lice_penalty)
    fish_health_penalty = get_fish_disease(a)

    # === 5. SEA LICE BURDEN (chronic parasite damage) ===
    # Separate from growth: osmoregulatory stress, secondary infections, welfare
    # Scales non-linearly (exponentially worse at high levels)
    # At 0.1: 0.10
    # At 0.5: 0.50
    # At 1.0: 1.00 × 1.10 = 1.10 (milder)
    # At 2.0: 2.00 × 1.30 = 2.60 (much milder than 3.50)
    sea_lice_penalty = adult_level * (1.0 + 0.2 * max(0, adult_level - 0.5))

    # === TOTAL REWARD ===
    return -(
        λ_trt * treatment_cost +
        λ_reg * regulatory_penalty +
        λ_bio * total_biomass_loss +
        λ_health * fish_health_penalty +
        λ_sea_lice * sea_lice_penalty
    )
end

# -------------------------
# Initial state
# -------------------------
function POMDPs.initialstate(pomdp::SeaLicePOMDP)

    # Get the distribution
    dist = truncated(Normal(pomdp.initial_mean, pomdp.adult_sd), pomdp.sea_lice_bounds...)

    # Get the states
    states = POMDPs.states(pomdp)

    # Calculate the probs using the cdf
    probs = discretize_distribution(dist, states)

    return SparseCat(states, probs)
end

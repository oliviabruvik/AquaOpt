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
"State representing predicted sea lice level, season, farm biomass, and treatment cooldown."
struct SeaLiceState
	SeaLiceLevel::Float64   # Adult lice per fish (natural space)
    Season::Int             # 1=Spring, 2=Summer, 3=Autumn, 4=Winter
    BiomassLevel::Float64   # Farm biomass in tonnes
    Cooldown::Int           # 0=no recent treatment, 1=treated last step
end

"Observation: noisy lice count plus perfectly observed season, biomass, and cooldown."
struct SeaLiceObservation
	SeaLiceLevel::Float64
    Season::Int
    BiomassLevel::Float64
    Cooldown::Int
end

# Action enum is imported from SharedTypes.jl

# -------------------------
# SeaLicePOMDP Definition
# -------------------------
"Sea lice POMDP with 4D state: (lice level, season, biomass, treatment cooldown)."
@with_kw struct SeaLicePOMDP <: POMDP{SeaLiceState, Action, SeaLiceObservation}

    # Parameters
    reward_lambdas::Vector{Float64} = [0.5, 0.5, 0.0, 0.0, 0.0] # [treatment, regulatory, biomass, health, sea_lice]
	discount_factor::Float64 = 0.95
    full_observability_solver::Bool = false
    location::String = "north"
    reproduction_rate::Float64 = 2.0
    motile_ratio::Float64 = 1.0
    sessile_ratio::Float64 = 1.0
    base_temperature::Float64 = 10.0

    # Regulation parameters — season-dependent thresholds
    regulation_limit::Float64 = 0.5  # default (summer/autumn/winter)
    season_regulation_limits::Vector{Float64} = [0.2, 0.5, 0.5, 0.5]  # [Spring, Summer, Autumn, Winter]

    # Season parameters
    # Midpoint weeks used to look up temperature for each season
    season_midpoint_weeks::Vector{Int} = [19, 32, 45, 6]  # Spring, Summer, Autumn, Winter

    # Biomass parameters
    biomass_range::Vector{Float64} = [50.0, 220.0, 390.0, 560.0, 730.0, 900.0]
    biomass_bounds::Tuple{Float64, Float64} = (50.0, 900.0)
    biomass_max::Float64 = 900.0
    k_biomass::Float64 = 0.015          # Weekly biomass growth rate
    temp_sensitivity::Float64 = 0.03    # Temperature effect on growth rate
    nat_mort_rate::Float64 = 0.0008     # Weekly natural mortality fraction
    biomass_sd::Float64 = 15.0          # Biomass transition noise (tonnes)
    max_growth_loss_fraction::Float64 = 0.10  # Max weekly growth loss from lice

    # Cooldown stress multiplier
    cooldown_stress_multiplier::Float64 = 1.5  # Extra stress when treating after recent treatment

    # Financial parameters — all reward components output MNOK
    salmon_price_MNOK_per_tonne::Float64 = 0.07  # ~70 NOK/kg Norwegian salmon spot price
    regulatory_violation_cost_MNOK::Float64 = 10.0  # Forced emergency treatment + production disruption + license risk
    welfare_cost_MNOK::Float64 = 1.0  # Per stress-score unit: veterinary, secondary infections, reduced performance
    chronic_lice_cost_MNOK::Float64 = 0.5  # Per burden-unit/week: growth reduction, feed conversion loss

    # Parameters from Aldrin et al. 2023
    n_sample::Int = 20                 # number of fish counted (ntc)
    rho_nb::Float64 = 0.175            # aggregation/over-dispersion "ρ" (adult default)
    use_underreport::Bool = false      # toggle logistic under-count correction
    beta0_Scount_f::Float64 = -1.535   # farm intercept for under-count (if used)
    beta1_Scount::Float64 = 0.039      # weight slope for under-count (if used)
    mean_fish_weight_kg::Float64 = 1.5 # mean fish weight
    W0::Float64 = 0.1                  # weight centering (kg)

    # Sea lice bounds and discretization
    sea_lice_bounds::Tuple{Float64, Float64} = (0.0, 10.0)
    initial_bounds::Tuple{Float64, Float64} = (0.0, 0.25)
    initial_mean::Float64 = 0.13

    # Sampling parameters
    adult_sd::Float64 = 0.1
    rng::AbstractRNG = Random.GLOBAL_RNG

    # Discretization
    discretization_step::Float64 = 0.1
    sea_lice_range::Vector{Float64} = collect(sea_lice_bounds[1]:discretization_step:sea_lice_bounds[2])
    initial_range::Vector{Float64} = collect(initial_bounds[1]:discretization_step:initial_bounds[2])

    # Discretizers
    lice_lindisc::LinearDiscretizer = LinearDiscretizer(collect(sea_lice_bounds[1]:discretization_step:(sea_lice_bounds[2]+discretization_step)))
    biomass_step::Float64 = length(biomass_range) > 1 ? biomass_range[2] - biomass_range[1] : 170.0
    biomass_lindisc::LinearDiscretizer = LinearDiscretizer(collect(biomass_bounds[1]:biomass_step:(biomass_bounds[2]+biomass_step)))
    catdisc::CategoricalDiscretizer = CategoricalDiscretizer([NoTreatment, MechanicalTreatment, ChemicalTreatment, ThermalTreatment])

    # Dimension sizes (precomputed for indexing)
    n_lice::Int = length(sea_lice_range)
    n_season::Int = 4
    n_biomass::Int = length(biomass_range)
    n_cooldown::Int = 2

    # Initial state
    production_start_week::Int = 34  # Week 34 ≈ late August → Summer (season 2)
end

# -------------------------
# POMDPs.jl Interface
# -------------------------
function POMDPs.states(pomdp::SeaLicePOMDP)
    state_list = SeaLiceState[]
    sizehint!(state_list, pomdp.n_lice * pomdp.n_season * pomdp.n_biomass * pomdp.n_cooldown)
    for c in 0:1
        for (bi, b) in enumerate(pomdp.biomass_range)
            for s in 1:4
                for l in pomdp.sea_lice_range
                    push!(state_list, SeaLiceState(l, s, b, c))
                end
            end
        end
    end
    return state_list
end

function POMDPs.observations(pomdp::SeaLicePOMDP)
    obs_list = SeaLiceObservation[]
    sizehint!(obs_list, pomdp.n_lice * pomdp.n_season * pomdp.n_biomass * pomdp.n_cooldown)
    for c in 0:1
        for (bi, b) in enumerate(pomdp.biomass_range)
            for s in 1:4
                for l in pomdp.sea_lice_range
                    push!(obs_list, SeaLiceObservation(l, s, b, c))
                end
            end
        end
    end
    return obs_list
end

POMDPs.actions(pomdp::SeaLicePOMDP) = [NoTreatment, MechanicalTreatment, ChemicalTreatment, ThermalTreatment]
POMDPs.discount(pomdp::SeaLicePOMDP) = pomdp.discount_factor
POMDPs.isterminal(pomdp::SeaLicePOMDP, s::SeaLiceState) = false

# Multi-dimensional state index: lice (fastest) → season → biomass → cooldown (slowest)
function POMDPs.stateindex(pomdp::SeaLicePOMDP, s::SeaLiceState)
    li = encode(pomdp.lice_lindisc, s.SeaLiceLevel)
    si = s.Season
    bi = encode(pomdp.biomass_lindisc, s.BiomassLevel)
    ci = s.Cooldown + 1  # 0→1, 1→2
    return li + pomdp.n_lice * ((si - 1) + pomdp.n_season * ((bi - 1) + pomdp.n_biomass * (ci - 1)))
end

POMDPs.actionindex(pomdp::SeaLicePOMDP, a::Action) = encode(pomdp.catdisc, a)

function POMDPs.obsindex(pomdp::SeaLicePOMDP, o::SeaLiceObservation)
    li = encode(pomdp.lice_lindisc, o.SeaLiceLevel)
    si = o.Season
    bi = encode(pomdp.biomass_lindisc, o.BiomassLevel)
    ci = o.Cooldown + 1
    return li + pomdp.n_lice * ((si - 1) + pomdp.n_season * ((bi - 1) + pomdp.n_biomass * (ci - 1)))
end

# -------------------------
# Transition
# -------------------------
function POMDPs.transition(pomdp::SeaLicePOMDP, s::SeaLiceState, a::Action)

    # === 1. Cooldown (deterministic) ===
    next_cooldown = (a != NoTreatment) ? 1 : 0

    # === 2. Season transition (stochastic cycling) ===
    # Each season lasts ~13 weeks; P(advance) = 1/13 per week
    next_season_same = s.Season
    next_season_advance = (s.Season % 4) + 1
    p_stay = 12.0 / 13.0
    p_advance = 1.0 / 13.0

    # === 3. Sea lice transition (temperature from season) ===
    temp = get_temperature(pomdp.season_midpoint_weeks[s.Season], pomdp.location)

    # Apply treatment in raw space
    rf_a, rf_m, rf_s = get_treatment_effectiveness(a)
    adult_raw = max(s.SeaLiceLevel * (1 - rf_a), 0.0)
    motile_raw = max(adult_raw * pomdp.motile_ratio * (1 - rf_m), 0.0)
    sessile_raw = max(adult_raw * pomdp.sessile_ratio * (1 - rf_s), 0.0)

    # Predict next adult level using the biological drift
    pred_adult_raw, _, _ = predict_next_abundances(
        adult_raw, motile_raw, sessile_raw,
        temp, pomdp.location, pomdp.reproduction_rate,
    )

    # Discretize lice distribution
    μ_lice = clamp(pred_adult_raw, pomdp.sea_lice_bounds...)
    lice_dist = truncated(Normal(μ_lice, pomdp.adult_sd), pomdp.sea_lice_bounds...)
    lice_probs = discretize_values(lice_dist, pomdp.sea_lice_range)

    # === 4. Biomass transition (growth model) ===
    # Growth depends on current lice level and season temperature
    lice_growth_factor = 1.0 / (1.0 + exp(5.0 * (s.SeaLiceLevel - 0.5)))
    temp_factor = 1.0 + pomdp.temp_sensitivity * (temp - 10.0)
    growth = pomdp.k_biomass * max(temp_factor, 0.0) * lice_growth_factor * (pomdp.biomass_max - s.BiomassLevel)
    mortality = (pomdp.nat_mort_rate + get_treatment_mortality_rate(a)) * s.BiomassLevel
    μ_biomass = clamp(s.BiomassLevel + growth - mortality, pomdp.biomass_bounds...)
    biomass_dist = truncated(Normal(μ_biomass, pomdp.biomass_sd), pomdp.biomass_bounds...)
    biomass_probs = discretize_values(biomass_dist, pomdp.biomass_range)

    # === 5. Build joint SparseCat ===
    out_states = SeaLiceState[]
    out_probs = Float64[]

    for (next_season, season_prob) in ((next_season_same, p_stay), (next_season_advance, p_advance))
        for (li, lice_level) in enumerate(pomdp.sea_lice_range)
            lp = lice_probs[li]
            lp < 1e-2 && continue
            for (bi, biomass_level) in enumerate(pomdp.biomass_range)
                bp = biomass_probs[bi]
                bp < 1e-2 && continue
                push!(out_states, SeaLiceState(lice_level, next_season, biomass_level, next_cooldown))
                push!(out_probs, season_prob * lp * bp)
            end
        end
    end

    # Normalize
    total = sum(out_probs)
    if total > 0
        out_probs ./= total
    end

    return SparseCat(out_states, out_probs)
end

# -------------------------
# Observation function: sea lice measured with negative binomial;
# season, biomass, and cooldown are perfectly observed.
# -------------------------
function POMDPs.observation(pomdp::SeaLicePOMDP, a::Action, sp::SeaLiceState)

    # If full observability solver, return the exact state as the observation
    if pomdp.full_observability_solver
        all_obs = POMDPs.observations(pomdp)
        target_idx = obsindex(pomdp, SeaLiceObservation(sp.SeaLiceLevel, sp.Season, sp.BiomassLevel, sp.Cooldown))
        probs = zeros(length(all_obs))
        probs[target_idx] = 1.0
        return SparseCat(all_obs, probs)
    end

    # (Optional) under-counting correction
    p_scount = if pomdp.use_underreport
        η = pomdp.beta0_Scount_f + pomdp.beta1_Scount*(pomdp.mean_fish_weight_kg - pomdp.W0)
        logistic(η)
    else
        1.0
    end

    # NB parameters on TOTAL counts over n_sample fish
    μ_total_adult = max(1e-12, pomdp.n_sample * p_scount * sp.SeaLiceLevel)
    k = max(1e-9, pomdp.n_sample * pomdp.rho_nb)
    r, p = nb_params_from_mean_k(μ_total_adult, k)
    nb = NegativeBinomial(r, p)

    # Build CDF at each lice grid value
    lice_range = pomdp.sea_lice_range
    n_lice = length(lice_range)
    cdfs = Vector{Float64}(undef, n_lice)
    @inbounds for i in 1:n_lice
        adult_threshold = clamp(lice_range[i], pomdp.sea_lice_bounds...)
        total_adult_threshold = max(0, floor(Int, pomdp.n_sample * adult_threshold))
        ci = cdf(nb, total_adult_threshold)
        cdfs[i] = isfinite(ci) ? ci : 0.0
    end
    cdfs[end] = 1.0

    # Convert CDFs to bin probabilities
    lice_probs = Vector{Float64}(undef, n_lice)
    past = 0.0
    @inbounds for i in 1:n_lice
        ci = cdfs[i]
        if !isfinite(ci) || ci < past
            ci = past
        end
        lice_probs[i] = max(ci - past, 0.0)
        past = ci
    end

    # Normalize (and fallback if degenerate)
    lice_total = sum(lice_probs)
    if lice_total <= 0 || !isfinite(lice_total)
        idx = findmin(abs.(lice_range .- μ_total_adult / pomdp.n_sample))[2]
        lice_probs .= 0.0
        lice_probs[idx] = 1.0
    else
        lice_probs ./= lice_total
    end

    # Build observation SparseCat: only lice dimension varies;
    # season, biomass, cooldown are perfectly observed from sp
    out_obs = SeaLiceObservation[]
    out_probs = Float64[]
    for (i, lice_level) in enumerate(lice_range)
        lp = lice_probs[i]
        lp < 1e-2 && continue
        push!(out_obs, SeaLiceObservation(lice_level, sp.Season, sp.BiomassLevel, sp.Cooldown))
        push!(out_probs, lp)
    end

    # Normalize after pruning
    total = sum(out_probs)
    if total > 0
        out_probs ./= total
    end

    return SparseCat(out_obs, out_probs)
end

# -------------------------
# Reward
# -------------------------
function POMDPs.reward(pomdp::SeaLicePOMDP, s::SeaLiceState, a::Action)

    λ_trt, λ_reg, λ_bio, λ_health, λ_sea_lice = pomdp.reward_lambdas

    adult_level = s.SeaLiceLevel

    # === 1. DIRECT TREATMENT COSTS ===
    treatment_cost = get_treatment_cost(a)

    # === 2. REGULATORY PENALTY (MNOK) — season-dependent threshold ===
    # Spring uses stricter 0.2 limit during smolt migration
    reg_limit = pomdp.season_regulation_limits[s.Season]
    regulatory_penalty = adult_level > reg_limit ? pomdp.regulatory_violation_cost_MNOK : 0.0

    # === 3. BIOMASS LOSS (MNOK) — from actual biomass state ===
    # 3a. Mortality loss (acute) — scales with actual farm biomass
    mortality_biomass_loss = get_treatment_mortality_rate(a) * s.BiomassLevel

    # 3b. Growth reduction from sea lice (chronic)
    if adult_level > 0.5
        lice_severity = min((adult_level - 0.5) / 1.5, 1.0)  # 0 at 0.5, 1.0 at 2.0+
        growth_biomass_loss = s.BiomassLevel * pomdp.max_growth_loss_fraction * lice_severity
    else
        growth_biomass_loss = 0.0
    end

    biomass_loss_MNOK = (mortality_biomass_loss + growth_biomass_loss) * pomdp.salmon_price_MNOK_per_tonne

    # === 4. FISH HEALTH (MNOK) — cooldown-dependent stress multiplier ===
    # Back-to-back treatments compound fish stress
    base_stress = get_fish_disease(a)
    stress_multiplier = s.Cooldown == 1 ? pomdp.cooldown_stress_multiplier : 1.0
    fish_health_MNOK = base_stress * stress_multiplier * pomdp.welfare_cost_MNOK

    # === 5. SEA LICE BURDEN (MNOK) — chronic parasite damage ===
    sea_lice_MNOK = adult_level * (1.0 + 0.2 * max(0, adult_level - 0.5)) * pomdp.chronic_lice_cost_MNOK

    # === TOTAL REWARD (all components in MNOK) ===
    return -(
        λ_trt * treatment_cost +
        λ_reg * regulatory_penalty +
        λ_bio * biomass_loss_MNOK +
        λ_health * fish_health_MNOK +
        λ_sea_lice * sea_lice_MNOK
    )
end

# -------------------------
# Observation-based reward function (all components in MNOK)
# POMDPs.jl prefers reward(m, s, a, sp, o) when available during simulation.
# The regulatory penalty is based on the observed (sampled) lice count,
# matching real-world enforcement where regulators assess compliance
# from sampled counts, not the true underlying population.
# -------------------------
function POMDPs.reward(pomdp::SeaLicePOMDP, s::SeaLiceState, a::Action, sp::SeaLiceState, o::SeaLiceObservation)

    λ_trt, λ_reg, λ_bio, λ_health, λ_sea_lice = pomdp.reward_lambdas

    adult_level = s.SeaLiceLevel

    # === 1. DIRECT TREATMENT COSTS (MNOK) ===
    treatment_cost = get_treatment_cost(a)

    # === 2. REGULATORY PENALTY (MNOK) — based on OBSERVATION, season-dependent threshold ===
    reg_limit = pomdp.season_regulation_limits[s.Season]
    regulatory_penalty = o.SeaLiceLevel > reg_limit ? pomdp.regulatory_violation_cost_MNOK : 0.0

    # === 3. BIOMASS LOSS (MNOK) — from actual biomass state ===
    mortality_biomass_loss = get_treatment_mortality_rate(a) * s.BiomassLevel

    if adult_level > 0.5
        lice_severity = min((adult_level - 0.5) / 1.5, 1.0)
        growth_biomass_loss = s.BiomassLevel * pomdp.max_growth_loss_fraction * lice_severity
    else
        growth_biomass_loss = 0.0
    end

    biomass_loss_MNOK = (mortality_biomass_loss + growth_biomass_loss) * pomdp.salmon_price_MNOK_per_tonne

    # === 4. FISH HEALTH (MNOK) — cooldown-dependent stress multiplier ===
    base_stress = get_fish_disease(a)
    stress_multiplier = s.Cooldown == 1 ? pomdp.cooldown_stress_multiplier : 1.0
    fish_health_MNOK = base_stress * stress_multiplier * pomdp.welfare_cost_MNOK

    # === 5. SEA LICE BURDEN (MNOK) — chronic parasite damage ===
    sea_lice_MNOK = adult_level * (1.0 + 0.2 * max(0, adult_level - 0.5)) * pomdp.chronic_lice_cost_MNOK

    # === TOTAL REWARD (all components in MNOK) ===
    return -(
        λ_trt * treatment_cost +
        λ_reg * regulatory_penalty +
        λ_bio * biomass_loss_MNOK +
        λ_health * fish_health_MNOK +
        λ_sea_lice * sea_lice_MNOK
    )
end

# -------------------------
# Initial state
# -------------------------
function POMDPs.initialstate(pomdp::SeaLicePOMDP)

    # Sea lice distribution (uncertain)
    dist = truncated(Normal(pomdp.initial_mean, pomdp.adult_sd), pomdp.sea_lice_bounds...)
    lice_probs = discretize_values(dist, pomdp.sea_lice_range)

    # Fixed initial values for other dimensions
    initial_season = week_to_season(pomdp.production_start_week)
    initial_biomass = pomdp.biomass_range[1]  # Start of production cycle
    initial_cooldown = 0

    # Build distribution: only lice dimension is uncertain
    out_states = SeaLiceState[]
    out_probs = Float64[]
    for (i, lice_level) in enumerate(pomdp.sea_lice_range)
        if lice_probs[i] > 1e-2
            push!(out_states, SeaLiceState(lice_level, initial_season, initial_biomass, initial_cooldown))
            push!(out_probs, lice_probs[i])
        end
    end

    return SparseCat(out_states, out_probs)
end

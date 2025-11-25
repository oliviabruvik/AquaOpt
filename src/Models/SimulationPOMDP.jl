

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
using Random

# -------------------------
# State, Observation, Action
# -------------------------
"State representing the predicted sea lice level the following week and the current sea lice level."
struct EvaluationState
	SeaLiceLevel::Float64 # The predicted adult sea lice level the following week (without treatment)
    Adult::Float64 # The adult sea lice level this week
    Motile::Float64 # The motile sea lice level this week
    Sessile::Float64 # The sessile sea lice level this week
    Temperature::Float64 # The mean temperature (°C) over the last 7 days at the farm (based on approximately daily measurements)
    ProductionWeek::Int64 # The number of weeks since production start
    AnnualWeek::Int64 # The week of the year
    NumberOfFish::Int64 # The number of fish in the pen
    AvgFishWeight::Float64 # The average weight of the fish in the pen (kg)
    Salinity::Float64 # The salinity of the water (psu)
end

"Observation representing the predicted observed sea lice level the following week and the current observed sea lice level."
struct EvaluationObservation
	SeaLiceLevel::Float64 # The observed predicted adult sea lice level the following week (without treatment)
    Adult::Float64 # The observed adult sea lice level this week
    Motile::Float64 # The observed motile sea lice level this week
    Sessile::Float64 # The observed sessile sea lice level this week
    Temperature::Float64 # The observed mean temperature (°C) over the last 7 days at the farm (based on approximately daily measurements)
    ProductionWeek::Int64 # The number of weeks since production start
    AnnualWeek::Int64 # The week of the year
    NumberOfFish::Int64 # The number of fish in the pen
    AvgFishWeight::Float64 # The average weight of the fish in the pen (kg)
    Salinity::Float64 # The salinity of the water (psu)
end

# -------------------------
# SimulationPOMDP Definition
# -------------------------
"Sea lice simulation POMDP with growth dynamics and treatment effects."
@with_kw struct SeaLiceSimPOMDP <: POMDP{EvaluationState, Action, EvaluationObservation}

    # Parameters
	lambda::Float64 = 0.5
    reward_lambdas::Vector{Float64} = [0.5, 0.5, 0.0, 0.0, 0.0] # [treatment, regulatory, biomass, health, sea_lice]
	costOfTreatment::Float64 = 10.0
	growthRate::Float64 = 0.3
    reproduction_rate::Float64 = 2.0  # Number of sessile larvae produced per adult female per week
    discount_factor::Float64 = 0.95

    # Regulation parameters
    regulation_limit::Float64 = 0.5

    # Weight parameters
    w_max::Float64 = 5.0                        # asymptotic harvest weight (kg)
    k_growth::Float64 = 0.01                    # weekly von Bertalanffy/logistic-like rate
    temp_sensitivity::Float64 = 0.03            # temperature effect on growth rate (°C)

    # Fish count parameters
    nat_mort_rate::Float64 = 0.0008             # weekly natural mortality fraction
    trt_mort_bump::Float64 = 0.005              # extra mortality fraction in treatment weeks
    harvest_schedule::Function = (week::Int)->0 # fish harvested this week (kg)
    move_in_fn::Function = (week::Int)->0       # fish moved in (kg)
    move_out_fn::Function = (week::Int)->0      # fish moved out (kg)

    # Parameters from Aldrin et al. 2023
    n_sample::Int = 20                         # number of fish counted (ntc)
    ρ_adult::Float64 = 0.175                   # aggregation parameter for adults
    ρ_motile::Float64 = 0.187                   # aggregation parameter for motile
    ρ_sessile::Float64 = 0.037                  # aggregation parameter for sessile

    # Under-reporting parameters from Aldrin et al. 2023
    use_underreport::Bool = false      # toggle logistic under-count correction
    beta0_Scount_f::Float64 = -1.535           # farm-specific intercept (can vary by farm)
    beta1_Scount::Float64 = 0.039              # common weight slope
    mean_fish_weight_kg::Float64 = 1.5         # mean fish weight (kg)
    W0::Float64 = 0.1 # kg

    # Bounds
    sea_lice_bounds::Tuple{Float64, Float64} = (0.0, 10.0)
    initial_bounds::Tuple{Float64, Float64} = (0.0, 0.25)
    weight_bounds::Tuple{Float64, Float64} = (0.0, 7.0)
    number_of_fish_bounds::Tuple{Float64, Float64} = (0.0, 200000.0)

    # Means: Empirical from Aldrin et al. 2023
    adult_mean::Float64 = 0.13
    motile_mean::Float64 = 0.47
    sessile_mean::Float64 = 0.12

    # Transition and Observation Noise
    adult_obs_sd::Float64 = 0.17
    motile_obs_sd::Float64 = 0.327
    sessile_obs_sd::Float64 = 0.10
    adult_sd::Float64 = 0.1
    motile_sd::Float64 = 0.29
    sessile_sd::Float64 = 0.16
    temp_sd::Float64 = 0.3
    weight_sd::Float64 = 0.05
    number_of_fish_sd::Float64 = 0.0

    # Distributions
    adult_dist::Distribution = Normal(0, adult_sd)
    motile_dist::Distribution = Normal(0, motile_sd)
    sessile_dist::Distribution = Normal(0, sessile_sd)
    temp_dist::Distribution = Normal(0, temp_sd)

    # Sampling parameters
    rng::AbstractRNG = Random.GLOBAL_RNG
    production_start_week::Int64 = 34 # Week 34 is approximately July 1st

    # Location for biological and temperature model
    location::String = "south" # "north", "west", or "south"
end

# -------------------------
# POMDPs.jl Interface
# -------------------------
POMDPs.actions(pomdp::SeaLiceSimPOMDP) = [NoTreatment, MechanicalTreatment, ChemicalTreatment, ThermalTreatment]
POMDPs.discount(pomdp::SeaLiceSimPOMDP) = pomdp.discount_factor
POMDPs.isterminal(pomdp::SeaLiceSimPOMDP, s::EvaluationState) = false


# -------------------------
# Transition function: the current sea lice level and the predicted sea lice level the following week
# are affected by the treatment and growth rate. The predicted sea lice level the following week will 
# have an additional e^r term because it is a week later.
# -------------------------
function POMDPs.transition(pomdp::SeaLiceSimPOMDP, s::EvaluationState, a::Action)
    ImplicitDistribution(pomdp, s, a) do pomdp, s, a, rng

        # Apply treatment effects
        rf_a, rf_m, rf_s = get_treatment_effectiveness(a)
        treated_adult = s.Adult * (1 - rf_a)
        treated_motile = s.Motile * (1 - rf_m)
        treated_sessile = s.Sessile * (1 - rf_s)

        # Predict next abundances using biological model with reproduction
        next_adult, next_motile, next_sessile = predict_next_abundances(
            treated_adult, treated_motile, treated_sessile, s.Temperature, pomdp.location, pomdp.reproduction_rate
        )

        # Update temperature for next week
        next_annual_week = (s.AnnualWeek + 1) % 52
        next_temp = get_temperature(next_annual_week, pomdp.location)

        # # Biomass loss
        # # TODO: add a function to calculate the biomass loss
        # # https://ars.els-cdn.com/content/image/1-s2.0-S0044848623005239-mmc1.pdf
        # lambda_mech = 1.210

        # Add noise
        next_adult = next_adult + rand(rng, pomdp.adult_dist)
        next_motile = next_motile + rand(rng, pomdp.motile_dist)
        next_sessile = next_sessile + rand(rng, pomdp.sessile_dist)
        next_temp = next_temp + rand(rng, pomdp.temp_dist)

        # Clamp the sea lice levels to be positive and within the bounds of the SeaLicePOMDP
        next_adult = max(next_adult, 0.0)
        next_motile = max(next_motile, 0.0)
        next_sessile = max(next_sessile, 0.0)
        next_pred = clamp(next_adult, pomdp.sea_lice_bounds...)
 
        # Calculate the weight transition based on a von Bertalanffy / logistic-like weekly update
        # W_{t+1} = W_t + (k0 * f(T) * f(lice)) * (w_max - W_t)
        # Sea lice reduce growth: higher lice = slower growth
        # Using logistic function: growth_reduction = 1 / (1 + exp(5 * (adult - 0.5)))
        # At adult=0.5 (regulation limit): ~50% growth reduction
        # At adult=1.0: ~99% growth reduction
        lice_growth_factor = 1.0 / (1.0 + exp(5.0 * (s.Adult - 0.5)))

        k0_base = pomdp.k_growth * (1.0 + pomdp.temp_sensitivity * (s.Temperature - 10.0))
        k0 = max(k0_base * lice_growth_factor, 0.0)
        next_average_weight = s.AvgFishWeight + k0 * (pomdp.w_max - s.AvgFishWeight)
        next_average_weight = next_average_weight * (1 - get_weight_loss(a))
        next_average_weight = clamp(next_average_weight, pomdp.weight_bounds...)

        # Calculate the next number of fish
        survival_rate = (1 - pomdp.nat_mort_rate) * (1 - get_treatment_mortality_rate(a))
        harvest = pomdp.harvest_schedule(s.ProductionWeek)
        move_in = pomdp.move_in_fn(s.ProductionWeek)
        move_out = pomdp.move_out_fn(s.ProductionWeek)
        survived_fish = round(Int, floor(s.NumberOfFish * survival_rate))
        next_number_of_fish = survived_fish + move_in - move_out - harvest
        next_number_of_fish = clamp(next_number_of_fish, pomdp.number_of_fish_bounds...)

        return EvaluationState(
            next_pred, # SeaLiceLevel
            next_adult, # Adult
            next_motile, # Motile
            next_sessile, # Sessile
            next_temp, # Temperature
            s.ProductionWeek + 1, # ProductionWeek
            (s.AnnualWeek + 1) % 52, # AnnualWeek
            next_number_of_fish, # NumberOfFish
            next_average_weight, # AvgFishWeight
            s.Salinity, # Salinity is constant
        )
    end
end

# -------------------------
# Observation function: the current counts of adults, motiles, and sessiles are measured
# with a negative binomial distribution. The predicted adult level the following week is
# calculated based on the observed adult count this week and the temperature.
# -------------------------
function POMDPs.observation(pomdp::SeaLiceSimPOMDP, a::Action, sp::EvaluationState)
    ImplicitDistribution(pomdp, sp, a) do pomdp, sp, a, rng

        # (Optional) under-counting correction like p^Scount_ftc
        # paper uses (W - 0.1) with W in kg
        # Accounts for the fact that sessiles are harder to count than motiles and adults
        p_scount = if pomdp.use_underreport
            η = pomdp.beta0_Scount_f + pomdp.beta1_Scount * (sp.AvgFishWeight - pomdp.W0)
            logistic(η)  # ∈ (0,1)
        else
            1.0
        end

        # Expected total lice counted across n_sample fish
        μ_total_adult = max(1e-12, pomdp.n_sample * p_scount * sp.Adult)
        μ_total_motile = max(1e-12, pomdp.n_sample * p_scount * sp.Motile)
        μ_total_sessile = max(1e-12, pomdp.n_sample * p_scount * sp.Sessile)

        # Dispersion parameters for the NB distributions
        # Aggregation (NB size) scales with sample size (n * ρ)
        k_adult = max(1e-9, pomdp.n_sample * pomdp.ρ_adult)
        k_motile = max(1e-9, pomdp.n_sample * pomdp.ρ_motile)
        k_sessile = max(1e-9, pomdp.n_sample * pomdp.ρ_sessile)

        # Converts the biological parameters (mean μ, dispersion k) to the mathematical
        # parameters (r, p) that Julia's NB distribution expects
        r_adult, p_adult = nb_params_from_mean_k(μ_total_adult, k_adult)
        r_motile, p_motile = nb_params_from_mean_k(μ_total_motile, k_motile)
        r_sessile, p_sessile = nb_params_from_mean_k(μ_total_sessile, k_sessile)

        # Sample the total counts randomly from the NB distributions
        total_adult = rand(rng, NegativeBinomial(r_adult, p_adult))
        total_motile = rand(rng, NegativeBinomial(r_motile, p_motile))
        total_sessile = rand(rng, NegativeBinomial(r_sessile, p_sessile))

        # Calculate the observed sea lice levels
        observed_adult = total_adult / pomdp.n_sample
        observed_motile = total_motile / pomdp.n_sample
        observed_sessile = total_sessile / pomdp.n_sample

        # Calculate the observed temperature
        observed_temperature = rand(rng, Normal(sp.Temperature, pomdp.temp_sd))

        # Clamp the sea lice levels to be positive
        observed_adult = max(observed_adult, 0.0)
        observed_motile = max(observed_motile, 0.0)
        observed_sessile = max(observed_sessile, 0.0)

        # Predict the next adult sea lice level based on the current state and temperature
        pred_adult, _, _ = predict_next_abundances(observed_adult, observed_motile, observed_sessile, observed_temperature, pomdp.location, pomdp.reproduction_rate)

        # Clamp the sea lice levels to be positive and within the bounds of the SeaLicePOMDP
        pred_adult = clamp(pred_adult, pomdp.sea_lice_bounds...)

        # Observe the number of fish and average weight
        observed_number_of_fish = round(Int, floor(sp.NumberOfFish + rand(rng, Normal(0, pomdp.number_of_fish_sd))))
        observed_number_of_fish = clamp(observed_number_of_fish, pomdp.number_of_fish_bounds...)
        observed_average_weight = sp.AvgFishWeight + rand(rng, Normal(0, pomdp.weight_sd))
        observed_average_weight = clamp(observed_average_weight, pomdp.weight_bounds...)

        return EvaluationObservation(
            pred_adult, # SeaLiceLevel
            observed_adult, # Adult
            observed_motile, # Motile
            observed_sessile, # Sessile
            observed_temperature, # Temperature
            sp.ProductionWeek, # ProductionWeek is fully observable
            sp.AnnualWeek, # AnnualWeek is fully observable
            observed_number_of_fish,
            observed_average_weight,
            sp.Salinity, # Salinity is fully observable
        )
    end
end

# -------------------------
# Reward function
# Penalizes:
# - Treatment costs (direct operational costs)
# - Regulatory non-compliance (exponential penalty above limit)
# - Biomass loss (mortality only - growth reduction is in transition dynamics)
# - Fish health impacts (treatment side effects)
# - Sea lice burden (chronic parasite damage)
# -------------------------
function POMDPs.reward(pomdp::SeaLiceSimPOMDP, s::EvaluationState, a::Action, sp::EvaluationState)

    λ_trt, λ_reg, λ_bio, λ_health, λ_sea_lice = pomdp.reward_lambdas

    # === 1. DIRECT TREATMENT COSTS ===
    treatment_cost = get_treatment_cost(a)

    # === 2. REGULATORY PENALTY (exponential above limit) ===
    # Reflects escalating consequences: fines, production caps, license restrictions
    if s.Adult > pomdp.regulation_limit
        excess_ratio = s.Adult / pomdp.regulation_limit
        # Penalty grows as: 100 * (excess%)^2 * ratio
        # At 0.6 (20% over): 100 * 0.2^2 * 1.2 = 4.8
        # At 0.75 (50% over): 100 * 0.5^2 * 1.5 = 37.5
        # At 1.0 (100% over): 100 * 1.0^2 * 2.0 = 200
        regulatory_penalty = 100.0 # * ((excess_ratio - 1.0)^2) * excess_ratio
    else
        regulatory_penalty = 0.0
    end

    # === 3. BIOMASS LOSS (expected growth shortfall) ===
    next_biomass = biomass_tons(sp)

    # Predict what biomass should be next week if no mortality occurs.
    # 1) Project fish count forward with only natural/treatment survival.
    ideal_survival_rate = 1 - pomdp.nat_mort_rate
    expected_fish = max(s.NumberOfFish * ideal_survival_rate, 0.0)
    
    # 2) Project average weight using the same growth rule as the transition.
    k0_base = pomdp.k_growth * (1.0 + pomdp.temp_sensitivity * (s.Temperature - 10.0))
    ideal_k0 = max(k0_base, 0.0)
    expected_weight = s.AvgFishWeight + ideal_k0 * (pomdp.w_max - s.AvgFishWeight)
    expected_weight = clamp(expected_weight, pomdp.weight_bounds...)
    expected_biomass = biomass_tons(expected_weight, expected_fish)

    total_biomass_loss = max(expected_biomass - next_biomass, 0.0)

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
    sea_lice_penalty = s.Adult * (1.0 + 0.2 * max(0, s.Adult - 0.5))

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
# Initial state distribution
# -------------------------
function POMDPs.initialstate(pomdp::SeaLiceSimPOMDP)
    ImplicitDistribution(pomdp) do pomdp, rng

        # Initial temperature upon production start
        temperature = get_temperature(pomdp.production_start_week, pomdp.location) + rand(rng, pomdp.temp_dist)

        # Initial sea lice level upon production start
        adult = pomdp.adult_mean + rand(rng, pomdp.adult_dist)
        motile  = pomdp.motile_mean + rand(rng, pomdp.motile_dist)
        sessile = pomdp.sessile_mean + rand(rng, pomdp.sessile_dist)

        # Next week's predicted adult sea lice level
        pred_adult, _, _ = predict_next_abundances(adult, motile, sessile, temperature, pomdp.location, pomdp.reproduction_rate)

        # Clamp the sea lice levels to be positive
        adult = max(adult, 0.0)
        motile = max(motile, 0.0)
        sessile = max(sessile, 0.0)
        pred_adult = clamp(pred_adult, pomdp.sea_lice_bounds...)

        return EvaluationState(
            pred_adult, # Predicted adult sea lice level the following week
            adult, # Adult
            motile, # Motile
            sessile, # Sessile
            temperature, # Temperature
            1, # ProductionWeek
            pomdp.production_start_week, # AnnualWeek
            200000, # NumberOfFish at the start of production
            0.1, # AvgFishWeight at the start of production (initial weight)
            30.0, # Salinity at the start of production
        )
    end
end

# -------------------------
# Shared types used across multiple files
# -------------------------
using POMDPs
using Distributions
using Random
using Parameters

# Action enum
@enum Action NoTreatment Treatment ThermalTreatment

# Action configuration struct
@with_kw struct ActionConfig
    # Action type
    action::Action
    
    # Cost information
    cost::Float64  # Cost per treatment
    
    # Treatment effectiveness (reduction factors)
    adult_reduction::Float64      # Reduction in adult sea lice (0.0 = no effect, 1.0 = 100% reduction)
    motile_reduction::Float64     # Reduction in motile sea lice
    sessile_reduction::Float64    # Reduction in sessile sea lice
    
    # Metadata
    name::String                  # Human-readable name
    description::String           # Description of the treatment
    duration_days::Int            # How long the treatment lasts
    frequency_limit::Int          # Maximum treatments per year (0 = no limit)
    fish_disease::Float64         # Penalty for fish health
    mortality_rate::Float64       # Mortality rate
    weight_loss::Float64          # Weight loss
    regulatory_penalty::Float64   # Regulatory penalty
end

# Treatment effectiveness based on results from Table 5 of 
# https://ars.els-cdn.com/content/image/1-s2.0-S0044848623005239-mmc1.pdf
if !isdefined(Main, :ACTION_CONFIGS)
    const ACTION_CONFIGS = Dict(
        NoTreatment => ActionConfig(
            action = NoTreatment,
            cost = 0.0,
            adult_reduction = 0.0,
            motile_reduction = 0.0,
            sessile_reduction = 0.0,
            name = "No Treatment",
            description = "No treatment applied",
            duration_days = 0,
            frequency_limit = 0,
            fish_disease = 0.0,
            mortality_rate = 0.0,
            weight_loss = 0.0,
            regulatory_penalty = 100.0
        ),
        
        Treatment => ActionConfig(
            action = Treatment,
            cost = 10.0,  # MNOK per treatment
            adult_reduction = 0.75,   # 75% reduction
            motile_reduction = 0.84,  # 84% reduction
            sessile_reduction = 0.74, # 74% reduction
            name = "Chemical Treatment",
            description = "Standard chemical treatment for sea lice control",
            duration_days = 7,
            frequency_limit = 4,  # Maximum 4 treatments per year
            fish_disease = 10.0,
            mortality_rate = 0.1,
            weight_loss = 0.0,
            regulatory_penalty = 100.0
        ),
        
        ThermalTreatment => ActionConfig(
            action = ThermalTreatment,
            cost = 13.0,  # MNOK per treatment (higher cost) Increased from 12
            adult_reduction = 0.88,   # 88% reduction
            motile_reduction = 0.87,  # 87% reduction
            sessile_reduction = 0.70, # 70% reduction
            name = "Thermal Treatment",
            description = "Thermal treatment for sea lice control",
            duration_days = 5,
            frequency_limit = 6,  # Maximum 6 treatments per year
            fish_disease = 15.0,
            mortality_rate = 0.1,
            weight_loss = 0.0,
            regulatory_penalty = 100.0
        )
    )
end

# Utility functions for working with ActionConfig
function get_action_config(action::Action)
    return ACTION_CONFIGS[action]
end

function get_treatment_cost(action::Action)
    @assert action in keys(ACTION_CONFIGS)
    return get_action_config(action).cost
end

function get_regulatory_penalty(action::Action)
    @assert action in keys(ACTION_CONFIGS)
    return get_action_config(NoTreatment).regulatory_penalty
end

function get_fish_disease(action::Action)
    @assert action in keys(ACTION_CONFIGS)
    return get_action_config(action).fish_disease
end

function get_treatment_mortality_rate(action::Action)
    @assert action in keys(ACTION_CONFIGS)
    return get_action_config(action).mortality_rate
end

function get_weight_loss(action::Action)
    @assert action in keys(ACTION_CONFIGS)
    return get_action_config(action).weight_loss
end

function get_treatment_effectiveness(action::Action)
    @assert action in keys(ACTION_CONFIGS)
    config = get_action_config(action)
    return (config.adult_reduction, config.motile_reduction, config.sessile_reduction)
end

function get_stochastic_treatment_effectiveness(action::Action, rng::AbstractRNG=Random.GLOBAL_RNG)
    @assert action in keys(ACTION_CONFIGS)
    config = get_action_config(action)
    
    # Add some variability to the treatment effectiveness
    # Sample from normal distributions around the base effectiveness values
    adult_eff = rand(rng, Normal(config.adult_reduction, 0.05))
    motile_eff = rand(rng, Normal(config.motile_reduction, 0.05))
    sessile_eff = rand(rng, Normal(config.sessile_reduction, 0.05))
    
    # Clamp to valid range [0, 1]
    adult_eff = clamp(adult_eff, 0.0, 1.0)
    motile_eff = clamp(motile_eff, 0.0, 1.0)
    sessile_eff = clamp(sessile_eff, 0.0, 1.0)
    
    return (adult_eff, motile_eff, sessile_eff)
end

# -------------------------
# Harvest schedule
# -------------------------
function harvest_schedule(week::Int)
    return 0
end

# -------------------------
# Move in function
# -------------------------
function move_in_fn(week::Int)
    return 0
end

# -------------------------
# Move out function
# -------------------------
function move_out_fn(week::Int)
    return 0
end

# Export the Action enum and related functions
export Action, NoTreatment, Treatment, ThermalTreatment
export ActionConfig, ACTION_CONFIGS
export get_action_config, get_treatment_cost, get_treatment_effectiveness, get_stochastic_treatment_effectiveness, get_regulatory_penalty, get_fish_disease, get_treatment_mortality_rate, get_weight_loss

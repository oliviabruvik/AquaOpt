using Distributions
using POMDPs
using Random

# Include shared types
include("SharedTypes.jl")

# -------------------------
# Utility functions for discretizing distributions
# -------------------------
function discretize_distribution(dist::Distribution, space::Any)

    probs = zeros(length(space))
    past_cdf = 0.0
    
    # Add safety check for NaN values
    for (i, s) in enumerate(space)
        curr_cdf = cdf(dist, s.SeaLiceLevel)
        
        # Check for NaN values
        if isnan(curr_cdf)
            @warn("NaN detected in CDF calculation. Distribution: $dist, State: $s")
            # Fallback to uniform distribution
            return fill(1.0/length(space), length(space))
        end
        
        probs[i] = curr_cdf - past_cdf
        past_cdf = curr_cdf
    end

    # Check for negative probabilities (shouldn't happen with CDF)
    if any(x -> x < -1e-10, probs)
        @warn("Negative probabilities detected, clamping to 0")
        probs = max.(probs, 0.0)
    end

    if sum(probs) < 0.01
        println("type of space: $(typeof(space))")
        println("dist: $dist")
        @warn("Probs over distribution very small")
    end

    probs = normalize(probs, 1)

    # Check that the probs sum to 1.0
    try
        @assert sum(probs) ≈ 1.0
    catch
        println("probs: $probs")
        println("sum(probs): $(sum(probs))")
        error("Probs do not sum to 1.0")
    end

    return probs
end

# -------------------------
# Temperature model: Return estimated weekly sea surface temperature (°C) for Norwegian salmon farms.
# -------------------------
function get_temperature(annual_week)
    # return 9.0
    T_mean = 12.0 #9.0      # average annual temperature (°C)
    T_amp = 4.5       # amplitude (°C)
    peak_week = 27      # aligns peak with July (week ~27)
    return T_mean + T_amp * cos(2π * (annual_week - peak_week) / 52)
end

# -------------------------
# Predict the next adult sea lice level based on the current state and temperature
# Development rate model: Return the expected development rate based on the temperature.
# Based on A salmon lice prediction model, Stige et al. 2025.
# https://www.sciencedirect.com/science/article/pii/S0167587724002915
# -------------------------
function predict_next_abundances(adult, motile, sessile, temp)
    
    # Weekly survival probabilities from Table 1 of Stige et al. 2025.
    s1 = 0.49  # sessile
    s2 = 2.3   # sessile → motile scaling
    s3 = 0.88  # motile
    s4 = 0.61  # adult

    # TODO: Remember to change this back to the original values
    # Original: d1_val = 1 / (1 + exp(-(-2.4 + 0.37 * (temp - 9))))
    # Original: d2_val = 1 / (1 + exp(-(-2.1 + 0.037 * (temp - 9))))
    d1_val = 1 / (1 + exp(-(-1.5 + 0.5 * (temp - 9))))  # Faster sessile to motile
    d2_val = 1 / (1 + exp(-(-1.0 + 0.1 * (temp - 9))))  # Faster motile to adult

    # Get the predicted sea lice levels
    pred_sessile = s1 * sessile
    pred_motile = s3 * (1 - d2_val) * motile + s2 * d1_val * sessile
    pred_adult = s4 * adult + d2_val * 0.5 * (s3 + s4) * motile

    # Add an influx of sessiles from the sea
    pred_sessile += 0.01

    # Clamp the sea lice levels to be positive
    pred_adult = max(pred_adult, zero(pred_adult))
    pred_motile = max(pred_motile, zero(pred_motile))
    pred_sessile = max(pred_sessile, zero(pred_sessile))

    return pred_adult, pred_motile, pred_sessile
end

# -------------------------
# Helper: logistic and NB parameterization
# -------------------------
logistic(x) = 1 / (1 + exp(-x))

# -------------------------
# Given NB "size" k and mean μ, Distributions.jl uses NegativeBinomial(r, p) with mean = r*(1-p)/p. 
# Solve p = k/(k+μ), r = k.
# -------------------------
nb_params_from_mean_k(μ, k) = (r = k, p = k/(k + μ))
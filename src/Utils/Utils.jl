using Distributions
using POMDPs
using Random



# -------------------------
# Utility functions for discretizing distributions
# -------------------------
function discretize_distribution(dist::Distribution, space::Any)

    n = length(space)
    probs = zeros(n)

    # Each state represents the CENTER of a bin
    # We calculate bin edges as midpoints between consecutive state centers
    for (i, s) in enumerate(space)
        if i == 1
            # First bin: from -∞ to midpoint between states[1] and states[2]
            if n > 1
                upper = (space[1].SeaLiceLevel + space[2].SeaLiceLevel) / 2
                curr_cdf = cdf(dist, upper)
            else
                # Only one state - gets all probability
                curr_cdf = 1.0
            end

            # Check for NaN values
            if isnan(curr_cdf)
                @warn("NaN detected in CDF calculation. Distribution: $dist, State: $s")
                return fill(1.0/n, n)
            end

            probs[i] = curr_cdf

        elseif i == n
            # Last bin: from midpoint to +∞
            lower = (space[n-1].SeaLiceLevel + space[n].SeaLiceLevel) / 2
            lower_cdf = cdf(dist, lower)

            # Check for NaN values
            if isnan(lower_cdf)
                @warn("NaN detected in CDF calculation. Distribution: $dist, State: $s")
                return fill(1.0/n, n)
            end

            probs[i] = 1.0 - lower_cdf

        else
            # Middle bins: from midpoint below to midpoint above
            lower = (space[i-1].SeaLiceLevel + space[i].SeaLiceLevel) / 2
            upper = (space[i].SeaLiceLevel + space[i+1].SeaLiceLevel) / 2
            lower_cdf = cdf(dist, lower)
            upper_cdf = cdf(dist, upper)

            # Check for NaN values
            if isnan(lower_cdf) || isnan(upper_cdf)
                @warn("NaN detected in CDF calculation. Distribution: $dist, State: $s")
                return fill(1.0/n, n)
            end

            probs[i] = upper_cdf - lower_cdf
        end
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
function get_temperature(annual_week, location="north")
    params = get_location_params(location)
    return params.T_mean + params.T_amp * cos(2π * (annual_week - params.peak_week) / 52)
end

# -------------------------
# Predict the next adult sea lice level based on the current state and temperature
# Development rate model: Return the expected development rate based on the temperature.
# Based on A salmon lice prediction model, Stige et al. 2025.
# https://www.sciencedirect.com/science/article/pii/S0167587724002915
# -------------------------
function predict_next_abundances(adult, motile, sessile, temp, location="north", reproduction_rate=2.0)

    # Get location-specific parameters
    params = get_location_params(location)

    # Calculate development rates using logistic functions
    d1_val = 1 / (1 + exp(-(params.d1_intercept + params.d1_temp_coef * (temp - params.d1_temp_offset))))  # Sessile to motile
    d2_val = 1 / (1 + exp(-(params.d2_intercept + params.d2_temp_coef * (temp - params.d2_temp_offset))))  # Motile to adult

    # Get the predicted sea lice levels using weekly survival probabilities
    pred_sessile = params.s1_sessile * sessile
    pred_motile = params.s3_motile * (1 - d2_val) * motile + params.s2_scaling * d1_val * sessile
    pred_adult = params.s4_adult * adult + d2_val * 0.5 * (params.s3_motile + params.s4_adult) * motile

    # Add reproduction: adult females produce new sessile larvae
    pred_sessile += reproduction_rate * adult

    # Add an influx of sessiles from the sea (external larval pressure)
    pred_sessile += params.external_influx

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
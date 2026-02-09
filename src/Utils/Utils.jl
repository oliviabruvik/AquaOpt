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
    d1_val = 1 / (1 + exp(-(params.d1_intercept + params.d1_temp_coef * (temp - params.T_mean))))  # Sessile to motile
    d2_val = 1 / (1 + exp(-(params.d2_intercept + params.d2_temp_coef * (temp - params.T_mean))))  # Motile to adult

    # Get the predicted sea lice levels using weekly survival probabilities
    pred_sessile = params.s1_sessile * sessile
    pred_motile = params.s3_motile * (1 - d2_val) * motile + params.s2_scaling * d1_val * sessile
    pred_adult = params.s4_adult * adult + d2_val * 0.5 * (params.s3_motile + params.s4_adult) * motile

    # Add reproduction: adult females produce new sessile larvae
    pred_sessile += reproduction_rate * adult

    # Add an influx of sessiles from the sea (external larval pressure)
    pred_sessile += params.external_influx

    # Clamp the sea lice levels to be positive and cap adult lice at 30 per fish
    pred_adult = clamp(pred_adult, 0.0, 10.0)
    pred_motile = max(pred_motile, zero(pred_motile))
    pred_sessile = max(pred_sessile, zero(pred_sessile))

    return pred_adult, pred_motile, pred_sessile
end

# -------------------------
# Discretize a continuous distribution over a sorted vector of bin centers.
# Returns a probability vector (one per bin center) that sums to 1.
# Unlike discretize_distribution, this works on plain Float64 values
# rather than state objects with a .SeaLiceLevel field.
# -------------------------
function discretize_values(dist::Distribution, values::AbstractVector{<:Real})
    n = length(values)
    probs = zeros(n)

    for i in 1:n
        if i == 1
            if n > 1
                upper = (values[1] + values[2]) / 2
                probs[i] = cdf(dist, upper)
            else
                probs[i] = 1.0
            end
        elseif i == n
            lower = (values[n-1] + values[n]) / 2
            probs[i] = 1.0 - cdf(dist, lower)
        else
            lower = (values[i-1] + values[i]) / 2
            upper = (values[i] + values[i+1]) / 2
            probs[i] = cdf(dist, upper) - cdf(dist, lower)
        end
        if !isfinite(probs[i])
            probs[i] = 0.0
        end
    end

    probs = max.(probs, 0.0)
    total = sum(probs)
    if total > 0
        probs ./= total
    else
        probs .= 1.0 / n
    end

    return probs
end

# -------------------------
# Map annual week to season index: 1=Spring, 2=Summer, 3=Autumn, 4=Winter
# Spring (weeks 13-25): smolt migration period with stricter lice limits
# -------------------------
function week_to_season(annual_week::Int)
    w = mod(annual_week, 52)
    if 13 <= w <= 25
        return 1  # Spring
    elseif 26 <= w <= 38
        return 2  # Summer
    elseif 39 <= w <= 51
        return 3  # Autumn
    else
        return 4  # Winter (weeks 0-12)
    end
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
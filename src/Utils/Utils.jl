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

    if location == "north"
        T_mean = 12.0 # average annual temperature (°C)
        T_amp = 4.5 # amplitude (°C)
        peak_week = 27 # aligns peak with July (week ~27)
    elseif location == "west"
        T_mean = 16.0 # average annual temperature (°C)
        T_amp = 4.5
        peak_week = 27
    elseif location == "south"
        T_mean = 20.0
        T_amp = 4.5
        peak_week = 27
    else
        error("Invalid location: $location. Must be 'north', 'west', or 'south'")
    end
    return T_mean + T_amp * cos(2π * (annual_week - peak_week) / 52)
end

# -------------------------
# Predict the next adult sea lice level based on the current state and temperature
# Development rate model: Return the expected development rate based on the temperature.
# Based on A salmon lice prediction model, Stige et al. 2025.
# https://www.sciencedirect.com/science/article/pii/S0167587724002915
# -------------------------
function predict_next_abundances(adult, motile, sessile, temp, location="north", reproduction_rate=2.0)

    if location == "north"
        d1_val = 1 / (1 + exp(-(-2.4 + 0.37 * (temp - 9))))  # Sessile to motile
        d2_val = 1 / (1 + exp(-(-2.1 + 0.037 * (temp - 9))))  # Motile to adult
    elseif location == "west"
        d1_val = 1 / (1 + exp(-(-1.5 + 0.5 * (temp - 16))))  # Faster sessile to motile
        d2_val = 1 / (1 + exp(-(-1.0 + 0.1 * (temp - 16))))  # Faster motile to adult
    elseif location == "south"
        d1_val = 1 / (1 + exp(-(-1.5 + 0.5 * (temp - 20))))  # Faster sessile to motile
        d2_val = 1 / (1 + exp(-(-1.0 + 0.1 * (temp - 20))))  # Faster motile to adult
    else
        error("Invalid location: $location. Must be 'north', 'west', or 'south'")
    end

    # Weekly survival probabilities from Table 1 of Stige et al. 2025.
    if location == "north"
        s1 = 0.49  # sessile
        s2 = 2.3   # sessile → motile scaling
        s3 = 0.88  # motile
        s4 = 0.61  # adult
    elseif location == "west"
        s1 = 0.6  # sessile
        s2 = 3.0   # sessile → motile scaling
        s3 = 0.95  # motile
        s4 = 0.70  # adult
    elseif location == "south"
        s1 = 0.8  # sessile
        s2 = 5.0   # sessile → motile scaling
        s3 = 0.99  # motile
        s4 = 0.99  # adult
    end

    # Get the predicted sea lice levels
    pred_sessile = s1 * sessile
    pred_motile = s3 * (1 - d2_val) * motile + s2 * d1_val * sessile
    pred_adult = s4 * adult + d2_val * 0.5 * (s3 + s4) * motile

    # Add reproduction: adult females produce new sessile larvae
    # This is the key biological process that was missing!
    pred_sessile += reproduction_rate * adult

    # Add an influx of sessiles from the sea (external larval pressure)
    if location == "north"
        pred_sessile += 0.01
    elseif location == "west"
        pred_sessile += 0.1
    elseif location == "south"
        pred_sessile += 0.2
    end

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
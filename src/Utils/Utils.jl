using Distributions
using POMDPs

# -------------------------
# Discretized Normal Sampling Utility
# -------------------------
# "Returns a transition distribution that ensures all states are reachable."
# function discretized_normal_points(mean::Float64, mdp::SeaLiceMDP, skew::Bool=false)

#     if skew
#         skew_factor = 2.0
#         dist = SkewNormal(mean, mdp.sampling_sd, skew_factor)
#     else
#         dist = Normal(mean, mdp.sampling_sd)
#     end

#     # Calculate the points
#     points = mean .+ mdp.sampling_sd .* [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3]
#     if skew
#         points = points .* (1 + 0.3 * skew_factor)
#     end
    
#     # Ensure points are within the range of the sea lice range
#     points = clamp.(points, mdp.min_lice_level, mdp.max_lice_level)

#     # Calculate and normalize the probabilities
#     probs = pdf.(dist, points)
#     probs = normalize(probs, 1)
    
#     # Ensure we have at least one transition
#     if length(points) == 0 || sum(probs) == 0
#         # Fallback to mean state
#         points = [mean]
#         probs = [1.0]
#     end

#     return points, probs
# end


# -------------------------
# Discretized Normal Distribution
# -------------------------
function discretize_distribution(dist::Distribution, space::Any, skew::Bool=false)

    probs = zeros(length(space))
    past_cdf = 0.0
    for (i, s) in enumerate(space)
        curr_cdf = cdf(dist, s.SeaLiceLevel)
        probs[i] = curr_cdf - past_cdf
        past_cdf = curr_cdf
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


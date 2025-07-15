include("../Algorithms/Evaluation.jl")
include("../Algorithms/Policies.jl")
include("../Algorithms/Simulation.jl")
include("../Data/Cleaning.jl")
include("../Models/SeaLicePOMDP.jl")
include("../Utils/Config.jl")

# Import required packages
using Logging
using DiscreteValueIteration
using GridInterpolations
using NativeSARSOP #: SARSOPSolver
using SARSOP
using POMDPs
using POMDPTools
using Plots: plot, plot!, scatter, scatter!, heatmap, heatmap!, histogram, histogram!, savefig
using LocalFunctionApproximation
using LocalApproximationValueIteration
using Dates

# ----------------------------
# Define algorithms
# ----------------------------

# No treat policy
no_treat_algo = Algorithm(solver_name="NoTreatment_Policy")

# Random policy
random_algo = Algorithm(solver_name="Random_Policy")

# Heuristic policy
heuristic_algo = Algorithm(solver_name="Heuristic_Policy", heuristic_config=HeuristicConfig(
    raw_space_threshold=5.0,
    belief_threshold=0.5,
    rho=0.8
))

# Native SARSOP
native_sarsop_algo = Algorithm(
    solver=NativeSARSOP.SARSOPSolver(max_time=5.0, verbose=true),
    solver_name="SARSOP_Policy",
)

# C++ SARSOPSolver
nus_sarsop_algo = Algorithm(
    solver=SARSOP.SARSOPSolver(
        timeout=5,
        verbose=false,
        policy_filename=joinpath("src", "Tests", "NUS_SARSOP.policy"),
        pomdp_filename=joinpath("src", "Tests", "NUS_SARSOP.pomdp")
    ),
    solver_name="NUS_SARSOP_Policy",
)

# Value iteration
vi_algo = Algorithm(solver=ValueIterationSolver(max_iterations=10), solver_name="VI_Policy")

# QMDP
qmdp_algo = Algorithm(solver=QMDPSolver(max_iterations=10), solver_name="QMDP_Policy")

# ----------------------------
# Run test function
# ----------------------------
function run_test(;log_space=true, skew=false, test_name="test", algo=nothing)

    @info "Running $test_name for $(algo.solver_name)."

    # Define experiment configuration
    exp_name = joinpath("Tests", string(Dates.now(), "_", test_name))

    EXPERIMENT_CONFIG = ExperimentConfig(
        num_episodes=2,
        steps_per_episode=10,
        log_space=log_space,
        skew=skew,
        experiment_name=exp_name,
        verbose=false,
        step_through=false,
        lambda_values=[0.4, 0.6],
        sarsop_max_time=5.0,
        VI_max_iterations=10,
        QMDP_max_iterations=10,
    )

    # Define heuristic configuration
    HEURISTIC_CONFIG = HeuristicConfig(
        raw_space_threshold=EXPERIMENT_CONFIG.heuristic_threshold,
        belief_threshold=EXPERIMENT_CONFIG.heuristic_belief_threshold,
        rho=EXPERIMENT_CONFIG.heuristic_rho
    )

   try
       results = test_optimizer(algo, EXPERIMENT_CONFIG)
   catch e
       @error "Error running test $test_name"
       @error e
   end

end

# ----------------------------
# Run tests for specific algo
# ----------------------------
function run_specific_tests(algo)
    @info "Running tests for $algo.solver_name \n"
    run_test(log_space=true, skew=false, test_name="test_log_space_non_skew", algo=algo)
    run_test(log_space=false, skew=false, test_name="test_non_log_space_non_skew", algo=algo)
    run_test(log_space=true, skew=true, test_name="test_log_space_skew", algo=algo)
    run_test(log_space=false, skew=true, test_name="test_non_log_space_skew", algo=algo)
end

# ----------------------------
# Run non-log space non-skew tests
# ----------------------------
function run_non_log_space_non_skew_tests(algos)
    @info "Running non-log space non-skew tests \n"
    for algo in algos
        run_test(log_space=false, skew=false, test_name="test_non_log_space_non_skew", algo=algo)
    end
end

# ----------------------------
# Run log space non-skew tests
# ----------------------------
function run_log_space_non_skew_tests(algos)
    @info "Running log space non-skew tests \n"
    for algo in algos
        run_test(log_space=true, skew=false, test_name="test_log_space_non_skew", algo=algo)
    end
end

# ----------------------------
# Run log space skew tests
# ----------------------------
function run_log_space_skew_tests(algos)
    @info "Running log space skew tests \n"
    for algo in algos
        run_test(log_space=true, skew=true, test_name="test_log_space_skew", algo=algo)
    end
end

# ----------------------------
# Run non-log space skew tests
# ----------------------------
function run_non_log_space_skew_tests(algos)
    @info "Running non-log space skew tests \n"
    for algo in algos
        run_test(log_space=false, skew=true, test_name="test_non_log_space_skew", algo=algo)
    end
    @info "Done running non-log space skew tests \n"
end

# ----------------------------
# Main function
# ----------------------------
policy_algos = [no_treat_algo, random_algo, heuristic_algo]
solver_algos = [native_sarsop_algo, nus_sarsop_algo, qmdp_algo, vi_algo]
all_algos = [policy_algos..., solver_algos...]

run_non_log_space_non_skew_tests(all_algos)
run_log_space_non_skew_tests(policy_algos)
# run_log_space_skew_tests(all_algos)
# run_non_log_space_skew_tests(all_algos)

module Optimize

using POMDPs, QuickPOMDPs, POMDPTools, QMDP, SARSOP, POMDPModels, DataFrames, JLD2

function sarsop_optimize(df::DataFrame)

    # initialize POMDP
    pomdp = TigerPOMDP()

    # initialize the solver
    solver = SARSOPSolver()

    # run the solver
    policy = solve(solver, pomdp)
    save("results/policies/tiger_policy.jld2", "policy", policy)

    return policy
end

end
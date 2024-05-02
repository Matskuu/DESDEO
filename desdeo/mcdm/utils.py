
from desdeo.problem import (
    Problem,
    pareto_navigator_test_problem
)
from desdeo.tools.generics import CreateSolverType, SolverResults
from desdeo.tools.scalarization import (
    add_asf_diff,
    add_asf_nondiff,
    add_scalarization_function
)
from desdeo.tools.utils import guess_best_solver

def project(
    problem: Problem,
    approximated_solution: dict[str, float],
    scalarization_function: str | None = None,
    create_solver: CreateSolverType | None = None
) -> SolverResults:
    if scalarization_function:
        problem_with_scalarization, target = add_scalarization_function(problem, scalarization_function, "target")
    else:
        problem_with_scalarization, target = add_asf_nondiff(problem, "target", approximated_solution)

    init_solver = guess_best_solver(problem) if create_solver is None else create_solver
    solver = init_solver(problem_with_scalarization)
    return solver.solve(target)

if __name__ == "__main__":
    problem = pareto_navigator_test_problem()
    reference_point = {"f_1": 0.35, "f_2": -0.51, "f_3": -26.26}
    res = project(problem, reference_point)
    print(reference_point, res.optimal_objectives)

    reference_point = {"f_1": -0.89, "f_2": 2.91, "f_3": -24.98}
    res = project(problem, reference_point)
    print(reference_point, res.optimal_objectives)

    reference_point = {"f_1": -0.32, "f_2": 2.33, "f_3": -27.85}
    res = project(problem, reference_point)
    print(reference_point, res.optimal_objectives)

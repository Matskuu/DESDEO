
import numpy as np
from typing import Callable
from desdeo.problem import (
    GenericEvaluator,
    Problem,
    dtlz2,
    pareto_navigator_test_problem
)
from desdeo.tools.generics import CreateSolverType, SolverResults
from desdeo.tools.scalarization import (
    add_asf_diff,
    add_asf_nondiff,
    add_scalarization_function
)
from desdeo.tools.utils import guess_best_solver
from desdeo.tools.pyomo_solver_interfaces import PyomoIpoptSolver

def project(
    problem: Problem,
    approximated_solution: dict[str, float],
    create_solver: CreateSolverType | None = None,
    scalarization_function: Callable | None = None,
    args: list | None = None
) -> SolverResults:
    if scalarization_function:
        problem_with_scalarization, target = scalarization_function(*args)
    else:
        problem_with_scalarization, target = add_asf_nondiff(problem, "target", approximated_solution)

    init_solver = guess_best_solver(problem) if create_solver is None else create_solver
    solver = init_solver(problem_with_scalarization)
    return solver.solve(target)

if __name__ == "__main__":
    #problem = pareto_navigator_test_problem()
    problem = dtlz2(8, 3)

    #xs = {f"{var.symbol}": [0.5] for var in problem.variables}
    #evaluator = GenericEvaluator(problem)
    #res = evaluator.evaluate(xs)
    #reference_point = {f"{obj.symbol}": res[obj.symbol][0] for obj in problem.objectives}

    reference_point = {"f_1": 0.4, "f_2": 0.7, "f_3": 0.5}
    #args = [problem, "target", reference_point]
    #print(args)
    #res = project(problem, reference_point, add_asf_nondiff, args)

    res = project(problem, reference_point, PyomoIpoptSolver)
    """
    print(reference_point, res.optimal_objectives, res.optimal_variables)
    values_sq = 0
    for obj in problem.objectives:
        values_sq += res.optimal_objectives[obj.symbol]
    print(values_sq)
    """

from dataclasses import dataclass, field
import cvxpy as cp


@dataclass(slots=True)
class LinearExpr:
    '''
        Compiled affine expression of moment variables
    '''
    terms: dict[int, float|complex] = field(default_factory=dict)
    const: float|complex = 0


@dataclass(slots=True)
class SDPData:
    var_cpx: bool
    n_vars: int
    M: list[list[LinearExpr]]
    constraints: list[LinearExpr]
    objective: LinearExpr


class SDPSolver:
    def __init__(self):
        self.problem = None
        self.objective = None
        self.status = None

    def _compile_expr(self, expr: LinearExpr) -> cp.Expression:
        assert self.m is not None
        e = cp.Constant(expr.const)
        for idx, coeff in expr.terms.items():
            e += coeff * self.m[idx]
        return e

    def build(self, data: SDPData):
        self.m = cp.Variable(data.n_vars, complex=data.var_cpx, name='vars')
        # [TODO] vectorize by pre-computing coeff matrix for each moment variable
        self.M = cp.bmat([
            [self._compile_expr(expr) for expr in row]
            for row in data.M
        ])
        self.constraints = [self.M >> 0]

        for expr in data.constraints:
            e = self._compile_expr(expr)
            self.constraints.append(e == 0)

        # objective must be real
        obj = self._compile_expr(data.objective)
        if obj.is_complex():
            obj = cp.real(obj)
        self.objective = cp.Minimize(obj)
        self.problem = cp.Problem(self.objective, self.constraints)
        return self.problem

    def solve(self, solver: str = 'SCS', **kwargs):
        if self.problem is None:
            raise ValueError('problem has not been built')
        value = self.problem.solve(solver=solver, **kwargs)
        self.status = self.problem.status
        return value

    def summary(self):
        assert self.problem is not None and self.status is not None
        return {
            'n_vars': self.m.shape[0],
            'n_constraints': len(self.constraints),
            'status': self.problem.status,
            'value': self.problem.value,
            'solver_name': self.problem.solver_stats.solver_name,
            'solve_time': self.problem.solver_stats.solve_time,
            'num_iters': self.problem.solver_stats.num_iters,
        }


if __name__ == '__main__':

    from ising.ising import IsingParams, IsingCompiler

    params=IsingParams(L=16, J=1., h=1.)
    basis0 = ['I', 'X', 'Y', 'Z', 'ZZ']
    basis1 = basis0 + ['XX', 'YY', 'XZ', 'XY', 'YX', 'YZ', 'ZX', 'ZY']
    basis2 = basis0 + ['XX', 'YY', 'XXX', 'YYY', 'ZZZ']

    compiler = IsingCompiler(params=params)
    compiler.compile(
        basis=basis1 # basis0, basis1, basis2
    )
    print(compiler.summary())

    solver = SDPSolver()
    solver.build(compiler.sdp_data())
    solver.solve(solver='SCS', eps=1e-4, max_iters=10_000)

    print(solver.problem.value/params.L)
    print(solver.problem.status)
    print(solver.summary())

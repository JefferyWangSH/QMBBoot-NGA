import time

from nga import NGAParams, NGARunner
from nga_scheduler import BaseScheduler
from compiler.hubbard import HubbardCompiler, HubbardParams, build_basis_reprs
from operators.majorana import MajoranaMonomial

if __name__ == '__main__':

    params = HubbardParams(L=10, t=1., U=4., n_particles=10)
    eps = 1e-5
    n_steps = 10

    runner = NGARunner(
        compiler=HubbardCompiler(params),
        basis_reprs=build_basis_reprs(
            params.L,
            max_degree=4,
            max_support=1,
            max_diameter=0,
        ),
        required_basis_reprs=[
            MajoranaMonomial.from_str(params.L, monomial).trans_canon_rep
            for monomial in ['I', '0u+', '0u-', '0u+ 0u-', '0d+', '0d-', '0d+ 0d-']
        ],
        nga_params=NGAParams(
            solver_backend='SCS_SSNAL',
            solver_kwargs={'eps': eps},
            drop_null_tol=eps,
            grow_null_tol=eps,
            max_drop_leverage=5e-2,
        ),
        scheduler=BaseScheduler(
            net_growth_min=1,
            net_growth_cap=8,
            drop_cap=8,
            reentry_penalty=0.5,
        ),
    )

    # the trajectory is driven by cold solves; from step 1 on, the same SDPData is also
    # solved warm from export_warm() of the previous step (before the cold solve, so that
    # the NGA proposals use the cold solution and the trajectory does not depend on warm start)
    warm = None
    totals = {'cold': [0, 0.], 'warm': [0, 0.]}
    for step in range(n_steps):
        runner.build()

        warmed = warm is not None
        line_warm = '-'
        if warmed:
            start = time.perf_counter()
            value_w = runner.solver.solve(warm_start=warm, **runner.nga_params.solver_kwargs)
            wall_w = time.perf_counter() - start
            iters_w = runner.solver.summary()['num_iters']
            line_warm = f'{iters_w:6d} it {wall_w:7.2f} s {runner.solver.status}'

        runner.solve()
        record = runner.record
        warm = runner.solver.export_warm()
        if warmed:
            line_warm += f' dE {value_w - record.value:+.1e}'
            for name, iters, wall in [
                ('cold', record.n_iters, record.time['solve_time']),
                ('warm', iters_w, wall_w),
            ]:
                totals[name][0] += iters
                totals[name][1] += wall

        print(
            f"step {step:2d} | vars {record.n_vars:5d} | "
            f"cold {record.n_iters:6d} it {record.time['solve_time']:7.2f} s {record.status} "
            f"E {record.value:.8f} | warm {line_warm}",
            flush=True,
        )

        runner.diagonalize()
        runner.propose_pruning()
        runner.propose_growth()
        runner.update()

    (iters_cold, time_cold), (iters_warm, time_warm) = totals['cold'], totals['warm']
    print(
        f'steps 1-{n_steps - 1}: cold {iters_cold} it {time_cold:.1f} s | warm {iters_warm} it {time_warm:.1f} s | '
        f'warm/cold iters {iters_warm / iters_cold:.2f}, time speedup {time_cold / time_warm:.2f}x'
    )

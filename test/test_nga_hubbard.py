from nga import NGARunner
from hubbard.hubbard import HubbardCompiler, HubbardParams, MajoranaMonomial, load_basis_reprs

if __name__ == '__main__':

    params = HubbardParams(L=10, t=1., U=4., n_particles=10)
    basis_reprs = load_basis_reprs(
        params.L,
        max_degree=4,
        max_support=1,
        max_diameter=0,
    )
    required_basis_reprs = [
        MajoranaMonomial.from_str(params.L, monomial)
        for monomial in ['I', '0u+', '0u-', '0u+ 0u-', '0d+', '0d-', '0d+ 0d-']
    ]

    runner = NGARunner(
        compiler=HubbardCompiler(params),
        basis_reprs=basis_reprs,
        required_basis_reprs=required_basis_reprs,
        solver_backend='SCS',
        null_tol=1e-5,
        descendant_tol=1e-5,
        max_drop_leverage=5e-2,
        min_net_growth_per_step=1,
        max_net_growth_per_step=8,
        max_drop_per_step=4,
    )

    n_steps = 16
    for i in range(n_steps):
        summary, record = runner.step()
        print(i, summary)
        print(i, record)

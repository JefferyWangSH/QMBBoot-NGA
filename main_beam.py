import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

from nga import NGAParams
from nga_beam import NGABeamRunner

@dataclass
class BeamModelAdapter:
    model_type: str
    model_params: object
    compiler: object
    nga_params: object
    scheduler: object
    basis: list
    required_basis: list
    start_step: int
    records: list
    events: dict
    drop_counts: dict

    @staticmethod
    def build_compiler_kwargs(config_data: dict, build_obj, build_obs):
        obj_config = config_data.get('objective', {})
        obj_name = obj_config.get('obj', 'hamil')
        obj_op = None if obj_name == 'hamil' else build_obj(obj_name)

        obj_sense = obj_config.get('obj_sense', 'min')
        if obj_sense not in ('min', 'max'):
            raise ValueError(f'unknown objective sense: {obj_sense}')
        if obj_name == 'hamil' and obj_sense != 'min':
            raise ValueError('Hamiltonian objective only supports min')

        e_lb = obj_config.get('e_lb')
        e_ub = obj_config.get('e_ub')
        if obj_name == 'hamil' and (e_lb is not None or e_ub is not None):
            raise ValueError('Hamiltonian objective does not use energy bounds e_lb or e_ub')
        if obj_name != 'hamil' and (e_lb is None or e_ub is None):
            raise ValueError('observable objective requires energy bounds e_lb and e_ub')

        obs_ops = {name: build_obs(name) for name in config_data.get('observables', [])}
        return {
            'obj_op': obj_op,
            'obj_sense': obj_sense,
            'e_lb': e_lb,
            'e_ub': e_ub,
            'obs_ops': obs_ops,
        }

    @classmethod
    def load(cls, config: Path, resume: Path | None = None):
        data = json.loads(Path(config).read_text())

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ model
        model_type = data['model_type']
        model_config = data['model']
        basis_config = data['basis']

        if model_type == 'hubbard':
            from compiler.hubbard import HubbardCompiler, HubbardParams, build_basis_reprs, build_hamil, build_szz, build_double_occ
            from operators.majorana import MajoranaMonomial
            model_params = HubbardParams(**model_config)

            def build_obj(name):
                if name == 'hamil':
                    return build_hamil(model_params)
                if name == 'szz':
                    return build_szz(model_params, 1)
                if name == 'double_occ':
                    return build_double_occ(model_params)
                raise ValueError(f'unknown Hubbard objective: {name}')

            def build_obs(name):
                if name == 'hamil':
                    return build_hamil(model_params)
                if name == 'szz':
                    return [build_szz(model_params, r) for r in range(model_params.L//2 + 1)]
                if name == 'double_occ':
                    return build_double_occ(model_params)
                raise ValueError(f'unknown Hubbard observable: {name}')

            compiler = HubbardCompiler(
                model_params,
                **cls.build_compiler_kwargs(data, build_obj, build_obs),
            )
            build_basis = lambda strings: [
                MajoranaMonomial.from_str(model_params.L, s).trans_canon_rep
                for s in strings
            ]
            build_initial_basis = lambda initial: (
                build_basis(initial)
                if isinstance(initial, list)
                else build_basis_reprs(model_params.L, **initial)
            )
            required_basis = build_basis(basis_config['required'])

        elif model_type == 'hubbard_square':
            from compiler.hubbard_square import HubbardSquareCompiler, HubbardSquareParams, build_basis_reprs, build_hamil, build_szz, build_double_occ
            from operators.majorana_square import MajoranaMonomialSquare
            model_params = HubbardSquareParams(**model_config)

            def build_obj(name):
                if name == 'hamil':
                    return build_hamil(model_params)
                if name == 'szz':
                    return build_szz(model_params, 1, 0)
                if name == 'double_occ':
                    return build_double_occ(model_params)
                raise ValueError(f'unknown Hubbard square objective: {name}')

            def build_obs(name):
                if name == 'hamil':
                    return build_hamil(model_params)
                if name == 'szz':
                    return [
                        build_szz(model_params, dx, dy)
                        for dx in range(model_params.Lx)
                        for dy in range(model_params.Ly)
                    ]
                if name == 'double_occ':
                    return build_double_occ(model_params)
                raise ValueError(f'unknown Hubbard square observable: {name}')

            compiler = HubbardSquareCompiler(
                model_params,
                **cls.build_compiler_kwargs(data, build_obj, build_obs),
            )
            build_basis = lambda strings: [
                MajoranaMonomialSquare.from_str(model_params.Lx, model_params.Ly, s).trans_canon_rep
                for s in strings
            ]
            build_initial_basis = lambda initial: (
                build_basis(initial)
                if isinstance(initial, list)
                else build_basis_reprs(model_params.Lx, model_params.Ly, **initial)
            )
            required_basis = build_basis(basis_config['required'])

        elif model_type == 'ising':
            from compiler.ising import IsingCompiler, IsingParams, build_basis_reprs, build_hamil, build_zz
            model_params = IsingParams(**model_config)

            def build_obj(name):
                if name == 'hamil':
                    return build_hamil(model_params)
                if name == 'zz':
                    return build_zz(model_params, 1)
                raise ValueError(f'unknown Ising objective: {name}')

            def build_obs(name):
                if name == 'hamil':
                    return build_hamil(model_params)
                if name == 'zz':
                    return [build_zz(model_params, r) for r in range(model_params.L//2 + 1)]
                raise ValueError(f'unknown Ising observable: {name}')

            compiler = IsingCompiler(
                model_params,
                **cls.build_compiler_kwargs(data, build_obj, build_obs),
            )
            build_basis = lambda strings: build_basis_reprs(model_params.L, strings)
            build_initial_basis = build_basis
            required_basis = build_basis(basis_config['required'])

        elif model_type == 'heisenberg':
            from compiler.heisenberg import HeisenbergCompiler, HeisenbergParams, build_basis_reprs, build_hamil, build_szz
            model_params = HeisenbergParams(**model_config)

            def build_obj(name):
                if name == 'hamil':
                    return build_hamil(model_params)
                if name == 'szz':
                    return build_szz(model_params, 1)
                raise ValueError(f'unknown Heisenberg objective: {name}')

            def build_obs(name):
                if name == 'hamil':
                    return build_hamil(model_params)
                if name == 'szz':
                    return [build_szz(model_params, r) for r in range(model_params.L//2 + 1)]
                raise ValueError(f'unknown Heisenberg observable: {name}')

            compiler = HeisenbergCompiler(
                model_params,
                **cls.build_compiler_kwargs(data, build_obj, build_obs),
            )
            build_basis = lambda strings: build_basis_reprs(model_params.L, strings)
            build_initial_basis = build_basis
            required_basis = build_basis(basis_config['required'])

        else:
            raise ValueError(f'unknown model type: {model_type}')

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ NGA params
        nga_params = NGAParams(**data['nga'])

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ NGA scheduler
        scheduler_type = data['scheduler_type']
        scheduler_config = data['scheduler']

        if scheduler_type == 'beam_base':
            from nga_scheduler import BaseBeamScheduler
            scheduler = BaseBeamScheduler(**scheduler_config)

        elif scheduler_type == 'beam_rate':
            from nga_scheduler import RateBeamScheduler
            scheduler = RateBeamScheduler(**scheduler_config)

        else:
            raise ValueError(f'unknown scheduler type: {scheduler_type}')

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ resume
        if resume:
            resume_state = Path(resume) / 'state.json'
            resume_basis = Path(resume) / 'basis.json'
            resume_events = Path(resume) / 'events.json'
            state = json.loads(resume_state.read_text())
            if state['model_type'] != model_type or state['model'] != asdict(model_params):
                raise ValueError('resume state model parameters do not match config')
            basis = build_basis(json.loads(resume_basis.read_text()))
            start_step = state.get('step', -1) + 1
            records = state.get('records', [])
            events = json.loads(resume_events.read_text()) if resume_events.exists() else {'steps': []}
            drop_counts = {}
            for step_event in events.get('steps', []):
                for rep in build_basis(step_event.get('drop', [])):
                    key = rep.trans_canon
                    drop_counts[key] = drop_counts.get(key, 0) + 1
            # apply the last move
            if events.get('steps'):
                basis_map = {rep.trans_canon: rep.trans_canon_rep for rep in basis}
                for rep in build_basis(events['steps'][-1].get('drop', [])):
                    basis_map.pop(rep.trans_canon, None)
                for rep in build_basis(events['steps'][-1].get('grow', [])):
                    basis_map[rep.trans_canon] = rep.trans_canon_rep
                basis = list(basis_map.values())
        else:
            basis = build_initial_basis(basis_config['initial'])
            # merge required basis into initial basis by default
            basis_map = {rep.trans_canon: rep.trans_canon_rep for rep in basis}
            for rep in required_basis:
                basis_map[rep.trans_canon] = rep.trans_canon_rep
            basis = list(basis_map.values())

            start_step, records = 0, []
            events = {
                'initial_basis': [str(rep) for rep in basis],
                'steps': [],
            }
            drop_counts = {}

        return cls(model_type, model_params, compiler, nga_params, scheduler, basis, required_basis, start_step, records, events, drop_counts)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('config/hubbard_beam.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    parser.add_argument('--resume', type=Path, default=None)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--max-workers', type=int, default=1)
    args = parser.parse_args()

    adapter = BeamModelAdapter.load(args.config, args.resume)
    output_basis = args.output_dir / 'basis.json'
    output_state = args.output_dir / 'state.json'
    output_events = args.output_dir / 'events.json'

    runner = NGABeamRunner(
        compiler=adapter.compiler,
        basis_reprs=adapter.basis,
        required_basis_reprs=adapter.required_basis,
        nga_params=adapter.nga_params,
        scheduler=adapter.scheduler,
        drop_counts=adapter.drop_counts,
        max_workers=args.max_workers,
    )

    start_step = adapter.start_step
    events = adapter.events
    records = adapter.records

    t0 = time.perf_counter()
    for step in range(start_step, start_step + args.steps):
        step_basis = list(runner.basis_reprs)
        record = runner.step()

        events['steps'].append({
            'step': step,
            'selected': record.selected,
            'drop': [str(rep.trans_canon_rep) for rep in runner.to_drop],
            'grow': [str(rep.trans_canon_rep) for rep in runner.to_grow],
            'candidates': [
                {
                    key: [str(rep.trans_canon_rep) for rep in basis]
                    for key, basis in cand.move.items()
                }
                for cand in runner.cands
            ],
        })
        records.append(record.to_dict())

        output_state.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'step': step,
            'model_type': adapter.model_type,
            'model': asdict(adapter.model_params),
            'nga': asdict(adapter.nga_params),
            'scheduler': runner.scheduler.to_dict(),
            'records': records,
        }
        output_state.write_text(json.dumps(state, indent=2))

        output_events.parent.mkdir(parents=True, exist_ok=True)
        output_events.write_text(json.dumps(events, indent=2))

        output_basis.parent.mkdir(parents=True, exist_ok=True)
        output_basis.write_text(
            json.dumps([str(rep.trans_canon_rep) for rep in step_basis], indent=2)
        )

        print(
            f'[{step:03d}] '
            f'{record.status} | '
            f'selected={record.selected} | '
            f'value={record.value:.12f} | '
            f'basis={record.basis_reps} '
            f'vars={record.n_vars} '
            f'aff_rank={record.affine_rank} '
            f'psd_dims={sum(record.psd_dims)} | '
            f'to_drop={record.to_drop} '
            f'to_grow={record.to_grow} '
            f'net={record.net_growth} | '
            f'max_compile_s={max(cand["record"]["time"]["compile_time"] for cand in record.candidates):.1f} '
            f'max_build_s={max(cand["record"]["time"]["build_time"] for cand in record.candidates):.1f} '
            f'max_solve_s={max(cand["record"]["time"]["solve_time"] for cand in record.candidates):.1f} | '
            f'elapsed_s={time.perf_counter()-t0:.1f}',
            flush=True,
        )

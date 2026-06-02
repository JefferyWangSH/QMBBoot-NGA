import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

from nga import NGAParams, NGARunner

@dataclass
class ModelAdapter:
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

    @classmethod
    def load(cls, config: Path, resume: Path | None = None):
        data = json.loads(Path(config).read_text())

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ model
        model_type = data['model_type']
        model_config = data['model']
        basis_config = data['basis']

        if model_type == 'hubbard':
            from compiler.hubbard import HubbardCompiler, HubbardParams, build_basis_reprs
            from operators.majorana import MajoranaMonomial
            model_params = HubbardParams(**model_config)
            compiler = HubbardCompiler(model_params)
            parse_basis = lambda strings: [MajoranaMonomial.from_str(model_params.L, s).trans_canon_rep for s in strings]
            initial_basis = lambda initial: parse_basis(initial) if isinstance(initial, list) else build_basis_reprs(model_params.L, **initial)
            required_basis = parse_basis(basis_config['required'])

        elif model_type == 'ising':
            from compiler.ising import IsingCompiler, IsingParams, build_basis_reprs
            model_params = IsingParams(**model_config)
            compiler = IsingCompiler(model_params)
            parse_basis = lambda strings: build_basis_reprs(model_params.L, strings)
            initial_basis = lambda initial: build_basis_reprs(model_params.L, initial)
            required_basis = parse_basis(basis_config['required'])

        elif model_type == 'heisenberg':
            from compiler.heisenberg import HeisenbergCompiler, HeisenbergParams, build_basis_reprs
            model_params = HeisenbergParams(**model_config)
            compiler = HeisenbergCompiler(model_params)
            parse_basis = lambda strings: build_basis_reprs(model_params.L, strings)
            initial_basis = lambda initial: build_basis_reprs(model_params.L, initial)
            required_basis = parse_basis(basis_config['required'])

        else:
            raise ValueError(f'unknown model type: {model_type}')

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ NGA params
        nga_params = NGAParams(**data['nga'])

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ NGA scheduler
        scheduler_type = data['scheduler_type']
        scheduler_config = data['scheduler']

        if scheduler_type == 'base':
            from nga_scheduler import BaseScheduler
            scheduler = BaseScheduler(**scheduler_config)

        elif scheduler_type == 'rate':
            from nga_scheduler import RateScheduler
            scheduler = RateScheduler(**scheduler_config)

        elif scheduler_type == 'decay':
            from nga_scheduler import DecayScheduler
            scheduler = DecayScheduler(**scheduler_config)

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
            basis = parse_basis(json.loads(resume_basis.read_text()))
            start_step = state.get('step', -1) + 1
            records = state.get('records', [])
            events = json.loads(resume_events.read_text()) if resume_events.exists() else {'steps': []}
            drop_counts = {}
            for step_event in events.get('steps', []):
                for rep in parse_basis(step_event.get('drop', [])):
                    key = rep.trans_canon
                    drop_counts[key] = drop_counts.get(key, 0) + 1
        else:
            basis = initial_basis(basis_config['initial'])
            start_step, records = 0, []
            events = {
                'initial_basis': [str(rep.trans_canon_rep) for rep in basis],
                'steps': [],
            }
            drop_counts = {}

        return cls(model_type, model_params, compiler, nga_params, scheduler, basis, required_basis, start_step, records, events, drop_counts)

    def get_basis_rep(self, key):
        return type(self.basis[0])(self.model_params.L, key).trans_canon_rep


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('config/hubbard.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    parser.add_argument('--resume', type=Path, default=None)
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()

    adapter = ModelAdapter.load(args.config, args.resume)
    output_basis = args.output_dir / 'basis.json'
    output_state = args.output_dir / 'state.json'
    output_events = args.output_dir / 'events.json'

    runner = NGARunner(
        compiler=adapter.compiler,
        basis_reprs=adapter.basis,
        required_basis_reprs=adapter.required_basis,
        nga_params=adapter.nga_params,
        scheduler=adapter.scheduler,
        drop_counts=adapter.drop_counts,
    )

    start_step = adapter.start_step
    events = adapter.events
    records = adapter.records

    t0 = time.perf_counter()
    for step in range(start_step, start_step + args.steps):
        _, record = runner.step()

        events['steps'].append({
            'step': step,
            'drop': [str(adapter.get_basis_rep(key)) for key in runner.to_drop],
            'grow': [str(rep.trans_canon_rep) for rep in runner.to_grow],
        })
        record = record.to_dict()
        records.append(record)

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
            json.dumps([str(rep.trans_canon_rep) for rep in runner.basis_reprs], indent=2)
        )

        print(
            f'[{step:03d}] '
            f'{record["status"]} '
            f'value={record["value"]:.12f} | '
            f'basis={record["basis_reps"]} '
            f'vars={record["n_vars"]} '
            f'aff_rank={record["affine_rank"]} '
            f'psd_dims={sum(record["psd_dims"])} | '
            f'drop_null={record["drop_null_count"]} '
            f'grow_null={record["grow_null_count"]} | '
            f'to_drop={record["to_drop"]} '
            f'to_grow={record["to_grow"]} '
            f'net={record["net_growth"]} | '
            f'compile_s={record["time"]["compile_time"]:.1f} '
            f'build_s={record["time"]["build_time"]:.1f} '
            f'solve_s={record["time"]["solve_time"]:.1f} | '
            f'elapsed_s={time.perf_counter()-t0:.1f}',
            flush=True,
        )

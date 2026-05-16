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
    basis: list
    required_basis: list
    start_step: int
    records: list
    events: dict

    @classmethod
    def load(cls, config: Path, resume: Path | None = None):
        data = json.loads(Path(config).read_text())
        model_type = data['model_type']
        model_config = data['model']
        basis_config = data['basis']
        nga_params = NGAParams(**data['nga'])

        if model_type == 'hubbard':
            from hubbard.hubbard import (
                HubbardCompiler, HubbardParams, MajoranaMonomial,
                build_basis_reprs,
            )
            model_params = HubbardParams(**model_config)
            compiler = HubbardCompiler(model_params)
            parse_basis = lambda strings: [MajoranaMonomial.from_str(model_params.L, s).canon_rep for s in strings]
            initial_basis = lambda initial: parse_basis(initial) if isinstance(initial, list) else build_basis_reprs(model_params.L, **initial)
            required_basis = parse_basis(basis_config['required'])

        elif model_type == 'ising':
            from ising.ising import (
                IsingCompiler, IsingParams,
                build_basis_reprs,
            )
            model_params = IsingParams(**model_config)
            compiler = IsingCompiler(model_params)
            parse_basis = lambda strings: build_basis_reprs(model_params.L, strings)
            initial_basis = lambda initial: build_basis_reprs(model_params.L, initial)
            required_basis = parse_basis(basis_config['required'])

        elif model_type == 'heisenberg':
            from heisenberg.heisenberg import (
                HeisenbergCompiler, HeisenbergParams,
                build_basis_reprs,
            )
            model_params = HeisenbergParams(**model_config)
            compiler = HeisenbergCompiler(model_params)
            parse_basis = lambda strings: build_basis_reprs(model_params.L, strings)
            initial_basis = lambda initial: build_basis_reprs(model_params.L, initial)
            required_basis = parse_basis(basis_config['required'])

        else:
            raise ValueError(f'unknown model type: {model_type}')

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
        else:
            basis = initial_basis(basis_config['initial'])
            start_step, records = 0, []
            events = {
                'initial_basis': [str(rep.canon_rep) for rep in basis],
                'steps': [],
            }

        return cls(model_type, model_params, compiler, nga_params, basis, required_basis, start_step, records, events)

    def get_basis_rep(self, key):
        return type(self.basis[0])(self.model_params.L, key).canon_rep


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('config-hubbard.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    parser.add_argument('--resume', type=Path, default=None)
    parser.add_argument('--steps', type=int, default=16)
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
    )

    start_step = adapter.start_step
    events = adapter.events
    records = adapter.records

    t0 = time.perf_counter()
    for step in range(start_step, start_step + args.steps):
        s0 = time.perf_counter()
        _, record = runner.step()

        events['steps'].append({
            'step': step,
            'drop': [str(adapter.get_basis_rep(key)) for key in runner.to_drop],
            'grow': [str(rep.canon_rep) for rep in runner.to_grow],
        })
        record = record.to_dict()
        records.append(record)

        output_state.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'step': step,
            'model_type': adapter.model_type,
            'model': asdict(adapter.model_params),
            'nga': asdict(adapter.nga_params),
            'records': records,
        }
        output_state.write_text(json.dumps(state, indent=2))

        output_events.parent.mkdir(parents=True, exist_ok=True)
        output_events.write_text(json.dumps(events, indent=2))

        output_basis.parent.mkdir(parents=True, exist_ok=True)
        output_basis.write_text(
            json.dumps([str(rep.canon_rep) for rep in runner.basis_reprs], indent=2)
        )

        print(
            f'[{step:03d}] {record["status"]} value={record["value"]:.12f} | '
            f'basis={record["basis_reps"]} vars={record["n_vars"]} aff_rank={record["affine_rank"]} psd_dims={sum(record["psd_dims"])} | '
            f'drop_null={record["drop_null_count"]} grow_null={record["grow_null_count"]} | '
            f'to_drop={record["to_drop"]} to_grow={record["to_grow"]} net={record["net_growth"]} | '
            f'step_s={time.perf_counter()-s0:.1f} elapsed_s={time.perf_counter()-t0:.1f}',
            flush=True,
        )

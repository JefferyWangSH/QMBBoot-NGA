import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from hubbard.hubbard import (
    HubbardCompiler, HubbardParams, MajoranaMonomial,
    load_basis_reprs
)
from nga import NGAParams, NGARunner

default_config = {
    'model': {
        'L': 10,
        't': 1.,
        'U': 4.,
        'n_particles': 10,
    },
    'nga': {
        'solver_backend': 'SCS',
        'drop_null_tol': 1e-5,
        'grow_null_tol': 1e-6,
        'max_drop_leverage': 5e-2,
        'min_net_growth_per_step': 1,
        'max_net_growth_per_step': 8,
        'drop_cap_base_per_step': 8,
        'drop_cap_rate': 0.15,
    },
    'initial_basis': {
        'max_degree': 4,
        'max_support': 1,
        'max_diameter': 0,
    },
}
required = ['I', '0u+', '0u-', '0u+ 0u-', '0d+', '0d-', '0d+ 0d-']


def config(path):
    data = json.loads(path.read_text()) if path else {}
    model = default_config['model'] | data.get('model', {})
    nga = default_config['nga'] | data.get('nga', {})
    initial_basis = default_config['initial_basis'] | data.get('initial_basis', {})
    return HubbardParams(**model), nga, initial_basis


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('config.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--steps', type=int, default=16)
    args = parser.parse_args()

    params, nga_config, initial_basis_config = config(args.config)
    required_basis = [MajoranaMonomial.from_str(params.L, s) for s in required]
    output_basis = args.output_dir / 'basis.json'
    output_state = args.output_dir / 'state.json'
    output_events = args.output_dir / 'events.json'

    if args.resume:
        resume_basis = args.resume / 'basis.json'
        resume_state = args.resume / 'state.json'
        resume_events = args.resume / 'events.json'
        state = json.loads(resume_state.read_text())
        if state['params'] != asdict(params):
            raise ValueError('resume state model parameters do not match config')
        strings = json.loads(resume_basis.read_text())
        basis = [MajoranaMonomial.from_str(params.L, s).canon_rep for s in strings]
        start_step = state.get('step', -1) + 1
        records = state.get('records', [])
        events = json.loads(resume_events.read_text()) if resume_events.exists() else {'steps': []}
    else:
        basis = load_basis_reprs(
            params.L,
            max_degree=initial_basis_config['max_degree'],
            max_support=initial_basis_config['max_support'],
            max_diameter=initial_basis_config['max_diameter'],
        )
        start_step, records = 0, []
        events = {
            'initial_basis': [str(rep.canon_rep) for rep in basis],
            'steps': [],
        }

    runner = NGARunner(
        compiler=HubbardCompiler(params),
        basis_reprs=basis,
        required_basis_reprs=required_basis,
        nga_params=NGAParams(**nga_config),
    )

    t0 = time.perf_counter()
    for step in range(start_step, start_step + args.steps):
        s0 = time.perf_counter()
        _, record = runner.step()
        events['steps'].append({
            'step': step,
            'drop': [str(MajoranaMonomial(params.L, key).canon_rep) for key in runner.to_drop],
            'grow': [str(rep.canon_rep) for rep in runner.to_grow],
        })
        record = record.to_dict()
        records.append(record)

        output_basis.parent.mkdir(parents=True, exist_ok=True)
        output_basis.write_text(
            json.dumps([str(rep.canon_rep) for rep in runner.basis_reprs], indent=2)
        )
        output_events.write_text(json.dumps(events, indent=2))

        output_state.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'step': step,
            'params': asdict(params),
            'nga': nga_config,
            'records': records,
        }
        output_state.write_text(json.dumps(data, indent=2))

        print(
            f'[{step:03d}] {record["status"]} value={record["value"]:.12f} | '
            f'basis={record["basis_reps"]} vars={record["n_vars"]} aff_rank={record["affine_rank"]} psd_dims={sum(record["psd_dims"])} | '
            f'drop_null={record["drop_null_count"]} grow_null={record["grow_null_count"]} | '
            f'to_drop={record["to_drop"]} to_grow={record["to_grow"]} net={record["net_growth"]} | '
            f'step_s={time.perf_counter()-s0:.1f} elapsed_s={time.perf_counter()-t0:.1f}',
            flush=True,
        )

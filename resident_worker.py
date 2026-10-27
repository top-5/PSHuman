"""Opt-in serial worker: keep one immutable diffusion pipeline across requests.

This controlled batch entrypoint owns no system service and changes no default.
Caller must retain the host GPU flock for the worker lifetime. Quality settings
come from the existing YAML; each request only names input/output/prior paths.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time


def run(config_path, requests_path):
    if os.environ.get('PSHUMAN_RESIDENT_PIPELINE') != '1':
        raise ValueError('Explicit PSHUMAN_RESIDENT_PIPELINE=1 required')
    import inference
    from omegaconf import OmegaConf
    from utils.misc import load_config
    from seed.obs.command import command_run
    from seed.obs.report import load
    requests_path = Path(requests_path).resolve(strict=True)
    root = requests_path.parent
    requests = json.loads(requests_path.read_text())
    base = OmegaConf.merge(OmegaConf.structured(inference.TestConfig), load_config(config_path))
    results = []
    for request in requests:
        output = (root / request['output']).resolve()
        if not output.is_relative_to(root) or output == root or output.exists():
            raise ValueError('New contained request output required')
        output.mkdir()
        cfg = copy.deepcopy(base)
        cfg.save_dir = str(output)
        cfg.recon_opt.res_path = str(output / 'pshuman')
        cfg.validation_dataset.root_dir = str(Path(request['input_directory']).resolve(strict=True))
        cfg.mv_dump_dir = str(output / 'mv_dump')
        os.environ['PSHUMAN_BODY_MODEL'] = 'soma'
        os.environ['SEED_NO_SMPLX'] = '1'
        os.environ['PSHUMAN_BODY_PRIOR_NPZ'] = str(Path(request['prior']).resolve(strict=True))
        start = time.perf_counter()
        status, error = 'complete', None
        try:
            with command_run(output / 'spans.jsonl', 'PSHuman resident request'):
                inference.main(cfg)
        except Exception as exc:
            status, error = 'failed', repr(exc)
        totals = {}
        for span in load(output / 'spans.jsonl'):
            name = span['name']
            totals[name] = totals.get(name, 0) + (span['end_unix_nano'] - span['start_unix_nano']) / 1e9
        row = dict(output=str(output), status=status, error=error, seconds=time.perf_counter()-start,
                   phases_inclusive=totals, prior_sha256=hashlib.sha256(Path(os.environ['PSHUMAN_BODY_PRIOR_NPZ']).read_bytes()).hexdigest())
        (output / 'resident-request.json').write_text(json.dumps(row, indent=2)+'\n')
        results.append(row)
        (root / 'resident-results.json').write_text(json.dumps(results, indent=2)+'\n')
        print(row, flush=True)
        if status != 'complete':
            raise RuntimeError(error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--requests', required=True)
    args = parser.parse_args()
    run(args.config, args.requests)

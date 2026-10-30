import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from utils.arm_normal_confidence import verify_prior_binding


def test_declared_capability_checks_declared_prior_and_keeps_external_unbound(tmp_path):
    declaration = json.loads((Path(__file__).parents[1] / 'seed_backend_capabilities.json').read_text())
    assert declaration['contract'] == 'seed.pshuman.backend_capabilities.v1'
    feature = declaration['features']['arm_normal_binding']
    assert feature['version'] == 1 and feature['samePriorSha256Verified'] is True
    assert feature['priorHashBehavior'] == 'required_and_must_match_loaded_prior'
    prior_a = hashlib.sha256(b'first prior with same topology').hexdigest()
    prior_b = hashlib.sha256(b'second prior with same topology').hexdigest()
    binding = tmp_path / 'bound.npz'
    np.savez(binding, prior_sha256=prior_a)
    assert verify_prior_binding(binding.read_bytes(), prior_a)['status'] == 'matched'
    with pytest.raises(ValueError, match='does not match'):
        verify_prior_binding(binding.read_bytes(), prior_b)
    with pytest.raises(ValueError, match='no SOMA prior'):
        verify_prior_binding(binding.read_bytes(), None)
    np.savez(binding, external=True)
    with pytest.raises(ValueError, match='requires prior_sha256'):
        verify_prior_binding(binding.read_bytes(), prior_b)
    with pytest.raises(ValueError, match='requires prior_sha256'):
        verify_prior_binding(binding.read_bytes(), None)


def test_reconstruction_consumer_uses_strict_verifier_only_when_enabled():
    import ast

    source = (Path(__file__).parents[1] / 'reconstruct.py').read_text()
    tree = ast.parse(source)
    checks = [n for n in ast.walk(tree) if isinstance(n, ast.If)
              and isinstance(n.test, ast.Name) and n.test.id == 'arm_normal_enabled'
              and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                      and c.func.id == 'verify_prior_binding' for c in ast.walk(n))]
    assert len(checks) == 1
    calls = [c for c in ast.walk(checks[0]) if isinstance(c, ast.Call)
             and isinstance(c.func, ast.Name) and c.func.id == 'verify_prior_binding']
    assert len(calls) == 1 and not any(k.arg == 'allow_unbound' for k in calls[0].keywords)
    assert 'loaded_sha256' in ast.get_source_segment(source, calls[0])

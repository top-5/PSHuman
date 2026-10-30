import io

import numpy as np
import pytest
from utils import arm_normal_confidence as arm
from utils import runtime_capabilities as capabilities


def test_receipt_checks_real_consumers_and_is_stable():
    first = capabilities.capability_receipt()
    second = capabilities.capability_receipt()
    assert first == second
    assert first['contract'] == 'pshuman.runtime_capabilities.v1'
    assert first['schema_version'] == 1
    assert all(first['checks'].values())
    assert first['features']['arm_normal_prior_binding']['supported'] is True
    for source in ('core/opt.py', 'core/remesh.py', 'utils/render.py', 'utils/soma_prior.py'):
        assert source in first['source_sha256']


def test_unsupported_prior_validator_cannot_advertise(monkeypatch):
    monkeypatch.setattr(arm, 'verify_prior_binding', lambda *args, **kwargs: {'status': 'matched'})
    with pytest.raises(RuntimeError, match='swapped_prior_refused'):
        capabilities.capability_receipt()


def test_source_drift_or_incomplete_checks_cannot_advertise(monkeypatch):
    manifests = iter([{'source': 'a'}, {'source': 'b'}])
    monkeypatch.setattr(capabilities, '_source_manifest', lambda root: next(manifests))
    with pytest.raises(RuntimeError, match='source changed'):
        capabilities.capability_receipt()
    monkeypatch.setattr(capabilities, '_source_manifest', lambda root: {})
    monkeypatch.setattr(capabilities, '_run_checks', lambda root: {})
    with pytest.raises(RuntimeError, match='Incomplete'):
        capabilities.capability_receipt()


def test_required_mode_rejects_downgrade_and_invalid_settings():
    data = io.BytesIO()
    np.savez(data, faces_sha256='unused')
    with pytest.raises(ValueError, match='requires prior_sha256'):
        arm.verify_prior_binding(data.getvalue(), 'a' * 64, require_bound=True)
    assert arm.verify_prior_binding(data.getvalue(), 'a' * 64)['status'] == 'external_unbound'
    with pytest.raises(ValueError, match='require_bound'):
        arm.verify_prior_binding(data.getvalue(), 'a' * 64, require_bound=1)
    for invalid in ('yes', 'garbage', ''):
        with pytest.raises(ValueError):
            arm.required_prior_binding({'PSHUMAN_ARM_NORMAL_REQUIRE_PRIOR_BINDING': invalid})


def test_cli_failure_has_no_positive_features(monkeypatch, capsys):
    def refuse():
        raise RuntimeError('bad consumer')
    monkeypatch.setattr(capabilities, 'capability_receipt', refuse)
    assert capabilities.main() == 1
    import json
    assert json.loads(capsys.readouterr().out)['features'] == {}


def test_manifest_tracks_geometry_file_changes_including_untracked_source(tmp_path):
    for name in capabilities._REQUIRED:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture source')
    untracked = tmp_path / 'core/new_geometry.py'
    untracked.write_text('geometry_v1')
    before = capabilities._source_manifest(tmp_path)
    untracked.write_text('geometry_v2')
    (tmp_path / 'core/opt.py').write_text('optimizer_v2')
    after = capabilities._source_manifest(tmp_path)
    assert before['core/new_geometry.py'] != after['core/new_geometry.py']
    assert before['core/opt.py'] != after['core/opt.py']

"""Versioned, executable PSHuman option capabilities; CPU checks only.

Run ``python -m utils.runtime_capabilities`` in the selected backend interpreter.
Source hashes are provenance/cache inputs, not an allowlist. No positive receipt
is emitted when conformance fails or source changes during the checks.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path

CONTRACT = "pshuman.runtime_capabilities.v1"
SCHEMA_VERSION = 1
_REQUIRED = (
    "reconstruct.py", "inference.py", "core/opt.py", "core/remesh.py",
    "core/deterministic_scatter.py", "utils/render.py",
    "utils/arm_normal_confidence.py", "utils/soma_prior.py",
    "utils/runtime_capabilities.py", "lib/dataset/mesh_util.py",
)
_SUFFIXES = {
    ".py", ".cu", ".cuh", ".cpp", ".h", ".hpp", ".yaml", ".yml", ".json",
    ".txt", ".toml", ".ini", ".cfg", ".sh",
}
_CHECKS = frozenset({
    "consumed_prior_bytes", "explicit_preinset_carve", "fallback_carve",
    "same_prior_binding", "swapped_prior_refused", "absent_prior_refused", "malformed_hash_refused",
    "missing_required_hash_refused", "external_unbound_explicit",
    "ordered_topology_remap", "normal_only_attenuation", "required_mode",
})


def _source_manifest(root: Path) -> dict[str, str]:
    files = set(_REQUIRED)
    try:
        result = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                                capture_output=True, check=True, timeout=5)
        files.update(p for p in result.stdout.decode().split("\0") if p and Path(p).suffix in _SUFFIXES)
    except (OSError, subprocess.SubprocessError):
        pass
    # Include relevant untracked working source too, and never walk run outputs
    # or compiled caches. A commit ID alone cannot describe a shared dirty tree.
    excluded = {".git", "__pycache__", "build", "build-linux", "out", ".cache", "node_modules"}
    for folder in ("core", "utils", "mvdiffusion", "lib", "configs"):
        for directory, folders, names in os.walk(root / folder):
            folders[:] = [name for name in folders if name not in excluded]
            files.update(str((Path(directory) / name).relative_to(root)) for name in names
                         if Path(name).suffix in _SUFFIXES)
    files.update(p.name for p in root.glob("*.py"))
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sorted(files)}


def _run_checks(root: Path) -> dict[str, bool]:
    import numpy as np
    import torch
    from utils import arm_normal_confidence as arm
    from utils import soma_prior as prior

    for module, relative in ((arm, "utils/arm_normal_confidence.py"), (prior, "utils/soma_prior.py")):
        if Path(module.__file__).resolve() != (root / relative).resolve():
            raise RuntimeError("Capability helper imported from a different backend")
    checks = {}

    def check(name, condition):
        if not condition:
            raise RuntimeError("Capability conformance failed: " + name)
        checks[name] = True

    def refuses(name, function):
        try:
            function()
        except ValueError:
            check(name, True)
        else:
            raise RuntimeError("Capability conformance failed: " + name)

    def binding_bytes(**extra):
        stream = io.BytesIO()
        np.savez(stream, faces_sha256=arm.topology_digest(faces), vertex_count=4,
                 vertex_ids=np.array([0]), known_non_arm_ids=np.array([1]), **extra)
        return stream.getvalue()

    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int64)
    carve = vertices + np.array([0., 0., .125], np.float32)
    with tempfile.TemporaryDirectory(prefix="pshuman-capability-") as temporary:
        path = Path(temporary) / "body.npz"
        payload = dict(faces=faces, carve_faces=faces, carve_vertices=carve,
                       **{name: np.array([0], np.int64) for name in prior.SOMA_PRIOR_GROUPS})
        np.savez(path, vertices=vertices, **payload)
        consumed = hashlib.sha256(path.read_bytes()).hexdigest()
        loaded = prior.load_soma_body_prior_npz(path, "cpu", bind_prior=True)
        check("consumed_prior_bytes", loaded["loaded_sha256"] == consumed
              and np.array_equal(loaded["vertices"].numpy(), vertices))
        _, _, cv, cf, ids = prior.select_carve_start(loaded, loaded["vertices"], loaded["faces"])
        check("explicit_preinset_carve", np.array_equal(cv.numpy(), carve)
              and np.array_equal(cf.numpy(), faces) and np.array_equal(ids.numpy(), np.arange(4)))
        _, _, fallback, _, _ = prior.select_carve_start(None, loaded["vertices"], loaded["faces"])
        check("fallback_carve", np.array_equal(fallback.numpy(), vertices))
        bound = binding_bytes(prior_sha256=consumed)
        check("same_prior_binding", arm.verify_prior_binding(
            bound, loaded["loaded_sha256"], require_bound=True)["status"] == "matched")
        np.savez(path, vertices=vertices + 1, **payload)
        swapped = prior.load_soma_body_prior_npz(path, "cpu", bind_prior=True)
        refuses("swapped_prior_refused", lambda: arm.verify_prior_binding(
            bound, swapped["loaded_sha256"], require_bound=True))
        refuses("absent_prior_refused", lambda: arm.verify_prior_binding(bound, None, require_bound=True))
        refuses("malformed_hash_refused", lambda: arm.verify_prior_binding(
            binding_bytes(prior_sha256="x" * 64), consumed, require_bound=True))
        unbound = binding_bytes()
        refuses("missing_required_hash_refused", lambda: arm.verify_prior_binding(
            unbound, consumed, require_bound=True))
        check("external_unbound_explicit", arm.verify_prior_binding(unbound, consumed)["status"] == "external_unbound")
        weights = arm.remap_weights(bound, faces, 4, ids.numpy(), cf.numpy())
        known = arm.remap_known_non_arm(bound, faces, 4, ids.numpy(), weights)
        check("ordered_topology_remap", np.array_equal(weights, [1, 0, 0, 0])
              and np.array_equal(known, [0, 1, 0, 0]))

    class FixtureRenderer:
        def render(self, v, f, colors):
            result = torch.zeros((1, 3, 3, 4))
            result[0, 1, 1, :3] = colors[0]
            return result

    state = torch.zeros((4, 11))
    state[:, 9] = torch.from_numpy(weights)
    state[:, 10] = torch.from_numpy(known)
    alpha = torch.ones((1, 3, 3, 1))
    original_alpha = alpha.clone()
    confidence = arm.normal_confidence(FixtureRenderer(), torch.from_numpy(vertices),
                                       torch.from_numpy(faces), state, alpha, dilation=0, attenuation=0)
    check("normal_only_attenuation", confidence[0, 1, 1, 0].item() == 0
          and confidence[0, 0, 0, 0].item() == 1 and torch.equal(alpha, original_alpha))
    check("required_mode", arm.required_prior_binding({"PSHUMAN_ARM_NORMAL_REQUIRE_PRIOR_BINDING": "1"})
          and not arm.required_prior_binding({}))
    return checks


def capability_receipt() -> dict:
    import sys

    import numpy as np
    import torch
    root = Path(__file__).resolve().parents[1]
    before = _source_manifest(root)
    checks = _run_checks(root)
    if set(checks) != _CHECKS or not all(value is True for value in checks.values()):
        raise RuntimeError("Incomplete capability conformance checks")
    if before != _source_manifest(root):
        raise RuntimeError("Backend source changed during capability checks")
    try:
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {
        "contract": CONTRACT, "schema_version": SCHEMA_VERSION,
        "features": {
            "preinset_carve": {"version": 1, "supported": True},
            "arm_normal_confidence": {"version": 1, "supported": True},
            "arm_normal_prior_binding": {
                "version": 1, "supported": True, "validation": "consumed_prior_sha256",
                "require_bound_env": "PSHUMAN_ARM_NORMAL_REQUIRE_PRIOR_BINDING",
                "external_unbound_supported": True,
            },
        },
        "checks": checks, "backend": {"revision": revision}, "source_sha256": before,
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": str(torch.__version__)},
        "scope": "Executable CPU protocol conformance; not GPU/model readiness or avatar quality",
    }


def main(argv=()) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='JSON is also the default output')
    parser.parse_args(argv)
    try:
        receipt = capability_receipt()
    except Exception as error:
        print(json.dumps({"contract": CONTRACT, "schema_version": SCHEMA_VERSION,
                          "features": {}, "error": f"{type(error).__name__}: {error}"}, sort_keys=True))
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))

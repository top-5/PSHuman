"""SEED_NO_SMPLX=1 (seed's SOMA runs pass it to this subprocess): this process must not touch SMPL-X.

The same three doors as seed.body_model.no_smplx, for the fork's own process: importing the smplx package, and opening
SMPL-X/SMPL/MANO model data (anything under smpl_related/ or checkpoints/smplx, or an smplx/smpl/mano .npz/.pkl that is
not a project output). install() is a no-op unless the variable is set. Imported first by reconstruct.py.
"""

import importlib.abc
import os
import re
import sys

ENV = "SEED_NO_SMPLX"
_MODEL_DATA = re.compile(r"(?i)(^|/)(checkpoints/smplx|smpl_related)(/|$)")
_MODEL_FILE = re.compile(r"(?i)(smplx|smpl|mano)[^/]*\.(npz|pkl)$")


class SmplxForbidden(RuntimeError):
    pass


def forbidden():
    return os.environ.get(ENV, "").strip().lower() not in ("", "0", "false", "off")


def is_smplx_model_path(path):
    p = os.path.abspath(str(path)).replace("\\", "/")
    if _MODEL_DATA.search(p):
        return True
    return bool(_MODEL_FILE.search(p)) and "/projects/" not in p and "/seed-projects/" not in p


def refuse(what):
    raise SmplxForbidden(f"{ENV}=1 (SOMA run): {what} is SMPL-X")


class _BlockSmplx(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if forbidden() and (fullname == "smplx" or fullname.startswith("smplx.")):
            refuse(f"importing the {fullname!r} package")
        return None


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)) and forbidden():
        path = args[0] if isinstance(args[0], str) else os.fsdecode(args[0])
        if is_smplx_model_path(path):
            refuse(f"opening {path}")


_installed = False


def install():
    global _installed
    if forbidden() and not _installed:
        sys.meta_path.insert(0, _BlockSmplx())
        sys.addaudithook(_audit)
        _installed = True
    return _installed

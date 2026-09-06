"""Run gradual motor DAgger with a frozen source-schema compatibility shim.

V1 froze the source model as ``source.model.path`` while the inherited DAgger
runner still reads ``source.model_path``.  This wrapper adds that legacy alias
to an in-memory copy after V1 has fully validated every frozen reference.  It
does not alter the source bytes, schedule, seeds, samples, optimizer, action
masks, environment, or final single-policy runtime semantics.
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import distill_alphazuma_55_motor_observable_gradual_v1 as gradual
from tools import distill_alphazuma_55_polar_dagger_v1 as dagger


SCRIPT_PATH = Path(__file__).resolve()
_BASE_DAGGER_RUN = dagger.run


def _compatibility_run(
    *, preregistration_path: Path, original_root: Path
) -> dict[str, Any]:
    original_validate = dagger._validate_preregistration

    def validate_with_legacy_source_alias(path: Path) -> dict[str, Any]:
        prereg = copy.deepcopy(original_validate(path))
        source = prereg.get("source")
        if not isinstance(source, dict):
            raise ValueError("validated gradual source is not an object")
        model = source.get("model")
        if not isinstance(model, dict) or not model.get("path"):
            raise ValueError("validated gradual source model path is missing")
        existing = source.get("model_path")
        if existing is not None and str(existing) != str(model["path"]):
            raise ValueError("legacy source alias conflicts with frozen model")
        source["model_path"] = str(model["path"])
        return prereg

    dagger._validate_preregistration = validate_with_legacy_source_alias
    try:
        return _BASE_DAGGER_RUN(
            preregistration_path=preregistration_path,
            original_root=original_root,
        )
    finally:
        dagger._validate_preregistration = original_validate


def main(argv: list[str] | None = None) -> int:
    original_gradual_script = gradual.SCRIPT_PATH
    original_dagger_run = dagger.run
    try:
        gradual.SCRIPT_PATH = SCRIPT_PATH
        dagger.run = _compatibility_run
        return gradual.main(argv)
    finally:
        dagger.run = original_dagger_run
        gradual.SCRIPT_PATH = original_gradual_script


if __name__ == "__main__":
    raise SystemExit(main())

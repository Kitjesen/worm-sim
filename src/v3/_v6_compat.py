"""Compatibility helpers for V6 modules moved to ``src/v6``.

The current paper-facing robot pipeline is V6.  The old ``src/v3`` entry
points remain as thin wrappers so existing commands and long-running scripts
do not break during the directory migration.
"""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path


def _v6_path(wrapper_file: str) -> Path:
    wrapper = Path(wrapper_file).resolve()
    project_root = wrapper.parents[2]
    target = project_root / "src" / "v6" / wrapper.name
    if not target.exists():
        raise FileNotFoundError(f"V6 target not found: {target}")
    v6_dir = str(target.parent)
    if v6_dir not in sys.path:
        sys.path.insert(0, v6_dir)
    return target


def export_or_run(wrapper_file: str, namespace: dict[str, object]) -> None:
    target = _v6_path(wrapper_file)
    if namespace.get("__name__") == "__main__":
        runpy.run_path(str(target), run_name="__main__")
        return

    module_name = f"_worm_sim_v6_{target.stem}"
    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load V6 module: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    skipped = {
        "__builtins__",
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__spec__",
    }
    for name, value in vars(module).items():
        if name not in skipped:
            namespace[name] = value

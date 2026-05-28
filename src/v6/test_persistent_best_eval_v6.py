"""
Smoke tests for persistent best-eval protection across training chunks.
"""

import json
import math
import os
import sys
import tempfile

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train_v6 import (  # noqa: E402
    best_eval_summary_path,
    load_persistent_best_eval,
    write_best_eval_summary,
)


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("x")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = os.path.join(tmp, "run")
        log_dir = os.path.join(run_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        assert math.isinf(load_persistent_best_eval(run_dir, log_dir))

        touch(os.path.join(run_dir, "best_model.zip"))
        touch(os.path.join(run_dir, "best_model_vecnormalize.pkl"))
        np.savez(
            os.path.join(log_dir, "evaluations.npz"),
            timesteps=np.array([100, 200]),
            results=np.array([[1.0, 3.0], [2.0, 8.0]]),
            ep_lengths=np.array([[10, 10], [10, 10]]),
        )
        assert load_persistent_best_eval(run_dir, log_dir) == 5.0

        write_best_eval_summary(best_eval_summary_path(run_dir), 7.5, 300)
        assert load_persistent_best_eval(run_dir, log_dir) == 7.5

        with open(best_eval_summary_path(run_dir), encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["best_timestep"] == 300

    print("persistent best eval helpers passed")


if __name__ == "__main__":
    main()

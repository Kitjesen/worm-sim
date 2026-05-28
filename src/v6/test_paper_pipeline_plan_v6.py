"""
Smoke test for the formal paper pipeline command plan.
"""

import os
import sys
from types import SimpleNamespace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from run_paper_pipeline_v6 import (  # noqa: E402
    FIXED_MODES,
    build_formal_records,
    filter_records,
    limit_records,
)


def main():
    args = SimpleNamespace(
        terrain=["flat", "sand", "slope"],
        train_modes=["worm", "snake", "mixed", "random"],
        timesteps=1_000_000,
        train_chunk_timesteps=None,
        n_envs=4,
        device="auto",
        episodes=5,
        eval_time=20.0,
        blends="0.0,0.25,0.5,0.75,1.0",
        video=False,
        resume_partial=False,
    )
    records = build_formal_records(args)
    stages = [record["stage"] for record in records]

    assert stages.count("hardware") == 2
    assert stages.count("train") == 12
    assert stages.count("baseline") == 3 * len(FIXED_MODES)
    assert stages.count("deploy") == 3
    assert stages.count("eval") == 2 * 3 * len(FIXED_MODES)
    assert stages.count("scan") == 3
    assert stages.count("summary") == 1
    assert stages.count("audit") == 2

    train = next(record for record in records
                 if record["stage"] == "train"
                 and record["key"] == "flat_random")
    assert "--timesteps" in train["cmd"]
    assert "1000000" in train["cmd"]
    assert "--device" in train["cmd"]
    assert "auto" in train["cmd"]
    assert "--robust" in train["cmd"]

    args.train_chunk_timesteps = 20000
    chunked_train = next(record for record in build_formal_records(args)
                         if record["stage"] == "train"
                         and record["key"] == "flat_random")
    assert "--train-chunk-timesteps" in chunked_train["cmd"]
    assert "20000" in chunked_train["cmd"]
    args.train_chunk_timesteps = None

    robust_eval = next(record for record in records
                       if record["key"] == "slope_mixed_robust")
    assert "--method" in robust_eval["cmd"]
    assert "robust-eval" in robust_eval["cmd"]

    baseline = next(record for record in records
                    if record["stage"] == "baseline"
                    and record["key"] == "flat_worm_cmaes")
    assert "--method" in baseline["cmd"]
    assert "cmaes" in baseline["cmd"]
    assert "--max-gen" in baseline["cmd"]

    scan = next(record for record in records
                if record["stage"] == "scan" and record["key"] == "sand_random")
    assert "0.0,0.25,0.5,0.75,1.0" in scan["cmd"]

    preflight = next(record for record in records
                     if record["key"] == "hardware_deploy_preflight")
    assert "preflight_hardware_deploy_v6.py" in " ".join(preflight["cmd"])

    train_only = filter_records(records, ["train"])
    assert len(train_only) == 12
    assert all(record["stage"] == "train" for record in train_only)

    eval_scan = filter_records(records, ["eval", "scan"])
    assert len(eval_scan) == 2 * 3 * len(FIXED_MODES) + 3
    assert {record["stage"] for record in eval_scan} == {"eval", "scan"}

    first_two_train = limit_records(train_only, 2)
    assert [record["key"] for record in first_two_train] == [
        "flat_worm",
        "flat_snake",
    ]
    assert limit_records(train_only, None) == train_only

    args.resume_partial = True
    resumable_train = next(record for record in build_formal_records(args)
                           if record["stage"] == "train"
                           and record["key"] == "flat_random")
    assert "--resume-partial" in resumable_train["cmd"]
    print("paper pipeline formal plan contract passed")


if __name__ == "__main__":
    main()

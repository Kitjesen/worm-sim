"""
Training-side contracts for the V6 deployable residual policy.

The deployed actor outputs deterministic joint commands. PPO training is still
stochastic, so its exploration scale is part of the experimental contract:
too much residual noise overwhelms the gait prior and makes train-episode
rewards look catastrophically negative even when the deterministic policy walks.
"""

import math


RESIDUAL_EXPLORATION_CONTRACT_VERSION = "low_noise_residual_v1"
DEFAULT_ENT_COEF = 0.005
DEFAULT_LOG_STD_INIT = -2.5


def residual_exploration_contract(
        ent_coef=DEFAULT_ENT_COEF,
        log_std_init=DEFAULT_LOG_STD_INIT):
    return {
        "version": RESIDUAL_EXPLORATION_CONTRACT_VERSION,
        "policy": "PPO diagonal Gaussian over normalized residual action",
        "reason": (
            "Residual exploration must stay small enough that the deployable "
            "gait prior remains the dominant initial behavior."),
        "ent_coef": float(ent_coef),
        "log_std_init": float(log_std_init),
        "initial_std": float(math.exp(float(log_std_init))),
    }

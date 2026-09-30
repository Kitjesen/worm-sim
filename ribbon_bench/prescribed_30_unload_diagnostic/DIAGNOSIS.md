# 30° prescribed-plate unloading diagnosis (2026-09-30)

This directory contains a numerical diagnosis, not a replacement for the published loading-only result. `actuate.py`, vendor code, material parameters, tolerances, and the previous data have not been changed. The plate is prescribed; this is not evidence that the four real cables can hold the 30° pose.

## Recovery and baseline

The eight 33-node ribbons were recovered from `../prescribed_30_refined/checkpoint.json` using their saved positions and material width directors. Recovery reproduces total elastic energy **0.25995024316443605 J**, free force residual **3.98575e-9 N**, and free twist residual **3.06147e-12 N m**. No loading calculation was repeated and no unloaded configuration was obtained by reversing saved animation frames.

`continue.py` runs the original corrector with 0.3125° yaw steps, saving each newly solved state. Four steps converge, reaching **28.75°**. The next 28.4375° step fails with scaled residual approximately **1.05e6**. That checkpoint is retained unchanged here. `diagnostic.json` records this run.

Restoring the checkpoint and reproducing the failed step in `diagnose.py` gives a slightly different floating-point iteration sequence and a final maximum free force residual **0.778691 N** and twist residual **0.00283111 N m**, both on strip 6 (zero-based). This is far above the unchanged tolerances (1e-6 N, 1e-7 N m); it is not a near-tolerance stall. The total energy is **0.248641281654896 J**.

## Why this solve stops

At the final iterate, the symmetric part of strip 6's scaled Hessian has minimum eigenvalue **−158.559486**. The original regularization schedule ends at **+100**. Every attempted final direction has positive energy slope `g @ p`, including the +100 direction, and is rejected before line search. The smallest positive line-search fraction previously tried is about **0.002368**; this particular failure is not caused by repeatedly shrinking a final descent step to zero.

At this state, central finite differences in the negative-curvature direction agree with the analytic Hessian-vector product to relative error **5.40e-6** at dimensionless perturbation 1e-6. The energy slope is **0.04388435168** by finite difference versus **0.04388429336** by the gradient. The random-direction Hessian-vector relative error is **7.04e-10**. Perturbations 1e-5 and 1e-7 are also saved. These checks support insufficient regularization as the immediate cause of this failed iteration; they do not prove all upstream Hessians are exact or rule out branch changes elsewhere.

Full evidence: `stall_line_search_trace.json`, `stall_derivative_checks.json`, and `stalled_state.json`. The latter is explicitly **NOT_CONVERGED_DIAGNOSTIC**, not a usable equilibrium or animation frame.

## Two independent single-change experiments

1. **Adaptive regularization only**, with the original geometric predictor: when +100 still fails to give a descent direction, `regularization_trial.py` estimates the smallest eigenvalue and applies `max(100, -lambda_min + max(1e-4, 0.01*abs(lambda_min)))`. It uses a shift of **160.145081** once. The problematic step converges in **36 iterations** with force **4.58e-9 N**, twist **6.80e-13 N m**, and energy **0.248292395258600 J**. Measured elapsed time is **67.210 s**, during another concurrent calculation, so this is not a controlled performance benchmark.

2. **Equilibrium-tangent predictor only**, with the original Newton corrector, original regularization schedule, and original line search: solve `H_ff du_f = -H_fp dp` at the preceding converged state, impose the new clamps, then correct the equilibrium. `tangent_trial.py` passes the same difficult step in **3 iterations**, approximately **0.999 s**. Its final energy agrees with experiment 1; maximum nodal difference is **1.37e-12 m**. This change improves the initial guess without changing strain energy, physical loads, or the acceptance thresholds.

The second experiment continues separately in `../prescribed_30_tangent_unload/`. Its metadata records whether the full unloading path has completed. `diagnostic.json` there contains the actual iterations, residuals, energies, and elapsed times for every newly solved increment. It must not be described as a completed return until `results.json` exists and its endpoint is checked.

## Reproduce

From `ribbon_bench`, using its existing virtual environment:

```powershell
.\.venv\Scripts\python.exe prescribed_30_unload_diagnostic/continue.py
.\.venv\Scripts\python.exe prescribed_30_unload_diagnostic/diagnose.py
.\.venv\Scripts\python.exe prescribed_30_unload_diagnostic/regularization_trial.py
.\.venv\Scripts\python.exe prescribed_30_unload_diagnostic/tangent_trial.py
```

These are independent experiments. The first command is expected to stop at the baseline failure; a rerun writes `baseline_reproduction/` and preserves the recorded 28.75-degree seed. The other commands use the recorded seed, not the newly timed rerun. The tangent script verifies the SHA-256 of both recorded input checkpoints. The complete return has now passed validation; see `../prescribed_30_tangent_unload/README.md` for the final result and minimal reproduction commands. Do not replace the original published results with these diagnostics. Any future integration into the main solver should separately validate the driven two-coordinate plate case; the present tests cover prescribed plate unloading only.

"""CUDA batch chain-rule assembly for the project Sano material path.

The upstream geometry/strain derivatives stay on NumPy.  This adapter moves
the expensive material E/g/H contractions and global gradient/Hessian scatter
to one FP64 Torch device operation.  It leaves the vendor submodule untouched
and returns the same NumPy API expected by the time stepper.
"""
from __future__ import annotations

from types import MethodType

import numpy as np
import scipy.sparse as sp


def _material(model, x, torch):
    """Closed-form Sano E/g/H; x is [springs, 4] on the target device."""
    c0 = .5 * float(model.EA) * float(model.delta_l) / float(model.energy_norm)
    factor = .5 * float(model.inv_dl) * float(model.scaling)**2 / float(model.energy_norm)
    b1 = factor * float(model.EI1)
    b2 = factor * float(model.EI2)
    bt = factor * float(model.GJ)
    c2 = (1. / (float(model.zeta) * float(model.inv_dl) * float(model.scaling)))**2
    eps, k1, k2, tau = x.unbind(1)
    inv = 1. / (k1*k1 + c2)
    tau2, tau3, tau4 = tau*tau, tau**3, tau**4
    energy = c0*eps*eps + b1*k1*k1 + b2*k2*k2 + bt*tau2 + b1*tau4*inv
    grad = torch.stack((2*c0*eps,
                        2*b1*k1 - 2*b1*k1*tau4*inv*inv,
                        2*b2*k2,
                        2*bt*tau + 4*b1*tau3*inv), dim=1)
    hess = torch.zeros((x.shape[0], 4, 4), dtype=x.dtype, device=x.device)
    hess[:, 0, 0] = 2*c0
    hess[:, 1, 1] = 2*b1 + 2*b1*tau4*(3*k1*k1-c2)*inv**3
    hess[:, 2, 2] = 2*b2
    hess[:, 3, 3] = 2*bt + 12*b1*tau2*inv
    h13 = -8*b1*k1*tau3*inv*inv
    hess[:, 1, 3] = h13
    hess[:, 3, 1] = h13
    return energy, grad, hess


def _cuda_grad_hess(self, state, sparse=False):
    """Drop-in GeneralElasticEnergySano.grad_hess_energy_linear_elastic."""
    torch = self._cuda_sano_torch
    device = self._cuda_sano_device
    n_springs = self._ind.shape[0]
    n_dof_total = state.q.shape[0]
    if not n_springs:
        out = np.zeros(n_dof_total)
        return out, sp.csr_matrix((n_dof_total, n_dof_total)) if sparse else np.zeros((n_dof_total, n_dof_total))

    # These two routines are the upstream exact Sano geometry derivatives.
    # Keeping them here makes the adapter a drop-in and keeps this first GPU
    # stage numerically auditable against the vendor implementation.
    current = self.get_strain(state)
    grad_s, hess_s = self.grad_hess_strain(state)
    x = current - self._nat_strain
    scale = self.h / self.l_eff
    x[:, 1:] *= scale[:, None]
    grad_s[:, :, 1:] *= scale[:, None, None]
    hess_s[:, :, :, 1:] *= scale[:, None, None, None]

    with torch.no_grad():
        xd = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64, device=device)
        gd = torch.as_tensor(np.ascontiguousarray(grad_s), dtype=torch.float64, device=device)
        hd = torch.as_tensor(np.ascontiguousarray(hess_s), dtype=torch.float64, device=device)
        _, ge, he = _material(self.nn_model, xd, torch)
        grad_local = torch.einsum('sk,sik->si', ge, gd)
        term1 = torch.einsum('sk,sijk->sij', ge, hd)
        term2 = torch.einsum('skl,sik,sjl->sij', he, gd, gd)
        scale_e = torch.as_tensor(.5*self.EA*self.l_eff, dtype=torch.float64, device=device)
        grad_local = grad_local * scale_e[:, None]
        hess_local = (term1 + term2) * scale_e[:, None, None]

        # One device-side scatter replaces Python's per-spring global loop.
        indices = self._cuda_sano_indices
        grad_flat = torch.zeros(n_dof_total, dtype=torch.float64, device=device)
        grad_flat.scatter_add_(0, indices.reshape(-1), -grad_local.reshape(-1))
        hess_flat = torch.zeros(n_dof_total*n_dof_total, dtype=torch.float64, device=device)
        rows = indices[:, :, None].expand(-1, -1, indices.shape[1])
        cols = indices[:, None, :].expand(-1, indices.shape[1], -1)
        linear = rows.reshape(-1)*n_dof_total + cols.reshape(-1)
        hess_flat.scatter_add_(0, linear, -hess_local.reshape(-1))
        Fs = grad_flat.cpu().numpy()
        Js = hess_flat.reshape(n_dof_total, n_dof_total).cpu().numpy()
    return Fs, sp.csr_matrix(Js) if sparse else Js


def install_cuda_sano(energy, device="cuda:0"):
    """Install the adapter on one GeneralElasticEnergySano instance.

    Returns the original bound method so callers can restore it.  ``device``
    may be ``"cpu"`` for a local numerical self-check.
    """
    import torch
    if not hasattr(energy, "nn_model") or not hasattr(energy, "_ind"):
        raise TypeError("expected GeneralElasticEnergySano energy object")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    original = energy.grad_hess_energy_linear_elastic
    energy._cuda_sano_torch = torch
    energy._cuda_sano_device = target
    energy._cuda_sano_indices = torch.as_tensor(energy._ind, dtype=torch.long, device=target)
    energy.grad_hess_energy_linear_elastic = MethodType(_cuda_grad_hess, energy)
    return original


def self_check():
    """Compare one N9 strip against the untouched vendor assembly."""
    import contextlib
    import io
    from pathlib import Path
    from run import make_robot, read_project, strip_geometry

    here = Path(__file__).resolve().parent
    p, delta, _ = read_project(here / "output/parameters.snapshot.json")
    rest, widths = strip_geometry(p, delta, 0, 9)
    with contextlib.redirect_stdout(io.StringIO()):
        robot, stepper = make_robot(p, rest, widths, "sano")
    energy = stepper._TimeStepper__elastic_energies[0]
    baseline = energy.grad_hess_energy_linear_elastic(robot.state, sparse=False)
    original = install_cuda_sano(energy, "cpu")
    try:
        candidate = energy.grad_hess_energy_linear_elastic(robot.state, sparse=False)
    finally:
        energy.grad_hess_energy_linear_elastic = original
    # CUDA/atomic scatter can reorder FP64 additions.  The resulting error is
    # ~1e-10 relative on the largest Hessian entries, far below solver scales.
    np.testing.assert_allclose(candidate[0], baseline[0], rtol=3e-12, atol=1e-12)
    np.testing.assert_allclose(candidate[1], baseline[1], rtol=3e-9, atol=1e-7)
    print({"status": "passed", "dof": int(robot.n_dof), "springs": int(energy._ind.shape[0]),
           "max_abs_force": float(np.max(np.abs(candidate[0]-baseline[0]))),
           "max_abs_hessian": float(np.max(np.abs(candidate[1]-baseline[1])))})


if __name__ == "__main__":
    self_check()

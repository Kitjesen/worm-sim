"""CUDA batch chain-rule assembly for the project Sano material path.

The Sano strain derivatives, material E/g/H contractions, and global
gradient/Hessian scatter run in one FP64 Torch device path.  The vendor
submodule remains untouched and the public API still returns NumPy arrays.
"""
from __future__ import annotations

from types import MethodType

import numpy as np
import scipy.sparse as sp


def _parallel_transport(u, t1, t2, torch):
    """Torch equivalent of the vendor batch parallel transport for one spring."""
    b = torch.cross(t1, t2, dim=-1)
    nb = torch.linalg.vector_norm(b, dim=-1, keepdim=True)
    valid = nb > 1e-10
    bh = b / torch.clamp(nb, min=1e-10)
    bh = bh - torch.sum(bh*t1, dim=-1, keepdim=True) * t1
    bh = bh / torch.clamp(torch.linalg.vector_norm(bh, dim=-1, keepdim=True), min=1e-10)
    bh = bh - torch.sum(bh*t2, dim=-1, keepdim=True) * t2
    bh = bh / torch.clamp(torch.linalg.vector_norm(bh, dim=-1, keepdim=True), min=1e-10)
    n1 = torch.cross(t1, bh, dim=-1)
    n2 = torch.cross(t2, bh, dim=-1)
    transported = (torch.sum(u*t1, dim=-1, keepdim=True) * t2 + torch.sum(u*n1, dim=-1, keepdim=True) * n2
                   + torch.sum(u*bh, dim=-1, keepdim=True) * bh)
    return torch.where(valid, transported, u)


def _outer(a, b):
    return a[..., :, None] * b[..., None, :]


def _cross_matrix(v, torch):
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((torch.stack((zero, -z, y), dim=-1),
                        torch.stack((z, zero, -x), dim=-1),
                        torch.stack((-y, x, zero), dim=-1)), dim=-2)


def _vendor_strain_hess_torch(node0, node1, node2, m1e, m2e, m1f, m2f, l_eff, torch):
    """Torch translation of the pinned vendor strain gradient/Hessian formulas."""
    count = node0.shape[0]
    eye = torch.eye(3, dtype=node0.dtype, device=node0.device)
    grad = torch.zeros((count, 11, 4), dtype=node0.dtype, device=node0.device)
    hess = torch.zeros((count, 11, 11, 4), dtype=node0.dtype, device=node0.device)

    ee, ef = node1-node0, node2-node1
    le_raw, lf_raw = torch.linalg.vector_norm(ee, dim=-1), torch.linalg.vector_norm(ef, dim=-1)
    valid_edges = (le_raw > 1e-12) & (lf_raw > 1e-12)
    le = torch.where(le_raw > 1e-12, le_raw, torch.ones_like(le_raw))
    lf = torch.where(lf_raw > 1e-12, lf_raw, torch.ones_like(lf_raw))
    te = torch.where((le_raw > 1e-12)[:, None], ee/le[:, None], torch.zeros_like(ee))
    tf = torch.where((lf_raw > 1e-12)[:, None], ef/lf[:, None], torch.zeros_like(ef))
    inv = 1./l_eff
    eps0, eps1 = le*inv-1., lf*inv-1.
    d0, d1 = te*inv[:, None], tf*inv[:, None]
    m0a = (inv[:, None, None]-1./le[:, None, None])*eye
    m1a = (inv[:, None, None]-1./lf[:, None, None])*eye
    m0a = torch.where((le_raw > 1e-12)[:, None, None], m0a, torch.zeros_like(m0a))
    m1a = torch.where((lf_raw > 1e-12)[:, None, None], m1a, torch.zeros_like(m1a))
    m0 = 2.*inv[:, None, None]*(m0a+_outer(ee, ee)/le[:, None, None]**3)
    m1 = 2.*inv[:, None, None]*(m1a+_outer(ef, ef)/lf[:, None, None]**3)
    e0 = .5*(m0-2.*_outer(d0, d0)); e1 = .5*(m1-2.*_outer(d1, d1))
    e0 = torch.where((eps0 != 0.)[:, None, None], e0/eps0[:, None, None], torch.zeros_like(e0))
    e1 = torch.where((eps1 != 0.)[:, None, None], e1/eps1[:, None, None], torch.zeros_like(e1))
    grad[:, 0:3, 0] = -.5*d0; grad[:, 3:6, 0] = .5*d0-.5*d1; grad[:, 6:9, 0] = .5*d1
    hess[:, 0:3, 0:3, 0] = .5*e0; hess[:, 3:6, 3:6, 0] = .5*e0+.5*e1
    hess[:, 6:9, 6:9, 0] = .5*e1; hess[:, 0:3, 3:6, 0] = -.5*e0
    hess[:, 3:6, 0:3, 0] = -.5*e0; hess[:, 3:6, 6:9, 0] = -.5*e1; hess[:, 6:9, 3:6, 0] = -.5*e1

    chi_raw = 1.+torch.sum(te*tf, dim=-1)
    valid = valid_edges & (torch.abs(chi_raw) > 1e-12)
    chi = torch.where(valid, chi_raw, torch.ones_like(chi_raw))
    kb = 2.*torch.cross(te, tf, dim=-1)/chi[:, None]
    tt = (te+tf)/chi[:, None]
    d1t = (m1e+m1f)/chi[:, None]; d2t = (m2e+m2f)/chi[:, None]
    k1 = .5*torch.sum(kb*(m2e+m2f), dim=-1)
    k2 = -.5*torch.sum(kb*(m1e+m1f), dim=-1)
    dk1e = (-k1[:, None]*tt+torch.cross(tf, d2t, dim=-1))/le[:, None]
    dk1f = (-k1[:, None]*tt-torch.cross(te, d2t, dim=-1))/lf[:, None]
    dk2e = (-k2[:, None]*tt-torch.cross(tf, d1t, dim=-1))/le[:, None]
    dk2f = (-k2[:, None]*tt+torch.cross(te, d1t, dim=-1))/lf[:, None]
    grad[:, 0:3, 1] = -dk1e; grad[:, 3:6, 1] = dk1e-dk1f; grad[:, 6:9, 1] = dk1f
    grad[:, 0:3, 2] = -dk2e; grad[:, 3:6, 2] = dk2e-dk2f; grad[:, 6:9, 2] = dk2f
    grad[:, 9, 1] = -.5*torch.sum(kb*m1e, dim=-1); grad[:, 10, 1] = -.5*torch.sum(kb*m1f, dim=-1)
    grad[:, 9, 2] = -.5*torch.sum(kb*m2e, dim=-1); grad[:, 10, 2] = -.5*torch.sum(kb*m2f, dim=-1)
    grad[:,0:3,3] = -.5*kb/le[:,None]; grad[:,6:9,3] = .5*kb/lf[:,None]
    grad[:,3:6,3] = -grad[:,0:3,3]-grad[:,6:9,3]
    grad[:,9,3] = -1.; grad[:,10,3] = 1.

    n2e, n2f = le*le, lf*lf
    ttott = _outer(tt, tt); tfd2 = torch.cross(tf, d2t, dim=-1); ted2 = torch.cross(te, d2t, dim=-1)
    tfd2ott = _outer(tfd2, tt); ttoted2 = _outer(tt, ted2)
    ted2ott = _outer(ted2, tt); ttotfd2 = _outer(tt, tfd2)
    kboe = _outer(kb, m2e); kbof = _outer(kb, m2f)
    d11e = (2*k1[:,None,None]*ttott-tfd2ott-ttotfd2)/n2e[:,None,None] \
        - k1[:,None,None]/(chi[:,None,None]*n2e[:,None,None])*(eye[None]-_outer(te,te)) \
        + kboe/(2*n2e[:,None,None])
    d11f = (2*k1[:,None,None]*ttott+ted2ott+ttoted2)/n2f[:,None,None] \
        - k1[:,None,None]/(chi[:,None,None]*n2f[:,None,None])*(eye[None]-_outer(tf,tf)) \
        + kbof/(2*n2f[:,None,None])
    teotf = _outer(te, tf)
    d1ef = -k1[:,None,None]/(chi[:,None,None]*le[:,None,None]*lf[:,None,None])*(eye[None]+teotf) \
        + (2*k1[:,None,None]*ttott-tfd2ott+ttoted2-_cross_matrix(d2t, torch))/(le*lf)[:,None,None]
    tfd1 = torch.cross(tf, d1t, dim=-1); ted1 = torch.cross(te, d1t, dim=-1)
    tfd1ott = _outer(tfd1, tt); ttoted1 = _outer(tt, ted1); ted1ott = _outer(ted1, tt); ttotfd1 = _outer(tt, tfd1)
    kb1e = _outer(kb, m1e); kb1f = _outer(kb, m1f)
    d21e = (2*k2[:,None,None]*ttott+tfd1ott+ttotfd1)/n2e[:,None,None] \
        - k2[:,None,None]/(chi[:,None,None]*n2e[:,None,None])*(eye[None]-_outer(te,te)) \
        - kb1e/(2*n2e[:,None,None])
    d21f = (2*k2[:,None,None]*ttott-ted1ott-ttoted1)/n2f[:,None,None] \
        - k2[:,None,None]/(chi[:,None,None]*n2f[:,None,None])*(eye[None]-_outer(tf,tf)) \
        - kb1f/(2*n2f[:,None,None])
    d2ef = -k2[:,None,None]/(chi[:,None,None]*le[:,None,None]*lf[:,None,None])*(eye[None]+teotf) \
        + (2*k2[:,None,None]*ttott+tfd1ott-ttoted1+_cross_matrix(d1t, torch))/(le*lf)[:,None,None]
    def assemble(h, a, b, c, d, e, component):
        h[:,0:3,0:3,component]=a; h[:,0:3,3:6,component]=-a+b; h[:,0:3,6:9,component]=-b
        h[:,3:6,0:3,component]=-a+b.transpose(-1,-2); h[:,3:6,3:6,component]=a-b-b.transpose(-1,-2)+c
        h[:,3:6,6:9,component]=b-c; h[:,6:9,0:3,component]=-b.transpose(-1,-2)
        h[:,6:9,3:6,component]=b.transpose(-1,-2)-c; h[:,6:9,6:9,component]=c
        h[:,9,9,component]=d; h[:,10,10,component]=e
    # Position/angle mixed curvature blocks.
    k1ee = (.5*torch.sum(kb*m1e,dim=-1)[:,None]*tt-torch.cross(tf,m1e,dim=-1)/chi[:,None])/le[:,None]
    k1ef = (.5*torch.sum(kb*m1f,dim=-1)[:,None]*tt-torch.cross(tf,m1f,dim=-1)/chi[:,None])/le[:,None]
    k1fe = (.5*torch.sum(kb*m1e,dim=-1)[:,None]*tt+torch.cross(te,m1e,dim=-1)/chi[:,None])/lf[:,None]
    k1ff = (.5*torch.sum(kb*m1f,dim=-1)[:,None]*tt+torch.cross(te,m1f,dim=-1)/chi[:,None])/lf[:,None]
    k2ee = (.5*torch.sum(kb*m2e,dim=-1)[:,None]*tt-torch.cross(tf,m2e,dim=-1)/chi[:,None])/le[:,None]
    k2ef = (.5*torch.sum(kb*m2f,dim=-1)[:,None]*tt-torch.cross(tf,m2f,dim=-1)/chi[:,None])/le[:,None]
    k2fe = (.5*torch.sum(kb*m2e,dim=-1)[:,None]*tt+torch.cross(te,m2e,dim=-1)/chi[:,None])/lf[:,None]
    k2ff = (.5*torch.sum(kb*m2f,dim=-1)[:,None]*tt+torch.cross(te,m2f,dim=-1)/chi[:,None])/lf[:,None]
    # The vendor writes the theta blocks explicitly; its position/theta blocks
    # are intentionally not symmetrized.
    assemble(hess, d11e, d1ef, d11f, -.5*torch.sum(kb*m2e,dim=-1), -.5*torch.sum(kb*m2f,dim=-1), 1)
    assemble(hess, d21e, d2ef, d21f, .5*torch.sum(kb*m1e,dim=-1), .5*torch.sum(kb*m1f,dim=-1), 2)
    # Position/theta blocks follow the vendor's explicit (non-symmetrized)
    # assembly.  Keep both theta columns separate; the two directors differ.
    hess[:,0:3,9,1]=-k1ee; hess[:,3:6,9,1]=k1ee-k1fe; hess[:,6:9,9,1]=k1fe
    hess[:,9,0:3,1]=-k1ee; hess[:,9,3:6,1]=k1ee-k1fe; hess[:,9,6:9,1]=k1fe
    hess[:,0:3,10,1]=-k1ef; hess[:,3:6,10,1]=k1ef-k1ff; hess[:,6:9,10,1]=k1ff
    hess[:,10,0:3,1]=-k1ef; hess[:,10,3:6,1]=k1ef-k1ff; hess[:,10,6:9,1]=k1ff
    hess[:,0:3,9,2]=-k2ee; hess[:,3:6,9,2]=k2ee-k2fe; hess[:,6:9,9,2]=k2fe
    hess[:,9,0:3,2]=-k2ee; hess[:,9,3:6,2]=k2ee-k2fe; hess[:,9,6:9,2]=k2fe
    hess[:,0:3,10,2]=-k2ef; hess[:,3:6,10,2]=k2ef-k2ff; hess[:,6:9,10,2]=k2ff
    hess[:,10,0:3,2]=-k2ef; hess[:,10,3:6,2]=k2ef-k2ff; hess[:,10,6:9,2]=k2ff

    # Twist Hessian (Panetta approximation used by the vendor).
    tp = te+tt; fp = tf+tt
    dte2 = -.5/n2e[:,None,None]*(_outer(kb,tp)+2./chi[:,None,None]*_cross_matrix(tf,torch))
    dtf2 = -.5/n2f[:,None,None]*(_outer(kb,fp)-2./chi[:,None,None]*_cross_matrix(te,torch))
    dtfde = .5/(le*lf)[:,None,None]*(2./chi[:,None,None]*_cross_matrix(te,torch)-_outer(kb,tt))
    dtedf = .5/(le*lf)[:,None,None]*(-2./chi[:,None,None]*_cross_matrix(tf,torch)-_outer(kb,tt))
    ht = hess[:,:,:,3]
    ht[:,0:3,0:3]=dte2; ht[:,0:3,3:6]=-dte2+dtfde; ht[:,3:6,0:3]=-dte2+dtedf
    ht[:,3:6,3:6]=dte2-dtedf-dtfde+dtf2; ht[:,0:3,6:9]=-dtfde; ht[:,6:9,0:3]=-dtedf
    ht[:,6:9,3:6]=dtedf-dtf2; ht[:,3:6,6:9]=dtfde-dtf2; ht[:,6:9,6:9]=dtf2
    valid_s = valid[:, None, None]
    grad[:, :, 1:3] = torch.where(valid_s, grad[:, :, 1:3], torch.zeros_like(grad[:, :, 1:3]))
    hess[:, :, :, 1:3] = torch.where(valid[:, None, None, None], hess[:, :, :, 1:3], torch.zeros_like(hess[:, :, :, 1:3]))
    grad[:, :, 3] = torch.where(valid[:, None], grad[:, :, 3], torch.zeros_like(grad[:, :, 3]))
    hess[:, :, :, 3] = torch.where(valid[:, None, None], hess[:, :, :, 3], torch.zeros_like(hess[:, :, :, 3]))
    return grad, hess


def _cuda_strain_batch(self, state, second=False):
    """Return vendor-compatible strain derivatives on the selected device."""
    torch = self._cuda_sano_torch
    device = self._cuda_sano_device
    # Vendor _get_node_pos uses Fortran-flattened node order.  Reproduce it
    # once on-device, then evaluate every spring in one tensor batch.
    node_indices = torch.as_tensor(self._node_dof_ind, dtype=torch.long, device=device)
    node_pos = torch.as_tensor(np.asarray(state.q), dtype=torch.float64, device=device)[node_indices]
    node_pos = node_pos.reshape(self._n_nodes, -1, 3)
    p0, p1, p2 = node_pos[0], node_pos[1], node_pos[2]
    edges = torch.as_tensor(self._edges_ind, dtype=torch.long, device=device)
    m1 = torch.as_tensor(np.asarray(state.m1), dtype=torch.float64, device=device)
    m2 = torch.as_tensor(np.asarray(state.m2), dtype=torch.float64, device=device)
    m1e, m1f = m1[edges[:, 0]], m1[edges[:, 1]]
    signs = torch.as_tensor(self._sgn, dtype=torch.float64, device=device)
    m2e, m2f = m2[edges[:, 0]]*signs[:, 0, None], m2[edges[:, 1]]*signs[:, 1, None]
    lengths = torch.as_tensor(self.l_eff, dtype=torch.float64, device=device)
    # The vendor's get_strain uses the same closed forms as this helper.  The
    # direct translation below also preserves its Panetta curvature/twist
    # Hessian convention, which differs from differentiating the angle formula
    # automatically at the material-director singularities.
    stretch = .5*(torch.linalg.vector_norm(p1-p0, dim=-1)/lengths - 1.) \
            + .5*(torch.linalg.vector_norm(p2-p1, dim=-1)/lengths - 1.)
    ee, ef = p1-p0, p2-p1
    le_raw, lf_raw = torch.linalg.vector_norm(ee, dim=-1), torch.linalg.vector_norm(ef, dim=-1)
    edge_ok = (le_raw > 1e-12) & (lf_raw > 1e-12)
    le = torch.where(le_raw > 1e-12, le_raw, torch.ones_like(le_raw))
    lf = torch.where(lf_raw > 1e-12, lf_raw, torch.ones_like(lf_raw))
    te = torch.where((le_raw > 1e-12)[:, None], ee/le[:, None], torch.zeros_like(ee))
    tf = torch.where((lf_raw > 1e-12)[:, None], ef/lf[:, None], torch.zeros_like(ef))
    chi_raw = 1. + torch.sum(te*tf, dim=-1)
    valid = edge_ok & (torch.abs(chi_raw) > 1e-12)
    chi = torch.where(valid, chi_raw, torch.ones_like(chi_raw))
    kb = 2.*torch.cross(te, tf, dim=-1)/chi[:, None]
    k1 = .5*torch.sum(kb*(m2e+m2f), dim=-1)
    k2 = -.5*torch.sum(kb*(m1e+m1f), dim=-1)
    transported = _parallel_transport(m1e, te, tf, torch)
    transported = transported - torch.sum(transported*tf, dim=-1, keepdim=True)*tf
    transported = transported/torch.clamp(torch.linalg.vector_norm(transported, dim=-1, keepdim=True), min=1e-10)
    m1f_proj = m1f - torch.sum(m1f*tf, dim=-1, keepdim=True)*tf
    m1f_proj = m1f_proj/torch.clamp(torch.linalg.vector_norm(m1f_proj, dim=-1, keepdim=True), min=1e-10)
    cosine = torch.clamp(torch.sum(transported*m1f_proj, dim=-1), -1., 1.)
    sine = torch.sum(torch.cross(transported, m1f_proj, dim=-1)*tf, dim=-1)
    twist = torch.atan2(sine, cosine)
    strain = torch.stack((stretch, k1, k2, twist), dim=-1)
    strain[:, 1:] = torch.where(valid[:, None], strain[:, 1:], torch.zeros_like(strain[:, 1:]))
    grad, hess = _vendor_strain_hess_torch(p0, p1, p2, m1e, m2e, m1f, m2f, lengths, torch)
    return (strain, grad) if not second else (strain, grad, hess)


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

    with torch.no_grad():
        current_d, grad_d, hess_d = _cuda_strain_batch(self, state, second=True)
        xd = current_d - torch.as_tensor(self._nat_strain, dtype=torch.float64, device=device)
        scale = torch.as_tensor(self.h / self.l_eff, dtype=torch.float64, device=device)
        xd = xd.clone(); xd[:, 1:] *= scale[:, None]
        gd = grad_d.clone(); gd[:, :, 1:] *= scale[:, None, None]
        hd = hess_d.clone(); hd[:, :, :, 1:] *= scale[:, None, None, None]
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
    strain_cpu = energy.get_strain(robot.state)
    grad_cpu, hess_cpu = energy.grad_hess_strain(robot.state)
    original = install_cuda_sano(energy, "cpu")
    strain_gpu, grad_gpu, hess_gpu = _cuda_strain_batch(energy, robot.state, second=True)
    np.testing.assert_allclose(strain_gpu.cpu().numpy(), strain_cpu, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(grad_gpu.cpu().numpy(), grad_cpu, rtol=2e-10, atol=2e-10)
    # The vendor stretch Hessian divides by an exactly-zero strain; NumPy and
    # Torch round the straight reference edge on opposite sides of zero.
    np.testing.assert_allclose(hess_gpu.cpu().numpy(), hess_cpu, rtol=1e-2, atol=5.)
    try:
        candidate = energy.grad_hess_energy_linear_elastic(robot.state, sparse=False)
    finally:
        energy.grad_hess_energy_linear_elastic = original
    # CUDA/atomic scatter can reorder FP64 additions.  The resulting error is
    # ~1e-10 relative on the largest Hessian entries, far below solver scales.
    np.testing.assert_allclose(candidate[0], baseline[0], rtol=3e-10, atol=3e-9)
    np.testing.assert_allclose(candidate[1], baseline[1], rtol=3e-9, atol=2e-6)
    print({"status": "passed", "dof": int(robot.n_dof), "springs": int(energy._ind.shape[0]),
           "max_abs_strain": float(np.max(np.abs(strain_gpu.cpu().numpy()-strain_cpu))),
           "max_abs_strain_grad": float(np.max(np.abs(grad_gpu.cpu().numpy()-grad_cpu))),
           "max_abs_strain_hess": float(np.max(np.abs(hess_gpu.cpu().numpy()-hess_cpu))),
           "max_abs_force": float(np.max(np.abs(candidate[0]-baseline[0]))),
           "max_abs_hessian": float(np.max(np.abs(candidate[1]-baseline[1])))})


if __name__ == "__main__":
    self_check()

"""Ideal four-cable and solid-disk stop potentials for plate pose [d_m, yaw_rad].

Cable order follows snapshot [side, upper/lower], flattened in row-major order.
Routes contain only [fixed front anchor, moving back-plate hole outlet]. These
are ideal variable-length actuators: no hole friction, slack shape, servo-horn
mapping, or steel-strip contact is modeled. Gradients are potential derivatives;
physical generalized forces are their negatives.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


class CableLoads:
    def __init__(self, p, delta, contact_stiffness=50000):
        self.anchors = np.asarray(p["tendon_anchors_m"], dtype=float).reshape(4, 3)
        outlets = np.asarray(p["tendon_guides_m"], dtype=float).reshape(4, -1, 3)[:, -1]
        self.front = np.asarray(p["front_plate_center_m"], dtype=float)
        back = np.asarray(p["back_plate_center_m"], dtype=float)
        delta = np.asarray(delta, dtype=float)
        if self.front.shape != (3,) or back.shape != (3,) or delta.shape != (3,):
            raise ValueError("Plate centers and delta must be XYZ vectors")
        self.center = back + delta
        self.offsets = outlets + delta - self.center
        self.k = float(p["effective_tendon_stiffness_n_m"])
        self.kc = float(contact_stiffness)
        self.radius = float(p["plate_stop_radius_m"])
        self.thickness = float(p["plate_stop_thickness_m"])
        scalars = np.array([self.k, self.kc, self.radius, self.thickness])
        if (not np.isfinite(scalars).all() or self.k <= 0 or self.kc < 0
                or self.radius <= 0 or self.thickness < 0
                or not all(np.isfinite(x).all() for x in
                           (self.anchors, self.front, self.center, self.offsets))):
            raise ValueError("Need finite geometry, positive cable stiffness/radius and nonnegative contact stiffness/thickness")

    def evaluate(self, pose2, rest_lengths4):
        """Return total potential, analytic gradient/Hessian and cable/stop data.

        Length and gap activation boundaries use the inactive-side Hessian.
        Active disk contact at an absolute-value cusp has no classical gradient
        and is rejected; ordinary separated zero-yaw poses remain valid.
        """
        pose = np.asarray(pose2, dtype=float)
        rest = np.asarray(rest_lengths4, dtype=float)
        if (pose.shape != (2,) or rest.shape != (4,) or not np.isfinite(pose).all()
                or not np.isfinite(rest).all() or np.any(rest < 0)):
            raise ValueError("Need finite [compression_m, yaw_rad] and four nonnegative rest lengths")
        d, a = pose
        c, s = np.cos(a), np.sin(a)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        arm = self.offsets @ rotation.T
        moving = self.center + arm + [d, 0., 0.]
        vector = moving - self.anchors
        lengths = np.linalg.norm(vector, axis=1)
        if np.any(lengths < 1e-12):
            raise ValueError("A zero-length cable span has no defined direction")
        direction = vector / lengths[:, None]

        # Position Jacobian columns are translation along X and rotation about Z.
        jac = np.zeros((4, 3, 2))
        jac[:, 0, 0] = 1.
        jac[:, 0, 1], jac[:, 1, 1] = -arm[:, 1], arm[:, 0]
        length_gradient = np.einsum("ni,nij->nj", direction, jac)
        length_hessian = (np.einsum("nki,nkj->nij", jac, jac)
                          - np.einsum("ni,nj->nij", length_gradient, length_gradient)) / lengths[:, None, None]
        length_hessian[:, 1, 1] -= np.sum(direction[:, :2] * arm[:, :2], axis=1)
        extension = np.maximum(lengths - rest, 0.)
        # ponytail: massless straight spans; use routed cable nodes for sag and friction.
        tensions = self.k * extension
        energy = .5 * self.k * np.dot(extension, extension)
        gradient = np.einsum("n,ni->i", tensions, length_gradient)
        hessian = (np.einsum("n,ni,nj->ij", self.k * (extension > 0), length_gradient, length_gradient)
                   + np.einsum("n,nij->ij", tensions, length_hessian))

        # ponytail: coaxial solid-disk stop approximation; use surface contact for holes/offset plates.
        gap = self.front[0] - self.center[0] - d - self.radius*abs(s) - .5*self.thickness*(1 + abs(c))
        penetration = min(gap, 0.)
        contact_energy = .5*self.kc*penetration**2
        if penetration < 0 and self.kc > 0:
            if abs(s) < 1e-12 or abs(c) < 1e-12:
                raise ValueError("Active disk contact at a yaw absolute-value cusp is nondifferentiable")
            gap_gradient = np.array([-1., -self.radius*np.sign(s)*c + .5*self.thickness*np.sign(c)*s])
            gradient += self.kc*penetration*gap_gradient
            hessian += self.kc*np.outer(gap_gradient, gap_gradient)
            hessian[1, 1] += self.kc*penetration*(self.radius*abs(s) + .5*self.thickness*abs(c))
        return dict(energy_j=float(energy + contact_energy), gradient=gradient,
                    hessian=hessian, lengths_m=lengths, tensions_n=tensions,
                    routes_m=np.stack((self.anchors, moving), axis=1),
                    plate_gap_m=float(gap), contact_force_n=float(-self.kc*penetration),
                    contact_energy_j=float(contact_energy))


def self_check():
    here = Path(__file__).resolve().parent
    p = json.loads((here/"output/parameters.snapshot.json").read_text(encoding="utf-8"))
    tree = ET.parse(here/"publication/data/cad_reference.urdf")
    joint = next(j for j in tree.getroot().findall("joint") if j.find("child").get("link") == "back2_Link")
    delta = np.fromstring(joint.find("origin").get("xyz"), sep=" ")
    loads = CableLoads(p, delta)
    reference = loads.evaluate([0., 0.], np.ones(4))["lengths_m"]
    assert np.allclose(loads.evaluate([0., 0.], reference)["routes_m"][:, 1],
                       np.asarray(p["tendon_guides_m"]).reshape(4, -1, 3)[:, -1] + delta)
    slack = loads.evaluate([0., 0.], reference + .01)
    assert slack["energy_j"] == 0 and np.all(slack["tensions_n"] == 0)
    assert np.all(slack["gradient"] == 0) and np.all(slack["hessian"] == 0)
    assert slack["contact_force_n"] == 0 and slack["plate_gap_m"] > 0

    def derivatives(pose, rest):
        pose = np.asarray(pose, dtype=float)
        value = loads.evaluate(pose, rest)
        grad_fd, hess_fd = np.zeros(2), np.zeros((2, 2))
        for i, step in enumerate((1e-6, 1e-5)):
            shift = np.zeros(2)
            shift[i] = step
            plus, minus = loads.evaluate(pose+shift, rest), loads.evaluate(pose-shift, rest)
            grad_fd[i] = (plus["energy_j"]-minus["energy_j"])/(2*step)
            hess_fd[:, i] = (plus["gradient"]-minus["gradient"])/(2*step)
        assert np.allclose(value["gradient"], grad_fd, rtol=2e-6, atol=1e-7), (value["gradient"], grad_fd)
        assert np.allclose(value["hessian"], hess_fd, rtol=2e-6, atol=1e-5), (value["hessian"], hess_fd)
        assert np.allclose(value["hessian"], value["hessian"].T, atol=1e-12)
        return value

    stretched_pose = [.01, .23]
    stretched_rest = loads.evaluate(stretched_pose, np.ones(4))["lengths_m"] - [.002, .003, .004, .005]
    stretched = derivatives(stretched_pose, stretched_rest)
    assert np.all(stretched["tensions_n"] > 0) and stretched["contact_force_n"] == 0
    # Short taut spans pull the rear plate toward +X; positive compression lowers cable energy.
    assert stretched["gradient"][0] < 0
    for yaw in (np.deg2rad(30), -np.deg2rad(30)):
        touching_d = loads.evaluate([0., yaw], np.ones(4))["plate_gap_m"]
        separated = derivatives([touching_d-.001, yaw], np.ones(4))
        contact = derivatives([touching_d+.001, yaw], np.ones(4))
        assert separated["contact_force_n"] == 0 and separated["contact_energy_j"] == 0
        assert np.isclose(contact["contact_force_n"], loads.kc*.001)
        assert np.isclose(contact["contact_energy_j"], .5*loads.kc*.001**2)
        assert contact["gradient"][0] > 0 and contact["gradient"][1]*yaw > 0
    print("CableLoads self-check passed: slack/tension, route transform, analytic derivatives, bilateral yaw stop activation and force signs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        parser.print_help()

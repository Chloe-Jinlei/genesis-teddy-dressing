"""Distance-gated Velcro approximation using Genesis's native PBD stretch solve.

Verified against local Genesis 1.4.3. This deliberately uses PRIVATE entity edge
arrays and solver fields: Genesis exposes attachment to rigid links, but no public
cloth-to-cloth adhesion API. Construct after adding the cloth and before building
the scene (and before adding another PBD entity). Call init_after_build BEFORE the
first scene.step. No particle positions or velocities are overwritten here.

The seam represents permanent, compliant bonds, not calibrated hook-and-loop
microstructure, peel strength, or breakage. Native self-collision stays enabled;
bond rest distance must therefore be at least the PBD particle diameter.
"""

from __future__ import annotations

import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class NativeVelcro:
    """Preallocate native distance edges, then activate only measured contact pairs.

    In matching='material' (default), opposite x strips are matched in INITIAL
    flat coordinates. With circumference=None, matching uses the same y and
    mirrored distance from each free end. Set circumference to instead match
    x_right=x_left+circumference; the strip width must include that overlap.
    In matching='contact', every opposite-strip pair with initial |dy| < 35 mm
    is a dormant candidate. Actual close/slow contacts activate greedily by
    distance, with at most one active bond incident on each material node.
    This helper currently supports one environment, matching this reconstruction.
    """

    def __init__(
        self,
        cloth,
        *,
        strip_width=0.025,
        matching="material",
        circumference=None,
        capture_distance=0.018,
        max_relative_speed=0.2,
        rest_gap=None,
        compliance=1e-8,
        relaxation=0.3,
        max_match_error=None,
        normal_dot_threshold=-0.25,
    ):
        self.cloth = cloth
        self.solver = cloth.solver
        self.particle_size = float(self.solver.particle_size)
        self.strip_width = float(strip_width)
        self.matching = str(matching)
        if self.matching not in ("material", "contact"):
            raise ValueError("matching must be 'material' or 'contact'")
        self.circumference = None if circumference is None else float(circumference)
        self.capture_distance = float(capture_distance)
        self.max_relative_speed = float(max_relative_speed)
        self.rest_gap = self.particle_size if rest_gap is None else float(rest_gap)
        self.compliance = float(compliance)
        self.relaxation = float(relaxation)
        self.normal_dot_threshold = float(normal_dot_threshold)
        if min(self.strip_width, self.capture_distance, self.rest_gap) <= 0:
            raise ValueError("strip width, capture distance and rest gap must be positive")
        if self.rest_gap < self.particle_size * (1 - 1e-5):
            raise ValueError("rest_gap must be at least particle_size for native self-collision")
        if self.capture_distance <= self.rest_gap:
            raise ValueError("capture_distance must exceed rest_gap so contact can activate bonds")
        if self.max_relative_speed < 0 or self.compliance <= 0 or not 0 < self.relaxation <= 1:
            raise ValueError("Require nonnegative speed, positive compliance, relaxation in (0,1]")
        if not -1 <= self.normal_dot_threshold <= 1:
            raise ValueError("normal_dot_threshold must lie in [-1,1]")
        flat = np.asarray(cloth._particles, dtype=float)
        if flat.ndim != 2 or flat.shape[1] != 3 or not np.isfinite(flat).all():
            raise ValueError("Expected finite sampled cloth positions with shape (N,3)")
        self.n_particles = len(flat)
        xmin, xmax = float(flat[:, 0].min()), float(flat[:, 0].max())
        if self.strip_width >= 0.5 * (xmax - xmin):
            raise ValueError("Velcro strips must occupy disjoint opposite ends of the garment")
        left = np.flatnonzero(flat[:, 0] <= xmin + self.strip_width + 1e-7)
        right = np.flatnonzero(flat[:, 0] >= xmax - self.strip_width - 1e-7)
        if not len(left) or not len(right):
            raise ValueError("Could not locate both Velcro strips")
        self.left_patch_nodes = left.copy()
        self.right_patch_nodes = right.copy()
        self.contact_y_tolerance = 0.035

        pairs, errors = [], []
        if self.matching == "contact":
            offsets = np.abs(flat[left, 1, None] - flat[right, 1][None, :])
            rows, cols = np.nonzero(offsets < self.contact_y_tolerance)
            pairs = list(zip(left[rows].tolist(), right[cols].tolist()))
            errors = offsets[rows, cols].tolist()
        else:
            target = flat[left].copy()
            target[:, 0] = (
                xmin + xmax - target[:, 0]
                if self.circumference is None
                else target[:, 0] + self.circumference
            )
            cost = np.linalg.norm(target[:, None, :] - flat[right][None, :, :], axis=-1)
            tolerance = 2.0 * self.particle_size if max_match_error is None else float(max_match_error)
            used_left, used_right = set(), set()
            for key in np.argsort(cost, axis=None, kind="stable"):
                i, j = np.unravel_index(key, cost.shape)
                if cost[i, j] > tolerance:
                    break
                if i in used_left or j in used_right:
                    continue
                used_left.add(i)
                used_right.add(j)
                pairs.append((int(left[i]), int(right[j])))
                errors.append(float(cost[i, j]))
        if not pairs:
            raise ValueError("No compatible material pairs; check strip width and circumference")
        order = np.argsort([flat[a, 1] for a, _ in pairs], kind="stable")
        self.pairs = np.asarray(pairs, dtype=cloth._edges.dtype)[order]
        self.match_errors = np.asarray(errors, dtype=float)[order]
        self.edge_start_local = len(cloth._edges)
        self.edge_start_global = int(cloth._edge_start) + self.edge_start_local
        self.n_pairs = len(self.pairs)
        self.edge_indices = np.arange(self.edge_start_global, self.edge_start_global + self.n_pairs)
        cloth._edges = np.concatenate((cloth._edges, self.pairs), axis=0)
        cloth._edges_len_rest = np.concatenate(
            (cloth._edges_len_rest, np.full(self.n_pairs, self.rest_gap, dtype=cloth._edges_len_rest.dtype))
        )
        self.active = np.zeros(self.n_pairs, dtype=bool)
        self.events = []
        self._ready = False
        self._updates = 0
        self._last_distances = np.linalg.norm(flat[self.pairs[:, 0]] - flat[self.pairs[:, 1]], axis=1)
        self._last_relative_speeds = np.zeros(self.n_pairs)
        self._normals_used = False
        self._allow_new = True

    def init_after_build(self):
        """Validate native indices and make every new seam edge dormant."""
        if int(self.solver._B) != 1:
            raise ValueError("NativeVelcro currently supports a single simulation environment")
        info = self.solver.edges_info
        actual_a = info.v1.to_numpy()[self.edge_indices]
        actual_b = info.v2.to_numpy()[self.edge_indices]
        expected = self.pairs + int(self.cloth._particle_start)
        if not (np.array_equal(actual_a, expected[:, 0]) and np.array_equal(actual_b, expected[:, 1])):
            raise RuntimeError("Native seam edge indices differ: construct before other PBD entities")
        compliance = info.stretch_compliance.to_numpy()
        compliance[self.edge_indices] = self.compliance
        info.stretch_compliance.from_numpy(compliance)
        rest = info.len_rest.to_numpy()
        rest[self.edge_indices] = self.rest_gap
        info.len_rest.from_numpy(rest)
        self._ready = True
        self.disable()
        return self.diagnostics()

    def disable(self):
        """Release every seam bond without changing any cloth particle state."""
        if not self._ready:
            raise RuntimeError("Call init_after_build before disable")
        relaxation = self.solver.edges_info.stretch_relaxation.to_numpy()
        relaxation[self.edge_indices] = 0
        self.solver.edges_info.stretch_relaxation.from_numpy(relaxation)
        self.active[:] = False

    def _positions(self, value, name):
        arr = _array(value)
        if arr.shape == (1, self.n_particles, 3):
            arr = arr[0]
        if arr.shape != (self.n_particles, 3) or not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite with shape ({self.n_particles},3)")
        return arr

    def update(self, positions=None, velocities=None, *, normals=None, step=None, allow_new=True):
        """Latch close, slow pairs; optional supplied surface normals must oppose.

        Call after scene.step, supplying measured particle state or letting the
        helper query it. New bonds act on the following native solver substep.
        Existing bonds persist until disable; no force-based breakage is assumed.
        Set allow_new=False outside pressing phases to suppress new activation
        while continuing to measure all pair gaps/speeds and retain existing bonds.
        """
        if not self._ready:
            raise RuntimeError("Call init_after_build before the first scene.step")
        p = self._positions(self.cloth.get_particles_pos() if positions is None else positions, "positions")
        v = self._positions(self.cloth.get_particles_vel() if velocities is None else velocities, "velocities")
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        self._last_distances = np.linalg.norm(p[a] - p[b], axis=1)
        self._last_relative_speeds = np.linalg.norm(v[a] - v[b], axis=1)
        self._allow_new = bool(allow_new)
        eligible = (self._last_distances <= self.capture_distance) & (
            self._last_relative_speeds <= self.max_relative_speed
        )
        self._normals_used = normals is not None
        if normals is not None:
            n = self._positions(normals, "normals")
            na, nb = n[a], n[b]
            norm_product = np.linalg.norm(na, axis=1) * np.linalg.norm(nb, axis=1)
            dot = np.sum(na * nb, axis=1) / np.maximum(norm_product, 1e-12)
            eligible &= (norm_product > 1e-10) & (dot <= self.normal_dot_threshold)
        # Candidate edges share particles in contact mode. Prevent one material
        # node from binding several opposite nodes, including bonds from prior
        # updates; nearest actual contact wins deterministically.
        candidates = np.flatnonzero(eligible & ~self.active & self._allow_new)
        candidates = candidates[np.argsort(self._last_distances[candidates], kind="stable")]
        used_nodes = set(self.pairs[self.active].reshape(-1).tolist())
        new = np.zeros(self.n_pairs, dtype=bool)
        for pair_index in candidates:
            left_node, right_node = self.pairs[pair_index]
            if left_node in used_nodes or right_node in used_nodes:
                continue
            new[pair_index] = True
            used_nodes.add(int(left_node))
            used_nodes.add(int(right_node))
        if new.any():
            relaxation = self.solver.edges_info.stretch_relaxation.to_numpy()
            relaxation[self.edge_indices[new]] = self.relaxation
            self.solver.edges_info.stretch_relaxation.from_numpy(relaxation)
            self.active |= new
            self.events.append({
                "step": int(self._updates if step is None else step),
                "pair_indices": np.flatnonzero(new).tolist(),
                "distance_m": self._last_distances[new].tolist(),
                "relative_speed_m_s": self._last_relative_speeds[new].tolist(),
                "active_count": int(self.active.sum()),
            })
        self._updates += 1
        return self.diagnostics()

    def diagnostics(self):
        active_gaps = self._last_distances[self.active]
        active_pairs = self.pairs[self.active]
        bonded_left = len(np.unique(active_pairs[:, 0]))
        bonded_right = len(np.unique(active_pairs[:, 1]))
        return {
            "model": "Genesis 1.4.3 native PBD distance bonds (private extension)",
            "matching": self.matching,
            "allow_new": self._allow_new,
            "pair_count": self.n_pairs,
            "active_count": int(self.active.sum()),
            "active_fraction": float(self.active.mean()),
            "active_fraction_meaning": "active constraints / preallocated candidate pairs; not patch coverage",
            "left_patch_node_count": len(self.left_patch_nodes),
            "right_patch_node_count": len(self.right_patch_nodes),
            "bonded_left_fraction": bonded_left / len(self.left_patch_nodes),
            "bonded_right_fraction": bonded_right / len(self.right_patch_nodes),
            "capture_distance_m": self.capture_distance,
            "max_relative_speed_m_s": self.max_relative_speed,
            "rest_gap_m": self.rest_gap,
            "compliance_m_per_N": self.compliance,
            "relaxation": self.relaxation,
            "min_pair_distance_m": float(self._last_distances.min()),
            "max_active_distance_m": float(active_gaps.max()) if len(active_gaps) else None,
            "max_material_match_error_m": float(self.match_errors.max()) if self.matching == "material" else None,
            "candidate_flat_y_tolerance_m": self.contact_y_tolerance if self.matching == "contact" else None,
            "normals_gated_on_last_update": self._normals_used,
            "breakage_model": None,
        }

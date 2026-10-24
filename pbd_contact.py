"""Opt-in interleaving of existing Genesis 1.4.3 PBD/contact kernels.

This is a local, forward-only scheduling adapter, not an upstream library edit.
Install after creating all entities and before scene.build(). It supports only
PBD cloth plus rigid bodies whose COUPLED geometries are fixed in the world.
Uncoupled moving robot links may still drive native particle attachments.

The native schedule completes every stretch iteration before self/rigid contact
projection, so those final projections can reintroduce stretch. Here the same
total stretch budget is split between repeated native contact projections. No
particle poses, rest lengths, masses, compliances, thicknesses or task thresholds
are prescribed or changed. Convergence and physical adequacy must be measured;
this is not a triangle-level collision or collision-free guarantee.
"""

from __future__ import annotations

import math

import genesis as gs
from genesis.engine.couplers.legacy_coupler import CLAMPED_INV_DT, LegacyCoupler


class _InterleavedContact:
    def __init__(self, solver, iterations):
        self.solver = solver
        self.sim = solver._sim
        self.coupler = self.sim._coupler
        self.iterations = iterations
        self.runtime_checked = False
        self.original_pre = solver.substep_pre_coupling
        self.original_couple = self.coupler.couple
        self.original_stretch_iterations = int(solver._max_stretch_solver_iterations)
        self.stretch_per_iteration = max(1, math.ceil(self.original_stretch_iterations / iterations))
        self.metadata = {
            "adapter": "local interleaved native PBD and fixed-rigid contact",
            "genesis_version": gs.__version__,
            "outer_contact_iterations": iterations,
            "requested_total_stretch_iterations": self.original_stretch_iterations,
            "native_stretch_iterations_per_outer": self.stretch_per_iteration,
            "actual_total_stretch_iterations": self.stretch_per_iteration * iterations,
            "native_bending_iterations_per_substep": int(solver._max_bending_solver_iterations),
            "bending_schedule": "once after first stretch group; original total iterations retained",
            "prediction_integrations_per_substep": 1,
            "final_velocity_reconstructions_per_substep": 1,
            "collision_underrelaxation": None,
            "collision_diameter_m": float(solver.particle_size),
            "all_coupled_rigid_geometry_must_be_fixed": True,
            "limitations": [
                "private Genesis interfaces; forward simulation only",
                "native contact reactions on fixed obstacles are not calibrated force outputs",
                "no claim of converged stretch/contact or cloth self-intersection freedom",
            ],
        }

    def check_scope(self):
        if not isinstance(self.coupler, LegacyCoupler):
            raise TypeError("Interleaved PBD contact requires the Genesis legacy coupler")
        if getattr(self.sim, "_requires_grad", False):
            raise ValueError("Interleaved PBD contact supports forward simulation only")
        if not self.solver.entities or not all(
            isinstance(entity.material, gs.materials.PBD.Cloth) for entity in self.solver.entities
        ):
            raise ValueError("The PBD solver must contain only cloth entities")
        for name in ("mpm_solver", "sph_solver", "fem_solver", "sf_solver"):
            if getattr(self.sim, name).is_active:
                raise ValueError(f"Interleaved PBD adapter does not support active {name}")
        movable_contacts = [
            geom for geom in self.sim.rigid_solver.geoms if geom.needs_coup and not geom.is_fixed
        ]
        if movable_contacts:
            raise ValueError(
                "Repeated native contact projections require fixed coupled rigid geometry; "
                "moving robot links must have needs_coup=False"
            )

    def pre(self, f):
        solver = self.solver
        if not solver.is_active:
            return
        if not self.runtime_checked:
            self.check_scope()
            self.runtime_checked = True
        # Keep one initial position and one force/gravity integration per true
        # substep. All inner projections reference this same time interval.
        solver._kernel_store_initial_pos(f)
        solver._kernel_apply_external_force(f, self.sim.cur_t)

    def _rigid_contact(self):
        rigid = self.sim.rigid_solver
        self.coupler.kernel_pbd_rigid_collide(
            geoms_state=rigid.dyn_state.geoms,
            geoms_info=rigid.dyn_info.geoms,
            links_state=rigid.dyn_state.links,
            sdf_info=rigid.collider._sdf._sdf_info,
            rigid_info=rigid.rigid_info,
            collider_static_config=rigid.collider.collider_config,
        )

    def couple(self, f):
        solver = self.solver
        if not solver.is_active:
            return
        for iteration in range(self.iterations):
            if solver._n_edges:
                solver._kernel_solve_stretch(f)
            # Do not multiply the material bending solver's configured budget
            # by the number of outer contact sweeps.
            if iteration == 0 and solver._n_inner_edges:
                solver._kernel_solve_bending(f)
            solver._kernel_reorder_particles(f)
            solver._kernel_solve_collision(f)
            if self.coupler._rigid_pbd:
                self._rigid_contact()
            if iteration + 1 < self.iterations:
                solver._kernel_copy_from_reordered(f)

        # Native rigid contact computes provisional contact velocities internally.
        # Reconstruct the final free-particle velocity once from final positions
        # and the original substep ipos; then restore the native link-following
        # corrective velocities, which must not be overwritten afterward.
        solver._kernel_compute_velocity(f)
        if self.coupler._rigid_pbd:
            inv_dt = min(1.0 / solver.substep_dt, CLAMPED_INV_DT)
            self.coupler.kernel_pbd_rigid_solve_animate_particles_by_link(
                inv_dt, self.sim.rigid_solver.dyn_state.links
            )
        # Genesis's unmodified post-coupling hook copies the final reordered
        # state and applies the domain boundary exactly once.


def install_interleaved_contact(scene_or_solver, iterations=8):
    """Install locally before build; return JSON-serializable experiment metadata.

    The solver's configured stretch count is a TOTAL budget, rounded up to an
    integer number per outer iteration. For example 40 iterations become 8x5,
    and 80 become 8x10. Restart a fresh scene/process to compare against baseline.
    """
    solver = getattr(scene_or_solver, "pbd_solver", scene_or_solver)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if solver.is_built:
        raise RuntimeError(
            "Install BEFORE scene.build: build itself executes a simulation step to compile kernels, "
            "so changing the native stretch-loop constant afterward is unsafe"
        )
    if hasattr(solver, "_local_interleaved_contact_adapter"):
        raise RuntimeError("Interleaved PBD contact is already installed on this solver")
    adapter = _InterleavedContact(solver, iterations)
    adapter.check_scope()
    # This is the only native numeric setting changed: distribute the same
    # stretch budget across contact sweeps, instead of applying it all first.
    solver._max_stretch_solver_iterations = adapter.stretch_per_iteration
    solver.substep_pre_coupling = adapter.pre
    adapter.coupler.couple = adapter.couple
    solver._local_interleaved_contact_adapter = adapter
    return dict(adapter.metadata)

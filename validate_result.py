"""Independent, conservative geometry checks for the image-only Genesis trial.

Reads only the requested trial's saved, ACTUAL solver mesh and trajectory. No
original/user simulation files are accessed. A passing report is evidence for
this surrogate model, never identification of the original simulation or a
continuous-time collision-free certificate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from arm_geometry import derive_arm_centerlines

ROOT = Path(__file__).resolve().parent


def scalar(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def mesh_topology(rest, faces):
    edges, counts = np.unique(np.sort(np.concatenate([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0, return_counts=True)
    boundary = edges[counts == 1]
    graph = defaultdict(list)
    for a, b in boundary:
        graph[int(a)].append(int(b)); graph[int(b)].append(int(a))
    loops, unseen = [], set(graph)
    regular = all(len(neighbors) == 2 for neighbors in graph.values())
    if regular:
        while unseen:
            start = min(unseen); current, previous, loop = start, None, []
            while current not in loop:
                loop.append(current); unseen.discard(current)
                following = [q for q in graph[current] if q != previous]
                previous, current = current, following[0]
            if current != start:
                regular = False; break
            loops.append(np.asarray(loop, dtype=int))
    if loops:
        # The actual rest panel is planar XY. Its outer perimeter has largest area.
        area = [abs(np.sum(rest[l, 0]*np.roll(rest[l, 1], -1)-rest[l, 1]*np.roll(rest[l, 0], -1)))/2 for l in loops]
        outer = int(np.argmax(area))
        holes = sorted([l for i, l in enumerate(loops) if i != outer], key=lambda l: rest[l, 0].mean())
    else:
        outer, holes = None, []
    # Number of connected vertex components, including isolated/unused vertices.
    neighbors = defaultdict(list)
    for a, b in edges:
        neighbors[int(a)].append(int(b)); neighbors[int(b)].append(int(a))
    remaining, components = set(range(len(rest))), 0
    while remaining:
        queue = [remaining.pop()]; components += 1
        while queue:
            for v in neighbors[queue.pop()]:
                if v in remaining:
                    remaining.remove(v); queue.append(v)
    info = {'vertices': len(rest), 'faces': len(faces), 'edges': len(edges),
            'boundary_loops': len(loops), 'interior_holes': len(holes),
            'connected_components': components, 'nonmanifold_edges': int(np.sum(counts > 2)),
            'regular_boundary': regular, 'all_vertices_used': bool(len(np.unique(faces)) == len(rest)),
            'hole_vertex_indices': [l.tolist() for l in holes],
            'hole_rest_centers': [rest[l].mean(axis=0).tolist() for l in holes]}
    info['passed'] = bool(regular and len(holes) == 2 and components == 1 and not np.any(counts > 2) and info['all_vertices_used'])
    return edges, holes, info


def cross2(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def simple_polygon(poly):
    # Strict segment intersections; adjacent edges share legitimate endpoints.
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i+1) % n]
        for j in range(i+2, n):
            if (j+1) % n == i:
                continue
            c, d = poly[j], poly[(j+1) % n]
            if cross2(b-a, c-a)*cross2(b-a, d-a) < -1e-14 and cross2(d-c, a-c)*cross2(d-c, b-c) < -1e-14:
                return False
    return True


def inside_polygon(p, poly):
    a, b = poly, np.roll(poly, -1, axis=0)
    # Winding/ray test uses actual boundary edges, not ring-center distance.
    dy = b[:, 1]-a[:, 1]
    crossing = ((a[:, 1] > p[1]) != (b[:, 1] > p[1]))
    x = a[:, 0]+(p[1]-a[:, 1])*(b[:, 0]-a[:, 0])/np.where(abs(dy) > 1e-15, dy, 1)
    return bool(np.count_nonzero(crossing & (p[0] < x)) % 2)


def ring_intersection(ring, centerline):
    """Fit ring plane; intersect arm polyline with its actual projected polygon.

    Failure/uncertain results cannot pass the task gate. The plane fit is explicitly
    an approximation for a deformable nonplanar opening; no center-distance test.
    """
    center = ring.mean(axis=0)
    _, _, basis = np.linalg.svd(ring-center, full_matrices=False)
    uv = (ring-center) @ basis[:2].T
    deviations = (ring-center) @ basis[2]
    planar_error = float(np.max(abs(deviations)))
    is_simple = simple_polygon(uv)
    endpoints = (centerline[[0, -1]]-center) @ basis[2]
    crossings = []
    for i, (a, b) in enumerate(zip(centerline[:-1], centerline[1:])):
        d0, d1 = (a-center) @ basis[2], (b-center) @ basis[2]
        if abs(d0-d1) < 1e-10:
            continue
        t = d0/(d0-d1)
        if not 0 <= t <= 1:
            continue
        q = a+t*(b-a); q2 = (q-center) @ basis[:2].T
        if not inside_polygon(q2, uv):
            continue
        e0, e1 = uv, np.roll(uv, -1, axis=0)
        edge = e1-e0
        s = np.clip(np.sum((q2-e0)*edge, axis=1)/np.maximum(np.sum(edge*edge, axis=1), 1e-20), 0, 1)
        margin = float(np.linalg.norm(q2-e0-s[:, None]*edge, axis=1).min())
        if not any(np.linalg.norm(q-np.asarray(old['point'])) < 1e-6 for old in crossings):
            crossings.append({'segment': i, 'parameter': float(t), 'point': q.tolist(), 'projected_ring_clearance_m': margin})
    opposite = bool(endpoints[0]*endpoints[1] < 0 and np.min(abs(endpoints)) >= .002)
    certain = bool(planar_error <= .012 and is_simple and abs(cross2(uv, np.roll(uv, -1, axis=0)).sum()) > 1e-5)
    threaded = bool(certain and opposite and len(crossings) == 1 and crossings[0]['projected_ring_clearance_m'] >= .003)
    return {'threaded_proxy': threaded, 'geometry_certain': certain,
            'ring_plane_max_deviation_m': planar_error, 'ring_projection_simple': is_simple,
            'proximal_distal_opposite_sides': opposite, 'endpoint_plane_distances_m': endpoints.tolist(),
            'centerline_crossings': crossings}


def sleeve_check(particles, time, holes, manifest):
    stored = manifest.get('bear', {}).get('arm_centerlines', {})
    try:
        centerlines = derive_arm_centerlines(manifest.get('bear', {}).get('components', []))
    except (KeyError, ValueError):
        centerlines = {}
    if len(holes) != 2 or any(side not in centerlines for side in ['left', 'right']):
        return {'passed': False, 'status': 'unavailable', 'reason': 'Need two actual mesh holes and both reconstructed arm centerlines.'}
    audit = {'source': 'Re-derived from saved actual component geometry; stored arm_centerlines never drives the check.',
             'centerlines': centerlines, 'stored_centerlines_replaced': False,
             'stored_component_center_max_error_m': None,
             'endpoint_method': 'Proximal point at <=0.5 torso ellipsoid normalized radius, through shoulder/upper arm/forearm/paw centers, ending 5 mm beyond paw surface along the forearm-to-paw direction.'}
    errors = []
    for side in ['left', 'right']:
        old = np.asarray(stored.get(side, []), dtype=float)
        new = np.asarray(centerlines[side])
        if old.shape == (4, 3):
            errors.append(float(np.linalg.norm(old-new[1:5], axis=1).max()))
            audit['stored_centerlines_replaced'] = True
        elif old.shape == new.shape:
            errors.append(float(np.linalg.norm(old-new, axis=1).max()))
            audit['stored_centerlines_replaced'] |= not np.allclose(old, new, atol=1e-8)
        else:
            audit['stored_centerlines_replaced'] = True
    if errors:
        audit['stored_component_center_max_error_m'] = max(errors)
    def solid_q(points, components):
        distances = []
        for component in components:
            local = (np.asarray(points)-np.asarray(component['position'])) @ np.asarray(component.get('rotation_matrix', np.eye(3)))
            distances.append(np.linalg.norm(local/np.asarray(component['scale']), axis=-1))
        return np.min(distances, axis=0)
    audit['internal_curve_within_solid_union'] = {}
    result, states = {}, {}
    def intervals(flags):
        indices = np.flatnonzero(flags)
        if not len(indices):
            return []
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)
        return [[float(time[g[0]]), float(time[g[-1]])] for g in groups]
    for side, loop in zip(['left', 'right'], holes):
        line = np.asarray(centerlines[side], dtype=float)
        parts = [c for c in manifest['bear']['components'] if c['name'] in [side+'_'+name for name in ['shoulder', 'upper_arm', 'forearm', 'paw']]]
        torso = next(c for c in manifest['bear']['components'] if c['name'] == 'torso')
        internal = np.concatenate([a+np.linspace(0, 1, 33)[:, None]*(b-a) for a, b in zip(line[:-2], line[1:-1])])
        curve_in_solid = bool(solid_q(internal, [torso]+parts).max() <= 1+1e-6)
        audit['internal_curve_within_solid_union'][side] = curve_in_solid
        tests = [ring_intersection(p[loop], line) for p in particles]
        for test in tests:
            crossings = test['centerline_crossings']
            q = solid_q([crossing['point'] for crossing in crossings], parts) if crossings else np.array([])
            # An artificial endpoint extension must not create a false positive
            # when the real paw has not protruded through the opening.
            test['crossings_within_actual_arm_solid'] = bool(len(q) == 1 and q[0] <= 1+1e-6)
            test['crossing_arm_ellipsoid_normalized_radius'] = q.tolist()
            test['threaded_proxy'] &= test['crossings_within_actual_arm_solid'] and curve_in_solid
        flags = np.asarray([t['threaded_proxy'] for t in tests])
        states[side] = flags
        tail = time >= time[-1]-1.0
        result[side] = {'initial': tests[0], 'final': tests[-1],
                        'first_detected_time_s': float(time[np.flatnonzero(flags)[0]]) if flags.any() else None,
                        'threading_intervals_s': intervals(flags),
                        'last_second_threaded_fraction': float(flags[tail].mean()),
                        'passed': bool(not flags[0] and flags[-1] and flags[tail].all())}
    simultaneous = states['left'] & states['right']
    return {'passed': all(v['passed'] for v in result.values()), 'arms': result,
            'centerline_audit': audit,
            'simultaneous_threading_intervals_s': intervals(simultaneous),
            'simultaneous_threading_saved_frame_count': int(simultaneous.sum()),
            'method': 'Actual boundary loop plane/polygon crossed exactly once by actual-component arm centerline extended from torso interior to beyond paw, with endpoints on opposite sides. Intersection must lie inside an actual shoulder/arm/paw solid, not just the artificial extension. Plane error <=12 mm and intersection margin >=3 mm.',
            'limitation': 'Projected disk and reconstructed fixed arm centerline are proxies. Sampled frames do not certify continuous threading or contact-free insertion.'}


def penetration_check(particles, faces, manifest):
    components = [c for c in manifest.get('bear', {}).get('components', []) if c.get('collision', False)]
    if not components:
        return {'passed': False, 'status': 'unavailable', 'reason': 'No collision components in manifest.'}
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    all_worst, final_worst, final_inside = 0., 0., 0
    per_component = {c['name']: {'max_penetration_proxy_m': 0., 'final_max_penetration_proxy_m': 0.} for c in components}
    worst_frame = 0
    for frame_idx, p in enumerate(particles):
        # Triangle centroids and edge midpoints expose crossings missed by vertices.
        samples = np.concatenate([p, p[faces].mean(axis=1), p[edges].mean(axis=1)])
        for c in components:
            shape = c.get('analytic_shape', {})
            center = np.asarray(shape.get('center', c['position']))
            radii = np.asarray(shape.get('radii', c['scale']))
            rotation = np.asarray(shape.get('rotation_matrix', c.get('rotation_matrix', np.eye(3))))
            q = np.linalg.norm(((samples-center) @ rotation)/radii, axis=1)
            # min(radius)*(1-q) is a conservative depth proxy, not exact SDF.
            penetration = np.maximum(1-q, 0)*radii.min()
            worst = float(penetration.max())
            per_component[c['name']]['max_penetration_proxy_m'] = max(worst, per_component[c['name']]['max_penetration_proxy_m'])
            if worst > all_worst:
                all_worst, worst_frame = worst, frame_idx
            if frame_idx == len(particles)-1:
                per_component[c['name']]['final_max_penetration_proxy_m'] = worst
                final_worst = max(final_worst, worst)
                final_inside += int(np.sum(penetration > .006))
    table_top = manifest.get('table', {}).get('top_z', .75)
    table_depth = np.maximum(table_top-particles[..., 2], 0)
    return {'passed': bool(all_worst <= .010 and final_worst <= .006 and table_depth.max() <= .006),
            'max_penetration_proxy_m': all_worst, 'worst_saved_frame': worst_frame,
            'final_max_penetration_proxy_m': final_worst, 'final_sample_component_pairs_deeper_than_6mm': final_inside,
            'table_max_vertex_penetration_m': float(table_depth.max()),
            'components': per_component,
            'thresholds_m': {'all_frames_bear': .010, 'final_bear': .006, 'table': .006},
            'method': 'All saved frames, actual vertices, triangle centroids and edge midpoints, against the union of manifest collision ellipsoids.',
            'limitation': 'Discrete analytical proxy, not exact Genesis mesh/SDF clearance or exhaustive triangle contact. Does not test cloth self-intersection or robot collision.'}


def material_section_at_y(rest, faces, cut):
    """Intersect actual rest triangles; reject every gap, including real holes."""
    intervals, points = [], {}
    for face in faces:
        local = []
        for a, b in zip(face, np.roll(face, -1)):
            ya, yb = rest[a, 1], rest[b, 1]
            if (ya < cut) != (yb < cut):
                w = float((cut-ya)/(yb-ya)); x = float(rest[a, 0]*(1-w)+rest[b, 0]*w)
                key = round(x, 9)
                points[key] = (int(a), int(b), w)
                local.append(x)
        if len(local) == 2:
            intervals.append((min(local), max(local)))
    intervals.sort()
    if not intervals:
        return None
    right, continuous = intervals[0][1], True
    for left, high in intervals[1:]:
        if left-right > 1e-6:
            continuous = False
        right = max(high, right)
    keys = sorted(points)
    spans = abs(keys[0]-rest[:, 0].min()) < 1e-5 and abs(keys[-1]-rest[:, 0].max()) < 1e-5
    return ([points[k] for k in keys], cut) if continuous and spans else None


def material_sections(rest, faces):
    """Enumerate complete material rows from actual boundary geometry.

    Between consecutive boundary-vertex Y levels the cross-section's topology
    cannot change. Test every open band midpoint. This avoids numerical boundary
    coincidences and never bridges a sleeve hole. Material Y orientation does
    not imply world height after the panel has been lifted or turned over.
    """
    edges, counts = np.unique(np.sort(np.concatenate([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0, return_counts=True)
    boundary_vertices = np.unique(edges[counts == 1])
    levels = np.unique(np.round(rest[boundary_vertices, 1], 9))
    sections = []
    for low, high in zip(levels[:-1], levels[1:]):
        if high-low <= 1e-7:
            continue
        section = material_section_at_y(rest, faces, float((low+high)/2))
        if section is not None:
            sections.append(section)
    return sections


def material_section(rest, faces):
    """Compatibility helper: return the lowest complete rest-material row."""
    sections = material_sections(rest, faces)
    return sections[0] if sections else None


def horizontal_coverage(p, faces, center, radii, z, bins=180, return_mask=False):
    occupied = np.zeros(bins, dtype=bool)
    scale = np.sqrt(max(0., 1-((z-center[2])/radii[2])**2))
    for face in faces:
        triangle = p[face]; cuts = []
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
            if (a[2] < z) != (b[2] < z):
                cuts.append(a+(b-a)*(z-a[2])/(b[2]-a[2]))
        if len(cuts) != 2:
            continue
        xy = (np.asarray(cuts)[:, :2]-center[:2])/(radii[:2]*scale)
        # Only cloth plausibly adjacent to the belly counts toward coverage.
        if np.linalg.norm(xy, axis=1).min() < .90 or np.linalg.norm(xy, axis=1).max() > 1.65:
            continue
        a, b = np.arctan2(xy[:, 1], xy[:, 0])
        delta = np.arctan2(np.sin(b-a), np.cos(b-a))
        sample = a+np.linspace(0, delta, max(2, int(abs(delta)/np.pi*bins)+2))
        occupied[np.floor((sample % (2*np.pi))/(2*np.pi)*bins).astype(int) % bins] = True
    return occupied if return_mask else float(occupied.mean())


def hole_cap_faces(points, loop):
    """Ear-clip a simple, near-planar actual hole polygon for envelope diagnostics.

    No cap is added to simulation or collision geometry. These triangles use only
    vertices of an observed open boundary and never join different boundaries.
    """
    ring = points[loop]
    _, _, axes = np.linalg.svd(ring-ring.mean(axis=0), full_matrices=False)
    uv = (ring-ring.mean(axis=0)) @ axes[:2].T
    if not simple_polygon(uv) or np.max(abs((ring-ring.mean(axis=0)) @ axes[2])) > .012:
        return None
    signed_area = float(cross2(uv, np.roll(uv, -1, axis=0)).sum())
    if abs(signed_area) <= 1e-5:
        return None
    order = list(range(len(loop)))
    if signed_area < 0:
        order.reverse()
    triangles = []
    while len(order) > 3:
        found = False
        for i, current in enumerate(order):
            previous, following = order[i-1], order[(i+1) % len(order)]
            a, b, c = uv[[previous, current, following]]
            if cross2(b-a, c-b) <= 1e-12:
                continue
            other = [j for j in order if j not in [previous, current, following]]
            inside = ((cross2(b-a, uv[other]-a) >= -1e-12) &
                      (cross2(c-b, uv[other]-b) >= -1e-12) &
                      (cross2(a-c, uv[other]-c) >= -1e-12))
            if inside.any():
                continue
            triangles.append(loop[[previous, current, following]])
            del order[i]; found = True; break
        if not found:
            return None
    triangles.append(loop[order])
    return np.asarray(triangles, dtype=int)


def wrap_check(rest, particles, faces, manifest, holes=None, sleeves=None):
    torso = next((c for c in manifest.get('bear', {}).get('components', []) if c['name'] == 'torso'), None)
    sections = material_sections(rest, faces)
    if torso is None or not sections:
        return {'passed': False, 'status': 'unavailable', 'reason': 'Need torso and continuous material path across actual rest mesh.'}
    center, radii = np.asarray(torso['position']), np.asarray(torso['scale'])
    p = particles[-1]
    path_checks = []
    for triples, cut in sections:
        a, b, w = [np.asarray(v) for v in zip(*triples)]; a, b = a.astype(int), b.astype(int)
        path = p[a]*(1-w[:, None])+p[b]*w[:, None]
        xy = (path[:, :2]-center[:2])/radii[:2]
        nxt = np.roll(xy, -1, axis=0)
        winding = float(np.sum(np.arctan2(cross2(xy, nxt), np.sum(xy*nxt, axis=1)))/(2*np.pi))
        endpoint_gap = float(np.linalg.norm(path[0]-path[-1]))
        height_fraction = float(np.mean((path[:, 2] > center[2]-.13) & (path[:, 2] < center[2]+.12)))
        path_checks.append({'rest_y': cut, 'inset_from_minimum_rest_y_m': float(cut-rest[:, 1].min()),
                            'winding': winding, 'endpoint_gap_m': endpoint_gap,
                            'height_range_m': [float(path[:, 2].min()), float(path[:, 2].max())],
                            'mean_height_m': float(path[:, 2].mean()), 'belly_height_fraction': height_fraction,
                            'passed': bool(abs(abs(winding)-1) <= .15 and endpoint_gap <= .04 and height_fraction >= .95)})
    passing_paths = [check for check in path_checks if check['passed']]
    selected = min(passing_paths or path_checks, key=lambda check: abs(check['mean_height_m']-center[2]))
    cut, winding, endpoint_gap = selected['rest_y'], selected['winding'], selected['endpoint_gap_m']
    levels = [float(center[2]+dz) for dz in [-.025, 0, .025]]
    raw_masks = [horizontal_coverage(p, faces, center, radii, z, return_mask=True) for z in levels]
    coverage = [float(mask.mean()) for mask in raw_masks]
    caps = [hole_cap_faces(p, loop) for loop in holes] if holes is not None and len(holes) == 2 else []
    caps_valid = len(caps) == 2 and all(cap is not None for cap in caps)
    cap_masks = [horizontal_coverage(p, np.concatenate(caps), center, radii, z, return_mask=True) for z in levels] if caps_valid else [np.zeros_like(mask) for mask in raw_masks]
    envelope = [float((raw | cap).mean()) for raw, cap in zip(raw_masks, cap_masks)]
    cap_additions = [float((cap & ~raw).mean()) for raw, cap in zip(raw_masks, cap_masks)]
    both_final_threaded = bool(sleeves and all(sleeves.get('arms', {}).get(side, {}).get('final', {}).get('threaded_proxy', False) for side in ['left', 'right']))
    use_envelope = caps_valid and both_final_threaded
    plausible_height = selected['belly_height_fraction'] >= .95
    assessed_coverage = envelope if use_envelope else coverage
    passed = bool(passing_paths) and min(assessed_coverage) >= .85
    return {'passed': bool(passed), 'closed_projection_winding_approx': winding,
            'material_path_endpoint_gap_m': endpoint_gap, 'material_path_rest_y': cut,
            'material_path_inset_from_lower_hem_m': float(cut-rest[:, 1].min()),
            'material_path_selection': 'Evaluate every full-width open boundary-Y band midpoint and require at least one actual material loop with winding, endpoint gap and belly-height criteria. Any material gap >1 micrometer rejects a row. No virtual caps are used for paths; no assumption that minimum rest Y is world-bottom. The displayed passing candidate is nearest torso-center height.',
            'material_path_candidates': path_checks,
            'material_path_candidate_count': len(path_checks),
            'material_path_passing_count': len(passing_paths),
            'material_path_height_range_m': selected['height_range_m'],
            'material_path_belly_height_compatible': plausible_height,
            'torso_cross_section_z_m': levels, 'torso_angular_coverage_fraction': coverage,
            'garment_envelope_angular_coverage_fraction': envelope,
            'angular_fraction_added_only_by_true_hole_caps': cap_additions,
            'hole_caps_valid': caps_valid,
            'hole_cap_triangle_counts': [len(cap) for cap in caps] if caps_valid else [],
            'both_actual_holes_threaded_at_final_state': both_final_threaded,
            'envelope_used_for_wrap_gate': use_envelope,
            'hole_envelope_method': 'Virtually cap only the two actual sleeve boundary loops by ear-clipped triangles; same height/radius filters and 85% threshold. Count their missing angular bins only when both actual hole/arm crossings are confirmed. Caps are analytical only, never added to the solver.',
            'thresholds': {'minimum_each_level_coverage': .85, 'maximum_path_endpoint_gap_m': .04, 'absolute_winding_target': 1},
            'limitation': 'Uses actual mesh/segments and optional true-hole virtual caps. XY projected winding and angular coverage are approximations; they do not establish a topological enclosure in 3D. Sleeve caps describe garment extent, not cloth covering the arm apertures.'}


def seam_check(output, particles, time, metrics):
    logs = metrics.get('logs', [])
    previously_attached, release, releases = False, None, []
    for row in logs:
        attached = bool(row.get('grasp_attached', False))
        if previously_attached and not attached:
            releases.append(float(row['time']))
        previously_attached = attached
    if releases:
        release = releases[-1]
    info = {'passed': False, 'grasp_release_logged': release is not None, 'release_time_s': release,
            'all_logged_release_times_s': releases,
            'model': 'Permanent compliant surrogate seam, not calibrated Velcro; no breakage law.'}
    if release is None:
        info['reason'] = 'No transition from attached grasps to released grasps is logged.'
        return info
    free = time >= release
    info['free_hold_duration_s'] = float(time[-1]-release)
    free_logs = [r for r in logs if float(r['time']) >= release]
    uninterrupted = bool(free_logs and all('grasp_attached' in r and not r['grasp_attached'] for r in free_logs))
    info['final_hold_uninterrupted_without_grasp'] = uninterrupted
    info['final_release_confirmed_in_metrics'] = metrics.get('released') is True
    if not uninterrupted or not info['final_release_confirmed_in_metrics']:
        info['reason'] = 'The last release is followed by another grasp, or final release is not confirmed. Temporary release for regrasp does not count as free hold.'
        return info
    if not free.any():
        info['reason'] = 'No saved particle frame after the final grasp release.'
        return info
    count = lambda r: int(r.get('velcro', {}).get('active_count', r.get('velcro', {}).get('latched_count', 0)))
    info['minimum_active_bonds_during_free_hold'] = min((count(r) for r in free_logs), default=0)
    state_path = output/'seam_state.npz'
    if state_path.exists():
        with np.load(state_path, allow_pickle=False) as state:
            pairs = np.asarray(state['pairs'], dtype=int)
            active = np.asarray(state['active'], dtype=bool)
    elif 'seam_pairs' in metrics and 'seam_active' in metrics:
        pairs = np.asarray(metrics['seam_pairs'], dtype=int)
        active = np.asarray(metrics['seam_active'], dtype=bool)
    else:
        info['reason'] = 'Missing seam_state.npz or seam_pairs/seam_active; logged active counts alone do not independently verify seam separation.'
        return info
    if pairs.ndim != 2 or pairs.shape[1] != 2 or active.shape != (len(pairs),) or not active.any():
        info['reason'] = 'No valid active seam pair indices.'
        return info
    if pairs.min() < 0 or pairs.max() >= particles.shape[1]:
        info['reason'] = 'Seam particle indices do not address this saved trajectory.'
        return info
    dt = metrics.get('solver_dt_s')
    events_path = output/'seam_events.json'
    events = json.loads(events_path.read_text()) if events_path.exists() else metrics.get('seam_events')
    if not isinstance(dt, (int, float)) or isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0 or not isinstance(events, list):
        info['reason'] = 'Need positive saved solver_dt_s and the actual seam activation-event ledger; sample spacing is not a substitute for solver dt.'
        return info
    activation_steps = np.full(len(pairs), np.inf)
    errors = []
    control_steps = np.rint(time/dt).astype(np.int64)
    if not np.allclose(time/dt, control_steps, atol=1e-6, rtol=0):
        errors.append('Saved control times are not integer multiples of solver_dt_s.')
    previous_step, activated_count = -1, 0
    for event in events:
        step, indices = event.get('step'), event.get('pair_indices')
        if not isinstance(step, int) or isinstance(step, bool) or step < previous_step or step < 0 or step > control_steps[-1]:
            errors.append('Activation steps must be nonnegative, ordered integers within the recorded control-step range.'); break
        if not isinstance(indices, list) or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(pairs) for i in indices):
            errors.append('Invalid activation pair indices.'); break
        if len(set(indices)) != len(indices) or np.isfinite(activation_steps[indices]).any():
            errors.append('A permanent seam pair is activated more than once.'); break
        activation_steps[indices] = step
        activated_count += len(indices)
        if event.get('active_count') != activated_count:
            errors.append('Event active_count disagrees with cumulative unique activations.'); break
        previous_step = step
    if not np.array_equal(np.isfinite(activation_steps), active):
        errors.append('Final active mask disagrees with the activation ledger; unlogged breakage or activation is not assumed.')
    for row in logs:
        step = float(row['time'])/dt
        if not np.isclose(step, round(step), atol=1e-6, rtol=0) or count(row) != int(np.sum(activation_steps <= round(step))):
            errors.append('Saved active-count timeline disagrees with activation event steps.'); break
    info['activation_ledger_consistent'] = not errors
    info['solver_dt_s'] = dt
    if errors:
        info['reason'] = 'Invalid or inconsistent seam activation timing.'
        info['activation_ledger_errors'] = errors
        return info
    selected = pairs[active]
    active_at_frame = activation_steps[active][None, :] <= control_steps[free, None]
    gaps = np.linalg.norm(particles[free][:, selected[:, 0]]-particles[free][:, selected[:, 1]], axis=-1)
    actual_counts = active_at_frame.sum(axis=1)
    if not actual_counts.all():
        info['reason'] = 'At least one free-hold frame has no active seam bonds.'
        return info
    max_active_gap = float(gaps[active_at_frame].max())
    release_active = active_at_frame[0]
    info.update({'active_pair_count': len(selected),
                 'minimum_active_bonds_during_free_hold': int(actual_counts.min()),
                 'release_active_pair_count': int(release_active.sum()),
                 'release_active_bonds_tracked_every_free_hold_frame': bool(active_at_frame[:, release_active].all()),
                 'max_release_active_pair_gap_during_free_hold_m': float(gaps[:, release_active].max()),
                 'unactivated_excluded_distinct_pair_count': int((~active_at_frame).any(axis=0).sum()),
                 'unactivated_excluded_pair_frame_count': int((~active_at_frame).sum()),
                 'max_active_pair_gap_during_free_hold_m': max_active_gap,
                 'final_active_pair_gap_p95_m': float(np.percentile(gaps[-1], 95)),
                 'cloth_center_drop_during_free_hold_m': float(particles[free][0, :, 2].mean()-particles[-1, :, 2].mean())})
    info['passed'] = bool(info['free_hold_duration_s'] >= 1 and info['minimum_active_bonds_during_free_hold'] >= 3 and max_active_gap <= .04 and np.percentile(gaps[-1], 95) <= .025)
    info['activation_time_method'] = 'A pair is evaluated only when event.step <= saved control time / saved solver_dt_s. Saved positions are post-step states; every bond active at release remains evaluated in every subsequent frame, and later bonds enter at activation.'
    info['thresholds_m'] = {'maximum_active_gap_during_free_hold': .04, 'final_active_gap_p95': .025}
    info['limitation'] = 'Survival only for the modeled permanent seam and tested duration, without an applied peel/load challenge. Not-yet-activated candidate pairs are not bonds and are excluded until their recorded activation.'
    return info


def validate(output):
    output = Path(output)
    required = [output/name for name in ['trajectory.npz', 'scene_manifest.json', 'metrics.json']]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        return {'task_success': False, 'status': 'incomplete', 'missing_files': missing}
    metrics = json.loads((output/'metrics.json').read_text())
    manifest = json.loads((output/'scene_manifest.json').read_text())
    with np.load(output/'trajectory.npz', allow_pickle=False) as data:
        particles = np.asarray(data['particles'], dtype=float)
        faces = np.asarray(data['faces'], dtype=int)
        rest = np.asarray(data['rest'], dtype=float)
        time = np.asarray(data['time'], dtype=float)
    shape_ok = particles.ndim == 3 and particles.shape[-1] == 3 and rest.shape == particles.shape[1:] and time.shape == (len(particles),) and len(particles) > 1
    valid = shape_ok and np.isfinite(particles).all() and np.isfinite(rest).all() and np.isfinite(time).all() and np.all(np.diff(time) > 0)
    face_ok = faces.ndim == 2 and faces.shape[1] == 3 and faces.size > 0 and faces.min() >= 0 and faces.max() < len(rest)
    report = {'schema_version': 1, 'task_success': False,
              'interpretation': 'All checks concern the inferred surrogate and actual saved Genesis solver state. Original parameters/motion are not observable from four initial images.',
              'numerical_finiteness': {'passed': bool(valid and face_ok), 'trajectory_shape': list(particles.shape), 'finite_particles': bool(np.isfinite(particles).all()), 'valid_faces': bool(face_ok)},
              'uncertainty_policy': 'Unavailable, uncertain or failing required checks fail the overall task gate.'}
    if not valid or not face_ok:
        report['status'] = 'invalid_saved_data'; return report
    edges, holes, topology = mesh_topology(rest, faces)
    rest_lengths = np.linalg.norm(rest[edges[:, 0]]-rest[edges[:, 1]], axis=-1)
    ratio = np.linalg.norm(particles[:, edges[:, 0]]-particles[:, edges[:, 1]], axis=-1)/np.maximum(rest_lengths, 1e-12)
    stretch = {'passed': bool(rest_lengths.min() > 1e-9 and ratio.max() <= 1.50 and np.percentile(ratio, 99) <= 1.20),
               'maximum_edge_length_ratio': float(ratio.max()), 'p99_edge_length_ratio': float(np.percentile(ratio, 99)),
               'final_p99_edge_length_ratio': float(np.percentile(ratio[-1], 99)),
               'minimum_edge_length_ratio': float(ratio.min()),
               'thresholds': {'maximum_ratio': 1.50, 'p99_ratio': 1.20},
               'note': 'Original triangle edges only; excludes newly added adhesion constraints. Thresholds are explicit engineering priors, not measured textile limits.'}
    sleeves = sleeve_check(particles, time, holes, manifest)
    report.update({'topology': topology, 'edge_stretch': stretch,
                   'sleeve_threading': sleeves,
                   'collision_proxy': penetration_check(particles, faces, manifest),
                   'torso_wrap': wrap_check(rest, particles, faces, manifest, holes, sleeves),
                   'seam_free_hold': seam_check(output, particles, time, metrics)})
    names = ['numerical_finiteness', 'topology', 'edge_stretch', 'sleeve_threading', 'collision_proxy', 'torso_wrap', 'seam_free_hold']
    report['failed_checks'] = [n for n in names if not report[n]['passed']]
    report['task_success'] = not report['failed_checks']
    report['status'] = 'surrogate_checks_passed' if report['task_success'] else 'task_not_demonstrated'
    report['not_certified'] = ['original simulation equivalence', 'continuous-time collision freedom', 'cloth self-intersection freedom', 'robot collision avoidance or force feasibility', 'real hook-and-loop strength', 'material identification']
    return report


def self_test():
    theta = np.linspace(0, 2*np.pi, 32, endpoint=False)
    ring = np.stack([.04*np.cos(theta), .04*np.sin(theta), np.zeros_like(theta)], axis=1)
    assert ring_intersection(ring, np.array([[0, 0, -.05], [0, 0, .05]]))['threaded_proxy']
    assert not ring_intersection(ring, np.array([[.1, 0, -.05], [.1, 0, .05]]))['threaded_proxy']
    assert not ring_intersection(ring, np.array([[0, 0, .01], [0, 0, .05]]))['threaded_proxy']
    folded = ring.copy(); folded[::2, 2] += .08
    assert not ring_intersection(folded, np.array([[0, 0, -.05], [0, 0, .15]]))['geometry_certain']
    print('Validation geometry self-checks passed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/native_trial', help='Absolute trial directory or path relative to this script.')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT/output
    report = validate(output)
    output.mkdir(parents=True, exist_ok=True)
    # Historical trajectories may be deliberately removed to free local disk.
    # Never replace a completed audit merely because its source was cleaned up.
    report_path = output/('validation_incomplete.json' if report['status'] == 'incomplete' else 'validation.json')
    report_path.write_text(json.dumps(report, indent=2, default=scalar)+'\n')
    print(json.dumps({'task_success': report['task_success'], 'status': report['status'], 'failed_checks': report.get('failed_checks'), 'report': str(report_path)}, indent=2))


if __name__ == '__main__':
    main()

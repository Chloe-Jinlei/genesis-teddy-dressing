"""Create image-inferred teddy/garment assets; Pillow generates the cloth texture.

All dimensions and geometry here are explicit reconstruction assumptions.  No
original simulation assets, hidden project data, or fitted physical parameters
are used.  Run this file directly, or call ``build_assets(output_dir)``.

The garment OBJ is a *single connected open sheet with two real holes*.  It is
in local XY coordinates: X is the long axis and Y is the short axis.  Its initial
world placement and the world-space teddy components are in manifest.json.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from arm_geometry import derive_arm_centerlines


BROWN = (0.43, 0.245, 0.115, 1.0)
HEAD_BROWN = (0.48, 0.285, 0.145, 1.0)
BEIGE = (0.66, 0.57, 0.38, 1.0)
MUZZLE = (0.72, 0.64, 0.46, 1.0)
DARK = (0.045, 0.027, 0.016, 1.0)
CLOTH = (0.77, 0.77, 0.66, 1.0)
HOOK = (0.055, 0.18, 0.22, 1.0)
LOOP = (0.42, 0.25, 0.12, 1.0)

# Estimated from the supplied overhead view. +Y is the panel's far edge;
# both holes lie toward the near edge, with the left hole farther forward.
GARMENT_HOLE_CENTERS = ((-.205, -.045), (.205, -.015))
GARMENT_HOLE_SIZE = (.066, .068)


def _normalized(p):
    s = math.sqrt(sum(a * a for a in p))
    return tuple(a / s for a in p)


def icosphere(subdivisions=3):
    t = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    vertices = [_normalized(v) for v in vertices]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    for _ in range(subdivisions):
        midpoints = {}

        def midpoint(a, b):
            key = tuple(sorted((a, b)))
            if key not in midpoints:
                midpoints[key] = len(vertices)
                vertices.append(_normalized(tuple((x + y) / 2 for x, y in zip(vertices[a], vertices[b]))))
            return midpoints[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        faces = new_faces
    return vertices, faces


def _rotation_xyz(degrees):
    # R = Rz @ Ry @ Rx, applied to local column vectors.
    x, y, z = [math.radians(a) for a in degrees]
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    return (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )


def _transformed(vertices, component):
    r = _rotation_xyz(component['euler_xyz_deg'])
    scale, position = component['scale'], component['position']
    out = []
    for v in vertices:
        p = [v[i] * scale[i] for i in range(3)]
        out.append(tuple(sum(r[i][j] * p[j] for j in range(3)) + position[i] for i in range(3)))
    return out


def teddy_components():
    parts = []

    def add(name, p, radii, color=BROWN, euler=(0, 0, 0), collision=True):
        r = _rotation_xyz(euler)
        parts.append({
            'name': name, 'mesh': 'unit_sphere.obj', 'position': list(p),
            'scale': list(radii), 'euler_xyz_deg': list(euler),
            'rotation_matrix': [list(row) for row in r], 'color': list(color),
            'collision': collision,
            'analytic_shape': {'type': 'ellipsoid', 'center': list(p), 'radii': list(radii),
                               'rotation_matrix': [list(row) for row in r]},
        })

    add('torso', (0, .07, .93), (.132, .100, .15))
    add('head', (0, .055, 1.095), (.111, .095, .095), HEAD_BROWN)
    # The ears add 1.5 cm to the nominal 44 cm head-to-table height.
    for sign, side in ((-1, 'left'), (1, 'right')):
        add(f'{side}_ear', (sign * .084, .059, 1.166), (.037, .029, .039), HEAD_BROWN)
        add(f'{side}_inner_ear', (sign * .084, .035, 1.166), (.024, .009, .025), BEIGE, collision=False)
        add(f'{side}_foot', (sign * .059, .027, .780), (.047, .063, .030), HEAD_BROWN)
        add(f'{side}_shoulder', (sign * .123, .065, 1.015), (.031, .030, .034), BROWN,
            euler=(0, -sign * 28, 0))
        add(f'{side}_upper_arm', (sign * .145, .055, .994), (.026, .028, .041), HEAD_BROWN,
            euler=(0, -sign * 32, 0))
        add(f'{side}_forearm', (sign * .166, .042, .964), (.022, .026, .034), HEAD_BROWN,
            euler=(0, -sign * 16, 0))
        add(f'{side}_paw', (sign * .175, .035, .941), (.025, .027, .026), HEAD_BROWN)
    add('belly_patch', (0, -.016, .929), (.090, .025, .111), BEIGE, collision=True)
    add('muzzle', (0, -.026, 1.068), (.060, .036, .038), MUZZLE, collision=False)
    for sign, side in ((-1, 'left'), (1, 'right')):
        add(f'{side}_eye', (sign * .037, -.030, 1.109), (.0076, .0053, .0090), DARK, collision=False)
        add(f'{side}_eye_glint', (sign * .038 - .0015, -.0351, 1.112), (.0016, .0008, .0017),
            (.90, .88, .79, 1), collision=False)
    add('nose', (0, -.061, 1.081), (.010, .005, .0074), DARK, collision=False)
    add('mouth_stem', (0, -.0615, 1.069), (.0014, .0018, .0050), DARK, collision=False)
    add('mouth_left', (-.0051, -.0608, 1.065), (.0060, .0018, .0015), DARK,
        euler=(0, -15, 0), collision=False)
    add('mouth_right', (.0051, -.0608, 1.065), (.0060, .0018, .0015), DARK,
        euler=(0, 15, 0), collision=False)
    return parts


def _write_obj(path, vertices, faces, materials=None, face_materials=None, uv=None):
    lines = ['# Procedural image-only reconstruction; coordinates in meters.']
    if materials:
        lines.append(f'mtllib {path.stem}.mtl')
    for v in vertices:
        lines.append('v ' + ' '.join(f'{a:.9f}' for a in v))
    if uv:
        for u, v in uv:
            lines.append(f'vt {u:.9f} {v:.9f}')
    active = None
    for index, face in enumerate(faces):
        if face_materials and face_materials[index] != active:
            active = face_materials[index]
            lines.append(f'usemtl {active}')
        if uv:
            lines.append('f ' + ' '.join(f'{i + 1}/{i + 1}' for i in face))
        else:
            lines.append('f ' + ' '.join(str(i + 1) for i in face))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if materials:
        mtl = []
        for name, rgba in materials.items():
            mtl.extend((f'newmtl {name}', 'Ka 0.08 0.08 0.08',
                        'Kd ' + ' '.join(f'{c:.6f}' for c in rgba[:3]),
                        'Ks 0.015 0.015 0.015', 'Ns 4', f'd {rgba[3]:.5f}', 'illum 2', ''))
        path.with_suffix('.mtl').write_text('\n'.join(mtl), encoding='utf-8')


def _write_ply(path, vertices, faces, colors):
    lines = [
        'ply', 'format ascii 1.0', 'comment Image-only procedural reconstruction',
        f'element vertex {len(vertices)}', 'property float x', 'property float y', 'property float z',
        'property uchar red', 'property uchar green', 'property uchar blue',
        f'element face {len(faces)}', 'property list uchar int vertex_indices', 'end_header',
    ]
    for p, c in zip(vertices, colors):
        lines.append(' '.join(f'{v:.9f}' for v in p) + ' ' + ' '.join(str(round(v * 255)) for v in c[:3]))
    lines.extend('3 ' + ' '.join(map(str, face)) for face in faces)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _write_garment_texture(path, width=1024, height=256):
    """Generate subtle woven cloth and two closure strips without source art."""
    from PIL import Image

    stripe_width = round(width * .035 / .80)
    pixels = []
    for y in range(height):
        for x in range(width):
            base = HOOK if x < stripe_width else LOOP if x >= width - stripe_width else CLOTH
            is_closure = x < stripe_width or x >= width - stripe_width
            warp = math.sin(2 * math.pi * x / (5 if is_closure else 6))
            weft = math.sin(2 * math.pi * y / (5 if is_closure else 6))
            noise = ((x * 374761393 + y * 668265263) % 17 - 8) / 8
            modulation = 1 + .018 * warp + .015 * weft + .005 * noise
            # Small contrasting rows reproduce the coarse Velcro texture.
            if is_closure and x % 5 <= 1 and y % 5 <= 1:
                modulation += .13
            # A faint stitched hem is part of the same material/texture.
            if (y in (3, 4, height - 4, height - 5)) and x % 8 < 5:
                modulation *= .90
            pixels.append(tuple(max(0, min(255, round(c * 255 * modulation))) for c in base[:3]))
    texture = Image.new('RGB', (width, height))
    texture.putdata(pixels)
    texture.save(path)


def _boundary_loops(vertices, faces):
    counts = Counter()
    for a, b, c in faces:
        counts.update((tuple(sorted(e)) for e in ((a, b), (b, c), (c, a))))
    graph = defaultdict(list)
    remaining = set()
    for edge, count in counts.items():
        if count > 2:
            raise AssertionError('Non-manifold garment edge')
        if count == 1:
            a, b = edge
            graph[a].append(b)
            graph[b].append(a)
            remaining.add(edge)
    assert all(len(neighbors) == 2 for neighbors in graph.values()), 'Non-manifold boundary vertex'
    loops = []
    while remaining:
        start, current = min(remaining)
        loop, previous = [start], start
        remaining.remove((start, current))
        while current != start:
            loop.append(current)
            a, b = graph[current]
            nxt = b if a == previous else a
            remaining.remove(tuple(sorted((current, nxt))))
            previous, current = current, nxt
        area = sum(vertices[a][0] * vertices[b][1] - vertices[b][0] * vertices[a][1]
                   for a, b in zip(loop, loop[1:] + loop[:1])) / 2
        if area < 0:
            loop.reverse()
            area = -area
        center = [sum(vertices[i][d] for i in loop) / len(loop) for d in range(3)]
        loops.append({'indices': loop, 'area_m2': area, 'centroid_local': center})
    return sorted(loops, key=lambda loop: -loop['area_m2'])


def garment_mesh(nx=100, ny=24):
    length, width = .80, .18
    holes = GARMENT_HOLE_CENTERS
    half_width, half_height = (size / 2 for size in GARMENT_HOLE_SIZE)
    corner = .010
    assert all(abs(y) + half_height < width / 2 for _, y in holes), 'Armholes must remain interior'
    vertices = [(-length / 2 + length * i / nx, -width / 2 + width * j / ny, 0.0)
                for j in range(ny + 1) for i in range(nx + 1)]
    faces = []

    def in_hole(x, y):
        for hx, hy in holes:
            dx = max(abs(x - hx) - (half_width - corner), 0)
            dy = max(abs(y - hy) - (half_height - corner), 0)
            if dx * dx + dy * dy < corner * corner:
                return True
        return False

    for j in range(ny):
        for i in range(nx):
            x = -length / 2 + length * (i + .5) / nx
            y = -width / 2 + width * (j + .5) / ny
            if in_hole(x, y):
                continue
            a = j * (nx + 1) + i
            b, c, d = a + 1, a + nx + 2, a + nx + 1
            if (i + j) % 2:
                faces.extend(((a, b, d), (b, c, d)))
            else:
                faces.extend(((a, b, c), (a, c, d)))
    used = sorted({index for face in faces for index in face})
    remap = {old: new for new, old in enumerate(used)}
    vertices = [vertices[i] for i in used]
    faces = [tuple(remap[i] for i in face) for face in faces]
    loops = _boundary_loops(vertices, faces)
    assert len(loops) == 3, 'Garment must have exactly two true interior holes'
    hole_loops = sorted(loops[1:], key=lambda loop: loop['centroid_local'][0])
    for side, loop in zip(('left', 'right'), hole_loops):
        loop['name'] = side + '_armhole'

    adjacency = defaultdict(set)
    edges = set()
    for face in faces:
        for a, b in zip(face, face[1:] + face[:1]):
            edges.add(tuple(sorted((a, b))))
            adjacency[a].add(b)
            adjacency[b].add(a)
    visited, todo = set(), [0]
    while todo:
        a = todo.pop()
        if a not in visited:
            visited.add(a)
            todo.extend(adjacency[a] - visited)
    assert len(visited) == len(vertices), 'Garment must be one connected component'

    def nearest(x, y):
        return min(range(len(vertices)), key=lambda i: (vertices[i][0] - x) ** 2 + (vertices[i][1] - y) ** 2)

    end_left = [i for i, p in enumerate(vertices) if p[0] <= -.365]
    end_right = [i for i, p in enumerate(vertices) if p[0] >= .365]
    edges = sorted(edges)
    material_names = ['hook_blue' if sum(vertices[i][0] for i in f) / 3 < -.365
                      else 'loop_brown' if sum(vertices[i][0] for i in f) / 3 > .365
                      else 'cloth_beige' for f in faces]
    sleeve_grasps = {}
    for side, (hx, hy) in zip(('left', 'right'), holes):
        sleeve_grasps[side + '_armhole_top'] = nearest(hx, hy + .0375)
        sleeve_grasps[side + '_armhole_bottom'] = nearest(hx, hy - .0375)
    topology = {
        'units': 'meters', 'coordinates': 'local XY sheet, +Z initial normal',
        'vertices': vertices, 'faces': faces, 'edges': edges,
        'edge_rest_lengths': [math.dist(vertices[a], vertices[b]) for a, b in edges],
        'outer_boundary': loops[0], 'armholes': hole_loops,
        'closure_patches': {'hook_left': end_left, 'loop_right': end_right},
        'closure_pairing': [[a, nearest(-vertices[a][0], vertices[a][1])] for a in end_left],
        'grasp_vertices': {
            'left_end_center': nearest(-.392, 0), 'right_end_center': nearest(.392, 0),
            'left_top_corner': nearest(-.392, .0825), 'right_top_corner': nearest(.392, .0825),
            **sleeve_grasps,
        },
        'face_materials': material_names,
        'validation': {'connected_components': 1, 'boundary_loops': 3, 'interior_holes': 2,
                       'all_vertices_used': True, 'nonmanifold_edges': 0},
    }
    return vertices, faces, topology


def build_assets(output_dir=None):
    output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent / 'assets'
    output_dir.mkdir(parents=True, exist_ok=True)
    sphere_vertices, sphere_faces = icosphere(3)
    _write_obj(output_dir / 'unit_sphere.obj', sphere_vertices, sphere_faces)
    components = teddy_components()
    merged_vertices, merged_faces, merged_materials, colors = [], [], [], []
    materials = {}
    for component in components:
        offset = len(merged_vertices)
        v = _transformed(sphere_vertices, component)
        merged_vertices.extend(v)
        merged_faces.extend(tuple(i + offset for i in f) for f in sphere_faces)
        merged_materials.extend([component['name']] * len(sphere_faces))
        colors.extend([component['color']] * len(v))
        materials[component['name']] = component['color']
    _write_obj(output_dir / 'bear_visual.obj', merged_vertices, merged_faces, materials, merged_materials)
    _write_ply(output_dir / 'bear_visual.ply', merged_vertices, merged_faces, colors)
    small_vertices, small_faces = icosphere(2)
    collision_vertices, collision_faces = [], []
    for component in components:
        if component['collision']:
            offset = len(collision_vertices)
            collision_vertices.extend(_transformed(small_vertices, component))
            collision_faces.extend(tuple(i + offset for i in f) for f in small_faces)
    _write_obj(output_dir / 'bear_collision.obj', collision_vertices, collision_faces)

    vertices, faces, topology = garment_mesh()
    garment_materials = {'cloth_beige': CLOTH, 'hook_blue': HOOK, 'loop_brown': LOOP}
    uv = [(x / .80 + .5, y / .18 + .5) for x, y, _ in vertices]
    _write_obj(output_dir / 'garment.obj', vertices, faces, garment_materials, topology['face_materials'], uv)
    # Genesis PBD expects one trimesh geometry.  A single textured material keeps
    # its topology shared while displaying both closure strips on the sheet.
    _write_obj(output_dir / 'garment_sim.obj', vertices, faces,
               {'garment_textured': (1, 1, 1, 1)}, ['garment_textured'] * len(faces), uv)
    with (output_dir / 'garment_sim.mtl').open('a', encoding='utf-8') as material_file:
        material_file.write('map_Kd garment_texture.png\n')
    _write_garment_texture(output_dir / 'garment_texture.png')
    garment_colors = [HOOK if x <= -.365 else LOOP if x >= .365 else CLOTH for x, _, _ in vertices]
    _write_ply(output_dir / 'garment.ply', vertices, faces, garment_colors)
    (output_dir / 'garment_topology.json').write_text(json.dumps(topology, separators=(',', ':')), encoding='utf-8')

    manifest = {
        'schema_version': 1,
        'provenance': 'Procedural reconstruction from four supplied initial-state images and the user task description only.',
        'not_identified_from_images': ['absolute scale', 'camera calibration', 'mass', 'cloth elasticity',
                                       'bending stiffness', 'friction', 'contact compliance', 'Velcro strength',
                                       'original motion and controller'],
        'coordinate_system': {'units': 'meters', 'up': '+Z', 'bear_front': '-Y', 'bear_left_right': 'X'},
        'table': {'size': [1.40, .95, .045], 'top_z': .75,
                  'position': [0, 0, .7275], 'color': [.57, .40, .25, 1]},
        'bear': {'position_reference': [0, .07, .75], 'nominal_height_m': .44,
                 'actual_ear_top_height_m': .455, 'visual_mesh': 'bear_visual.obj',
                 'vertex_color_visual_mesh': 'bear_visual.ply', 'collision_mesh': 'bear_collision.obj',
                 'mesh_coordinates': 'world', 'components': components,
                 'arm_centerlines': derive_arm_centerlines(components),
                 'arm_centerline_method': 'Derived from torso/shoulder/upper-arm/forearm/paw components; proximal point inside torso, distal point 5 mm beyond the paw ellipsoid.'},
        'garment': {'mesh': 'garment.obj', 'simulation_mesh': 'garment_sim.obj',
                    'simulation_texture': 'garment_texture.png', 'vertex_color_mesh': 'garment.ply',
                    'topology': 'garment_topology.json', 'mesh_coordinates': 'local',
                    'position': [0, -.28, .758], 'euler_xyz_deg': [0, 0, 0],
                    'length_m': .80, 'width_m': .18, 'visual_thickness_m': .0015,
                    'hole_centers_local': [[x, y, 0] for x, y in GARMENT_HOLE_CENTERS],
                    'nominal_hole_size_m': list(GARMENT_HOLE_SIZE),
                    'hole_pattern_inference': 'Overhead-image estimates: centers at roughly 24.2% and 75.7% of panel length, and 75.0% and 58.5% of width measured from the far edge. Rounded rectangular openings approximate the irregular visible outlines.',
                    'minimum_nominal_hole_to_hem_margin_m': min(.09 - abs(y) - GARMENT_HOLE_SIZE[1] / 2 for _, y in GARMENT_HOLE_CENTERS),
                    'closure_strip_width_m': .035, 'color': list(CLOTH),
                    'closure_colors': {'hook_left': list(HOOK), 'loop_right': list(LOOP)},
                    'vertex_count': len(vertices), 'face_count': len(faces),
                    'preserve_topology': True,
                    'topology_warning': 'Disable automatic filling, convexification, and remeshing that could erase armholes.',
                    'physical_properties': 'Set as explicit priors in the simulator, not recovered measurements.'},
        'placement_blocks': [
            {'name': 'front_left_blue', 'position': [-.54, -.29, .773], 'size': [.043, .043, .046], 'color': [.10, .32, .64, 1]},
            {'name': 'front_right_red', 'position': [.54, -.29, .773], 'size': [.043, .043, .046], 'color': [.68, .14, .11, 1]},
            {'name': 'back_left_green', 'position': [-.28, .37, .773], 'size': [.043, .043, .046], 'color': [.12, .46, .30, 1]},
            {'name': 'back_right_yellow', 'position': [.28, .37, .773], 'size': [.043, .043, .046], 'color': [.95, .61, .11, 1]},
        ],
        'geometric_notes': [
            'The panel has two actual open boundary loops, not painted sleeve markers.',
            'Its middle is intended to cover the chest; tails meet behind the teddy.',
            'Current torso semi-axes 0.132 and 0.100 m give an equatorial circumference of approximately 0.732 m (Ramanujan ellipse approximation); this larger width/depth prior follows the rear-view silhouette ratio and leaves about 0.068 m of a 0.80 m panel for back overlap and closure, rather than identifying true dimensions from the images.',
            'Armhole centers are staggered by 0.030 m across the panel width and separated by 0.410 m along its length, following the supplied overhead view. Their nominal dimensions are 0.066 by 0.068 m; the left opening leaves an approximately 0.011 m near-edge hem bridge.',
            'Decorative face and belly meshes are excluded from analytic collision components.',
        ],
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    result = build_assets(args.output_dir)
    print(json.dumps({'output_dir': str(args.output_dir or Path(__file__).resolve().parent / 'assets'),
                      'garment_vertices': result['garment']['vertex_count'],
                      'garment_faces': result['garment']['face_count'],
                      'armholes': 2, 'bear_components': len(result['bear']['components'])}, indent=2))

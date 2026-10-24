"""Derive audit centerlines from the reconstructed solid component geometry.

Centerlines extend inside the torso and beyond each paw's actual ellipsoid,
so a real opening near a shoulder or paw is not excluded by truncated endpoints.
No trajectory or garment positions influence the construction.
"""
from __future__ import annotations
import math


def derive_arm_centerlines(components, distal_extension_m=.005):
    by_name = {component['name']: component for component in components}
    torso = by_name['torso']
    origin, torso_radii = torso['position'], torso['scale']
    torso_rotation = torso.get('rotation_matrix', [[1,0,0],[0,1,0],[0,0,1]])
    lines = {}
    for side in ['left', 'right']:
        arm = [by_name[side+'_'+name] for name in ['shoulder', 'upper_arm', 'forearm', 'paw']]
        centers = [part['position'][:] for part in arm]
        delta = [centers[0][i]-origin[i] for i in range(3)]
        local = [sum(delta[i]*torso_rotation[i][j] for i in range(3)) for j in range(3)]
        q = math.sqrt(sum((local[i]/torso_radii[i])**2 for i in range(3)))
        fraction = min(.5, .5/max(q, 1e-12))
        proximal = [origin[i]+fraction*delta[i] for i in range(3)]
        direction = [centers[-1][i]-centers[-2][i] for i in range(3)]
        norm = math.sqrt(sum(value*value for value in direction))
        if norm <= 1e-10:
            raise ValueError('Paw and forearm centers coincide; no distal direction.')
        direction = [value/norm for value in direction]
        paw = arm[-1]
        rotation = paw.get('rotation_matrix', [[1,0,0],[0,1,0],[0,0,1]])
        local_direction = [sum(direction[i]*rotation[i][j] for i in range(3)) for j in range(3)]
        radius = 1/math.sqrt(sum((local_direction[i]/paw['scale'][i])**2 for i in range(3)))
        distal = [centers[-1][i]+direction[i]*(radius+distal_extension_m) for i in range(3)]
        lines[side] = [proximal]+centers+[distal]
    return lines

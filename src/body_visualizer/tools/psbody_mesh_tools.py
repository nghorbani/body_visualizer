import numpy as np
from psbody.mesh import Mesh
from psbody.mesh.sphere import Sphere
from body_visualizer.tools.psbody_mesh_cube import Cube

def rotateXYZ(mesh_v, Rxyz):
    angle = np.radians(Rxyz[0])
    rx = np.array([
        [1., 0., 0.           ],
        [0., np.cos(angle), -np.sin(angle)],
        [0., np.sin(angle), np.cos(angle) ]
    ])

    angle = np.radians(Rxyz[1])
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)],
        [0., 1., 0.           ],
        [-np.sin(angle), 0., np.cos(angle)]
    ])

    angle = np.radians(Rxyz[2])
    rz = np.array([
        [np.cos(angle), -np.sin(angle), 0. ],
        [np.sin(angle), np.cos(angle), 0. ],
        [0., 0., 1. ]
    ])
    # return rotateZ(rotateY(rotateX(mesh_v, Rxyz[0]), Rxyz[1]), Rxyz[2])
    return rz.dot(ry.dot(rx.dot(mesh_v.T))).T
    # return rx.dot(mesh_v.T).T

def points_to_spheres(points, radius=0.2, color=np.ones(3) * .5):
    spheres = Mesh(v=[], f=[])
    for pidx, center in enumerate(points):
        clr = color[pidx] if len(color) > 3 else color
        spheres.concatenate_mesh(Sphere(center, radius).to_mesh(color=clr))
    return spheres

def points_to_cubes(points, radius=0.2, color=np.ones(3) * .5):
    cubes = Mesh(v=[], f=[])
    for pidx, center in enumerate(points):
        clr = color[pidx] if len(color) > 3 else color
        cubes.concatenate_mesh(Cube(center, radius).to_mesh(color=clr))
    return cubes
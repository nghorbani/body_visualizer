
def apply_mesh_tranfsormations_(meshes, transf):
    '''
    apply inplace translations to meshes
    :param meshes: list of trimesh meshes
    :param transf:
    :return:
    '''
    for i in range(len(meshes)):
        meshes[i] = meshes[i].apply_transform(transf)

import numpy as np

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

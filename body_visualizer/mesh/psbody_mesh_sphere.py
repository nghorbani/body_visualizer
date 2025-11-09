import numpy as np
from psbody.mesh import Mesh
from psbody.mesh.sphere import Sphere
from body_visualizer.tools.vis_tools import colors

def points_to_spheres(points, radius=0.01, point_color = colors['red']):
    '''

    :param points: Nx3 numpy array
    :param radius:
    :param vc: either a 3-element normalized RGB vector or a list of them for each point
    :return:
    '''
    spheres = Mesh(v=[], f=[])
    for id in range(len(points)):
        if isinstance(radius, float):
            spheres.concatenate_mesh(Sphere( center= points[id].reshape(-1,3), radius=radius ).to_mesh( color = point_color if len(point_color)==3 and not isinstance(point_color[0], list) else point_color[id]))
        else:
            spheres.concatenate_mesh(Sphere( center= points[id].reshape(-1,3), radius=radius[id] ).to_mesh( color = point_color if len(point_color)==3 and not isinstance(point_color[0], list) else point_color[id]))
    return spheres


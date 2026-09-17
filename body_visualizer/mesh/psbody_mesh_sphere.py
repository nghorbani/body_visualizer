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
    shared_color = len(point_color) == 3 and not isinstance(point_color[0], list)
    for id in range(len(points)):
        cur_radius = radius if isinstance(radius, float) else radius[id]
        cur_color = point_color if shared_color else point_color[id]
        spheres.concatenate_mesh(Sphere(center=points[id].reshape(-1, 3), radius=cur_radius).to_mesh(color=cur_color))
    return spheres


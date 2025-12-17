import numpy as np
from psbody.mesh import Mesh
from psbody.mesh.colors import name_to_rgb
from body_visualizer.tools.vis_tools import colors

__all__ = ['Cube']

class Cube(object):
    def __init__(self, center, radius):
        if(center.flatten().shape != (3,)):
            raise Exception("Center should have size(1,3) instead of %s" % center.shape)
        self.center = center.flatten()
        self.radius = radius

    def __str__(self):
        return "%s:%s" % (self.center, self.radius)

    def to_mesh(self, point_color=name_to_rgb['red']):
        v = np.array([[-1., -1., -1.],
               [-1., -1.,  1.],
               [-1.,  1.,  1.],
               [-1.,  1., -1.],
               [-1.,  1., -1.],
               [-1.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 1.,  1., -1.],
               [ 1.,  1., -1.],
               [ 1.,  1.,  1.],
               [ 1., -1.,  1.],
               [ 1., -1., -1.],
               [-1., -1.,  1.],
               [-1., -1., -1.],
               [ 1., -1., -1.],
               [ 1., -1.,  1.],
               [ 1., -1., -1.],
               [-1., -1., -1.],
               [-1.,  1., -1.],
               [ 1.,  1., -1.],
               [ 1.,  1.,  1.],
               [-1.,  1.,  1.],
               [-1., -1.,  1.],
               [ 1., -1.,  1.]])


        f = np.array([[ 0,  1,  2],
                   [ 0,  2,  3],
                   [ 4,  5,  6],
                   [ 4,  6,  7],
                   [ 8,  9, 10],
                   [ 8, 10, 11],
                   [12, 13, 14],
                   [12, 14, 15],
                   [16, 17, 18],
                   [16, 18, 19],
                   [20, 21, 22],
                   [20, 22, 23]])

        return Mesh(v=v * self.radius + self.center, f=f, vc=np.tile(point_color, (v.shape[0], 1)))

def points_to_cubes(points, radius=0.01, point_color = colors['red']):
    '''
    :param points: Nx3 numpy array
    :param radius:
    :param vc: either a 3-element normalized RGB vector or a list of them for each point
    :return:
    '''
    cubes = Mesh(v=[], f=[])
    for id in range(len(points)):
        if isinstance(radius, float):
            cubes.concatenate_mesh(Cube( center= points[id].reshape(-1,3), radius=radius ).to_mesh( point_color = point_color if len(point_color)==3 and not isinstance(point_color[0], list) else point_color[id]))
        else:
            cubes.concatenate_mesh(Cube( center= points[id].reshape(-1,3), radius=radius[id] ).to_mesh( point_color = point_color if len(point_color)==3 and not isinstance(point_color[0], list) else point_color[id]))
    return cubes



if __name__ == '__main__':
    # a = Mesh(filename='/home/nghorbani/Downloads/cube.ply')
    # v = a.f
    # from pprint import pprint
    # pprint(v)
    a = Cube(np.array([5,5,5]), .1).to_mesh()
    b = Cube(np.array([0,0,0]), .5).to_mesh()
    a.concatenate_mesh(b).show()

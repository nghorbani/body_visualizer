import numpy as np
import trimesh
from body_visualizer.tools.vis_tools import colors

__all__ = ['Cube', 'points_to_cube']


class Cube(object):
    def __init__(self, center, scale):
        if(center.flatten().shape != (3,)):
            raise Exception("Center should have size(1,3) instead of %s" % center.shape)
        self.center = center.flatten()
        self.scale = scale

    def __str__(self):
        return "%s:%s" % (self.center, self.scale)

    def to_mesh(self, color=colors['red']):
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

        # return Mesh(v=v * self.scale + self.center, f=f, vc=np.tile(color, (v.shape[0], 1)))
        return trimesh.Trimesh(vertices=v * self.scale + self.center, faces=f, vertex_colors=np.tile(color, (v.shape[0], 1)))

def points_to_cube(points, scale=0.01, vc = colors['red']):
    '''

    :param points: Nx3 numpy array
    :param scale:
    :param vc: either a 3-element normalized RGB vector or a list of them for each point
    :return:
    '''
    cubes = None
    for id in range(len(points)):
        cur_cube = Cube( center= points[id].reshape(-1,3), scale=scale ).to_mesh( color = vc if len(vc)==3 and not isinstance(vc[0], list) else vc[id])
        if cubes is None: spheres = cur_cube
        else: cubes = trimesh.util.concatenate(cubes, cur_cube)
    return spheres


if __name__ == '__main__':
    # a = Mesh(filename='/home/nghorbani/Downloads/cube.ply')
    # v = a.f
    # from pprint import pprint
    # pprint(v)
    a = Cube(np.array([5,5,5]), .1).to_mesh()
    b = Cube(np.array([0,0,0]), .5).to_mesh()
    a.concatenate_mesh(b).show()

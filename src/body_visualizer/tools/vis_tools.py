# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2018.01.02

import numpy as np
import cv2
import os
import trimesh
# import platform
# if 'Ubuntu' in platform.version():
#     print('In Ubuntu, using osmesa mode for rendering')
#     os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
# else:
#     print('In other system, using egl mode for rendering')
os.environ['PYOPENGL_PLATFORM'] = 'egl'


colors = {
    'pink': [.6, .0, .4],
    'purple': [.9, .7, .7],
    'cyan': [.7, .75, .5],
    'red': [1.0, 0.0, 0.0],

    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [.5, .2, .1],
    'brown-light': [0.654, 0.396, 0.164],
    'blue': [.0, .0, 1.],

    'offwhite': [.8, .9, .9],
    'white': [1., 1., 1.],
    'orange': [1., .2, 0],

    'grey': [.7, .7, .7],
    'grey-blue': [0.345, 0.580, 0.713],
    'black': np.zeros(3),
    'white': np.ones(3),

    'yellowg': [0.83, 1, 0],
}

def imagearray2file(img_array, outpath=None, fps=30):
    '''
    :param nparray: RxCxTxwidthxheightx3
    :param outpath: the directory where T images will be dumped for each time point in range T
    :param fps: fps of the gif file
    :return:
        it will return an image list with length T
        if outpath is given as a png file, an image will be saved for each t in T.
        if outpath is given as a gif file, an animated image with T frames will be created.
    '''

    if outpath is not None:
        outdir = os.path.dirname(outpath)
        if not os.path.exists(outdir): os.makedirs(outdir)

    if not isinstance(img_array, np.ndarray) or img_array.ndim < 6:
        raise ValueError('img_array should be a numpy array of shape RxCxTxwidthxheightx3')

    R, C, T, img_h, img_w, img_c = img_array.shape

    out_images = []
    for tIdx in range(T):
        row_images = []
        for rIdx in range(R):
            col_images = []
            for cIdx in range(C):
                col_images.append(img_array[rIdx, cIdx, tIdx])
            row_images.append(np.hstack(col_images))
        t_image = np.vstack(row_images)
        out_images.append(t_image)

    if outpath is not None:
        ext = outpath.split('.')[-1]
        if ext in ['png', 'jpeg', 'jpg']:
            for tIdx in range(T):
                if T > 1:
                    cur_outpath = outpath.replace('.%s'%ext, '_%03d.%s'%(tIdx, ext))
                else:
                    cur_outpath = outpath
                    
                img = cv2.cvtColor(out_images[tIdx], cv2.COLOR_BGR2RGB)
                cv2.imwrite(cur_outpath, img)
                while not os.path.exists(cur_outpath): continue  # wait until the snapshot is written to the disk
        elif ext == 'gif':
            import imageio
            with imageio.get_writer(outpath, mode='I', fps = fps) as writer:
                for tIdx in range(T):
                    img = out_images[tIdx].astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    writer.append_data(img)
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video = cv2.VideoWriter(outpath, fourcc, fps, (img_w, img_h), True)
            for tIdx in range(T):
                img = out_images[tIdx].astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video.write(img)

            video.release()
            cv2.destroyAllWindows()
        elif ext == 'mp4':
            #
            # from moviepy.editor import ImageSequenceClip
            # animation = ImageSequenceClip(out_images, fps=fps)
            # animation.write_videofile(outpath, verbose=False)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(outpath, fourcc, fps, (img_w, img_h), True)
            for tIdx in range(T):
                img = out_images[tIdx].astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video.write(img)

            video.release()
            try:
                cv2.destroyAllWindows()
            except:
                pass

    return out_images

def render_smpl_params(bm, body_parms, rot_body=None):
    '''
    :param bm: pytorch body model with batch_size 1
    :param pose_body: Nx21x3
    :param trans: Nx3
    :param betas: Nxnum_betas
    :return: N x 400 x 400 x 3
    '''

    from human_body_prior.tools.omni_tools import copy2cpu as c2c
    from body_visualizer.mesh.mesh_viewer import MeshViewer
    from body_visualizer.tools.mesh_tools import rotateXYZ

    imw, imh = 800, 800

    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    mv.set_cam_trans([0, 0.5, 3.0])
    faces = c2c(bm.f)

    v = c2c(bm(**body_parms).v)

    T, num_verts = v.shape[:-1]

    images = []
    for fIdx in range(T):
        verts = v[fIdx]
        if rot_body is not None:
            verts = rotateXYZ(verts, rot_body)
        mesh = trimesh.base.Trimesh(verts, faces, vertex_colors=num_verts*colors['grey'])

        mv.set_meshes([mesh], 'static')

        images.append(mv.render())

    return np.array(images).reshape(T, imw, imh, 3)

def meshes_as_png(meshes, outpath=None, view_angles=[0, 180]):
    from body_visualizer.mesh.mesh_viewer import MeshViewer

    imw = 800
    imh = 800
    mv = MeshViewer(imh, imw)
    mv.set_cam_trans([0, -.5, 1.75])
    images = np.zeros([len(meshes), len(view_angles), 1, imw, imh, 3])
    for mIdx, mesh in enumerate(meshes):
        for rId, angle in enumerate(view_angles):
            if angle != 0: mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(angle), (0, 1, 0)))
            mv.set_meshes([mesh], group_name='static')
            images[mIdx, rId, 0] = cv2.cvtColor(mv.render(render_wireframe=False), cv2.COLOR_BGR2RGB)
            if angle != 0: mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-angle), (0, 1, 0)))

    if outpath is not None: imagearray2file(images, outpath)
    return images

def show_image(img_ndarray):
    '''
    Visualize rendered body images resulted from render_smpl_params in Jupyter notebook
    :param img_ndarray: Nxim_hxim_wx3
    '''
    import matplotlib.pyplot as plt
    import cv2
    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.gca()

    img = img_ndarray.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    plt.axis('off')

    # fig.canvas.draw()
    # return True
import bpy
import os
import os.path as osp
import sys
from loguru import logger
from math import radians


def pngs2mp4(png_file_pattern, out_path, fps=60):
    import subprocess
    import time

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    logger.info(f'running ffmpeg with png_file_pattern: {png_file_pattern}')
    # cmd = "ffmpeg -frame_rate %d -i %s -pix_fmt yuv420p %s"% (1, png_path_pattern, out_path)
    cmd = f"ffmpeg -loglevel quiet -framerate {fps:d} -i {png_file_pattern} -pix_fmt yuv420p {out_path}"
    subprocess.call(cmd.split(' '))

    count = 0
    while not os.path.exists(out_path) and count < 100:
        count += 1
        time.sleep(1.0)
        continue
    logger.sucess(f'Created {out_path}\n')
    return True


def pngs2gif(pngs, out_path):
    '''

    :param pngs:a sorted list of *.png files. first file will be the first frame and so on
    :param out_path: a .gif file path. the exention should be present
    :return:
    '''
    import imageio
    from skimage import io
    import time
    from skimage.transform import rescale

    with imageio.get_writer(out_path, mode='I') as writer:
        for png_Idx in range(0, len(pngs)):
            image = io.imread(pngs[png_Idx])
            image = rescale(image, 1.0 / 4.0, mode='reflect')

            writer.append_data(image)  # (255 * (image / np.max(image))).astype(np.uint8))
    count = 0
    while not os.path.exists(out_path) and count < 100:
        count += 1
        time.sleep(1)
        continue
    return True

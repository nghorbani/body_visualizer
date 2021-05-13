
def apply_mesh_tranfsormations_(meshes, transf):
    '''
    apply inplace translations to meshes
    :param meshes: list of trimesh meshes
    :param transf:
    :return:
    '''
    for i in range(len(meshes)):
        meshes[i] = meshes[i].apply_transform(transf)
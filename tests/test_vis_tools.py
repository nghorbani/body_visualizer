# -*- coding: utf-8 -*-
"""
Tests for body_visualizer.tools.vis_tools.
"""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from body_visualizer.tools import vis_tools


class FakeCV2:
    COLOR_BGR2RGB = 1

    def __init__(self):
        self.cvt_calls = []
        self.imwrite_calls = []

    def cvtColor(self, img, flag):
        self.cvt_calls.append(flag)
        return img

    def imwrite(self, path, img):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_bytes(b"fake")
        self.imwrite_calls.append(str(path_obj))
        return True

    # Placeholders for interfaces used elsewhere in vis_tools
    def VideoWriter_fourcc(self, *_args):
        return 0

    class VideoWriter:
        def __init__(self, *_args, **_kwargs):
            self.frames = []

        def write(self, img):
            self.frames.append(img)

        def release(self):
            return True


def test_imagearray2file_validates_input():
    with pytest.raises(ValueError):
        vis_tools.imagearray2file([1, 2, 3])


def test_imagearray2file_writes_pngs(monkeypatch, tmp_path):
    fake_cv2 = FakeCV2()
    monkeypatch.setattr(vis_tools, "cv2", fake_cv2)

    outpath = tmp_path / "grid.png"
    img_array = np.arange(6, dtype=np.uint8).reshape(1, 1, 2, 1, 1, 3)

    images = vis_tools.imagearray2file(img_array, outpath=str(outpath))

    assert len(images) == 2
    assert all(image.shape == (1, 1, 3) for image in images)

    expected_files = [str(outpath).replace(".png", "_000.png"), str(outpath).replace(".png", "_001.png")]
    assert fake_cv2.imwrite_calls == expected_files
    assert len(fake_cv2.cvt_calls) == 2


def test_imagearray2file_writes_gif(monkeypatch, tmp_path):
    fake_cv2 = FakeCV2()
    monkeypatch.setattr(vis_tools, "cv2", fake_cv2)

    appended = []

    class FakeWriter:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def append_data(self, img):
            appended.append(img)

    def fake_get_writer(path, mode, fps):
        return FakeWriter()

    monkeypatch.setitem(sys.modules, "imageio", SimpleNamespace(get_writer=fake_get_writer))

    outpath = tmp_path / "anim.gif"
    img_array = np.ones((1, 1, 1, 2, 2, 3), dtype=np.uint8)

    images = vis_tools.imagearray2file(img_array, outpath=str(outpath), fps=12)

    assert len(images) == 1
    assert len(appended) == 1
    assert fake_cv2.cvt_calls  # conversion called


def test_render_smpl_params_with_rotation(monkeypatch):
    class FakeMeshViewer:
        instances = []
        render_count = 0

        def __init__(self, width, height, use_offscreen):
            self.width = width
            self.height = height
            self.use_offscreen = use_offscreen
            self.mesh_calls = []
            self.cam_trans = None
            FakeMeshViewer.instances.append(self)

        def set_cam_trans(self, trans):
            self.cam_trans = trans

        def set_meshes(self, meshes, group_name):
            self.mesh_calls.append((meshes, group_name))

        def render(self):
            FakeMeshViewer.render_count += 1
            return np.full((800, 800, 3), FakeMeshViewer.render_count, dtype=np.uint8)

    fake_mesh_viewer_module = ModuleType("body_visualizer.mesh.mesh_viewer")
    fake_mesh_viewer_module.MeshViewer = FakeMeshViewer
    monkeypatch.setitem(sys.modules, "body_visualizer.mesh.mesh_viewer", fake_mesh_viewer_module)
    monkeypatch.setattr(vis_tools, "MeshViewer", FakeMeshViewer, raising=False)

    rotate_calls = []

    fake_mesh_tools = ModuleType("body_visualizer.tools.mesh_tools")

    def fake_rotateXYZ(verts, rot_body):
        rotate_calls.append(rot_body)
        return verts + rot_body

    fake_mesh_tools.rotateXYZ = fake_rotateXYZ
    monkeypatch.setitem(sys.modules, "body_visualizer.tools.mesh_tools", fake_mesh_tools)

    omni_module = ModuleType("human_body_prior.tools.omni_tools")
    omni_module.copy2cpu = lambda array: np.array(array)
    tools_module = ModuleType("human_body_prior.tools")
    tools_module.omni_tools = omni_module
    hbp_module = ModuleType("human_body_prior")
    hbp_module.tools = tools_module
    monkeypatch.setitem(sys.modules, "human_body_prior", hbp_module)
    monkeypatch.setitem(sys.modules, "human_body_prior.tools", tools_module)
    monkeypatch.setitem(sys.modules, "human_body_prior.tools.omni_tools", omni_module)

    class FakeBodyModel:
        def __init__(self):
            self.f = np.array([[0, 1, 2]], dtype=np.int32)

        def __call__(self, **_body_parms):
            verts = np.stack(
                (np.zeros((3, 3), dtype=np.float32), np.ones((3, 3), dtype=np.float32)),
                axis=0,
            )
            return SimpleNamespace(v=verts)

    body_visualizer_mesh = sys.modules.setdefault("body_visualizer.mesh", ModuleType("body_visualizer.mesh"))
    monkeypatch.setattr(body_visualizer_mesh, "mesh_viewer", fake_mesh_viewer_module, raising=False)

    monkeypatch.setattr(vis_tools.trimesh.base, "Trimesh", lambda verts, faces, vertex_colors: SimpleNamespace(verts=verts, faces=faces, colors=vertex_colors))

    result = vis_tools.render_smpl_params(FakeBodyModel(), {}, rot_body=np.array([1.0, 0.0, 0.0]))

    assert result.shape == (2, 800, 800, 3)
    assert len(FakeMeshViewer.instances[0].mesh_calls) == 2
    assert rotate_calls and np.allclose(rotate_calls[0], np.array([1.0, 0.0, 0.0]))


def test_meshes_as_png_calls_viewer(monkeypatch, tmp_path):
    class DummyMesh:
        def __init__(self):
            self.transforms = []

        def apply_transform(self, matrix):
            self.transforms.append(matrix)

    meshes = [DummyMesh()]
    angles = [0, 90]

    class FakeMeshViewer:
        def __init__(self, width, height):
            self.width = width
            self.height = height
            self.calls = []

        def set_cam_trans(self, trans):
            self.cam = trans

        def set_meshes(self, meshes, group_name="static"):
            self.calls.append(("set_meshes", meshes, group_name))

        def render(self, render_wireframe=False):
            return np.zeros((800, 800, 3), dtype=np.uint8)

    fake_mesh_viewer_module = ModuleType("body_visualizer.mesh.mesh_viewer")
    fake_mesh_viewer_module.MeshViewer = FakeMeshViewer
    monkeypatch.setitem(sys.modules, "body_visualizer.mesh.mesh_viewer", fake_mesh_viewer_module)
    monkeypatch.setattr(vis_tools, "MeshViewer", FakeMeshViewer, raising=False)

    rotation_calls = []

    def fake_rotation_matrix(angle, axis):
        rotation_calls.append((angle, axis))
        return np.eye(4)

    monkeypatch.setattr(vis_tools.trimesh.transformations, "rotation_matrix", fake_rotation_matrix)

    fake_cv2 = FakeCV2()
    monkeypatch.setattr(vis_tools, "cv2", fake_cv2)

    captured = {}

    def fake_imagearray2file(images, outpath):
        captured["images"] = images
        captured["outpath"] = outpath

    monkeypatch.setattr(vis_tools, "imagearray2file", fake_imagearray2file)

    outpath = tmp_path / "meshes.gif"
    images = vis_tools.meshes_as_png(meshes, outpath=str(outpath), view_angles=angles)

    assert images.shape[0] == len(meshes)
    assert captured["outpath"] == str(outpath)
    assert any(angle != 0 for angle, _axis in rotation_calls)
    assert len(meshes[0].transforms) == 2  # rotated forward and restored


def _install_matplotlib_stub(monkeypatch):
    captured = {}

    class FakeAxis:
        def imshow(self, img):
            captured["imshow"] = img

    class FakeFigure:
        def __init__(self, *args, **kwargs):
            captured["figure_args"] = (args, kwargs)

        def gca(self):
            captured["gca_called"] = True
            return FakeAxis()

    pyplot_module = ModuleType("matplotlib.pyplot")

    def fake_figure(*args, **kwargs):
        return FakeFigure(*args, **kwargs)

    def fake_axis(*args, **kwargs):
        captured["axis_args"] = (args, kwargs)

    monkeypatch.setitem(sys.modules, "matplotlib", ModuleType("matplotlib"))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot_module)
    pyplot_module.figure = fake_figure
    pyplot_module.axis = fake_axis
    return captured


def test_vis_tools_show_image(monkeypatch):
    class Cv2Stub:
        COLOR_BGR2RGB = FakeCV2.COLOR_BGR2RGB

        def __init__(self):
            self.calls = []

        def cvtColor(self, img, flag):
            self.calls.append(flag)
            return img

    cv2_stub = Cv2Stub()
    monkeypatch.setitem(sys.modules, "cv2", cv2_stub)
    captured = _install_matplotlib_stub(monkeypatch)

    img = np.ones((4, 4, 3), dtype=np.uint8)
    vis_tools.show_image(img)

    assert cv2_stub.calls[0] == Cv2Stub.COLOR_BGR2RGB
    np.testing.assert_array_equal(captured["imshow"], img)

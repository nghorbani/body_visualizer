# -*- coding: utf-8 -*-
"""
Tests for body_visualizer.mesh.mesh_viewer
"""
import io
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import trimesh

# Mock pyrender before importing MeshViewer to avoid OpenGL/EGL issues on Windows
sys.modules['pyrender'] = MagicMock()
sys.modules['pyrender.constants'] = MagicMock()

# These imports have to follow the mocking above, hence the noqa.
from body_visualizer.mesh import mesh_viewer as mesh_viewer_module  # noqa: E402
from body_visualizer.mesh.mesh_viewer import MeshViewer  # noqa: E402
from body_visualizer.tools.vis_tools import colors  # noqa: E402

FAKE_RENDER_FLAGS = SimpleNamespace(
    SHADOWS_DIRECTIONAL=1,
    RGBA=2,
    ALL_WIREFRAME=4,
)


class FakeDirectionalLight:
    def __init__(self, color=None, intensity=1.0):
        self.color = color
        self.intensity = intensity


class FakeNode:
    def __init__(self, light=None, matrix=None):
        self.light = light
        self.matrix = matrix


def _install_fake_pyrender_modules():
    prev_light = sys.modules.get('pyrender.light')
    prev_node = sys.modules.get('pyrender.node')

    light_module = ModuleType('pyrender.light')
    light_module.DirectionalLight = FakeDirectionalLight

    node_module = ModuleType('pyrender.node')
    node_module.Node = FakeNode

    sys.modules['pyrender.light'] = light_module
    sys.modules['pyrender.node'] = node_module
    return prev_light, prev_node


def _restore_module(name, previous):
    if previous is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous


class TestMeshViewerInitialization:
    """Test MeshViewer initialization"""

    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_initialization_offscreen(self, mock_scene, mock_renderer):
        """Test MeshViewer initialization with offscreen renderer"""
        viewer = MeshViewer(width=800, height=600, use_offscreen=True)

        assert viewer.width == 800
        assert viewer.height == 600
        assert viewer.use_offscreen is True
        mock_renderer.assert_called_once_with(800, 600)

    @patch('body_visualizer.mesh.mesh_viewer.Viewer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_initialization_interactive(self, mock_scene, mock_viewer):
        """Test MeshViewer initialization with interactive viewer"""
        viewer = MeshViewer(width=800, height=600, use_offscreen=False)

        assert viewer.width == 800
        assert viewer.height == 600
        assert viewer.use_offscreen is False
        mock_viewer.assert_called_once()

    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_default_dimensions(self, mock_scene, mock_renderer):
        """Test default width and height"""
        viewer = MeshViewer()

        assert viewer.width == 1200
        assert viewer.height == 800

    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_scene_created(self, mock_scene, mock_renderer):
        """Test that scene is created during initialization"""
        viewer = MeshViewer()

        mock_scene.assert_called_once()
        assert viewer.scene is not None


class TestMeshViewerCamera:
    """Test camera-related methods"""

    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_set_cam_trans_with_list(self, mock_scene, mock_renderer):
        """Test setting camera translation with list"""
        mock_scene_instance = Mock()
        mock_scene.return_value = mock_scene_instance

        viewer = MeshViewer()
        viewer.set_cam_trans([1, 2, 3])

        # Verify set_pose was called
        mock_scene_instance.set_pose.assert_called_once()

    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_set_cam_trans_with_array(self, mock_scene, mock_renderer):
        """Test setting camera translation with numpy array"""
        mock_scene_instance = Mock()
        mock_scene.return_value = mock_scene_instance

        viewer = MeshViewer()
        trans = np.array([1, 2, 3])
        viewer.set_cam_trans(trans)

        # Verify set_pose was called
        mock_scene_instance.set_pose.assert_called_once()

    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_update_camera_pose(self, mock_scene, mock_renderer):
        """Test updating camera pose with custom matrix"""
        mock_scene_instance = Mock()
        mock_scene.return_value = mock_scene_instance

        viewer = MeshViewer()
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = [5, 6, 7]

        viewer.update_camera_pose(camera_pose)

        # Verify set_pose was called
        mock_scene_instance.set_pose.assert_called_once()


class TestMeshViewerMeshManagement:
    """Test mesh management methods"""

    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_set_static_meshes_single(self, mock_scene, mock_renderer):
        """Test setting a single static mesh"""
        mock_scene_instance = Mock()
        mock_scene_instance.get_nodes.return_value = []
        mock_scene.return_value = mock_scene_instance

        viewer = MeshViewer()

        # Create a simple mesh
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        viewer.set_static_meshes([mesh])

        # Verify mesh was added
        assert mock_scene_instance.add.call_count >= 1

    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_set_dynamic_meshes_multiple(self, mock_scene, mock_renderer):
        """Test setting multiple dynamic meshes"""
        mock_scene_instance = Mock()
        mock_scene_instance.get_nodes.return_value = []
        mock_scene.return_value = mock_scene_instance

        viewer = MeshViewer()

        # Create multiple meshes
        mesh1 = trimesh.Trimesh(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
                                faces=np.array([[0, 1, 2]]))
        mesh2 = trimesh.Trimesh(vertices=np.array([[2, 2, 2], [3, 2, 2], [2, 3, 2]]),
                                faces=np.array([[0, 1, 2]]))

        viewer.set_dynamic_meshes([mesh1, mesh2])

        # Verify meshes were added
        assert mock_scene_instance.add.call_count >= 2

    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_set_meshes_with_poses(self, mock_scene, mock_renderer):
        """Test setting meshes with custom poses"""
        mock_scene_instance = Mock()
        mock_scene_instance.get_nodes.return_value = []
        mock_scene.return_value = mock_scene_instance

        viewer = MeshViewer()

        mesh = trimesh.Trimesh(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
                              faces=np.array([[0, 1, 2]]))

        pose = np.eye(4)
        pose[:3, 3] = [1, 2, 3]

        viewer.set_static_meshes([mesh], poses=[pose])

        # Verify mesh was added with pose
        assert mock_scene_instance.add.call_count >= 1

    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_set_meshes_replaces_existing(self, mock_scene, mock_renderer):
        """Test that setting meshes replaces existing ones in the same group"""
        # Create mock nodes with names
        mock_node1 = Mock()
        mock_node1.name = 'static-mesh-00'
        mock_node2 = Mock()
        mock_node2.name = 'other-mesh'

        mock_scene_instance = Mock()
        mock_scene_instance.get_nodes.return_value = [mock_node1, mock_node2]
        mock_scene.return_value = mock_scene_instance

        viewer = MeshViewer()

        mesh = trimesh.Trimesh(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
                              faces=np.array([[0, 1, 2]]))

        viewer.set_static_meshes([mesh])

        # Verify old static mesh was removed
        mock_scene_instance.remove_node.assert_called_with(mock_node1)


class TestMeshViewerBackgroundAndLighting:
    """Test background color and lighting methods"""

    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_set_background_color(self, mock_scene, mock_renderer):
        """Test setting background color"""
        mock_scene_instance = Mock()
        mock_scene.return_value = mock_scene_instance

        viewer = MeshViewer()
        viewer.set_background_color(colors['red'])

        assert viewer.scene.bg_color == colors['red']

    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_use_raymond_lighting_offscreen(self, mock_scene, mock_renderer):
        """Test using raymond lighting with offscreen renderer"""
        mock_scene_instance = Mock()
        mock_scene_instance.has_node.return_value = False
        mock_scene.return_value = mock_scene_instance

        viewer = MeshViewer(use_offscreen=True)
        viewer.use_raymond_lighting(intensity=2.0)

        # Verify lights were added to scene
        assert mock_scene_instance.add_node.call_count >= 1

    @patch('body_visualizer.mesh.mesh_viewer.Viewer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_use_raymond_lighting_interactive_warns(self, mock_scene, mock_viewer):
        """Interactive viewer should warn about Raymond lighting"""
        mock_scene_instance = Mock()
        mock_scene.return_value = mock_scene_instance
        viewer = MeshViewer(use_offscreen=False)
        fake_stderr = io.StringIO()

        with patch.object(MeshViewer, '_add_raymond_light') as mock_add_light, \
             patch('body_visualizer.mesh.mesh_viewer.sys.stderr', fake_stderr):
            viewer.use_raymond_lighting()

        mock_add_light.assert_not_called()
        assert 'Interactive viewer already uses raymond lighting!' in fake_stderr.getvalue()
        mock_scene_instance.add_node.assert_not_called()

    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_add_raymond_light_imports_pyrender_submodules(self, mock_scene, mock_renderer):
        """Ensure pyrender light/node modules are imported when available"""
        mock_scene.return_value = Mock()
        viewer = MeshViewer()
        prev_light, prev_node = _install_fake_pyrender_modules()

        try:
            nodes = viewer._add_raymond_light()
        finally:
            _restore_module('pyrender.light', prev_light)
            _restore_module('pyrender.node', prev_node)

        assert len(nodes) == 3
        assert all(isinstance(node.light, FakeDirectionalLight) for node in nodes)

    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_add_raymond_light_handles_zero_norm_axis(self, mock_scene, mock_renderer):
        """Branch for zero-norm x axis should fall back to default basis"""
        mock_scene.return_value = Mock()
        viewer = MeshViewer()
        prev_light, prev_node = _install_fake_pyrender_modules()
        original_norm = mesh_viewer_module.np.linalg.norm
        triggered = {'flag': False}

        def fake_norm(value, *args, **kwargs):
            if (not triggered['flag'] and isinstance(value, np.ndarray) and value.shape == (3,)
                    and np.isclose(value[2], 0.0)):
                triggered['flag'] = True
                return 0.0
            return original_norm(value, *args, **kwargs)

        try:
            with patch('body_visualizer.mesh.mesh_viewer.np.linalg.norm', side_effect=fake_norm):
                nodes = viewer._add_raymond_light()
        finally:
            _restore_module('pyrender.light', prev_light)
            _restore_module('pyrender.node', prev_node)

        assert triggered['flag'] is True
        assert len(nodes) == 3
        assert all(isinstance(node.light, FakeDirectionalLight) for node in nodes)


class TestMeshViewerRendering:
    """Test rendering methods"""

    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_render_returns_image(self, mock_scene, mock_renderer):
        """Test that render returns an image"""
        mock_renderer_instance = Mock()
        mock_renderer_instance.render.return_value = (np.zeros((600, 800, 3)), np.zeros((600, 800)))
        mock_renderer.return_value = mock_renderer_instance

        viewer = MeshViewer(width=800, height=600, use_offscreen=True)
        result = viewer.render()

        assert result is not None
        mock_renderer_instance.render.assert_called_once()

    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_render_with_wireframe(self, mock_scene, mock_renderer):
        """Test rendering with wireframe mode"""
        mock_renderer_instance = Mock()
        mock_renderer_instance.render.return_value = (np.zeros((600, 800, 3)), np.zeros((600, 800)))
        mock_renderer.return_value = mock_renderer_instance

        viewer = MeshViewer(width=800, height=600, use_offscreen=True)
        result = viewer.render(render_wireframe=True)

        assert result is not None
        mock_renderer_instance.render.assert_called_once()

    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_render_with_rgba(self, mock_scene, mock_renderer):
        """Test rendering with RGBA output"""
        mock_renderer_instance = Mock()
        mock_renderer_instance.render.return_value = (np.zeros((600, 800, 4)), np.zeros((600, 800)))
        mock_renderer.return_value = mock_renderer_instance

        viewer = MeshViewer(width=800, height=600, use_offscreen=True)
        result = viewer.render(RGBA=True)

        assert result is not None
        mock_renderer_instance.render.assert_called_once()

    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    @patch('pyrender.constants.RenderFlags', new=FAKE_RENDER_FLAGS)
    def test_render_uses_internal_wireframe_flag(self, mock_scene, mock_renderer):
        """Ensure internal render_wireframe flag toggles RenderFlags"""
        mock_renderer_instance = Mock()
        mock_renderer_instance.render.return_value = (np.zeros((600, 800, 3)), np.zeros((600, 800)))
        mock_renderer.return_value = mock_renderer_instance

        viewer = MeshViewer(width=800, height=600, use_offscreen=True)
        viewer.render_wireframe = True
        viewer.render()

        call_kwargs = mock_renderer_instance.render.call_args.kwargs
        expected_flags = FAKE_RENDER_FLAGS.SHADOWS_DIRECTIONAL | FAKE_RENDER_FLAGS.ALL_WIREFRAME
        assert call_kwargs['flags'] == expected_flags

    @patch('body_visualizer.mesh.mesh_viewer.cv2.imwrite')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.OffscreenRenderer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_save_snapshot(self, mock_scene, mock_renderer, mock_imwrite):
        """Test saving a snapshot"""
        mock_renderer_instance = Mock()
        mock_renderer_instance.render.return_value = (np.zeros((600, 800, 3)), np.zeros((600, 800)))
        mock_renderer.return_value = mock_renderer_instance

        viewer = MeshViewer(width=800, height=600, use_offscreen=True)
        viewer.save_snapshot('test.png')

        # Verify cv2.imwrite was called
        mock_imwrite.assert_called_once_with('test.png', mock_renderer_instance.render.return_value[0])

    @patch('body_visualizer.mesh.mesh_viewer.cv2.imwrite')
    @patch('body_visualizer.mesh.mesh_viewer.Viewer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_save_snapshot_interactive_warns(self, mock_scene, mock_viewer, mock_imwrite):
        """Interactive viewer should not attempt to save snapshots"""
        mock_scene.return_value = Mock()
        fake_stderr = io.StringIO()

        with patch('body_visualizer.mesh.mesh_viewer.sys.stderr', fake_stderr):
            viewer = MeshViewer(use_offscreen=False)
            viewer.save_snapshot('interactive.png')

        mock_imwrite.assert_not_called()
        assert 'Currently saving snapshots only works with off-screen renderer!' in fake_stderr.getvalue()


class TestMeshViewerClosing:
    """Test viewer closing"""

    @patch('body_visualizer.mesh.mesh_viewer.Viewer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_close_viewer_when_active(self, mock_scene, mock_viewer):
        """Test closing an active viewer"""
        mock_viewer_instance = Mock()
        mock_viewer_instance.is_active = True
        mock_viewer.return_value = mock_viewer_instance

        viewer = MeshViewer(use_offscreen=False)
        viewer.close_viewer()

        mock_viewer_instance.close_external.assert_called_once()

    @patch('body_visualizer.mesh.mesh_viewer.Viewer')
    @patch('body_visualizer.mesh.mesh_viewer.pyrender.Scene')
    def test_close_viewer_when_inactive(self, mock_scene, mock_viewer):
        """Test closing an inactive viewer"""
        mock_viewer_instance = Mock()
        mock_viewer_instance.is_active = False
        mock_viewer.return_value = mock_viewer_instance

        viewer = MeshViewer(use_offscreen=False)
        viewer.close_viewer()

        # Should not call close_external if not active
        mock_viewer_instance.close_external.assert_not_called()

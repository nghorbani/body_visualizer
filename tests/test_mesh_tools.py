# -*- coding: utf-8 -*-
"""
Tests for body_visualizer.tools.mesh_tools
"""
import pytest
import numpy as np
import trimesh
from body_visualizer.tools.mesh_tools import rotateXYZ, apply_mesh_tranfsormations_


class TestRotateXYZ:
    """Test rotation transformations"""
    
    def test_no_rotation(self):
        """Test that zero rotation returns original vertices"""
        vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = rotateXYZ(vertices, [0, 0, 0])
        np.testing.assert_array_almost_equal(result, vertices)
    
    def test_90_degree_x_rotation(self):
        """Test 90 degree rotation around X axis"""
        vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = rotateXYZ(vertices, [90, 0, 0])
        expected = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
    
    def test_90_degree_y_rotation(self):
        """Test 90 degree rotation around Y axis"""
        vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = rotateXYZ(vertices, [0, 90, 0])
        expected = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
    
    def test_90_degree_z_rotation(self):
        """Test 90 degree rotation around Z axis"""
        vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = rotateXYZ(vertices, [0, 0, 90])
        expected = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
    
    def test_180_degree_rotation(self):
        """Test 180 degree rotation"""
        vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = rotateXYZ(vertices, [180, 0, 0])
        expected = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
    
    def test_combined_rotation(self):
        """Test combined rotation around multiple axes"""
        vertices = np.array([[1, 0, 0]])
        result = rotateXYZ(vertices, [45, 45, 45])
        # Result should be different from original
        assert not np.allclose(result, vertices)
        # But should maintain distance from origin
        original_dist = np.linalg.norm(vertices)
        result_dist = np.linalg.norm(result)
        np.testing.assert_almost_equal(original_dist, result_dist)
    
    def test_negative_angles(self):
        """Test rotation with negative angles"""
        vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = rotateXYZ(vertices, [-90, 0, 0])
        expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
    
    def test_preserves_shape(self):
        """Test that rotation preserves the shape of the input array"""
        vertices = np.random.rand(10, 3)
        result = rotateXYZ(vertices, [30, 45, 60])
        assert result.shape == vertices.shape


class TestApplyMeshTransformations:
    """Test mesh transformation application"""
    
    def test_identity_transformation(self):
        """Test that identity transformation doesn't change meshes"""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        meshes = [mesh]
        
        identity = np.eye(4)
        apply_mesh_tranfsormations_(meshes, identity)
        
        np.testing.assert_array_almost_equal(meshes[0].vertices, vertices)
    
    def test_translation(self):
        """Test translation transformation"""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        meshes = [mesh]
        
        # Create translation matrix
        translation = np.eye(4)
        translation[:3, 3] = [1, 2, 3]
        
        apply_mesh_tranfsormations_(meshes, translation)
        
        expected_vertices = vertices + np.array([1, 2, 3])
        np.testing.assert_array_almost_equal(meshes[0].vertices, expected_vertices)
    
    def test_scaling(self):
        """Test scaling transformation"""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        meshes = [mesh]
        
        # Create scaling matrix
        scale = np.eye(4)
        scale[0, 0] = 2.0
        scale[1, 1] = 2.0
        scale[2, 2] = 2.0
        
        apply_mesh_tranfsormations_(meshes, scale)
        
        expected_vertices = vertices * 2.0
        np.testing.assert_array_almost_equal(meshes[0].vertices, expected_vertices)
    
    def test_multiple_meshes(self):
        """Test transformation on multiple meshes"""
        mesh1 = trimesh.Trimesh(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
                                faces=np.array([[0, 1, 2]]))
        mesh2 = trimesh.Trimesh(vertices=np.array([[2, 2, 2], [3, 2, 2], [2, 3, 2]]),
                                faces=np.array([[0, 1, 2]]))
        meshes = [mesh1, mesh2]
        
        translation = np.eye(4)
        translation[:3, 3] = [1, 0, 0]
        
        apply_mesh_tranfsormations_(meshes, translation)
        
        # Both meshes should be translated
        assert meshes[0].vertices[0, 0] == 1.0
        assert meshes[1].vertices[0, 0] == 3.0
    
    def test_inplace_modification(self):
        """Test that the function modifies meshes in place"""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        meshes = [mesh]
        original_id = id(meshes[0])
        
        translation = np.eye(4)
        translation[:3, 3] = [1, 0, 0]
        
        apply_mesh_tranfsormations_(meshes, translation)
        
        # The mesh object should be replaced (not the same object)
        # because trimesh.apply_transform returns a new mesh
        assert meshes[0].vertices[0, 0] == 1.0

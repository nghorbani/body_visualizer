# -*- coding: utf-8 -*-
"""
Tests for body_visualizer.mesh.cube
"""
import pytest
import numpy as np
import trimesh
from body_visualizer.mesh.cube import Cube, points_to_cubes
from body_visualizer.tools.vis_tools import colors


class TestCube:
    """Test Cube class"""
    
    def test_initialization(self):
        """Test cube initialization with valid inputs"""
        center = np.array([1, 2, 3])
        radius = 0.5
        cube = Cube(center, radius)
        
        np.testing.assert_array_equal(cube.center, center.flatten())
        assert cube.radius == radius
    
    def test_initialization_with_2d_center(self):
        """Test cube initialization with 2D array center"""
        center = np.array([[1, 2, 3]])
        radius = 0.5
        cube = Cube(center, radius)
        
        np.testing.assert_array_equal(cube.center, np.array([1, 2, 3]))
    
    def test_initialization_invalid_center_shape(self):
        """Test that invalid center shape raises exception"""
        center = np.array([1, 2])  # Only 2 elements
        radius = 0.5
        
        with pytest.raises(Exception) as exc_info:
            Cube(center, radius)
        assert "Center should have size(1,3)" in str(exc_info.value)
    
    def test_str_representation(self):
        """Test string representation"""
        center = np.array([1, 2, 3])
        radius = 0.5
        cube = Cube(center, radius)
        
        str_repr = str(cube)
        assert "1" in str_repr and "2" in str_repr and "3" in str_repr
        assert "0.5" in str_repr
    
    def test_to_mesh_creates_trimesh(self):
        """Test that to_mesh creates a valid trimesh"""
        center = np.array([0, 0, 0])
        radius = 1.0
        cube = Cube(center, radius)
        
        mesh = cube.to_mesh()
        
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
    
    def test_to_mesh_has_correct_number_of_faces(self):
        """Test that cube mesh has 12 triangular faces (6 square faces * 2 triangles)"""
        center = np.array([0, 0, 0])
        radius = 1.0
        cube = Cube(center, radius)
        
        mesh = cube.to_mesh()
        
        assert len(mesh.faces) == 12
    
    def test_to_mesh_with_custom_color(self):
        """Test to_mesh with custom color"""
        center = np.array([0, 0, 0])
        radius = 1.0
        cube = Cube(center, radius)
        
        custom_color = colors['blue']
        mesh = cube.to_mesh(color=custom_color)
        
        assert isinstance(mesh, trimesh.Trimesh)
        # Check that color is applied to all vertices
        np.testing.assert_array_equal(mesh.visual.vertex_colors[0, :3], 
                                     (np.array(custom_color) * 255).astype(np.uint8))
    
    def test_to_mesh_vertices_scaled_by_radius(self):
        """Test that mesh vertices are properly scaled by radius"""
        center = np.array([0, 0, 0])
        radius1 = 1.0
        radius2 = 2.0
        
        cube1 = Cube(center, radius1)
        cube2 = Cube(center, radius2)
        
        mesh1 = cube1.to_mesh()
        mesh2 = cube2.to_mesh()
        
        # Vertices should be scaled by radius
        max_dist1 = np.max(np.abs(mesh1.vertices))
        max_dist2 = np.max(np.abs(mesh2.vertices))
        
        assert max_dist2 > max_dist1
        np.testing.assert_almost_equal(max_dist2 / max_dist1, 2.0, decimal=5)
    
    def test_to_mesh_vertices_translated_by_center(self):
        """Test that mesh vertices are properly translated by center"""
        center = np.array([1, 2, 3])
        radius = 1.0
        cube = Cube(center, radius)
        
        mesh = cube.to_mesh()
        
        # Calculate the centroid of the mesh
        centroid = np.mean(mesh.vertices, axis=0)
        
        # Centroid should be close to the center
        np.testing.assert_array_almost_equal(centroid, center, decimal=1)
    
    def test_to_mesh_bounds(self):
        """Test that cube mesh has correct bounds"""
        center = np.array([0, 0, 0])
        radius = 1.0
        cube = Cube(center, radius)
        
        mesh = cube.to_mesh()
        
        # For a cube centered at origin with radius 1.0,
        # vertices should be between -1.0 and 1.0
        assert np.all(mesh.vertices >= -1.0)
        assert np.all(mesh.vertices <= 1.0)
        
        # Check that we actually reach the bounds
        assert np.any(np.isclose(mesh.vertices, -1.0))
        assert np.any(np.isclose(mesh.vertices, 1.0))
    
    def test_to_mesh_symmetry(self):
        """Test that cube mesh is symmetric around center"""
        center = np.array([0, 0, 0])
        radius = 1.0
        cube = Cube(center, radius)
        
        mesh = cube.to_mesh()
        
        # For a centered cube, the mean of all vertices should be near zero
        centroid = np.mean(mesh.vertices, axis=0)
        np.testing.assert_array_almost_equal(centroid, np.array([0, 0, 0]), decimal=5)


class TestPointsToCubes:
    """Test points_to_cubes function"""
    
    def test_single_point(self):
        """Test conversion of single point to cube"""
        points = np.array([[0, 0, 0]])
        radius = 0.1
        
        result = points_to_cubes(points, radius=radius)
        
        assert isinstance(result, trimesh.Trimesh)
        assert len(result.vertices) > 0
    
    def test_multiple_points(self):
        """Test conversion of multiple points to cubes"""
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        radius = 0.1
        
        result = points_to_cubes(points, radius=radius)
        
        assert isinstance(result, trimesh.Trimesh)
        # Should have more vertices than a single cube
        single_cube_verts = Cube(np.array([0, 0, 0]), radius).to_mesh().vertices.shape[0]
        assert len(result.vertices) >= 3 * single_cube_verts
    
    def test_with_single_radius(self):
        """Test with a single radius value for all points"""
        points = np.array([[0, 0, 0], [1, 1, 1]])
        radius = 0.2
        
        result = points_to_cubes(points, radius=radius)
        
        assert isinstance(result, trimesh.Trimesh)
    
    def test_with_different_radii(self):
        """Test with different radius for each point"""
        points = np.array([[0, 0, 0], [1, 1, 1]])
        radii = [0.1, 0.2]
        
        result = points_to_cubes(points, radius=radii)
        
        assert isinstance(result, trimesh.Trimesh)
    
    def test_with_single_color(self):
        """Test with single color for all points"""
        points = np.array([[0, 0, 0], [1, 1, 1]])
        radius = 0.1
        color = colors['blue']
        
        result = points_to_cubes(points, radius=radius, point_color=color)
        
        assert isinstance(result, trimesh.Trimesh)
    
    def test_with_different_colors(self):
        """Test with different color for each point"""
        points = np.array([[0, 0, 0], [1, 1, 1]])
        radius = 0.1
        colors_list = [colors['red'], colors['blue']]
        
        result = points_to_cubes(points, radius=radius, point_color=colors_list)
        
        assert isinstance(result, trimesh.Trimesh)
    
    def test_empty_points_array(self):
        """Test behavior with empty points array"""
        points = np.array([]).reshape(0, 3)
        radius = 0.1
        
        # Function should handle empty array gracefully
        # It returns None when no points are processed
        result = points_to_cubes(points, radius=radius)
        assert result is None
    
    def test_preserves_point_positions(self):
        """Test that cube centers match original point positions"""
        points = np.array([[1, 2, 3], [4, 5, 6]])
        radius = 0.1
        
        result = points_to_cubes(points, radius=radius)
        
        assert isinstance(result, trimesh.Trimesh)
        # The centroid of resulting mesh should be close to average of points
        centroid = np.mean(result.vertices, axis=0)
        expected_centroid = np.mean(points, axis=0)
        
        # Loose tolerance since we're averaging all vertices
        np.testing.assert_array_almost_equal(centroid, expected_centroid, decimal=0)
    
    def test_faces_count(self):
        """Test that combined cubes have correct number of faces"""
        points = np.array([[0, 0, 0], [1, 1, 1]])
        radius = 0.1
        
        result = points_to_cubes(points, radius=radius)
        
        # Each cube has 12 faces, so 2 cubes should have 24 faces
        assert len(result.faces) == 24

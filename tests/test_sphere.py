# -*- coding: utf-8 -*-
"""
Tests for body_visualizer.mesh.sphere
"""
import numpy as np
import pytest
import trimesh

from body_visualizer.mesh.sphere import Sphere, points_to_spheres
from body_visualizer.tools.vis_tools import colors


class TestSphere:
    """Test Sphere class"""

    def test_initialization(self):
        """Test sphere initialization with valid inputs"""
        center = np.array([1, 2, 3])
        radius = 0.5
        sphere = Sphere(center, radius)

        np.testing.assert_array_equal(sphere.center, center.flatten())
        assert sphere.radius == radius

    def test_initialization_with_2d_center(self):
        """Test sphere initialization with 2D array center"""
        center = np.array([[1, 2, 3]])
        radius = 0.5
        sphere = Sphere(center, radius)

        np.testing.assert_array_equal(sphere.center, np.array([1, 2, 3]))

    def test_initialization_invalid_center_shape(self):
        """Test that invalid center shape raises exception"""
        center = np.array([1, 2])  # Only 2 elements
        radius = 0.5

        with pytest.raises(Exception) as exc_info:
            Sphere(center, radius)
        assert "Center should have size(1,3)" in str(exc_info.value)

    def test_str_representation(self):
        """Test string representation"""
        center = np.array([1, 2, 3])
        radius = 0.5
        sphere = Sphere(center, radius)

        str_repr = str(sphere)
        assert "1" in str_repr and "2" in str_repr and "3" in str_repr
        assert "0.5" in str_repr

    def test_to_mesh_creates_trimesh(self):
        """Test that to_mesh creates a valid trimesh"""
        center = np.array([0, 0, 0])
        radius = 1.0
        sphere = Sphere(center, radius)

        mesh = sphere.to_mesh()

        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_to_mesh_with_custom_color(self):
        """Test to_mesh with custom color"""
        center = np.array([0, 0, 0])
        radius = 1.0
        sphere = Sphere(center, radius)

        custom_color = colors['blue']
        mesh = sphere.to_mesh(color=custom_color)

        assert isinstance(mesh, trimesh.Trimesh)
        # Check that color is applied to all vertices
        np.testing.assert_array_equal(mesh.visual.vertex_colors[0, :3],
                                     (np.array(custom_color) * 255).astype(np.uint8))

    def test_to_mesh_vertices_scaled_by_radius(self):
        """Test that mesh vertices are properly scaled by radius"""
        center = np.array([0, 0, 0])
        radius1 = 1.0
        radius2 = 2.0

        sphere1 = Sphere(center, radius1)
        sphere2 = Sphere(center, radius2)

        mesh1 = sphere1.to_mesh()
        mesh2 = sphere2.to_mesh()

        # Vertices should be scaled by radius
        max_dist1 = np.max(np.linalg.norm(mesh1.vertices, axis=1))
        max_dist2 = np.max(np.linalg.norm(mesh2.vertices, axis=1))

        assert max_dist2 > max_dist1
        np.testing.assert_almost_equal(max_dist2 / max_dist1, 2.0, decimal=5)

    def test_to_mesh_vertices_translated_by_center(self):
        """Test that mesh vertices are properly translated by center"""
        center = np.array([1, 2, 3])
        radius = 1.0
        sphere = Sphere(center, radius)

        mesh = sphere.to_mesh()

        # Calculate the centroid of the mesh
        centroid = np.mean(mesh.vertices, axis=0)

        # Centroid should be close to the center
        np.testing.assert_array_almost_equal(centroid, center, decimal=1)

    def test_has_inside_point_at_center(self):
        """Test that center point is inside sphere"""
        center = np.array([1, 2, 3])
        radius = 1.0
        sphere = Sphere(center, radius)

        assert sphere.has_inside(center)

    def test_has_inside_point_on_surface(self):
        """Test that point on surface is inside sphere"""
        center = np.array([0, 0, 0])
        radius = 1.0
        sphere = Sphere(center, radius)

        point_on_surface = np.array([1, 0, 0])
        assert sphere.has_inside(point_on_surface)

    def test_has_inside_point_outside(self):
        """Test that point outside sphere returns False"""
        center = np.array([0, 0, 0])
        radius = 1.0
        sphere = Sphere(center, radius)

        point_outside = np.array([2, 0, 0])
        assert not sphere.has_inside(point_outside)

    def test_intersects_overlapping_spheres(self):
        """Test intersection detection for overlapping spheres"""
        sphere1 = Sphere(np.array([0, 0, 0]), 1.0)
        sphere2 = Sphere(np.array([1, 0, 0]), 1.0)

        assert sphere1.intersects(sphere2)
        assert sphere2.intersects(sphere1)

    def test_intersects_touching_spheres(self):
        """Test intersection detection for touching spheres"""
        sphere1 = Sphere(np.array([0, 0, 0]), 1.0)
        sphere2 = Sphere(np.array([2, 0, 0]), 1.0)

        assert not sphere1.intersects(sphere2)

    def test_intersects_separate_spheres(self):
        """Test intersection detection for separate spheres"""
        sphere1 = Sphere(np.array([0, 0, 0]), 1.0)
        sphere2 = Sphere(np.array([5, 0, 0]), 1.0)

        assert not sphere1.intersects(sphere2)

    def test_intersection_vol_no_intersection(self):
        """Test volume calculation for non-intersecting spheres"""
        sphere1 = Sphere(np.array([0, 0, 0]), 1.0)
        sphere2 = Sphere(np.array([5, 0, 0]), 1.0)

        vol = sphere1.intersection_vol(sphere2)
        assert vol == 0

    def test_intersection_vol_small_spheres_inside_large(self):
        """Test volume when small sphere is completely inside large sphere"""
        large_sphere = Sphere(np.array([0, 0, 0]), 2.0)
        small_sphere = Sphere(np.array([0, 0, 0]), 1.0)

        vol = large_sphere.intersection_vol(small_sphere)
        expected_vol = (4 * np.pi * (1.0 ** 3)) / 3

        np.testing.assert_almost_equal(vol, expected_vol, decimal=5)

    def test_intersection_vol_partial_overlap(self):
        """Test volume for partially overlapping spheres"""
        sphere1 = Sphere(np.array([0, 0, 0]), 1.0)
        sphere2 = Sphere(np.array([1, 0, 0]), 1.0)

        vol = sphere1.intersection_vol(sphere2)

        # Volume should be positive and less than full sphere volume
        assert vol > 0
        assert vol < (4 * np.pi * (1.0 ** 3)) / 3


class TestPointsToSpheres:
    """Test points_to_spheres function"""

    def test_single_point(self):
        """Test conversion of single point to sphere"""
        points = np.array([[0, 0, 0]])
        radius = 0.1

        result = points_to_spheres(points, radius=radius)

        assert isinstance(result, trimesh.Trimesh)
        assert len(result.vertices) > 0

    def test_multiple_points(self):
        """Test conversion of multiple points to spheres"""
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        radius = 0.1

        result = points_to_spheres(points, radius=radius)

        assert isinstance(result, trimesh.Trimesh)
        # Should have more vertices than a single sphere
        single_sphere_verts = Sphere(np.array([0, 0, 0]), radius).to_mesh().vertices.shape[0]
        assert len(result.vertices) >= 3 * single_sphere_verts

    def test_with_single_radius(self):
        """Test with a single radius value for all points"""
        points = np.array([[0, 0, 0], [1, 1, 1]])
        radius = 0.2

        result = points_to_spheres(points, radius=radius)

        assert isinstance(result, trimesh.Trimesh)

    def test_with_different_radii(self):
        """Test with different radius for each point"""
        points = np.array([[0, 0, 0], [1, 1, 1]])
        radii = [0.1, 0.2]

        result = points_to_spheres(points, radius=radii)

        assert isinstance(result, trimesh.Trimesh)

    def test_with_single_color(self):
        """Test with single color for all points"""
        points = np.array([[0, 0, 0], [1, 1, 1]])
        radius = 0.1
        color = colors['blue']

        result = points_to_spheres(points, radius=radius, point_color=color)

        assert isinstance(result, trimesh.Trimesh)

    def test_with_different_colors(self):
        """Test with different color for each point"""
        points = np.array([[0, 0, 0], [1, 1, 1]])
        radius = 0.1
        colors_list = [colors['red'], colors['blue']]

        result = points_to_spheres(points, radius=radius, point_color=colors_list)

        assert isinstance(result, trimesh.Trimesh)

    def test_empty_points_array(self):
        """Test behavior with empty points array"""
        points = np.array([]).reshape(0, 3)
        radius = 0.1

        # Function should handle empty array gracefully
        # It returns None when no points are processed
        result = points_to_spheres(points, radius=radius)
        assert result is None

    def test_preserves_point_positions(self):
        """Test that sphere centers match original point positions"""
        points = np.array([[1, 2, 3], [4, 5, 6]])
        radius = 0.1

        result = points_to_spheres(points, radius=radius)

        assert isinstance(result, trimesh.Trimesh)
        # The centroid of resulting mesh should be close to average of points
        centroid = np.mean(result.vertices, axis=0)
        expected_centroid = np.mean(points, axis=0)

        # Loose tolerance since we're averaging all vertices
        np.testing.assert_array_almost_equal(centroid, expected_centroid, decimal=0)

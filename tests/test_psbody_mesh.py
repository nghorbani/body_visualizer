# -*- coding: utf-8 -*-
"""
Tests for psbody-backed mesh helpers.
"""
import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from body_visualizer.tools.vis_tools import colors


class FakeMesh:
    """Minimal psbody Mesh stub that tracks concatenations."""

    def __init__(self, v=None, f=None, vc=None):
        self.v = np.array(v, dtype=float) if v is not None else np.zeros((0, 3), dtype=float)
        self.f = np.array(f, dtype=int) if f is not None else np.zeros((0, 3), dtype=int)
        self.vc = np.array(vc, dtype=float) if vc is not None else None
        self.concatenated = []
        self.shown = False

    def concatenate_mesh(self, other):
        self.concatenated.append(other)
        return self

    def show(self):  # pragma: no cover - used only in manual scripts
        self.shown = True


class FakeSphere:
    """psbody Sphere stub that emits FakeMesh instances."""

    def __init__(self, center, radius):
        self.center = np.array(center, dtype=float)
        self.radius = radius

    def to_mesh(self, color):
        # Store provenance on the returned FakeMesh for assertions.
        vertices = self.center.copy()
        mesh = FakeMesh(v=vertices, f=np.array([[0, 1, 2]]), vc=np.array(color, dtype=float))
        mesh.generated_from = {"center": self.center, "radius": self.radius, "color": np.array(color, dtype=float)}
        return mesh


@pytest.fixture
def fake_psbody_modules():
    """Install lightweight psbody modules so optional dependency code runs under test."""

    saved_modules = {}

    def _install(name, module):
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    psbody_package = ModuleType("psbody")
    mesh_module = ModuleType("psbody.mesh")
    colors_module = ModuleType("psbody.mesh.colors")
    sphere_module = ModuleType("psbody.mesh.sphere")

    mesh_module.Mesh = FakeMesh
    colors_module.name_to_rgb = {"red": np.array([1.0, 0.0, 0.0])}
    sphere_module.Sphere = FakeSphere

    psbody_package.mesh = mesh_module
    mesh_module.colors = colors_module
    mesh_module.sphere = sphere_module

    _install("psbody", psbody_package)
    _install("psbody.mesh", mesh_module)
    _install("psbody.mesh.colors", colors_module)
    _install("psbody.mesh.sphere", sphere_module)

    yield SimpleNamespace(FakeMesh=FakeMesh, FakeSphere=FakeSphere)

    for name, original in saved_modules.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture
def psbody_cube_module(fake_psbody_modules):
    import body_visualizer.mesh.psbody_mesh_cube as cube_module

    return importlib.reload(cube_module)


@pytest.fixture
def psbody_sphere_module(fake_psbody_modules):
    import body_visualizer.mesh.psbody_mesh_sphere as sphere_module

    return importlib.reload(sphere_module)


def test_cube_to_mesh_scales_and_translates_vertices(psbody_cube_module):
    cube = psbody_cube_module.Cube(np.array([1.0, 2.0, 3.0]), radius=0.5)
    mesh = cube.to_mesh()

    assert isinstance(mesh, psbody_cube_module.Mesh)
    np.testing.assert_allclose(np.mean(mesh.v, axis=0), np.array([1.0, 2.0, 3.0]))
    # Edge length is twice the radius because vertices span [-1, 1] before scaling.
    np.testing.assert_allclose(
        mesh.v.max(axis=0) - mesh.v.min(axis=0),
        np.array([1.0, 1.0, 1.0]),
    )


def test_cube_repr_and_shape_validation(psbody_cube_module):
    with pytest.raises(Exception) as exc:
        psbody_cube_module.Cube(np.array([1.0, 2.0]), radius=1.0)
    assert "Center should have size(1,3)" in str(exc.value)

    cube = psbody_cube_module.Cube(np.array([0.0, 0.0, 0.0]), radius=0.25)
    assert str(cube) == f"{cube.center}:{cube.radius}"


def test_points_to_cubes_supports_iterable_radii_and_colors(psbody_cube_module):
    points = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 2.0]])
    radii = [0.1, 0.2]
    per_point_colors = [[0.2, 0.3, 0.4], [0.8, 0.1, 0.1]]

    mesh = psbody_cube_module.points_to_cubes(points, radius=radii, point_color=per_point_colors)

    assert isinstance(mesh, psbody_cube_module.Mesh)
    assert len(mesh.concatenated) == 2
    np.testing.assert_allclose(np.mean(mesh.concatenated[0].v, axis=0), points[0], atol=1e-8)
    np.testing.assert_allclose(np.mean(mesh.concatenated[1].v, axis=0), points[1], atol=1e-8)
    np.testing.assert_allclose(mesh.concatenated[1].vc[0], per_point_colors[1])


def test_points_to_cubes_with_scalar_radius(psbody_cube_module):
    points = np.array([[2.0, 0.0, -2.0]])
    mesh = psbody_cube_module.points_to_cubes(points, radius=0.3, point_color=colors["blue"])

    assert len(mesh.concatenated) == 1
    np.testing.assert_allclose(mesh.concatenated[0].vc[0], colors["blue"])


def test_points_to_spheres_with_scalar_radius(psbody_sphere_module):
    points = np.array([[0.0, 0.0, 0.0]])
    mesh = psbody_sphere_module.points_to_spheres(points, radius=0.05)

    assert isinstance(mesh, psbody_sphere_module.Mesh)
    assert len(mesh.concatenated) == 1
    np.testing.assert_allclose(mesh.concatenated[0].generated_from["center"].ravel(), points[0], atol=1e-8)


def test_points_to_spheres_with_per_point_radius_and_color(psbody_sphere_module):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    radii = [0.1, 0.2]
    colors_per_point = [[0.5, 0.5, 0.5], [0.1, 0.9, 0.2]]

    mesh = psbody_sphere_module.points_to_spheres(points, radius=radii, point_color=colors_per_point)

    assert len(mesh.concatenated) == 2
    np.testing.assert_allclose(mesh.concatenated[1].generated_from["radius"], radii[1])
    np.testing.assert_allclose(mesh.concatenated[0].generated_from["color"], colors_per_point[0])

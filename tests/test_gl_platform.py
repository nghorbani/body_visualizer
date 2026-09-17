# -*- coding: utf-8 -*-
"""
Tests for body_visualizer.gl_platform
"""
from body_visualizer import gl_platform


def test_default_platform_per_system():
    assert gl_platform.default_pyopengl_platform('Linux') == 'egl'
    assert gl_platform.default_pyopengl_platform('Darwin') == 'osmesa'
    assert gl_platform.default_pyopengl_platform('Windows') is None


def test_configure_respects_an_existing_choice():
    env = {gl_platform.ENV_VAR: 'osmesa'}
    assert gl_platform.configure_pyopengl_platform(env) == 'osmesa'
    assert env[gl_platform.ENV_VAR] == 'osmesa'


def test_configure_sets_the_default_when_unset(monkeypatch):
    monkeypatch.setattr(gl_platform.platform, 'system', lambda: 'Linux')
    env = {}
    assert gl_platform.configure_pyopengl_platform(env) == 'egl'
    assert env[gl_platform.ENV_VAR] == 'egl'


def test_configure_leaves_unknown_systems_alone(monkeypatch):
    monkeypatch.setattr(gl_platform.platform, 'system', lambda: 'Windows')
    env = {}
    assert gl_platform.configure_pyopengl_platform(env) is None
    assert gl_platform.ENV_VAR not in env


def test_empty_value_counts_as_unset(monkeypatch):
    monkeypatch.setattr(gl_platform.platform, 'system', lambda: 'Darwin')
    env = {gl_platform.ENV_VAR: ''}
    assert gl_platform.configure_pyopengl_platform(env) == 'osmesa'

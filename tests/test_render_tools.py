# -*- coding: utf-8 -*-
"""
Tests for body_visualizer.tools.render_tools.
"""
import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def render_tools(monkeypatch):
    """Reload render_tools with a stubbed bpy dependency."""

    monkeypatch.setitem(sys.modules, "bpy", ModuleType("bpy"))
    import body_visualizer.tools.render_tools as render_tools_module

    return importlib.reload(render_tools_module)


def test_pngs2mp4_invokes_ffmpeg(monkeypatch, tmp_path, render_tools):
    out_path = tmp_path / "video" / "clip.mp4"
    pattern = str(tmp_path / "frames_%04d.png")

    state = {"dir_exists": False, "file_checks": 0}

    def fake_exists(path):
        if path == str(out_path.parent):
            return state["dir_exists"]
        if path == str(out_path):
            state["file_checks"] += 1
            return state["file_checks"] > 1
        return False

    def fake_makedirs(path):
        state["dir_exists"] = True
        state["made_dir"] = path

    monkeypatch.setattr(render_tools.os.path, "exists", fake_exists)
    monkeypatch.setattr(render_tools.os, "makedirs", fake_makedirs)

    fake_logger = SimpleNamespace(info=[], sucess=[])

    def log_info(msg):
        fake_logger.info.append(msg)

    def log_success(msg):
        fake_logger.sucess.append(msg)

    monkeypatch.setattr(render_tools, "logger", SimpleNamespace(info=log_info, sucess=log_success))

    def fake_subprocess_call(cmd):
        fake_subprocess_call.called = cmd
        return 0

    fake_subprocess_call.called = None

    def fake_sleep(_seconds):
        fake_sleep.calls += 1

    fake_sleep.calls = 0

    monkeypatch.setitem(sys.modules, "subprocess", SimpleNamespace(call=fake_subprocess_call))
    monkeypatch.setitem(sys.modules, "time", SimpleNamespace(sleep=fake_sleep))

    assert render_tools.pngs2mp4(pattern, str(out_path), fps=30) is True
    assert state["made_dir"] == str(out_path.parent)
    assert fake_subprocess_call.called[0] == "ffmpeg"
    assert fake_sleep.calls == 1
    assert fake_logger.info
    assert fake_logger.sucess


def test_pngs2gif_writes_frames(monkeypatch, tmp_path, render_tools):
    out_path = str(tmp_path / "anim.gif")
    pngs = [f"frame_{idx}.png" for idx in range(2)]

    appended_frames = []

    class FakeWriter:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def append_data(self, img):
            appended_frames.append(img)

    def fake_get_writer(_path, mode):
        return FakeWriter()

    fake_imageio = SimpleNamespace(get_writer=fake_get_writer)
    monkeypatch.setitem(sys.modules, "imageio", fake_imageio)

    def fake_imread(path):
        return np.ones((4, 4, 3), dtype=np.uint8) * pngs.index(path)

    io_module = SimpleNamespace(imread=fake_imread)

    def fake_rescale(image, scale, mode):
        fake_rescale.calls.append((scale, mode))
        return image

    fake_rescale.calls = []
    transform_module = SimpleNamespace(rescale=fake_rescale)

    skimage_module = ModuleType("skimage")
    monkeypatch.setitem(sys.modules, "skimage", skimage_module)
    monkeypatch.setitem(sys.modules, "skimage.io", io_module)
    monkeypatch.setitem(sys.modules, "skimage.transform", transform_module)

    check_state = {"calls": 0}

    def fake_exists(path):
        if path == out_path:
            check_state["calls"] += 1
            return check_state["calls"] > 1
        return False

    def fake_sleep(_seconds):
        check_state["slept"] = True

    monkeypatch.setattr(render_tools.os.path, "exists", fake_exists)
    monkeypatch.setitem(sys.modules, "time", SimpleNamespace(sleep=fake_sleep))

    assert render_tools.pngs2gif(pngs, out_path) is True
    assert len(appended_frames) == len(pngs)
    assert fake_rescale.calls[0] == (0.25, "reflect")
    assert check_state["slept"] is True

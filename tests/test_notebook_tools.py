# -*- coding: utf-8 -*-
"""
Tests for body_visualizer.tools.notebook_tools.
"""
from types import ModuleType
import sys

import numpy as np


def _install_matplotlib_stub(monkeypatch):
    """Provide a tiny matplotlib.pyplot replacement."""

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

    pyplot_module.figure = fake_figure
    pyplot_module.axis = fake_axis

    monkeypatch.setitem(sys.modules, "matplotlib", ModuleType("matplotlib"))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot_module)

    return captured


def test_show_image_invokes_matplotlib(monkeypatch):
    matplotlib_calls = _install_matplotlib_stub(monkeypatch)

    cv2_module = ModuleType("cv2")
    monkeypatch.setitem(sys.modules, "cv2", cv2_module)

    from body_visualizer.tools.notebook_tools import show_image

    img = np.ones((4, 4, 3), dtype=np.uint8) * 127
    show_image(img)

    assert matplotlib_calls["figure_args"][1]["figsize"] == (4, 4)
    assert matplotlib_calls["figure_args"][1]["dpi"] == 300
    assert matplotlib_calls["gca_called"] is True
    np.testing.assert_array_equal(matplotlib_calls["imshow"], img.astype(np.uint8))
    assert matplotlib_calls["axis_args"][0][0] == "off"

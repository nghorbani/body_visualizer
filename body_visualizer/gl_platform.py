"""Selection of the PyOpenGL platform that pyrender renders with.

PyOpenGL reads ``PYOPENGL_PLATFORM`` once, when it is first imported, so the
variable has to be settled before anything imports pyrender. The package
``__init__`` calls :func:`configure_pyopengl_platform` for that reason.
"""
import os
import platform

ENV_VAR = 'PYOPENGL_PLATFORM'


def default_pyopengl_platform(system=None):
    """Return the default platform for ``system`` (``platform.system()`` when None).

    Linux defaults to ``egl``, which is what this package always used, macOS to
    ``osmesa`` because there is no EGL there, and every other system leaves
    PyOpenGL's own choice untouched (``None``). Set ``PYOPENGL_PLATFORM`` before
    importing body_visualizer to override, for example ``osmesa`` on a headless
    Linux machine without a GPU.
    """
    system = platform.system() if system is None else system
    if system == 'Linux':
        return 'egl'
    if system == 'Darwin':
        return 'osmesa'
    return None


def configure_pyopengl_platform(environ=None):
    """Set ``PYOPENGL_PLATFORM`` to the per-platform default unless the user already set it.

    Returns the platform in effect, or ``None`` when nothing was set.
    """
    environ = os.environ if environ is None else environ
    if environ.get(ENV_VAR):
        return environ[ENV_VAR]
    default = default_pyopengl_platform()
    if default is not None:
        environ[ENV_VAR] = default
    return default

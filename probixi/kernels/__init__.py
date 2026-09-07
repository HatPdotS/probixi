"""Optional accelerators; AUTO falls back before launch on unsupported inputs."""

from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from functools import lru_cache
from importlib import import_module

__all__ = ["Engine", "use_engine"]


class Engine(Enum):
    """Backend selection for optional processing and decoding kernels.

    AUTO selects supported accelerators, REFERENCE disables them, and
    ACCELERATED raises when an accelerated entry point is unsupported.
    """

    AUTO = "auto"
    REFERENCE = "reference"
    ACCELERATED = "accelerated"


_engine = ContextVar("probixi_engine", default=Engine.AUTO)


@contextmanager
def use_engine(engine):
    """Temporarily select a backend in the current execution context.

    Parameters
    ----------
    engine : Engine or {'auto', 'reference', 'accelerated'}
        Backend policy, restored on leaving the context, including exceptions.

    Yields
    ------
    None
        Execute processing or consume lazy frame iterators inside the context.

    Notes
    -----
    Calibration computations always use the reference implementations.
    """
    token = _engine.set(Engine(engine))
    try:
        yield
    finally:
        _engine.reset(token)


@lru_cache(None)
def _module(name):
    try:
        return import_module(f"{__name__}.{name}")
    except ImportError:
        return None


def select(name, supported):
    """Return a lazy kernel module, or None for the reference implementation."""
    engine = _engine.get()
    if engine is Engine.REFERENCE:
        return None
    module = _module(name) if supported else None
    if module is None and engine is Engine.ACCELERATED:
        raise RuntimeError(
            f"{name}: accelerated backend unavailable or unsupported inputs"
        )
    return module


def tensor_key(t):
    version = object() if t.is_inference() else t._version
    return (id(t), version, t.device, t.dtype, tuple(t.shape))

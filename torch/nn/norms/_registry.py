# mypy: allow-untyped-defs
"""Registry for norm implementations.
This module contains the registration system for norm implementations.
"""

import logging
from collections.abc import Callable
from typing import Protocol


logger = logging.getLogger(__name__)


class NormHandle(Protocol):
    def remove(self) -> None: ...


_RegisterFn = Callable[..., NormHandle | None]
_NORM_IMPLS: dict[str, _RegisterFn] = {}
_NORM_ACTIVE: tuple[str, NormHandle] | None = None


def register_norm_impl(
    impl: str,
    *,
    register_fn: _RegisterFn,
) -> None:
    """
    Register the callable that activates a norm impl.
    """
    global _NORM_IMPLS
    _NORM_IMPLS[impl] = register_fn


def activate_norm_impl(
    impl: str,
) -> None:
    """
    Activate into the dispatcher a previously registered norm impl.
    """
    global _NORM_ACTIVE, _NORM_IMPLS

    restore_norm_impl(_raise_warn=False)

    register_fn = _NORM_IMPLS.get(impl)
    if register_fn is None:
        raise ValueError(
            f"Unknown norm impl '{impl}'. "
            f"Available implementations: {list_norm_impls()}"
        )

    handle = register_fn()
    if handle is not None:
        _NORM_ACTIVE = (impl, handle)


def list_norm_impls() -> list[str]:
    """Return the names of all available norm implementations."""
    return sorted(_NORM_IMPLS.keys())


def current_norm_impl() -> str | None:
    """
    Return the currently activated norm impl name, if any.
    """
    return _NORM_ACTIVE[0] if _NORM_ACTIVE is not None else _NORM_ACTIVE


def restore_norm_impl(_raise_warn: bool = True) -> None:
    """
    Restore the default norm implementation.
    """
    global _NORM_ACTIVE

    handle = None
    if _NORM_ACTIVE is not None:
        handle = _NORM_ACTIVE[1]

    if handle is not None:
        handle.remove()
    elif _raise_warn:
        logger.warning(
            "Trying to restore default norm impl when no custom impl was activated"
        )

    _NORM_ACTIVE = None

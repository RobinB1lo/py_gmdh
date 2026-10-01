"""Deferred import of SymPy, which is only needed for symbolic output.

``sp`` behaves like the ``sympy`` module, but SymPy is imported on the first
attribute access, i.e. when an equation or another symbolic description is
requested, rather than when the package is imported.
"""

import importlib


class _LazyModule:
    __slots__ = ("_name", "_module")

    def __init__(self, name: str) -> None:
        self._name = name
        self._module = None

    def __getattr__(self, attr: str):
        if self._module is None:
            self._module = importlib.import_module(self._name)
        return getattr(self._module, attr)

    def __repr__(self) -> str:
        return f"<lazy module {self._name!r}>"


sp = _LazyModule("sympy")

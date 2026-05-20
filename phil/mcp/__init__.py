"""
Phil MCP server package.

Exposes a FastMCP server (``phil-mcp`` console script) that lets AI agents
run Phil imputation sweeps on file-backed pandas or polars datasets.

The server module imports ``fastmcp``, which is part of Phil's optional
``mcp`` extra. ``phil.mcp.server`` is therefore imported lazily so that
non-MCP submodules (``phil.mcp.config``, ``phil.mcp.registry``,
``phil.mcp.errors``, ``phil.mcp.prompts``) can be used without
installing the extras.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["main", "mcp"]


def __getattr__(name: str) -> Any:
    if name in {"main", "mcp"}:
        from phil.mcp.server import main as _main
        from phil.mcp.server import mcp as _mcp

        return {"main": _main, "mcp": _mcp}[name]
    raise AttributeError(f"module 'phil.mcp' has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover - hint only
    from phil.mcp.server import main, mcp

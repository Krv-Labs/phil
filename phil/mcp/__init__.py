"""
Phil MCP server package.

Exposes a FastMCP server (``phil-mcp`` console script) that lets AI agents
run Phil imputation sweeps on file-backed pandas or polars datasets.
"""

from phil.mcp.server import main, mcp

__all__ = ["main", "mcp"]

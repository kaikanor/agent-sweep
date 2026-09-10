"""Console-script shim for ``agentsweep-mcp``.

Lives in its own module so the entry point never imports ``mcp_server``
(and therefore ``fastmcp``) at load time: a base install without the
``mcp`` extra still gets a clear message instead of a traceback.
"""

from __future__ import annotations


def main() -> None:
    try:
        from agentsweep.mcp_server import mcp
    except ModuleNotFoundError as exc:
        if exc.name != "fastmcp":
            raise
        raise SystemExit(
            "agentsweep-mcp requires the optional MCP extra.\n"
            "Install it with: pip install 'agentsweep[mcp]'"
        ) from exc
    mcp.run()


if __name__ == "__main__":
    main()

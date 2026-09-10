"""Optional MCP server exposing agentsweep as read-only tools.

Run with ``agentsweep-mcp`` (installed only with the ``mcp`` extra) or
``python -m agentsweep.mcp_server``. Speaks stdio, same trust domain as
the CLI: local child process of the MCP client, fully offline.

Tool surface (read-only; the destructive fix path stays CLI-only):

  list_sources          which agents are installed here
  scan_history          scan one or all sources, masked findings only
  get_rotation_guidance provider-specific rotation steps

The plaintext secret boundary is the core invariant: ``Finding.value``
never leaves this process. Every payload is built with the same masked
fields the JSON CLI already emits (``_json_payload``).
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from . import __version__
from .pipeline import _scan_all, _source_rows
from .scanner import DETECTOR_IDS, ROTATION_GUIDANCE, RULES
from .sources import SOURCES

mcp = FastMCP("agentsweep", version=__version__)

# All tools are read-only by design: they scan and report, never modify.
_READ_ONLY = {"readOnlyHint": True}


def _parse_source(source: str | None) -> list[str]:
    """Resolve the ``source`` argument to a list of registered source keys."""
    if source in (None, "", "all"):
        return sorted(SOURCES)
    if source not in SOURCES:
        known = ", ".join(sorted(SOURCES))
        raise ValueError(f"unknown source {source!r}; known sources: {known}")
    return [source]


def _parse_rules(exclude_rules, only_rules) -> tuple[set[str] | None, set[str] | None]:
    def _split(raw) -> set[str] | None:
        if raw in (None, "", []):
            return None
        items = (
            {r.strip() for r in raw.split(",") if r.strip()}
            if isinstance(raw, str)
            else set(raw)
        )
        # Same validation set as the CLI: RULES ids plus detector ids (scan_text
        # emits e.g. "bip39-mnemonic", which is not a RULES entry).
        known_rules = {r[0] for r in RULES} | set(DETECTOR_IDS)
        unknown = items - known_rules
        if unknown:
            known = ", ".join(sorted(known_rules))
            raise ValueError(
                f"unknown rule(s): {', '.join(sorted(unknown))}; known rules: {known}"
            )
        return items

    return _split(exclude_rules), _split(only_rules)


@mcp.tool(annotations=_READ_ONLY)
def list_sources(detected_only: bool = False) -> list[dict[str, Any]]:
    """List every supported agent history source and whether it is installed.

    Read-only: reads no history files, touches nothing on disk.
    """
    rows = _source_rows()
    if detected_only:
        rows = [r for r in rows if r["detected"]]
    return rows


@mcp.tool(annotations=_READ_ONLY)
def scan_history(
    source: str | None = None,
    path: str | None = None,
    glob: str | None = None,
    root: str | None = None,
    exclude_rules: str | None = None,
    only_rules: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """Scan agent history files for leaked secrets. Read-only.

    Findings are masked (``sk-…abcd``); the plaintext never leaves the
    server process. Tell the human to run ``agentsweep fix`` in a
    terminal when this reports anything.

    Args:
        source: source key (see list_sources) or "all"/None for every source.
        path: scan exactly this file or directory (inside the source root)
            instead of the full source root walk.
        glob: shell-style filter on file paths, e.g. "*.jsonl". None = all.
        root: override the source's default root directory (like --root).
        exclude_rules: comma-separated rule ids to skip (e.g. "bip39-mnemonic,openai").
        only_rules: comma-separated rule ids to keep.
        limit: max findings returned (default 200); the counts in "summary"
            always reflect the full scan. Raise it for big sweeps.
    """
    if root is not None and source in (None, "", "all"):
        raise ValueError(
            "root= requires an explicit source; use list_sources to pick one"
        )
    if path is not None and source in (None, "", "all"):
        raise ValueError(
            "path= requires an explicit source; use list_sources to pick one"
        )
    if exclude_rules and only_rules:
        raise ValueError("pass either exclude_rules or only_rules, not both")
    if limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")

    exclude, only = _parse_rules(exclude_rules, only_rules)

    results: list[dict[str, Any]] = []
    files_scanned = 0
    strings_scanned = 0
    skipped_sources: list[str] = []

    for key in _parse_source(source):
        cls = SOURCES[key]
        src = cls(root=Path(root)) if root is not None else cls()

        # Validate path= before the detection skip: with the default root
        # absent, is_detected() would otherwise swallow a bad path into a
        # "skipped" result instead of the documented error.
        resolved_path: Path | None = None
        if path is not None:
            resolved_path = Path(path).resolve()
            root_dir = src.root.resolve()
            if not resolved_path.is_relative_to(root_dir):
                # Keep the scan inside the source root: the server otherwise
                # becomes an arbitrary-file oracle (readable-file metadata +
                # masked findings) for any path the caller names.
                raise ValueError(
                    f"path {path!r} is outside the source root {src.root!s}; "
                    "pass root= to scan a different tree"
                )
            if not (resolved_path.is_file() or resolved_path.is_dir()):
                raise ValueError(f"path {path!r} is not a file or directory")

        if not src.is_detected():
            skipped_sources.append(key)
            continue

        if resolved_path is not None:
            files = (
                [resolved_path]
                if resolved_path.is_file()
                else sorted(f for f in resolved_path.rglob("*") if f.is_file())
            )
        else:
            files = list(src.iter_files())
        if glob:
            files = [f for f in files if fnmatch.fnmatch(str(f), glob)]

        if not files:
            skipped_sources.append(key)
            continue

        found_by_file, sc, _suppressed, _truncated = _scan_all(
            src, files, exclude_rules=exclude, only_rules=only
        )
        strings_scanned += sc
        for f, items in found_by_file.items():
            rel = _relpath(f, src.root)
            for line_num, keypath, _val, finding in items:
                results.append(
                    {
                        "source": key,
                        "file": rel,
                        "line": line_num,
                        "keypath": keypath,
                        "rule": finding.rule,
                        "display": finding.display,
                        "masked": finding.masked,
                    }
                )
        files_scanned += len(files)

    by_rule: dict[str, int] = {}
    for f in results:
        by_rule[f["rule"]] = by_rule.get(f["rule"], 0) + 1

    truncated = len(results) > limit
    return {
        "version": __version__,
        "files_scanned": files_scanned,
        "strings_scanned": strings_scanned,
        "summary": {
            "total_findings": len(results),
            "by_rule": dict(sorted(by_rule.items())),
        },
        "findings": results[:limit],
        "findings_truncated": truncated,
        "skipped_sources": skipped_sources,
    }


def _relpath(f: Path, root: Path) -> str:
    try:
        return str(f.relative_to(root))
    except ValueError:
        return str(f)


@mcp.tool(annotations=_READ_ONLY)
def get_rotation_guidance(
    rule: str | None = None,
) -> dict[str, str] | list[dict[str, Any]]:
    """Provider-specific steps for rotating a leaked credential.

    Pass a rule id from scan_history findings for one provider's steps;
    omit it for the full mapping. Read-only reference data.
    """
    if rule is None:
        return ROTATION_GUIDANCE
    if rule not in ROTATION_GUIDANCE:
        known = ", ".join(sorted(ROTATION_GUIDANCE))
        raise ValueError(
            f"no rotation guidance for rule {rule!r}; known rules: {known}"
        )
    return {rule: ROTATION_GUIDANCE[rule]}


def main() -> None:
    """Entry point for the ``agentsweep-mcp`` console script."""
    mcp.run()


if __name__ == "__main__":
    main()

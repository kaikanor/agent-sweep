"""Tests for the optional MCP server (agentsweep[mcp]).

Covers the read-only tool surface with the plaintext boundary as the
core invariant: no tool output may carry ``Finding.value``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

fastmcp = pytest.importorskip("fastmcp")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentsweep.mcp_server import (  # noqa: E402
    _parse_rules,
    _parse_source,
    get_rotation_guidance,
    list_sources,
    mcp as mcp_app,
    scan_history,
)
from agentsweep.scanner import ROTATION_GUIDANCE, scan_text  # noqa: E402
from agentsweep.sources import SOURCES  # noqa: E402


# ---------------------------------------------------------------------------
# Plaintext boundary helper: the one rule the whole module exists to enforce.

PLAINTEXTS: list[str] = []


def _collect_plaintexts() -> None:
    """Scan one string containing a real-looking fake secret per rule family."""
    if PLAINTEXTS:
        return
    sample = "token=ghp_" + "A" * 36 + " key=AKIA" + "B" * 16 + " sk=sk-proj-" + "C" * 40
    for f in scan_text(sample):
        PLAINTEXTS.append(f.value)


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Point HOME at an empty dir so scans can't touch real agent histories."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    yield fake_home


# ---------------------------------------------------------------------------
# list_sources


def test_list_sources_matches_registry():
    rows = list_sources()
    assert {r["source"] for r in rows} == set(SOURCES)


def test_list_sources_detected_only_is_subset():
    every = list_sources()
    detected = list_sources(detected_only=True)
    assert {r["source"] for r in detected} <= {r["source"] for r in every}
    assert all(r["detected"] for r in detected)


# ---------------------------------------------------------------------------
# scan_history


def test_scan_history_unknown_source_raises():
    with pytest.raises(ValueError, match="unknown source"):
        scan_history(source="definitely-not-a-source")


def test_scan_history_unknown_rule_raises():
    with pytest.raises(ValueError, match="unknown rule"):
        scan_history(source="claude-code", only_rules="not-a-rule")


def test_scan_history_unknown_source_lists_known():
    with pytest.raises(ValueError, match="claude-code"):
        scan_history(source="definitely-not-a-source")


@pytest.fixture
def fake_history(tmp_path):
    """A leaky JSONL tree shaped like a claude-code projects dir."""
    secret = "ghp_" + "A" * 36
    root = tmp_path / "claude" / "projects"
    root.mkdir(parents=True)
    (root / "history.jsonl").write_text(
        json.dumps({"prompt": f"deploy with {secret}"}) + "\n", encoding="utf-8"
    )
    return root, secret


def test_scan_history_masks_plaintext(fake_history):
    root, secret = fake_history
    result = scan_history(source="claude-code", root=str(root))
    assert result["findings"], "expected at least one finding"
    blob = json.dumps(result)
    assert secret not in blob
    assert all(f["masked"] for f in result["findings"])


def test_scan_history_only_rules_filters(fake_history):
    root, _secret = fake_history
    result = scan_history(source="claude-code", root=str(root), only_rules="aws-access-key")
    assert result["findings"] == []


def test_scan_history_limit_truncates_but_counts_full(fake_history):
    root, _secret = fake_history
    result = scan_history(source="claude-code", root=str(root), limit=0)
    assert result["findings"] == []
    assert result["findings_truncated"] is True
    assert result["summary"]["total_findings"] >= 1
    assert "github-pat" in result["summary"]["by_rule"]


def test_scan_history_rejects_exclude_and_only(fake_history):
    root, _secret = fake_history
    with pytest.raises(ValueError, match="not both"):
        scan_history(
            source="claude-code", root=str(root), exclude_rules="github-pat", only_rules="aws-access-key"
        )


def test_scan_history_root_requires_explicit_source(tmp_path):
    # root= only makes sense for one source; "all" + root must be rejected.
    with pytest.raises(ValueError, match="requires an explicit source"):
        scan_history(source="all", root=str(tmp_path))


def test_scan_history_empty_root_lands_in_skipped(tmp_path):
    result = scan_history(source="claude-code", root=str(tmp_path))
    assert result["skipped_sources"] == ["claude-code"]
    assert result["findings"] == []


# ---------------------------------------------------------------------------
# get_rotation_guidance


def test_rotation_guidance_full_mapping():
    guidance = get_rotation_guidance()
    assert isinstance(guidance, dict)
    assert guidance == ROTATION_GUIDANCE


def test_rotation_guidance_single_rule():
    rule = sorted(ROTATION_GUIDANCE)[0]
    guidance = get_rotation_guidance(rule)
    assert guidance == {rule: ROTATION_GUIDANCE[rule]}


def test_rotation_guidance_unknown_rule():
    with pytest.raises(ValueError, match="no rotation guidance"):
        get_rotation_guidance("not-a-rule")


# ---------------------------------------------------------------------------
# Argument parsing helpers


def test_parse_source_none_means_all():
    assert _parse_source(None) == sorted(SOURCES)
    assert _parse_source("all") == sorted(SOURCES)


def test_parse_rules_roundtrip():
    exclude, only = _parse_rules("github-pat,aws-access-key", None)
    assert exclude == {"github-pat", "aws-access-key"}
    assert only is None


# ---------------------------------------------------------------------------
# Server wiring


async def _tool_names() -> set[str]:
    return {t.name for t in await mcp_app.list_tools()}


def test_tools_registered():
    tools = asyncio.run(_tool_names())
    assert {"list_sources", "scan_history", "get_rotation_guidance"} <= tools

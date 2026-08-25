"""Tests for per-server MCP tool filtering.

A server that offers forty tools charges for forty tools on every
message. These cover the allow/deny patterns that make one usable.
"""

from __future__ import annotations

from dataclasses import dataclass

from chord.mcp_client import load_server_specs, select_tools


@dataclass
class _Tool:
    name: str


TOOLS = [
    _Tool("daiso_search_products"),
    _Tool("daiso_check_inventory"),
    _Tool("cgv_get_showtimes"),
    _Tool("gs25_search_products"),
    _Tool("submit_feedback"),
]


def _names(tools) -> list[str]:
    return [tool.name for tool in tools]


def test_no_patterns_means_every_tool():
    """Every config written before filtering existed still means this."""
    assert select_tools(TOOLS, None, None) == TOOLS
    assert select_tools(TOOLS, [], []) == TOOLS


def test_an_allowlist_keeps_only_what_matches():
    kept = select_tools(TOOLS, ["daiso_*"], None)

    assert _names(kept) == ["daiso_search_products", "daiso_check_inventory"]


def test_several_patterns_are_a_union():
    kept = select_tools(TOOLS, ["daiso_*", "cgv_*"], None)

    assert len(kept) == 3


def test_an_exact_name_works_as_a_pattern():
    assert _names(select_tools(TOOLS, ["submit_feedback"], None)) == ["submit_feedback"]


def test_a_denylist_removes_from_what_is_left():
    kept = select_tools(TOOLS, None, ["submit_*", "gs25_*"])

    assert "submit_feedback" not in _names(kept)
    assert "daiso_search_products" in _names(kept)


def test_exclude_applies_after_include():
    kept = select_tools(TOOLS, ["daiso_*"], ["*_check_inventory"])

    assert _names(kept) == ["daiso_search_products"]


def test_a_pattern_matching_nothing_registers_nothing():
    """Better an empty server than a silent forty-tool bill."""
    assert select_tools(TOOLS, ["nope_*"], None) == []


def test_patterns_match_the_servers_own_names_not_the_prefixed_ones():
    """Config is written against what the server's docs list."""
    kept = select_tools(TOOLS, ["daiso_daiso_*"], None)

    assert kept == []


def test_order_is_preserved():
    assert _names(select_tools(TOOLS, ["*_products", "cgv_*"], None)) == [
        "daiso_search_products",
        "cgv_get_showtimes",
        "gs25_search_products",
    ]


# -- Reading them out of the config ----------------------------------------------------


def test_filter_keys_survive_config_loading(tmp_path):
    config = tmp_path / "mcp.json"
    config.write_text(
        """{"mcpServers": {"daiso": {"url": "https://mcp.aka.page",
           "tools": ["daiso_*"], "excludeTools": ["*_feedback"]}}}""",
        encoding="utf-8",
    )

    specs = load_server_specs(config)

    assert specs["daiso"]["tools"] == ["daiso_*"]
    assert specs["daiso"]["excludeTools"] == ["*_feedback"]


def test_a_server_without_filters_loads_as_before(tmp_path):
    config = tmp_path / "mcp.json"
    config.write_text('{"mcpServers": {"x": {"url": "https://x.test"}}}', encoding="utf-8")

    specs = load_server_specs(config)

    assert specs["x"].get("tools") is None

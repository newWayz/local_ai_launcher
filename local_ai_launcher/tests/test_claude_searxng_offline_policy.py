"""
Unit tests verifying offline search policy for Claude Code CLI and Claude Desktop on Strata / LocalAI.
Ensures official Anthropic cloud search tools (WebSearch, WebFetch, web_search) are ALWAYS disallowed / stripped
in offline local mode, but OPTIONALLY disallowed in online normal mode via settings.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from local_ai_launcher.config import config, _flatten_overrides
from local_ai_launcher.harnesses.claude import (
    ClaudeAdapter,
    setup_claude_cli_mcp_config,
    setup_claude_desktop_config,
)
from local_ai_launcher.localai.native_server import (
    is_anthropic_cloud_tool,
    _convert_anthropic_to_openai_body,
    _convert_openai_to_anthropic_response,
)
import local_ai_launcher.bridges.searxng_mcp as searxng_mcp


def test_claude_cli_always_disallows_cloud_search_in_local_offline_mode(tmp_path, monkeypatch):
    """Verify local offline Claude Code profile ALWAYS denies WebSearch/WebFetch even if online toggle is False."""
    import local_ai_launcher.harnesses.claude as claude_mod

    test_claude_dir = tmp_path / ".claude"
    monkeypatch.setattr(config, "claude_dir", test_claude_dir)
    monkeypatch.setattr(config, "claude_mode", "local")
    # Even if online toggle is explicitly False, local mode must ALWAYS disallow cloud search
    monkeypatch.setattr(config, "claude_disallow_cloud_search_online", False)

    captured_cmds = []
    monkeypatch.setattr(claude_mod, "start_in_wt_tab", lambda cmd, cwd=None: captured_cmds.append(cmd))

    adapter = ClaudeAdapter(str(tmp_path), [])
    adapter.start("http://127.0.0.1:8080/v1", active_model="strata")

    local_profile = test_claude_dir / "local"
    settings_file = local_profile / "settings.json"
    assert settings_file.exists()

    settings_cfg = json.loads(settings_file.read_text(encoding="utf-8"))
    assert "permissions" in settings_cfg
    assert "deny" in settings_cfg["permissions"]
    assert "WebSearch" in settings_cfg["permissions"]["deny"]
    assert "WebFetch" in settings_cfg["permissions"]["deny"]

    # Verify CLAUDE.md was generated with search instructions
    claude_md = local_profile / "CLAUDE.md"
    assert claude_md.exists()
    claude_md_text = claude_md.read_text(encoding="utf-8")
    assert "WebSearch and WebFetch are disabled" in claude_md_text
    assert "searxng" in claude_md_text

    # Verify CLAUDE.md also carries the offline answer contract (C3 anti-deflection layer)
    assert "Offline Answer Contract" in claude_md_text
    assert "Accept the user's wording literally" in claude_md_text
    assert "private, offline, single-user box" in claude_md_text

    # Verify runner scripts unconditionally contain --disallowedTools
    ps1_txt = (local_profile / "claude-local.ps1").read_text(encoding="utf-8")
    sh_txt = (local_profile / "claude-local.sh").read_text(encoding="utf-8")
    cmd_txt = (local_profile / "claude-local.cmd").read_text(encoding="utf-8")
    assert '--disallowedTools "WebSearch,WebFetch"' in ps1_txt
    assert '--disallowedTools "WebSearch,WebFetch"' in sh_txt
    assert '--disallowedTools WebSearch,WebFetch' in cmd_txt

    # Verify command string launched in Windows Terminal
    assert len(captured_cmds) == 1
    assert '--disallowedTools "WebSearch,WebFetch"' in captured_cmds[0]


def test_claude_cli_online_mode_preserves_search_by_default(tmp_path, monkeypatch):
    """Verify online mode does not pass --disallowedTools and does not deny search tools by default."""
    import local_ai_launcher.harnesses.claude as claude_mod

    test_claude_dir = tmp_path / ".claude"
    monkeypatch.setattr(config, "claude_dir", test_claude_dir)
    monkeypatch.setattr(config, "claude_mode", "official")
    monkeypatch.setattr(config, "claude_mode_explicit", True)
    monkeypatch.setattr(config, "claude_disallow_cloud_search_online", False)
    monkeypatch.setattr(config, "use_litellm_proxy", False)

    captured_cmds = []
    monkeypatch.setattr(claude_mod, "start_in_wt_tab", lambda cmd, cwd=None: captured_cmds.append(cmd))

    adapter = ClaudeAdapter(str(tmp_path), [])
    adapter.start("http://127.0.0.1:8080/v1", active_model="gpt-4")

    assert len(captured_cmds) == 1
    assert "--disallowedTools" not in captured_cmds[0]

    # Verify setup_claude_cli_mcp_config in online mode doesn't deny WebSearch
    mcp_file = setup_claude_cli_mcp_config(
        project_dir=tmp_path,
        llm_base_url="http://127.0.0.1:8080",
        active_model="gpt-4",
        mode="official"
    )
    assert mcp_file.exists()
    settings_local = tmp_path / ".claude" / "settings.local.json"
    if settings_local.exists():
        data = json.loads(settings_local.read_text(encoding="utf-8"))
        deny_list = data.get("permissions", {}).get("deny", [])
        assert "WebSearch" not in deny_list
        assert "WebFetch" not in deny_list


def test_claude_cli_online_mode_disallows_search_when_configured(tmp_path, monkeypatch):
    """Verify online mode passes --disallowedTools and denies search tools when claude_disallow_cloud_search_online is True."""
    import local_ai_launcher.harnesses.claude as claude_mod

    test_claude_dir = tmp_path / ".claude"
    monkeypatch.setattr(config, "claude_dir", test_claude_dir)
    monkeypatch.setattr(config, "claude_mode", "official")
    monkeypatch.setattr(config, "claude_mode_explicit", True)
    monkeypatch.setattr(config, "claude_disallow_cloud_search_online", True)
    monkeypatch.setattr(config, "use_litellm_proxy", False)

    captured_cmds = []
    monkeypatch.setattr(claude_mod, "start_in_wt_tab", lambda cmd, cwd=None: captured_cmds.append(cmd))

    adapter = ClaudeAdapter(str(tmp_path), [])
    adapter.start("http://127.0.0.1:8080/v1", active_model="gpt-4")

    assert len(captured_cmds) == 1
    assert '--disallowedTools "WebSearch,WebFetch"' in captured_cmds[0]

    # Verify setup_claude_cli_mcp_config in online mode denies WebSearch when configured
    mcp_file = setup_claude_cli_mcp_config(
        project_dir=tmp_path,
        llm_base_url="http://127.0.0.1:8080",
        active_model="gpt-4",
        mode="official"
    )
    assert mcp_file.exists()
    settings_local = tmp_path / ".claude" / "settings.local.json"
    assert settings_local.exists()
    data = json.loads(settings_local.read_text(encoding="utf-8"))
    assert "WebSearch" in data.get("permissions", {}).get("deny", [])
    assert "WebFetch" in data.get("permissions", {}).get("deny", [])


def test_setup_claude_cli_mcp_config_offline_always_denies(tmp_path, monkeypatch):
    """Verify setup_claude_cli_mcp_config always denies WebSearch in offline local mode."""
    monkeypatch.setattr(config, "claude_disallow_cloud_search_online", False)

    mcp_file = setup_claude_cli_mcp_config(
        project_dir=tmp_path,
        llm_base_url="http://127.0.0.1:8080",
        active_model="strata",
        mode="local"
    )
    assert mcp_file.exists()

    settings_local = tmp_path / ".claude" / "settings.local.json"
    assert settings_local.exists()
    data = json.loads(settings_local.read_text(encoding="utf-8"))

    assert "permissions" in data
    assert "deny" in data["permissions"]
    assert "WebSearch" in data["permissions"]["deny"]
    assert "WebFetch" in data["permissions"]["deny"]


def test_setup_claude_desktop_gateway_always_disables_web_search(tmp_path, monkeypatch):
    """Verify Claude Desktop in gateway mode ALWAYS disables webSearchEnabled even if online setting is False."""
    monkeypatch.setattr(config, "claude_disallow_cloud_search_online", False)

    cfg_file = tmp_path / "claude_desktop_config.json"
    setup_claude_desktop_config(
        config_path=cfg_file,
        llm_base_url="http://127.0.0.1:8080/v1",
        active_model="strata-iq3s-vision",
        desktop_mode="gateway"
    )

    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert data.get("preferences", {}).get("webSearchEnabled") is False
    assert data.get("preferences", {}).get("coworkWebSearchEnabled") is False
    assert "searxng" in data.get("mcpServers", {})

    meta_file = tmp_path / "configLibrary" / "_meta.json"
    meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
    applied_id = meta_data["appliedId"]

    profile_file = tmp_path / "configLibrary" / f"{applied_id}.json"
    profile_data = json.loads(profile_file.read_text(encoding="utf-8"))
    # In configLibrary/{applied_id}.json, unrecognized keys must NOT be present to prevent warning banner
    assert "webSearchEnabled" not in profile_data
    assert "coworkWebSearchEnabled" not in profile_data


def test_setup_claude_desktop_gateway_syncs_claude_3p(tmp_path, monkeypatch):
    """Verify setup_claude_desktop_config syncs complete mcpServers including searxng to Claude-3p directory."""
    fake_localappdata = tmp_path / "localappdata"
    fake_appdata = tmp_path / "appdata"
    monkeypatch.setenv("LOCALAPPDATA", str(fake_localappdata))
    monkeypatch.setenv("APPDATA", str(fake_appdata))

    setup_claude_desktop_config(
        llm_base_url="http://127.0.0.1:8080/v1",
        active_model="strata-iq3s-vision",
        desktop_mode="gateway"
    )

    c3p_cfg = fake_localappdata / "Claude-3p" / "claude_desktop_config.json"
    assert c3p_cfg.exists()
    c3p_data = json.loads(c3p_cfg.read_text(encoding="utf-8"))
    assert "searxng" in c3p_data.get("mcpServers", {})
    assert "local-model" in c3p_data.get("mcpServers", {})

    # Check configLibrary in Claude-3p does not contain unrecognized keys
    profile_file = fake_localappdata / "Claude-3p" / "configLibrary" / "00000000-0000-0000-0000-000000000001.json"
    assert profile_file.exists()
    profile_data = json.loads(profile_file.read_text(encoding="utf-8"))
    assert "webSearchEnabled" not in profile_data
    assert "coworkWebSearchEnabled" not in profile_data


def test_setup_claude_desktop_mcp_preserves_search_by_default(tmp_path, monkeypatch):
    """Verify Claude Desktop in MCP mode preserves webSearchEnabled by default."""
    monkeypatch.setattr(config, "claude_disallow_cloud_search_online", False)

    cfg_file = tmp_path / "claude_desktop_config.json"
    setup_claude_desktop_config(
        config_path=cfg_file,
        llm_base_url="http://127.0.0.1:8080/v1",
        active_model="default",
        desktop_mode="mcp"
    )

    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    # In online MCP mode, preferences should not force webSearchEnabled to False
    prefs = data.get("preferences", {})
    assert prefs.get("webSearchEnabled") is not False


def test_setup_claude_desktop_mcp_disables_search_when_configured(tmp_path, monkeypatch):
    """Verify Claude Desktop in MCP mode disables webSearchEnabled when claude_disallow_cloud_search_online is True."""
    monkeypatch.setattr(config, "claude_disallow_cloud_search_online", True)

    cfg_file = tmp_path / "claude_desktop_config.json"
    setup_claude_desktop_config(
        config_path=cfg_file,
        llm_base_url="http://127.0.0.1:8080/v1",
        active_model="default",
        desktop_mode="mcp"
    )

    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert data.get("preferences", {}).get("webSearchEnabled") is False
    assert data.get("preferences", {}).get("coworkWebSearchEnabled") is False


def test_config_overrides_disallow_cloud_search_online():
    """Verify launcher.toml [claude] and [claude_desktop] flattening handles disallow_cloud_search_online."""
    raw_toml = {
        "claude": {
            "disallow_cloud_search_online": True,
        },
        "claude_desktop": {
            "disallow_cloud_search_online": True,
        }
    }
    flattened = _flatten_overrides(raw_toml)
    assert flattened.get("claude_disallow_cloud_search_online") is True


def test_native_server_is_anthropic_cloud_tool_classification():
    """Verify is_anthropic_cloud_tool correctly separates official cloud search tools from MCP tools."""
    # Official cloud tools to strip
    assert is_anthropic_cloud_tool({"name": "web_search"}) is True
    assert is_anthropic_cloud_tool({"name": "websearch"}) is True
    assert is_anthropic_cloud_tool({"name": "web_fetch"}) is True
    assert is_anthropic_cloud_tool({"name": "webfetch"}) is True
    assert is_anthropic_cloud_tool({"name": "claude_web_search"}) is True
    assert is_anthropic_cloud_tool({"name": "web_search_20250101"}) is True

    # Local MCP tools to preserve
    assert is_anthropic_cloud_tool({"name": "mcp__searxng__searxng_search"}) is False
    assert is_anthropic_cloud_tool({"name": "searxng_search"}) is False
    assert is_anthropic_cloud_tool({"name": "searxng_web_search"}) is False
    assert is_anthropic_cloud_tool({"name": "mcp__local-model__query_local_model"}) is False
    assert is_anthropic_cloud_tool({"name": "mcp__rea__run_command"}) is False
    assert is_anthropic_cloud_tool({"name": "bash"}) is False
    assert is_anthropic_cloud_tool({"name": "read_file"}) is False


def test_native_server_strips_cloud_tools_and_steers_searxng():
    """Verify _convert_anthropic_to_openai_body drops web_search and injects SearXNG steering."""
    incoming_body = {
        "model": "claude-3-7-sonnet-latest",
        "system": "You are a software engineer.",
        "messages": [
            {"role": "user", "content": "Look up recent documentation on SearXNG."}
        ],
        "tools": [
            {
                "name": "web_search",
                "description": "Anthropic cloud web search",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
            },
            {
                "name": "mcp__searxng__searxng_search",
                "description": "SearXNG metasearch tool",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        ]
    }

    openai_body = _convert_anthropic_to_openai_body(incoming_body)

    # 1. web_search must be stripped from tools schema
    tools = openai_body.get("tools", [])
    tool_names = [t["function"]["name"] for t in tools]
    assert "web_search" not in tool_names
    assert "mcp__searxng__searxng_search" in tool_names

    # 2. System prompt must include critical offline search steering policy
    messages = openai_body.get("messages", [])
    system_msg = next((m for m in messages if m.get("role") == "system"), None)
    assert system_msg is not None
    sys_content = system_msg.get("content", "")
    assert "CRITICAL SEARCH POLICY" in sys_content
    assert "mcp__searxng__searxng_search" in sys_content
    assert "Never attempt to invoke 'web_search'" in sys_content


def test_native_server_aliases_hallucinated_web_search():
    """Verify _convert_openai_to_anthropic_response aliases any web_search tool call to SearXNG."""
    fake_openai_res = {
        "id": "chatcmpl-test1234",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"q": "latest quantum news"})
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 15}
    }

    anthropic_res = _convert_openai_to_anthropic_response(
        fake_openai_res,
        model_name="claude-3-7-sonnet-latest",
        searxng_tool_name="mcp__searxng__searxng_search"
    )

    content_blocks = anthropic_res.get("content", [])
    assert len(content_blocks) == 1
    tool_block = content_blocks[0]
    assert tool_block["type"] == "tool_use"
    # Name must be aliased from 'web_search' to the active SearXNG tool
    assert tool_block["name"] == "mcp__searxng__searxng_search"
    # Argument 'q' must be mapped to 'query'
    assert tool_block["input"].get("query") == "latest quantum news"


def test_searxng_mcp_description_and_aliases(monkeypatch):
    """Verify SearXNG MCP exposes strongly-worded tool description and handles search aliases."""
    tools_res = searxng_mcp.handle_tools_list("req-1")
    search_tool = next(t for t in tools_res["result"]["tools"] if t["name"] == "searxng_search")
    assert "PRIMARY AND EXCLUSIVE WEB SEARCH ENGINE" in search_tool["description"]

    # Mock _execute_search
    monkeypatch.setattr(searxng_mcp, "_execute_search", lambda query, limit=5, categories="general": f"Results for: {query}")

    # Tool call with name="searxng_web_search" and argument "q"
    call_res = searxng_mcp.handle_tools_call("req-2", {
        "name": "searxng_web_search",
        "arguments": {"q": "python tutorial"}
    })
    assert "Results for: python tutorial" in call_res["result"]["content"][0]["text"]

    # Tool call with name="web_search" and argument "query"
    call_res2 = searxng_mcp.handle_tools_call("req-3", {
        "name": "web_search",
        "arguments": {"query": "fastapi docs"}
    })
    assert "Results for: fastapi docs" in call_res2["result"]["content"][0]["text"]


def test_searxng_options_and_github_provider():
    """Verify LauncherConfig supports custom SearXNG options including github provider and top 10 results."""
    from local_ai_launcher.config import LauncherConfig

    raw_toml = {
        "searxng": {
            "enabled": True,
            "providers": ["duckduckgo", "brave", "wikipedia", "google", "github"],
            "max_results": 10,
            "categories": "general",
            "auto_fallback_native": True,
            "timeout": 8.0,
        }
    }
    flat = _flatten_overrides(raw_toml)
    assert flat["searxng_upstream_engines"] == ["duckduckgo", "brave", "wikipedia", "google", "github"]
    assert flat["searxng_max_results"] == 10
    assert flat["searxng_categories"] == "general"
    assert flat["searxng_auto_fallback_native"] is True
    assert flat["searxng_timeout"] == 8.0

    # Default LauncherConfig includes github and max_results=10
    cfg = LauncherConfig()
    assert "github" in cfg.searxng_upstream_engines
    assert cfg.searxng_max_results == 10
    assert cfg.searxng_categories == "general"
    assert cfg.searxng_auto_fallback_native is True


def test_searxng_mcp_default_limit_and_native_fallback(monkeypatch):
    """Verify SearXNG MCP tool list defaults limit to 10 and falls back to native metasearch on unreachable endpoint."""
    tools_res = searxng_mcp.handle_tools_list("req-tools")
    search_tool = next(t for t in tools_res["result"]["tools"] if t["name"] == "searxng_search")
    assert search_tool["inputSchema"]["properties"]["limit"]["default"] == 10
    assert "github" in search_tool["inputSchema"]["properties"]["engines"]["default"]

    # Point to an unreachable port to trigger native fallback
    monkeypatch.setattr(searxng_mcp, "DEFAULT_SEARXNG_URL", "http://127.0.0.1:59999")
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:59999")

    # Mock run_metasearch so network calls during tests are hermetic
    fake_hits = [
        {"title": f"Result {i}", "url": f"https://example.com/{i}", "content": f"Snippet {i}", "engine": "github", "score": 1.0}
        for i in range(1, 11)
    ]
    with patch("local_ai_launcher.backends.searxng.run_metasearch", return_value={
        "query": "local ai launcher",
        "number_of_results": 10,
        "results": fake_hits
    }) as mock_native:
        call_res = searxng_mcp.handle_tools_call("req-fallback", {
            "name": "searxng_search",
            "arguments": {"query": "local ai launcher", "limit": 10, "engines": "github"}
        })
        assert mock_native.called
        text_out = call_res["result"]["content"][0]["text"]
        assert "Found 10 results" in text_out
        assert "Result 1" in text_out
        assert "Result 10" in text_out
        assert "`[github]`" in text_out


def test_claude_md_offline_contract_shared_fingerprint_with_chat_template():
    """Drift guard: the seeded CLAUDE.md contract and the shared chat-template persona clause
    must keep the same fingerprint phrase, so one copy cannot be edited/reverted without failing."""
    from local_ai_launcher.paths import AI_WORKSPACE_DIR
    from local_ai_launcher.harnesses.claude import OFFLINE_PERSONA_CONTRACT_MD

    fingerprint = "private, offline, single-user box"
    assert fingerprint in OFFLINE_PERSONA_CONTRACT_MD

    template = AI_WORKSPACE_DIR / "models" / "chat_template_qwen38_sharp_v22.4.0_medium-convergent.jinja"
    if not template.exists():
        pytest.skip(f"shared chat template not present on this checkout: {template}")
    assert fingerprint in template.read_text(encoding="utf-8")


def test_claude_md_reseed_regenerates_contract(tmp_path, monkeypatch):
    """Reseed drill: CLAUDE.md is seeder-owned — an existing (even user-edited) file is fully
    regenerated with both the search policy and the offline answer contract on every setup."""
    test_claude_dir = tmp_path / ".claude"
    monkeypatch.setattr(config, "claude_dir", test_claude_dir)

    local_profile = test_claude_dir / "local"
    local_profile.mkdir(parents=True)
    (local_profile / "CLAUDE.md").write_text("user junk that must not survive reseed", encoding="utf-8")

    adapter = ClaudeAdapter(str(tmp_path), [])
    adapter._setup_isolated_local_profile(
        anthropic_base="http://127.0.0.1:8080",
        claude_model="claude-sonnet-4-5",
        dummy_key="dummy-token",
    )

    claude_md_text = (local_profile / "CLAUDE.md").read_text(encoding="utf-8")
    assert "user junk" not in claude_md_text
    assert "WebSearch and WebFetch are disabled" in claude_md_text
    assert "Offline Answer Contract" in claude_md_text

    # Idempotent: a second reseed produces byte-identical content
    second = adapter._setup_isolated_local_profile(
        anthropic_base="http://127.0.0.1:8080",
        claude_model="claude-sonnet-4-5",
        dummy_key="dummy-token",
    )
    assert (second / "CLAUDE.md").read_text(encoding="utf-8") == claude_md_text


def test_run_metasearch_github_provider():
    """Verify run_metasearch calls fetch_github_search when github engine is requested."""
    from local_ai_launcher.backends.searxng import run_metasearch

    fake_github = [
        {"title": "owner/repo", "url": "https://github.com/owner/repo", "content": "A repo", "engine": "github", "score": 0.98}
    ]
    with patch("local_ai_launcher.backends.searxng.fetch_github_search", return_value=fake_github) as mock_gh, \
         patch("local_ai_launcher.backends.searxng.fetch_duckduckgo_api", return_value=[]), \
         patch("local_ai_launcher.backends.searxng.fetch_duckduckgo_lite", return_value=[]):
        res = run_metasearch("ai tools", limit=10, engines=["github"])
        assert mock_gh.called
        assert len(res["results"]) == 1
        assert res["results"][0]["engine"] == "github"
        assert res["results"][0]["url"] == "https://github.com/owner/repo"


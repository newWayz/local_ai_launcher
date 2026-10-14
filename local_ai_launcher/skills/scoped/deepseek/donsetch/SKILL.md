---
name: donsetch
description: First-class DonSeTch plugin for DeepSeek Harness (DSH): keyless stealth web search, URL fetching, and crawling via native donsetch_* tools.
---

# DonSeTch Web Research for DeepSeek Harness (DSH)

DonSeTch is integrated into DeepSeek Harness via the first-class `donsetch-dsh` plugin. Tools execute in-process on the DSH tool registry (with native call/result cards in the web workbench, cancellation support, and zero MCP wrapper latency).

## Native Tools

### 1. `donsetch_web_search`
Keyless multi-engine search across Bing, DuckDuckGo, and Brave with local reranking.
- Parameters:
  - `query`: Search query
  - `limit` (optional): Result count limit
- Also powers the harness's built-in `web_search` tool without requiring a DeepSeek web search API key.

### 2. `donsetch_web_fetch`
Fetches web content and converts HTML into clean Markdown for the LLM.
- Parameters:
  - `url`: URL to retrieve
  - `focus` (optional): Focus extraction mode
  - `must_contain` (optional): Required keyword filter
- Features:
  - Chrome-grade TLS fingerprinting and browser escalation for anti-bot bypass.
  - PDF document extraction with OCR.

### 3. `donsetch_web_crawl`
Deep multi-page site and documentation crawling.
- Parameters:
  - `url`: Target domain or base URL
  - `maxDepth` (optional): Crawl depth
  - `limit` (optional): Maximum pages

### 4. `donsetch_status`
Self-diagnostics and status report of the local DonSeTch daemon, provider health, and network connectivity.

## Configuration & BYOK
- By default, searches run keyless with zero accounts required.
- Providers can be added via `donsetch keys add <provider> <key>` (Exa, Tavily, Serper, Bright Data, etc.) and land live in DSH without restarting.

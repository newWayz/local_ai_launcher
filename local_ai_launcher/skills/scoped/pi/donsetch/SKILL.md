---
name: donsetch
description: Keyless stealth web search, URL fetching with focus mode and section extraction, and recursive site crawling using native DonSeTch tools.
---

# DonSeTch Native Web Research for Pi

DonSeTch provides first-class, keyless web search, URL fetching, and crawling capabilities via native Pi tools (`web_fetch`, `web_search`, `web_crawl`, `web_screenshot`). All traffic uses Chrome-grade TLS stealth to bypass anti-bot challenges without external API keys.

## Native Tools

### 1. `web_search`
Query multi-engine keyless search (aggregating Bing, DuckDuckGo, Brave with local cross-encoder reranking).
- Parameters:
  - `query`: Search query string
  - `num_results` (optional): Maximum results (default: 8)
- Returns:
  - Structured search results with title, URL, snippet, and S-handles (`S1`...`Sn`).
  - Follow up on specific results immediately: e.g. run `web_fetch` on `S1` or URL.

### 2. `web_fetch`
Fetch full webpage content converted into clean, token-efficient Markdown.
- Parameters:
  - `url`: Target webpage URL or `S#` handle
  - `focus` (optional): Extraction mode (e.g. `section`, `toc`, `article`, `pdf`)
  - `must_contain` (optional): Keyword filter to ensure target topic is present
- Features:
  - Stealth anti-bot bypass with real browser emulation.
  - Native PDF extraction with OCR fallback.
  - Section-aware focus extraction to minimize context token usage.

### 3. `web_crawl`
Recursively crawl a documentation or website domain.
- Parameters:
  - `url`: Root URL to begin crawling
  - `max_depth` (optional): Traversal depth (default: 2)
  - `max_pages` (optional): Page ceiling (default: 10)
  - `deadline_s` (optional): Total crawl timeout in seconds
- Features:
  - Sitemap discovery and adaptive pacing.
  - Aggregates crawled pages into structured Markdown sections.

### 4. `web_screenshot`
Capture stealth screenshots of dynamic web content.
- Parameters:
  - `url`: Webpage URL to capture

## Workflow Guidelines
1. **Search Before Browsing**: Use `web_search` to discover authoritative documentation or release notes.
2. **Focused Extraction**: Prefer fetching specific sections (`focus="section"`) when examining large documentation to preserve model context.
3. **Dead-Link Resurrection**: DonSeTch automatically attempts Wayback Machine snapshots if live URLs return 404.

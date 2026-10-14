# `docs/` Index

| Document | Topic | Status (2026-10-14) |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Unified LocalAI architecture, directory tree, ports, design decisions | Current — code-verified |
| [BLACKWELL_REFERENCE.md](BLACKWELL_REFERENCE.md) | sm_120a Blackwell llama.cpp tuning, WDDM/BAR1 diagnostics, engine landscape | Current — §5/§7/§8 updated for the Bruno/Huang/official router |
| [STRATA.md](STRATA.md) | Strata 125B MoE engine, VRAM budgeting, mmap-experts modes, memory clawback | Current |
| [CLAUDE_HARNESS.md](CLAUDE_HARNESS.md) | Claude Code/Desktop adapter family, routing modes, native `/v1/messages` | New |
| [GVS5H_SOLVER.md](GVS5H_SOLVER.md) | GVS5H deep-solve autonomous solver (3 invocation methods, full CLI table) | Current |
| [UV_CACHE_PRUNE.md](UV_CACHE_PRUNE.md) | uv trampoline failures & self-healing, safe uv-cache cleanup rules | Current |
| [analysis-local_ai_launcher-skill_routers.md](analysis-local_ai_launcher-skill_routers.md) | Skill-router consolidation proposal | **Proposal — not implemented** (see status header) |
| [doc-update-analysis.md](doc-update-analysis.md) | Doc-vs-code audit that drove this update pass (ground-truth facts with file:line evidence) | Point-in-time record |

## Diagrams (SVG)

| File | Caveat |
|---|---|
| `diagram-High-Level Topology The 4 Dedicated Engines.svg` | Predates the Strata/Bruno/Huang expansion — title says "4 Dedicated Engines" while the shim layer now spans 9940 (blackwell/bruno/huang variants), 9941, 9942, 9943. Treat as historical; redraw before reuse. |
| `diagram-cpu_only_aux_models.svg` | Verify it includes Decider-4B (8084) and SearXNG (8088) before reuse. |
| `diagram-llama-cpp Internal 3-Tier Routing Flow.svg` (+ `-checkhuang_details` variant) | The `-checkhuang_details` variant covers the Huang adaptive-KV fork; the base diagram predates the `llamacpp_router` three-fork selection. |

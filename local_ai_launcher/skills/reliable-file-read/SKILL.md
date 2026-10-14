---
name: reliable-file-read
description: "Auto-resolve paths with fullwidth/Unicode chars (「」, ［］, ／) via substitution → find → realpath when reads fail."
---

# reliable-file-read

When the user asks to read a file or view an image whose path contains special characters (fullwidth brackets `「」` `【】` `（）`, fullwidth slashes `／`, zero-width chars, trailing invisible chars, non-ASCII Unicode), first attempt a direct read; if it fails with ENOENT, apply character substitution (e.g. `「」` → `[ ]`, `／` → `/`), then fall back to `find`/`realpath` for canonical resolution. Prefer in-place resolution over copying files to temp directories.

## Triggers
- Any file read request from user message
- Image viewing from user message  
- Path with special characters: `「」`, `【】`, `（）`, `！`, `？`, etc.
- Paths containing non-ASCII characters that might cause encoding issues
- Previous tool results referencing files

## Use Whenever
- Reading a file whose path comes directly from the user's message
- Opening an image or viewing any visual content by path
- Accessing any file where the read tool fails with ENOENT but the file exists
- Working with paths that contain spaces, brackets, fullwidth characters, or Unicode

## Workflow

### Primary: Direct Read
1. Call `read`/`lsp_*`/etc. with the exact path from user message
2. If successful — done

### Fallback 1: Character Substitution
If read fails (ENOENT), try common path encoding issues:
- Fullwidth slashes `／` → regular `/`
- Fullwidth brackets `「」` `【】` `（）` → `[ ]`, `( )`
- Zero-width chars, BOM markers
- Trailing spaces or invisible Unicode characters
- Re-encode the path and retry

### Fallback 2: realpath / find
If character substitution doesn't help:
1. Use `find` to locate the file by name within the workspace
2. Use `realpath` (or equivalent) to get the canonical path
3. Retry with the canonical path

### Key Rules
- **Try character substitutions first** — most ENOENT errors are encoding issues, not missing files
- **Images should never go to `<workspace_root>/tmp/`** unless all else fails
- Prefer in-place resolution over copying files to temp directories
- Always show the original path alongside the resolved path so the user sees what happened

## Example
User: "read file `C:\aiWorkspace\test「project」/file.txt`"
→ Try direct read → ENOENT
→ Substitute fullwidth brackets: `C:\aiWorkspace\test[project]/file.txt`
→ Retry → success, return content with note about path correction

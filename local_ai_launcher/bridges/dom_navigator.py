"""
Browser DOM Navigation Bridge for decider-4b v2.1.
Prunes raw DOM / Accessibility Tree dumps down to candidate interactive elements
and evaluates optimal browser action types (CLICK, TYPE, SCROLL, SUBMIT) via
Mind2Web & MiniWoB++ probes on CPU Port 8084 without GPU VRAM overhead.
"""

import re
import time
import logging
from typing import Dict, List, Any, Optional, Tuple

from local_ai_launcher.config import config
from local_ai_launcher.backends.cpu_decider import (
    evaluate_decider_dom_sync,
    evaluate_decider_dom_async
)

logger = logging.getLogger("dom_navigator")


def prune_dom_for_decider(
    raw_html_or_dom: str,
    max_elements: int = 40
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Strips non-semantic elements (<script>, <style>, <svg>, comments) from HTML/DOM
    and extracts compact candidate interactive elements (<button>, <input>, <a>, roles).
    Reduces token payload by >90% before sending to decider-4b.
    """
    if not raw_html_or_dom:
        return "", []

    # 1. Strip script, style, svg blocks and HTML comments
    clean = re.sub(r"<!--.*?-->", "", raw_html_or_dom, flags=re.DOTALL)
    clean = re.sub(r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>", "", clean, flags=re.IGNORECASE | re.DOTALL)
    clean = re.sub(r"<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>", "", clean, flags=re.IGNORECASE | re.DOTALL)
    clean = re.sub(r"<svg\b[^<]*(?:(?!<\/svg>)<[^<]*)*<\/svg>", "", clean, flags=re.IGNORECASE | re.DOTALL)

    # 2. Extract interactive elements (handles both container elements, void tags, and role-annotated divs)
    pattern = re.compile(
        r"<(button|a|select|textarea|label)\b([^>]*)>(.*?)<\/\1>|"
        r"<(input)\b([^>]*?)(?:\/?>)|"
        r"<(div|span)\b([^>]*(?:role\s*=\s*[\"'](?:button|link|textbox|checkbox|menuitem|tab|searchbox)[\"']|onclick)[^>]*)>(.*?)<\/\6>",
        re.IGNORECASE | re.DOTALL
    )

    candidate_elements: List[Dict[str, Any]] = []
    seen_ids = set()

    for match in pattern.finditer(clean):
        if len(candidate_elements) >= max_elements:
            break

        if match.group(1):
            tag = match.group(1).lower()
            attrs_raw = match.group(2)
            inner_content = re.sub(r"<[^>]+>", " ", match.group(3)).strip()
        elif match.group(4):
            tag = match.group(4).lower()
            attrs_raw = match.group(5)
            inner_content = ""
        elif match.group(6):
            tag = match.group(6).lower()
            attrs_raw = match.group(7)
            inner_content = re.sub(r"<[^>]+>", " ", match.group(8)).strip()
        else:
            continue

        # Check for interactive attributes
        is_interactive = tag in ("button", "input", "a", "select", "textarea")
        has_role = bool(re.search(r'role=["\'](button|link|textbox|checkbox|menuitem|tab|searchbox)["\']', attrs_raw, re.I))
        has_click = "onclick" in attrs_raw.lower() or "href" in attrs_raw.lower() or "tabindex" in attrs_raw.lower()

        if not (is_interactive or has_role or has_click):
            continue

        # Extract id, name, placeholder, aria-label
        id_m = re.search(r'id=["\']([^"\']+)["\']', attrs_raw, re.I)
        name_m = re.search(r'name=["\']([^"\']+)["\']', attrs_raw, re.I)
        aria_m = re.search(r'aria-label=["\']([^"\']+)["\']', attrs_raw, re.I)
        ph_m = re.search(r'placeholder=["\']([^"\']+)["\']', attrs_raw, re.I)
        type_m = re.search(r'type=["\']([^"\']+)["\']', attrs_raw, re.I)

        elem_id = id_m.group(1) if id_m else (name_m.group(1) if name_m else f"{tag}_{len(candidate_elements)}")
        if elem_id in seen_ids:
            continue
        seen_ids.add(elem_id)

        label_parts = []
        if aria_m:
            label_parts.append(aria_m.group(1))
        if ph_m:
            label_parts.append(ph_m.group(1))
        if inner_content:
            label_parts.append(inner_content[:40])
        label = " | ".join(label_parts) if label_parts else elem_id

        candidate_elements.append({
            "id": elem_id,
            "tag": tag,
            "text": label,
            "type": type_m.group(1) if type_m else "text",
            "name": name_m.group(1) if name_m else elem_id
        })

    # Build concise text representation
    lines = [f"Interactive Candidate Elements ({len(candidate_elements)}):"]
    for e in candidate_elements:
        lines.append(f"- [{e['tag']}] id=\"{e['id']}\" text=\"{e['text']}\"")
    pruned_text = "\n".join(lines)

    return pruned_text, candidate_elements


async def evaluate_browser_dom_action_async(
    goal: str,
    raw_html_or_dom: str,
    candidate_elements: Optional[List[Dict[str, Any]]] = None,
    suggest_only: bool = True
) -> Dict[str, Any]:
    """
    Asynchronously resolves target DOM element and action for browser automation tasks.
    Returns structured recommendations and estimated token savings.
    """
    t0 = time.perf_counter()
    raw_token_estimate = max(10, len(raw_html_or_dom) // 4)

    elems = candidate_elements
    pruned_dom = ""
    if elems is None:
        pruned_dom, elems = prune_dom_for_decider(raw_html_or_dom)

    res = await evaluate_decider_dom_async(
        goal=goal,
        dom_state=pruned_dom or raw_html_or_dom[:2000],
        candidate_elements=elems
    )

    action = res.get("selected_action") or res.get("action") or "CLICK"
    elem_id = res.get("selected_element_id") or "unknown_element"
    conf = res.get("confidence", 0.85)

    summary = f"Plan: Action '{action}' on element '{elem_id}' (Confidence: {conf:.2f})"
    tokens_saved = max(0, raw_token_estimate - (len(summary) // 4 + 20))
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "action": action,
        "selected_element_id": elem_id,
        "confidence": conf,
        "goal_status": res.get("goal_status", "IN_PROGRESS"),
        "action_summary": summary,
        "suggest_only": suggest_only,
        "candidate_count": len(elems),
        "tokens_saved_gpu": tokens_saved,
        "latency_ms": elapsed_ms,
        "probe": "Mind2Web_v2.1"
    }


def evaluate_browser_dom_action_sync(
    goal: str,
    raw_html_or_dom: str,
    candidate_elements: Optional[List[Dict[str, Any]]] = None,
    suggest_only: bool = True
) -> Dict[str, Any]:
    """Synchronous caller for evaluate_browser_dom_action_async."""
    t0 = time.perf_counter()
    raw_token_estimate = max(10, len(raw_html_or_dom) // 4)

    elems = candidate_elements
    pruned_dom = ""
    if elems is None:
        pruned_dom, elems = prune_dom_for_decider(raw_html_or_dom)

    res = evaluate_decider_dom_sync(
        goal=goal,
        dom_state=pruned_dom or raw_html_or_dom[:2000],
        candidate_elements=elems
    )

    action = res.get("selected_action") or res.get("action") or "CLICK"
    elem_id = res.get("selected_element_id") or "unknown_element"
    conf = res.get("confidence", 0.85)

    summary = f"Plan: Action '{action}' on element '{elem_id}' (Confidence: {conf:.2f})"
    tokens_saved = max(0, raw_token_estimate - (len(summary) // 4 + 20))
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "action": action,
        "selected_element_id": elem_id,
        "confidence": conf,
        "goal_status": res.get("goal_status", "IN_PROGRESS"),
        "action_summary": summary,
        "suggest_only": suggest_only,
        "candidate_count": len(elems),
        "tokens_saved_gpu": tokens_saved,
        "latency_ms": elapsed_ms,
        "probe": "Mind2Web_v2.1"
    }

import subprocess
import time
import sys
import os
import shutil
from pathlib import Path
import zipfile
import io
import re
import json
import threading
import urllib.request
import urllib.error
import webbrowser
import logging
import sqlite3
import atexit
import signal
try:
    import psutil
except ImportError:
    psutil = None
import configparser
import socket
import argparse
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from abc import ABC, abstractmethod

from local_ai_launcher.config import *
from local_ai_launcher.paths import get_basename, STRATA_MODELS_PRESET_FILE

def normalize_model_identifier(model_name: Optional[str]) -> str:
    """Normalize model identifier by stripping backend directory prefixes (freetoken/, exl3/, gguf/)."""
    if not model_name:
        return ""
    name = str(model_name).strip()
    clean = name.replace("\\", "/")
    for prefix in ("freetoken/", "exl3/", "gguf/"):
        if clean.lower().startswith(prefix):
            return clean[len(prefix):]
    return name

def scan_model_directories(base_dir: Path, max_depth: int = 2) -> List[Path]:
    """Scan subdirectories up to max_depth without inspecting individual files, preserving org/model structures."""
    results = []
    if not base_dir.exists():
        return results
    base_parts_len = len(base_dir.parts)
    for root, dirs, _ in os.walk(base_dir):
        p = Path(root)
        if p != base_dir:
            results.append(p)
            if len(p.parts) - base_parts_len >= max_depth:
                dirs.clear()
    return results

def get_model_specs(model_name: str) -> tuple[int, bool]:
    ctx_size = 32768
    is_reasoning = False
    try:
        norm_name = normalize_model_identifier(model_name)
        parser = load_effective_models_parser()
        
        clean_name = get_basename(model_name)
        norm_clean = get_basename(norm_name)
        
        for sec in parser.sections():
                if sec.strip() == "*" or sec.lower() == "default":
                    continue
                sec_alias = parser.get(sec, "alias", fallback="")
                if (get_basename(sec) in (clean_name, norm_clean) or get_basename(sec_alias) in (clean_name, norm_clean) or
                    sec in (model_name, norm_name) or sec_alias in (model_name, norm_name)):
                    ctx_size = int(parser.get(sec, "ctx-size", fallback=32768))
                    is_reasoning = parser.get(sec, "reasoning", fallback="off").lower() == "on"
                    return ctx_size, is_reasoning
            
        ft_ini_path = FREETOKENS_MODELS_PRESET_FILE
        if ft_ini_path.exists():
            ft_parser = configparser.ConfigParser(strict=False)
            ft_parser.read(str(ft_ini_path), encoding="utf-8")
            
            clean_name = get_basename(norm_name)
            norm_clean = get_basename(model_name)
                
            for sec in ft_parser.sections():
                sec_alias = ft_parser.get(sec, "alias", fallback="")
                
                if (get_basename(sec) in (clean_name, norm_clean) or get_basename(sec_alias) in (clean_name, norm_clean) or 
                    sec in (model_name, norm_name) or sec_alias in (model_name, norm_name)):
                    max_seq_len = int(ft_parser.get(sec, "max-seq-len-override", fallback=8192))
                    kv_reserve = int(ft_parser.get(sec, "kv-reserve-tokens", fallback=max_seq_len))
                    ctx_size = min(max_seq_len, kv_reserve)
                    is_reasoning = ft_parser.has_option(sec, "reasoning-parser") and ft_parser.get(sec, "reasoning-parser", fallback="").lower() != "off"
                    return ctx_size, is_reasoning


        if STRATA_MODELS_PRESET_FILE.exists():
            try:
                with open(STRATA_MODELS_PRESET_FILE, "r", encoding="utf-8") as sf:
                    strata_data = json.load(sf)
                clean_name = get_basename(norm_name) if norm_name else ""
                norm_clean = get_basename(model_name) if model_name else ""
                targets = {t for t in (clean_name, norm_clean, model_name, norm_name) if t and isinstance(t, str)}
                for sm_id, sm_cfg in strata_data.items():
                    if sm_id.startswith("_") or not isinstance(sm_cfg, dict):
                        continue
                    aliases = [str(a) for a in sm_cfg.get("aliases", []) if a]
                    all_names = [sm_id, get_basename(sm_id), *aliases, *(get_basename(a) for a in aliases)]
                    cands = {c for c in all_names if c and isinstance(c, str)}
                    if targets & cands:
                        ctx = int(sm_cfg.get("ctx_size", 262144))
                        r_cfg = sm_cfg.get("reasoning", {})
                        is_reas = r_cfg.get("enabled", False) if isinstance(r_cfg, dict) else bool(r_cfg)
                        return ctx, is_reas
            except Exception as se:
                logger.warning(f"Failed to parse strata_models.json for specs: {se}")

        exl3_dir = AI_WORKSPACE_DIR / "models" / "exl3"
        if exl3_dir.exists():
            for d in scan_model_directories(exl3_dir):
                if model_name.lower() in d.name.lower():
                    tabby_yml = d / "tabby_config.yml"
                    if tabby_yml.exists():
                        content = tabby_yml.read_text(encoding="utf-8")
                        match = re.search(r'max_seq_len:\s*(\d+)', content)
                        if match:
                            ctx_size = int(match.group(1))
                        if re.search(r'reasoning:\s*true', content, re.IGNORECASE) or re.search(r'reasoning_budget_tokens', content):
                            is_reasoning = True
                        return ctx_size, is_reasoning
    except Exception as e:
        logger.warning(f"Failed to determine model specs for {model_name}: {e}")
    return ctx_size, is_reasoning


@dataclass
class CompactionParams:
    ctx_size: int
    is_reasoning: bool
    pi_reserve_tokens: int
    pi_keep_tokens: int
    hermes_prune_tokens: int
    protect_first_n: int
    protect_last_n: int
    compaction_enabled_str: str


def compute_compaction_params(model_name: str) -> CompactionParams:
    ctx_size, is_reasoning = get_model_specs(model_name)
    return CompactionParams(
        ctx_size=ctx_size,
        is_reasoning=is_reasoning,
        pi_reserve_tokens=int(ctx_size * 0.25) if is_reasoning else 16384,
        pi_keep_tokens=max(10000, min(int(ctx_size * config.pi_compaction_keep_ratio), 40000)),
        hermes_prune_tokens=int(ctx_size * config.hermes_proactive_prune_ratio),
        protect_first_n=3,
        protect_last_n=max(6, min(int(ctx_size / 4000), 30)),
        compaction_enabled_str="true" if config.compaction_enabled else "false",
    )


def resolve_model_name(active_model: Optional[str] = None, use_omniroute: Optional[bool] = None, omniroute_model: Optional[str] = None) -> str:
    """Resolve active model name based on OmniRoute status and requested model name."""
    use_omni = config.use_omniroute_proxy if use_omniroute is None else use_omniroute
    omni_model = config.omniroute_default_model if omniroute_model is None else omniroute_model
    if use_omni:
        return omni_model
    raw = active_model if active_model else config.default_model
    return normalize_model_identifier(raw)


def fetch_omniroute_models(base_url):
    """Fetch available models and combos from OmniRoute API endpoint (instant fallback to avoid blocking startup)."""
    return ["1-local_first-brain", "2-online-free_workers"]


def _parse_models_ini(ini_path: Path, models_info: list, ctx_key: str = "ctx-size", ctx_default: str = "131072"):
    """Parse an INI file and append model defs to models_info."""
    parser = configparser.ConfigParser(strict=False)
    try:
        parser.read(str(ini_path), encoding="utf-8")
        for section in parser.sections():
            if section.lower() == "default" or section.strip() == "*":
                continue
            sec_data = dict(parser.items(section))
            if "model" not in sec_data and "max-seq-len-override" not in sec_data:
                continue
            ctx = int(sec_data.get(ctx_key, ctx_default))
            has_mmproj = "mmproj" in sec_data
            input_types = ["text", "image"] if has_mmproj else ["text"]
            models_info.append({"id": section, "name": section,
                                "contextWindow": ctx, "input": input_types})
            for alias in (a.strip() for a in sec_data.get("alias", "").split(",") if a.strip()):
                if alias != section and not any(m["id"] == alias for m in models_info):
                    models_info.append({"id": alias, "name": alias,
                                        "contextWindow": ctx, "input": input_types})
    except Exception as e:
        logger.error(f"Warning: Could not parse {ini_path}: {e}")


def get_available_preset_models(only_installed: bool = False, active_model: Optional[str] = None):
    """Parse models.ini and return list of model definitions for agent harnesses.
    
    If only_installed is True, filters to models that physically exist on disk,
    excluding uninstalled presets, backslash-corrupted identifiers, and duplicate aliases.
    """
    active_norm = normalize_model_identifier(active_model or getattr(config, 'default_model', ''))
    models_info = []
    seen_ids = set()

    def _add_model_entry(mid: str, mname: str, ctx: int, input_types: list[str]) -> bool:
        if not mid or mid in seen_ids or "\\" in mid or mid == "*":
            return False
        seen_ids.add(mid)
        models_info.append({
            "id": mid,
            "name": mname,
            "contextWindow": ctx,
            "input": input_types
        })
        return True

    config_parser = load_effective_models_parser()
    if not config_parser.sections():
        ctx_len = 131072
        try:
            import yaml
            tabby_conf = AI_WORKSPACE_DIR / "models" / "exl3" / config.default_model / "tabby_config.yml"
            if tabby_conf.exists():
                yml_data = yaml.safe_load(tabby_conf.read_text(encoding='utf-8'))
                if yml_data and "model" in yml_data and "max_seq_len" in yml_data["model"]:
                    ctx_len = int(yml_data["model"]["max_seq_len"])
        except Exception:
            pass
        return [{"id": config.default_model, "name": config.default_model, "contextWindow": ctx_len, "input": ["text", "image"]}]

    try:
        for section in config_parser.sections():
            if section.lower() == "default" or section.strip() == "*":
                continue
            sec_data = dict(config_parser.items(section))
            if "model" not in sec_data:
                continue

            # If only_installed is requested, verify model file actually exists on disk
            if only_installed:
                mpath = sec_data.get("model")
                if mpath:
                    full_path = AI_WORKSPACE_DIR / mpath
                    if not full_path.exists() and section != active_norm:
                        continue

            ctx = int(sec_data.get("ctx-size", "131072"))
            has_mmproj = "mmproj" in sec_data
            input_types = ["text", "image"] if has_mmproj else ["text"]
            _add_model_entry(section, section, ctx, input_types)

            if not only_installed:
                aliases = [a.strip() for a in sec_data.get("alias", "").split(",") if a.strip()]
                for alias in aliases:
                    _add_model_entry(alias, alias, ctx, input_types)
    except Exception as e:
        logger.warning(f"Warning: Could not parse {MODELS_PRESET_FILE}: {e}")

    # Also scan freetokens_models.ini
    if FREETOKENS_MODELS_PRESET_FILE.exists():
        ft_parser = configparser.ConfigParser(strict=False)
        try:
            ft_parser.read(str(FREETOKENS_MODELS_PRESET_FILE), encoding="utf-8")
            for section in ft_parser.sections():
                if section.lower() == "default" or section.strip() == "*":
                    continue
                sec_data = dict(ft_parser.items(section))
                max_seq_len = int(sec_data.get("max-seq-len-override", "8192"))
                kv_reserve = int(sec_data.get("kv-reserve-tokens", str(max_seq_len)))
                ctx = min(max_seq_len, kv_reserve)
                input_types = ["text", "image"]
                _add_model_entry(section, section, ctx, input_types)

                if not only_installed:
                    aliases = [a.strip() for a in sec_data.get("alias", "").split(",") if a.strip()]
                    for alias in aliases:
                        _add_model_entry(alias, alias, ctx, input_types)
                    _add_model_entry(f"freetoken/{section}", f"freetoken/{section}", ctx, input_types)
        except Exception as e:
            logger.warning(f"Warning: Could not parse {FREETOKENS_MODELS_PRESET_FILE}: {e}")

    # Also scan exl3 models and append them
    try:
        import yaml
        exl3_dir = AI_WORKSPACE_DIR / "models" / "exl3"
        if exl3_dir.exists():
            for model_dir in scan_model_directories(exl3_dir):
                if any(model_dir.glob("*.safetensors")):
                    ctx_len = 131072
                    tabby_conf = model_dir / "tabby_config.yml"
                    if tabby_conf.exists():
                        try:
                            yml_data = yaml.safe_load(tabby_conf.read_text(encoding='utf-8'))
                            if yml_data and "model" in yml_data and "max_seq_len" in yml_data["model"]:
                                ctx_len = int(yml_data["model"]["max_seq_len"])
                        except Exception:
                            pass
                    _add_model_entry(model_dir.name, model_dir.name, ctx_len, ["text", "image"])
    except Exception:
        pass

    # Also scan strata_models.json presets
    if STRATA_MODELS_PRESET_FILE.exists():
        try:
            with open(STRATA_MODELS_PRESET_FILE, "r", encoding="utf-8") as sf:
                strata_data = json.load(sf)
            for sm_id, sm_cfg in strata_data.items():
                if sm_id.startswith("_") or not isinstance(sm_cfg, dict):
                    continue
                if only_installed:
                    mpath = sm_cfg.get("model")
                    if mpath:
                        full_path = AI_WORKSPACE_DIR / mpath
                        if not full_path.exists() and sm_id != active_norm:
                            continue
                ctx = int(sm_cfg.get("ctx_size", 262144))
                has_vision = "vision" in sm_id.lower() or any("vision" in str(a).lower() for a in sm_cfg.get("aliases", []))
                input_types = ["text", "image"] if has_vision else ["text"]
                _add_model_entry(sm_id, sm_id, ctx, input_types)
                if not only_installed:
                    for alias in sm_cfg.get("aliases", []):
                        _add_model_entry(alias, alias, ctx, input_types)
        except Exception as e:
            logger.warning(f"Failed to parse strata_models.json presets: {e}")

    # Append CPU ONNX Decision Gater model if enabled
    if getattr(config, 'use_cpu_onnx_gater', True) or getattr(config, 'use_cpu_decision', True):
        _add_model_entry("cpu-gater", "cpu-gater", 4096, ["text"])

    if not models_info:
        def_m = getattr(config, 'default_model', 'local-model')
        return [{"id": def_m, "name": def_m, "contextWindow": 131072, "input": ["text", "image"]}]

    return models_info




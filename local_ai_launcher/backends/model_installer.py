import os
import sys
import shutil
import subprocess
import configparser
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from local_ai_launcher.config import (
    config,
    logger,
    prompt_choice,
    AI_WORKSPACE_DIR,
    MODELS_PRESET_FILE,
)

# Registry mapping known GGUF weight files and multimodal projectors to their Hugging Face repos
GGUF_TO_HF_REGISTRY: Dict[str, str] = {
    # Qwen 3.8 Blackwell / Dense
    "Qwen3.8-27B-UD-IQ4_XS.gguf": "unsloth/Qwen3.8-27B-GGUF",
    "Qwen3.8-27B-UD-ASCII-IQ4_XS.gguf": "unsloth/Qwen3.8-27B-GGUF",
    "Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf": "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF",
    "Huihui-Qwen3.8-27B-abliterated-UD-ASCII-IQ4_XS.gguf": "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF",
    "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf": "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO",
    "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf": "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO",
    "mmproj-Qwen3.8-27B-BF16.gguf": "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO",
    # KatCoder 2.5 35B APEX MoE
    "KAT-Coder-V2.5-Dev-APEX-I-Quality.gguf": "mudler/KAT-Coder-V2.5-Dev-APEX-GGUF",
    "mmproj-Qwen_Qwen3.6-35B-A3B-f16.gguf": "mudler/KAT-Coder-V2.5-Dev-APEX-GGUF",
    # Qwen 3.6 35B MoE
    "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf": "unsloth/qwen-36_35B_A3B_UD-Q4_K_XL",
    # Ornith 1.5 35B APEX MoE
    "Ornith-1.5-35B-A3B-APEX-I-Quality.gguf": "mudler/Ornith-1.5-35B-A3B-APEX-GGUF",
    # Qwen 3.6 35B Uncensored Heretic MoE
    "Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-I-Quality.gguf": "SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF",
    "mmproj-Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-F16.gguf": "SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF",
    # Qwen 3.6 35B Uncensored Genesis MoE
    "Hermes3.6-35B-A3B-Uncensored-Genesis-V7-APEX.gguf": "LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V7-GGUF",
    # Gemma 4 12B
    "Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf": "HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced",
    "mmproj-Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf": "HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced",
    # Muse Glimmer 30B
    "darkc0de_Muse-Glimmer-30B-heretic-IQ3_M.gguf": "bartowski/Muse-Glimmer-30B-GGUF",
    "mmproj-kquant.gguf": "bartowski/Muse-Glimmer-30B-GGUF",
    # Strata 125B MoE (SC117 & Swift 1.5)
    "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf": "SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF",
    "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf": "SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF",
    "mmproj-Qwen3.8-Flash-Next-BF16.gguf": "SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF",
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf": "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf": "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
    "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf": "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
}



def resolve_model_repo(rel_path: str, sec_options: Optional[dict] = None) -> Optional[str]:
    """
    Resolves the Hugging Face repository (e.g. 'unsloth/Qwen3.8-27B-GGUF')
    using:
    1. Explicit 'repo' or 'hf-repo' in section options.
    2. Exact filename match in GGUF_TO_HF_REGISTRY.
    3. Path inference from 'models/{owner}/{repo_name}/{filename}'.
    """
    if sec_options:
        for key in ("repo", "hf-repo", "hf_repo", "huggingface_repo"):
            if key in sec_options and sec_options[key].strip():
                return sec_options[key].strip()

    filename = Path(rel_path).name
    if filename in GGUF_TO_HF_REGISTRY:
        return GGUF_TO_HF_REGISTRY[filename]

    norm = rel_path.replace("\\", "/").strip("/")
    if norm.lower().startswith("models/"):
        parts = norm[len("models/"):].split("/")
        if parts and parts[0].lower() in ("llama-server", "gguf"):
            parts = parts[1:]
        if len(parts) >= 3:
            # parts[0] is author/owner, parts[1] is repo
            return f"{parts[0]}/{parts[1]}"

    return None


def get_hf_downloader_executable() -> Optional[Path]:
    """Finds the hf or huggingface-cli executable on the system."""
    candidates = [
        AI_WORKSPACE_DIR / ".venv" / "Scripts" / "hf.exe",
        AI_WORKSPACE_DIR / ".venv" / "Scripts" / "huggingface-cli.exe",
    ]
    for c in candidates:
        if c.exists():
            return c

    for name in ("hf.exe", "hf", "huggingface-cli.exe", "huggingface-cli"):
        w = shutil.which(name)
        if w:
            return Path(w)
    return None


def download_model_file(repo_id: str, filename: str, target_dir: Path) -> bool:
    """
    Downloads a single model file from Hugging Face into target_dir.
    Uses hf.exe CLI when available, falling back to python huggingface_hub.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename

    logger.info(f"Downloading '{filename}' from Hugging Face repo '{repo_id}' into {target_dir}...")
    hf_exe = get_hf_downloader_executable()
    if hf_exe:
        try:
            cmd = [str(hf_exe), "download", repo_id, filename, "--local-dir", str(target_dir)]
            logger.info(f"Running: {' '.join(cmd)}")
            res = subprocess.run(cmd, text=True, encoding='utf-8', errors='replace')
            if res.returncode == 0 and target_path.exists() and target_path.stat().st_size > 0:
                logger.info(f"[SUCCESS] Downloaded {filename} ({target_path.stat().st_size // (1024*1024)} MiB)")
                return True
            else:
                logger.warning(f"hf.exe download returned code {res.returncode}. Attempting python fallback...")
        except Exception as e:
            logger.warning(f"hf.exe download encountered error: {e}. Trying python fallback...")

    # Python huggingface_hub fallback
    try:
        from huggingface_hub import hf_hub_download
        logger.info(f"Using huggingface_hub python API to download {filename} from {repo_id}...")
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(target_dir))
        if target_path.exists() and target_path.stat().st_size > 0:
            logger.info(f"[SUCCESS] Downloaded {filename} ({target_path.stat().st_size // (1024*1024)} MiB)")
            return True
    except Exception as e:
        logger.error(f"Failed to download {filename} from Hugging Face: {e}")

    return False


def find_model_section(parser: configparser.ConfigParser, model_name: str) -> Optional[Tuple[str, dict]]:
    """Locates the section in models.ini matching model_name or alias."""
    from local_ai_launcher.paths import get_basename
    clean_name = get_basename(model_name)
    for sec in parser.sections():
        if sec.strip() == "*" or sec.lower() == "default":
            continue
        sec_alias = parser.get(sec, "alias", fallback="")
        aliases = [a.strip() for a in sec_alias.split(",") if a.strip()]

        if (sec == model_name or
            model_name in aliases or
            get_basename(sec) == clean_name or
            any(get_basename(a) == clean_name for a in aliases)):
            return sec, dict(parser.items(sec))
    return None


def ensure_single_model_installed(selected_model_name: str) -> bool:
    """
    Verifies and installs ONLY the single model selected via config.py (config.default_model).
    Does NOT download or check all models in models.ini.
    """
    from local_ai_launcher.presets import load_effective_models_parser
    try:
        parser = load_effective_models_parser()
    except Exception as e:
        logger.warning(f"Could not load effective models parser: {e}")
        return True

    match = find_model_section(parser, selected_model_name)
    if not match:
        # Not a models.ini model (could be exl3 or freetoken); skip GGUF check
        return True

    sec_name, sec_options = match
    logger.info(f"Verifying local files for selected model '{selected_model_name}' (preset: [{sec_name}])...")

    # Keys that point to local files required for model execution
    file_keys = ["model", "mmproj", "model-draft", "chat-template-file"]
    files_to_check: List[str] = []
    for k in file_keys:
        val = sec_options.get(k, "").strip()
        if val and not val.startswith("#"):
            files_to_check.append(val)

    for rel_path in files_to_check:
        target_path = AI_WORKSPACE_DIR / rel_path
        if target_path.exists() and target_path.stat().st_size > 0:
            continue

        filename = target_path.name
        repo_id = resolve_model_repo(rel_path, sec_options)
        if not repo_id:
            logger.warning(f"Notice: Model file '{rel_path}' is missing, but no Hugging Face repo mapping could be resolved.")
            return False

        from local_ai_launcher.backends.ascii_pruner import (
            is_ascii_model,
            derive_base_model_filename,
            ensure_ascii_variant,
        )

        if is_ascii_model(filename):
            base_filename = derive_base_model_filename(filename)
            base_target_path = target_path.parent / base_filename

            if not base_target_path.exists() or base_target_path.stat().st_size == 0:
                logger.info(f"Notice: Required ASCII model '{filename}' is missing.")
                logger.info(f"  -> Base model '{base_filename}' must be downloaded from Hugging Face ({repo_id}) and pruned to ASCII (~12.25 GB).")

                behavior = getattr(config, 'models_install_behavior', 'existing').lower().strip()
                if behavior == 'never_install':
                    logger.info(f"  -> Skipping download of '{base_filename}' as per models_install_behavior='never_install'.")
                    return False
                elif behavior == 'existing':
                    prompt = f"Selected ASCII model '{filename}' requires base model '{base_filename}'. Download from Hugging Face ({repo_id}) and auto-prune? (y/n): "
                    if not prompt_choice(prompt, default='y', timeout=15):
                        logger.info(f"  -> User declined download of '{base_filename}'.")
                        return False

                success = download_model_file(repo_id, base_filename, target_path.parent)
                if not success or not base_target_path.exists() or base_target_path.stat().st_size == 0:
                    logger.error(f"Error: Failed to download required base model '{base_filename}' from Hugging Face.")
                    return False
            else:
                logger.info(f"Notice: Base model '{base_filename}' exists ({base_target_path.stat().st_size / 1e9:.2f} GB). Pruning to ASCII variant '{filename}'...")

            try:
                ensure_ascii_variant(base_target_path, target_path)
            except Exception as e:
                logger.error(f"Error: Failed to prune base model to ASCII variant '{filename}': {e}")
                return False

            if not target_path.exists() or target_path.stat().st_size == 0:
                logger.error(f"Error: ASCII variant '{filename}' was not created.")
                return False

            continue

        logger.info(f"Notice: Required model file is missing: {rel_path}")
        logger.info(f"  -> Mapped to Hugging Face repository: {repo_id}")

        behavior = getattr(config, 'models_install_behavior', 'existing').lower().strip()
        if behavior == 'never_install':
            logger.info(f"  -> Skipping download of '{filename}' as per models_install_behavior='never_install'.")
            return False
        elif behavior == 'existing':
            prompt = f"Selected model file '{filename}' is missing. Download from Hugging Face ({repo_id})? (y/n): "
            if not prompt_choice(prompt, default='y', timeout=15):
                logger.info(f"  -> User declined download of '{filename}'.")
                return False

        # Execute download for only this specific file
        success = download_model_file(repo_id, filename, target_path.parent)
        if not success:
            logger.error(f"Error: Failed to install required model file '{filename}'.")
            return False

    return True

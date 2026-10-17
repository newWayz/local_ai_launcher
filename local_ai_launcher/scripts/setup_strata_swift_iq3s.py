import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

# Paths (dynamically resolved for 100% portability)
WORKSPACE = Path(os.environ.get("AI_WORKSPACE_DIR", Path(__file__).resolve().parents[2]))
STRATA_DIR = WORKSPACE / "strata"
TARGET_DIR = WORKSPACE / "models" / "llama-server" / "ukisai" / "Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
REPO_ID = "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"

SHARDS = [
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
    "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf"
]

def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def find_hf_cli():
    for name in ["hf.exe", "hf", "huggingface-cli.exe", "huggingface-cli"]:
        w = shutil.which(name)
        if w:
            return w
    return None

def download_shards():
    hf_bin = find_hf_cli()
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    for shard in SHARDS:
        shard_path = TARGET_DIR / shard
        # Skip download if already complete
        min_size = 500 * 1024 * 1024 if "mmproj" in shard else 10 * 1024 * 1024 * 1024
        if shard_path.exists() and shard_path.stat().st_size > min_size:
            log(f"File already downloaded: {shard} ({shard_path.stat().st_size / (1024**3):.2f} GB)")
            continue
        
        log(f"Starting download for {shard} from {REPO_ID}...")
        if hf_bin:
            cmd = [
                hf_bin,
                "download",
                REPO_ID,
                shard,
                "--local-dir",
                str(TARGET_DIR)
            ]
            log(f"Running command: {' '.join(cmd)}")
            res = subprocess.run(cmd, cwd=str(WORKSPACE))
            if res.returncode != 0:
                log(f"hf.exe failed with returncode {res.returncode}, falling back to huggingface_hub...")
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo_id=REPO_ID, filename=shard, local_dir=str(TARGET_DIR))
        else:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=REPO_ID, filename=shard, local_dir=str(TARGET_DIR))
        
        log(f"Finished download for {shard}.")

def prepare_pack():
    first_shard = TARGET_DIR / SHARDS[0]
    if not first_shard.exists():
        log(f"Error: {first_shard} not found. Cannot prepare pack.")
        return False

    pack_dir = STRATA_DIR / "data" / "packs" / "swift-qwen3.8-flash-next-gsq-rco-iq3_s"
    pack_dir.mkdir(parents=True, exist_ok=True)
    
    native_experts = pack_dir / "native_experts.txt"
    if native_experts.exists():
        log(f"Pack already prepared at {pack_dir}")
        return True

    log(f"Preparing Strata native pack at {pack_dir}...")
    llama_dir = STRATA_DIR / "third_party" / "llama.cpp"
    env = dict(os.environ)
    if llama_dir.exists():
        env["STRATA_GGUF_PY"] = str(llama_dir / "gguf-py")

    cmd = [
        sys.executable,
        str(STRATA_DIR / "tools" / "iq_pack.py"),
        "--gguf", str(first_shard),
        "--out", str(pack_dir)
    ]
    res = subprocess.run(cmd, env=env, cwd=str(STRATA_DIR))
    if res.returncode == 0:
        log(f"Pack preparation complete: {pack_dir}")
        return True
    else:
        log(f"Pack preparation failed with code {res.returncode}")
        return False

def link_strata_rt():
    target_rt = TARGET_DIR / "strata" / "rt"
    if target_rt.exists():
        return
    source_rt = WORKSPACE / "models" / "llama-server" / "SC117" / "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF" / "strata" / "rt"
    if source_rt.exists():
        target_rt.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(str(source_rt), str(target_rt))
            log(f"Copied shared Strata MTP runtime pack to {target_rt}")
        except Exception as e:
            log(f"Notice: unable to copy shared rt: {e}")

def main():
    log("=== Strata Swift-1.5 Qwen3.8 Flash Next IQ3_S Automated Setup ===")
    download_shards()
    log("=== Shard Download Complete. Verifying Pack ===")
    prepare_pack()
    link_strata_rt()
    log("=== Strata Swift-1.5 IQ3_S Ready for LocalAI Launcher ===")

if __name__ == "__main__":
    main()

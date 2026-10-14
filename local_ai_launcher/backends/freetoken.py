from local_ai_launcher.environment import _get_cuda_build_env
from local_ai_launcher.config import prompt_choice
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
from local_ai_launcher.environment import *
from local_ai_launcher.models import *
from local_ai_launcher.process import *




def _patch_freetoken_windows_hotfixes(freetoken_dir: Path):
    """Apply any necessary hotfixes to the freetoken-windows codebase after cloning/pulling."""
    if sys.platform != 'win32':
        return
        
    ftw_path = freetoken_dir / "python" / "freetoken" / "checkpoint" / "ftw.py"
    if ftw_path.exists():
        content = ftw_path.read_text(encoding="utf-8")
        modified = False
        if "m = mmap.mmap(fd, 0, prot=mmap.PROT_READ)" in content:
            logger.info("  -> Applying Windows mmap hotfix to ftw.py...")
            content = content.replace("m = mmap.mmap(fd, 0, prot=mmap.PROT_READ)", "m = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)")
            modified = True
            
        if not re.search(r'^import sys\s*$', content, re.MULTILINE):
            logger.info("  -> Applying Windows sys import hotfix to ftw.py...")
            content = re.sub(r'^(import os\s*)$', r'\1\nimport sys', content, count=1, flags=re.MULTILINE)
            modified = True
            
        if modified:
            ftw_path.write_text(content, encoding="utf-8")
            
    fp8_path = freetoken_dir / "python" / "freetoken" / "kernel" / "triton" / "fp8_pertensor_linear.py"
    if fp8_path.exists():
        content = fp8_path.read_text(encoding="utf-8")
        modified = False
        
        if "_ROWWISE_SCALED_MM_SUPPORTED = True" not in content:
            logger.info("  -> Applying Windows FP8 rowwise scaling fallback hotfix to fp8_pertensor_linear.py...")
            content = content.replace("def _scaled_mm(", "_ROWWISE_SCALED_MM_SUPPORTED = True\n\ndef _scaled_mm(")
            content = content.replace("qa = _static_quant(a, input_scale)", "global _ROWWISE_SCALED_MM_SUPPORTED\n    qa = _static_quant(a, input_scale)")
            
            old_return = "    return torch._scaled_mm(\n        qa, wt,\n        scale_a=input_scale.reshape(1, 1).expand(a.shape[0], 1).contiguous(),\n        scale_b=weight_scale.reshape(1, -1),\n        out_dtype=out_dtype,\n    )"
            new_return = "    if not _ROWWISE_SCALED_MM_SUPPORTED:\n        raise NotImplementedError(\"Rowwise scaling unsupported\")\n    try:\n        return torch._scaled_mm(\n            qa, wt,\n            scale_a=input_scale.reshape(1, 1).expand(a.shape[0], 1).contiguous(),\n            scale_b=weight_scale.reshape(1, -1),\n            out_dtype=out_dtype,\n        )\n    except RuntimeError as e:\n        if \"Rowwise scaling is not currently supported\" in str(e):\n            _ROWWISE_SCALED_MM_SUPPORTED = False\n            raise NotImplementedError(\"Rowwise scaling unsupported\")\n        raise"
            content = content.replace(old_return, new_return)
            
            old_elif = "    elif input_scale is not None and e4m3_native():\n        out = _scaled_mm(\n            x.reshape(-1, K), weight, weight_scale, input_scale, uniform_scale, x.dtype,\n        ).reshape(*lead, N)"
            new_elif = "    elif input_scale is not None and e4m3_native():\n        try:\n            out = _scaled_mm(\n                x.reshape(-1, K), weight, weight_scale, input_scale, uniform_scale, x.dtype,\n            ).reshape(*lead, N)\n        except NotImplementedError:\n            if x.numel() // K == 1:\n                out = _gemv(x.reshape(K), e4m3_kernel_view(weight), weight_scale, x.dtype).reshape(*lead, N)\n            else:\n                out = _gemm(\n                    x.reshape(-1, K), e4m3_kernel_view(weight), weight_scale, x.dtype,\n                ).reshape(*lead, N)"
            content = content.replace(old_elif, new_elif)
            modified = True
            
        if modified:
            fp8_path.write_text(content, encoding="utf-8")
            
    patch_deps_path = freetoken_dir / "scripts" / "patch_deps_windows.py"
    if patch_deps_path.exists():
        content = patch_deps_path.read_text(encoding="utf-8")
        modified = False
        
        if "function_details.h" not in content:
            logger.info("  -> Applying Windows tvm-ffi MSVC template JIT hotfix to patch_deps_windows.py...")
            new_logic = """
    # Also patch function_details.h for MSVC template issue
    include_dir = p.parent.parent.parent / "include" / "tvm" / "ffi"
    func_details = include_dir / "function_details.h"
    if func_details.exists():
        s_func = func_details.read_text(encoding="utf-8")
        if "struct FunctionInfo<T&, void>" not in s_func:
            old_str = "template <typename T, typename = void>\\nstruct FunctionInfo : FunctionInfoHelper<decltype(&T::operator())> {};"
            new_str = old_str + "\\ntemplate <typename T>\\nstruct FunctionInfo<T&, void> : FunctionInfo<T, void> {};"
            s_func = s_func.replace(old_str, new_str)
            old_str_crlf = "template <typename T, typename = void>\\r\\nstruct FunctionInfo : FunctionInfoHelper<decltype(&T::operator())> {};"
            new_str_crlf = old_str_crlf + "\\r\\ntemplate <typename T>\\r\\nstruct FunctionInfo<T&, void> : FunctionInfo<T, void> {};"
            s_func = s_func.replace(old_str_crlf, new_str_crlf)
            func_details.write_text(s_func, encoding="utf-8")
            print(f"patched tvm-ffi function_details.h")
"""
            content = content.replace("    return 0", new_logic + "    return 0")
            modified = True
            
        if modified:
            patch_deps_path.write_text(content, encoding="utf-8")
def check_freetoken():
    """Check if FreeToken is installed and provide options to install or update."""
    logger.info("\nChecking for FreeToken...")
    freetoken_dir = AI_WORKSPACE_DIR / "freetoken"
    venv_python = freetoken_dir / "venv" / "Scripts" / "python.exe" if sys.platform == 'win32' else freetoken_dir / "venv" / "bin" / "python"
    
    ft_exe = freetoken_dir / "venv" / "Scripts" / "ft.exe" if sys.platform == 'win32' else freetoken_dir / "venv" / "bin" / "ft"
    is_installed = ft_exe.exists()
    install_target = "." if sys.platform == 'win32' else ".[accel]"
    build_env = _get_cuda_build_env()
    
    if is_installed:
        logger.info("  -> FreeToken is already installed.")
        update_behavior = config.freetoken_update_behavior if config.freetoken_update_behavior != 'existing' else 'existing'
        update_frequency_hrs = getattr(config, 'engine_update_frequency_hrs', {}).get(
            'freetoken', getattr(config, 'update_frequency_hrs', 168)
        )
        force = getattr(config, 'force_update', False)

        from local_ai_launcher.update_frequency import should_update_component, record_successful_update
        do_update = should_update_component(
            component_name="freetoken",
            category="engine",
            behavior=update_behavior,
            frequency=update_frequency_hrs,
            force=force,
            prompt_if_existing=True,
        )

        if do_update:
            logger.info("Updating FreeToken...")
            try:
                res = subprocess.run(["git", "pull", "--autostash"], cwd=str(freetoken_dir), timeout=300, check=True, capture_output=True, text=True)
                sys.stdout.write(res.stdout)
                if res.stderr: sys.stderr.write(res.stderr)
                _patch_freetoken_windows_hotfixes(freetoken_dir)
                if "Already up to date." not in res.stdout:
                    subprocess.run(["uv", "pip", "install", "-e", ".", "--no-build-isolation", "--no-deps", "--python", str(venv_python)], timeout=300, cwd=str(freetoken_dir), env=build_env, check=True)
                record_successful_update("engine:freetoken")
                return True
            except Exception as e:
                logger.error(f"  -> Failed to update FreeToken: {e}")
                return False
        return True
    else:
        logger.info("  -> FreeToken is not installed.")
        install_behavior = config.freetoken_install_behavior if config.freetoken_install_behavior != 'existing' else 'existing'
        if install_behavior == 'never_install':
            logger.info("  -> Skipping FreeToken installation as per configuration.")
            return False
        
        do_install = install_behavior == 'always_install'
        if not do_install:
            do_install = prompt_choice("Do you want to install FreeToken? (y/n): ", default='n')
            
        if do_install:
            logger.info("Installing FreeToken...")
            try:
                subprocess.run(["git", "clone", "https://github.com/UnsignedChad/freetoken-windows.git", "freetoken"], cwd=str(AI_WORKSPACE_DIR), check=True)
                _patch_freetoken_windows_hotfixes(freetoken_dir)
                subprocess.run(["uv", "venv", "--python", config.python_version, "venv"], cwd=str(freetoken_dir), check=True)
                
                subprocess.run(["uv", "pip", "install", "torch>=2.11,<2.12", "--index-url", "https://download.pytorch.org/whl/cu130", "--python", str(venv_python)], cwd=str(freetoken_dir), check=True)
                subprocess.run(["uv", "pip", "install", "triton-windows", "tornado", "--python", str(venv_python)], cwd=str(freetoken_dir), check=True)
                subprocess.run(["uv", "pip", "install", "nvidia-cuda-nvcc==13.0.88", "nvidia-nvvm<13.1", "nvidia-cuda-crt<13.1", "--python", str(venv_python)], cwd=str(freetoken_dir), check=True)
                subprocess.run(["uv", "pip", "install", "-e", ".", "--no-build-isolation", "--python", str(venv_python)], timeout=600, cwd=str(freetoken_dir), env=build_env, check=True)
                subprocess.run([str(venv_python), "scripts/patch_deps_windows.py"], cwd=str(freetoken_dir), check=True)
                
                logger.info(f"  -> FreeToken installed successfully in {freetoken_dir}.")
                return True
            except Exception as e:
                logger.error(f"  -> Failed to install FreeToken: {e}")
                return False
        else:
            return False



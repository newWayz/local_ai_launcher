# UV Cache Prune, Trampoline Maintenance & Self-Healing Guide

## 1. Overview of Trampoline Errors

On Windows, `uv` installs lightweight launcher stubs called **PE trampolines** (~45–109 KB `.exe` files) rather than standalone Python executables or console batch scripts. These trampolines embed hardcoded target paths to virtual environment interpreters and script entrypoints.

Two distinct trampoline failure modes can occur in the workspace:

---

### Failure Mode 1: Python Interpreter Missing (`os error 2`)

```text
error: uv trampoline failed to spawn Python child process
  Caused by: entity not found (os error 2)
```

#### Root Cause
1. **`uv-cache` contains both disposable caches AND active Python runtimes:**
   In `local_ai_launcher/environment.py`, the workspace configures:
   - `UV_CACHE_DIR = C:\aiWorkspace\uv-cache`
   - `UV_PYTHON_INSTALL_DIR = C:\aiWorkspace\uv-cache\python`

2. **The "uv trampoline" interpreter lookup:**
   Virtual environment launchers (`.venv\Scripts\python.exe`) read `.venv\pyvenv.cfg` for the `home` directory:
   ```ini
   home = C:\aiWorkspace\uv-cache\python\cpython-3.11-windows-x86_64-none
   ```
   The trampoline attempts to spawn `<home>\python.exe` as a child process.

3. **Accidental deletion of directory junctions during wheel pruning:**
   `uv` installs Python into full version directories (e.g. `cpython-3.11.15-windows-x86_64-none`) and creates a directory junction with the major.minor name (`cpython-3.11-windows-x86_64-none`) pointing to it.
   During manual disk cleanup of large `pytorch+cu1xx` wheels inside `uv-cache`, the directory junction `cpython-3.11-windows-x86_64-none` (or the entire `python\` folder) was deleted. Because the junction was missing, the trampoline failed with `os error 2` (`ERROR_FILE_NOT_FOUND`).

---

### Failure Mode 2: Script Canonicalization Failure (`failed to canonicalize script path`)

```text
error: uv trampoline failed to canonicalize script path
```

#### Root Cause
1. **Console Script Trampolines in `bin/` directories:**
   When tools or coding agents (such as Nous Research's **Hermes Agent**, `browser-use`, or CLI utilities) are installed via `uv tool` or virtual environments, `uv` writes console script `.exe` trampolines into workspace `bin` directories (e.g., `C:\aiWorkspace\.hermes\bin\hermes.exe`). These binaries embed absolute paths to their internal Python scripts or active virtual environment generation.

2. **Environment Relocation or Re-creation:**
   When an agent updates, prunes, or rebuilds its internal virtual environments (such as Hermes' package manager generations under `.hermes\installs\...`), the target script path referenced in the `.exe` header no longer exists. The stub crashes immediately upon launch before running any Python code.

3. **Windows `PATHEXT` Precedence Collision:**
   Windows executes commands by searching `PATHEXT` (`.COM;.EXE;.BAT;.CMD`). Even if a fully functional companion script exists (e.g., `hermes.cmd` or a shell script), Windows unconditionally prioritizes the broken `hermes.exe`.

---

## 2. Autonomous Self-Healing Engine (Zero AI Involvement)

`local_ai_launcher` incorporates an autonomous, startup-level healing architecture in [`local_ai_launcher/environment.py`](../environment.py) and [`local_ai_launcher/harnesses/hermes.py`](../harnesses/hermes.py):

1. **Startup Binary Scanner (`heal_workspace_bin_trampolines`)**:
   - Runs automatically inside `_setup_workspace_path()` on every launcher boot.
   - **Fast Binary Pre-Filter (< 0.05ms per file)**: Reads the first 128 KB of candidate `.exe` files. If `b"uv trampoline"` or `b"canonicalize"` is not present in the header, the file is skipped immediately without spawning any subprocess.
   - **Deterministic Probe**: For verified `uv` trampolines, probes the binary with `--version` and `CREATE_NO_WINDOW`.
   - **Auto-Purge**: If the binary exits non-zero with `uv trampoline failed to canonicalize script path`, the launcher automatically unlinks the corrupt `.exe`.
   - **Clean Fallback**: Once the stale `.exe` is removed, Windows falls back seamlessly to the healthy companion script (`hermes.cmd`) or allows the harness adapter to bootstrap a fresh install.

2. **Hardened NTFS Junction Management (`ensure_directory_junction`)**:
   - Safely detects dangling reparse points via `os.path.lexists()` and `FILE_ATTRIBUTE_REPARSE_POINT` (where standard `Path.exists()` falsely returns `False`).
   - Removes dangling links using `os.rmdir()` without deleting target files, preventing `Cannot create a file when that file already exists` errors during link recreation.

3. **Hermes Harness Multi-Tier CLI Resolution**:
   - `HermesAdapter` checks CLI candidates in preference order (`hermes.exe` -> `hermes.cmd` -> `venv/Scripts/hermes.exe` -> `PATH`), ensuring only verified executables are invoked.
   - Auto-bootstrap uses the official Nous Research `scripts/install.ps1 -NonInteractive` installer and maintains an active NTFS junction to the internal package manager venv generation.

---

## 3. Safe Disk Cleanup Rules

When cleaning up `uv-cache` to reclaim disk space:

### Safe to Delete:
- `C:\aiWorkspace\uv-cache\archive-v0\` (downloaded zip/tar archives)
- `C:\aiWorkspace\uv-cache\wheels-v6\` (downloaded `.whl` files)
- `C:\aiWorkspace\uv-cache\built-wheels-v0\` (locally built wheels)

### NEVER Delete or Wipe:
- `C:\aiWorkspace\uv-cache\python\` (holds Python interpreters and major.minor junctions)

---

## 4. Manual Recovery Procedures (If Needed)

If manual repair is ever required:

### Restoring Python Interpreter Junctions
```powershell
# Option A: Re-link uv-cache\python to the user's persistent uv store
New-Item -ItemType Junction -Path "C:\aiWorkspace\uv-cache\python" -Target "C:\Users\user\AppData\Roaming\uv\python"

# Option B: Recreate a specific major.minor junction manually
New-Item -ItemType Junction -Path "C:\aiWorkspace\uv-cache\python\cpython-3.11-windows-x86_64-none" -Target "C:\aiWorkspace\uv-cache\python\cpython-3.11.15-windows-x86_64-none"
```

### Manually Running Trampoline Healing
```powershell
python -c "from local_ai_launcher.environment import heal_workspace_bin_trampolines; print(heal_workspace_bin_trampolines())"
```

### Verification
```powershell
# Verify Python venv trampoline:
C:\aiWorkspace\.venv\Scripts\python.exe --version

# Verify Hermes CLI:
C:\aiWorkspace\.hermes\bin\hermes.cmd --version
```

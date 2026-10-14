# Workspace Root

All file creates/edits/deletes must be under C:\aiWorkspace\.

Temp files: C:\aiWorkspace\tmp\
Never use C:\tmp or other outside paths.

# ai_launcher

Code related to `ai_launcher` should ALWAYS be placed within `C:\aiWorkspace\ai_launcher`. 
Aside from the external entry points (like `2_start_all_services.py`) at the workspace root, all files, tests, and configurations for `ai_launcher` itself must be contained inside the `C:\aiWorkspace\ai_launcher` directory. Do NOT create new `ai_launcher` related files in `C:\aiWorkspace` directly.

"""Bridges package for connecting external harnesses and IDE environments."""

def __getattr__(name):
    if name == "GVS5HSolver":
        from local_ai_launcher.bridges.gvs5h_solver import GVS5HSolver
        return GVS5HSolver
    if name == "run_gvs5h_solve":
        from local_ai_launcher.bridges.gvs5h_solver import run_gvs5h_solve
        return run_gvs5h_solve
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["GVS5HSolver", "run_gvs5h_solve"]

"""
Blackwell Llama Shim Dispatcher / Compatibility Facade.
Dispatches modularly to HuangLlamaServicer (default) or BrunoLlamaServicer
based on config.blackwell_engine.
"""
import sys
import logging
import atexit
from local_ai_launcher.config import config, logger
from local_ai_launcher.shims.base_shim import BaseBackendServicer, serve_shim
from local_ai_launcher.shims.bruno_llama_shim import BrunoLlamaServicer
from local_ai_launcher.shims.huang_llama_shim import HuangLlamaServicer
import local_ai_launcher.shims.bruno_llama_shim as bruno_mod
import local_ai_launcher.shims.huang_llama_shim as huang_mod
from local_ai_launcher.backends.blackwell_builder import resolve_blackwell_server_exe

_active_servicer = None


def _cleanup_llama_on_exit():
    servicer = _active_servicer or huang_mod._active_servicer or bruno_mod._active_servicer
    if servicer and getattr(servicer, "process", None) and servicer.process.poll() is None:
        try:
            from local_ai_launcher.process import kill_process_tree
            kill_process_tree(servicer.process)
        except Exception:
            pass


atexit.register(_cleanup_llama_on_exit)


class BlackwellLlamaServicer(BaseBackendServicer):
    """
    Dispatcher servicer returning HuangLlamaServicer or BrunoLlamaServicer
    instance based on active config.blackwell_engine.
    """
    def __new__(cls, *args, **kwargs):
        engine = getattr(config, "blackwell_engine", "huang").lower().strip()
        if "bruno" in engine:
            inst = BrunoLlamaServicer(*args, **kwargs)
        else:
            inst = HuangLlamaServicer(*args, **kwargs)
        global _active_servicer
        _active_servicer = inst
        return inst


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Blackwell llama-server Modular gRPC Shim")
    parser.add_argument("--addr", default=f"127.0.0.1:{config.blackwell_shim_port}", help="gRPC address to bind")
    parser.add_argument("--port", type=int, help="Alternative port flag.")
    args, _ = parser.parse_known_args()

    port = args.port or int(args.addr.split(":")[-1])
    engine = getattr(config, "blackwell_engine", "huang").lower().strip()
    active_cls = BrunoLlamaServicer if "bruno" in engine else HuangLlamaServicer
    logging.info(f"Starting Blackwell ({engine}) llama-server gRPC shim on port {port}...")
    serve_shim(active_cls(), default_addr=args.addr, port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

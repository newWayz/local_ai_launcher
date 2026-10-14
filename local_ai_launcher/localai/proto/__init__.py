import sys
from pathlib import Path

# Add this directory to sys.path so backend_pb2_grpc can find backend_pb2
_proto_dir = str(Path(__file__).resolve().parent)
if _proto_dir not in sys.path:
    sys.path.insert(0, _proto_dir)

from . import backend_pb2
from . import backend_pb2_grpc

__all__ = ["backend_pb2", "backend_pb2_grpc"]

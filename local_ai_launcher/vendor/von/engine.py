"""Von Engine orchestrator.

Von supports its own decision backends only. The third-party encoders reachable
here exist purely so the benchmark suite can score Von against them on identical
inputs -- they are comparison baselines, not supported backends.
"""

import os
import threading
from typing import Any, Dict, List, Optional, Union

from .backends import (
    BaseBackend,
)
from .types import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    SystemOneResponse,
)


# Von ships exactly one model. Releases are identified by version number only --
# never by architecture name -- so callers never have to know or care how the
# current model is built. A model that changes enough to break parity gets the
# next version number.
VON_VERSION = "1.3"

# Aliases that resolve to the current model.
# "von-1.1" stays accepted: 1.2 is the same model family retrained, and callers
# that pinned the previous version should keep working rather than break on upgrade.
VON_CURRENT_ALIASES = ("von-1.3", "1.3", "von-1.2", "1.2", "von-1.1", "1.1", "von", "default", "latest", "von-latest")
SUPPORTED_BACKENDS = frozenset(VON_CURRENT_ALIASES)

# Von ships exactly one model. Superseded releases and third-party encoders used
# to be selectable here as benchmark baselines; they were removed because a name
# that loads is a name someone ships to production. Historical comparisons live
# in the benchmark results, not in the runtime.


class VonEngine:
    """System One inference engine orchestrator."""

    _instance: Optional["VonEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, backend_name: str = "von-1.3", device: Optional[str] = None):
        self.backend_name = backend_name.lower().strip()
        self.device = device or os.environ.get("VON_DEVICE")
        if self.backend_name in VON_CURRENT_ALIASES:
            from .backends.option_marker_backend import OptionMarkerBackend
            self.backend: BaseBackend = OptionMarkerBackend(device=self.device)
        else:
            raise ValueError(
                f"Unknown model '{self.backend_name}'. "
                f"Von {VON_VERSION} is the only model; accepted aliases: "
                f"{', '.join(sorted(VON_CURRENT_ALIASES))}."
            )

    @classmethod
    def get_instance(cls, backend: Optional[str] = None, device: Optional[str] = None) -> "VonEngine":
        with cls._lock:
            if cls._instance is None:
                b = backend or os.environ.get("VON_BACKEND", f"von-{VON_VERSION}")
                d = device or os.environ.get("VON_DEVICE")
                cls._instance = cls(backend_name=b, device=d)
            return cls._instance

    @classmethod
    def set_backend(cls, backend: str, device: Optional[str] = None):
        """Switch active engine backend."""
        with cls._lock:
            d = device or os.environ.get("VON_DEVICE")
            cls._instance = cls(backend_name=backend, device=d)

    def evaluate_choice(self, *args, **kwargs) -> ChoiceAnswer:
        return self.backend.evaluate_choice(*args, **kwargs)

    def evaluate_score(self, *args, **kwargs) -> ScoreAnswer:
        return self.backend.evaluate_score(*args, **kwargs)

    def evaluate_noul(self, *args, **kwargs) -> NoulAnswer:
        return self.backend.evaluate_noul(*args, **kwargs)

    def evaluate(
        self,
        state: Any,
        questions: Dict[str, Union[Question, Dict[str, Any]]],
        model: Optional[str] = None,
    ) -> SystemOneResponse:
        # Von ships exactly one model, so the response is always stamped with the
        # version actually served. Echoing the caller's requested id back would
        # let a stale client (JS SDK 1.0.1 still asks for "von-1.0.0") receive a
        # response labelled as a model that no longer exists, served by a
        # different one. Old ids are accepted, never reflected.
        resolved_model = f"von-{VON_VERSION}.0"
        return self.backend.evaluate(state=state, questions=questions, model=resolved_model)

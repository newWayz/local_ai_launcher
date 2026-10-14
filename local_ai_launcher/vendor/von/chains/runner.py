"""Chain-of-options runtime.

A chain is a TOML file: a route description, typed slots, deterministic
steps over the operator library, and an answer rule. At inference the
controller makes only typed Choice decisions with the same backend that
answers ordinary questions:

  level 0  gate     -- cheap: the state has >= 2 numerals, else skip
  level 1  route    -- Choice over chain descriptions (+ "none")
  level 2  bind     -- one Choice per slot over proposed spans of its kind
  execute           -- run the steps; any exception => fall back to plain
  answer            -- describe the computed result in one sentence and ask
                       the ORIGINAL question about that sentence (Choice /
                       Noul / Score), or match a formatted value to an option

Zero generated tokens. Regex proposes; the model chooses; code computes.
"""

from __future__ import annotations

import glob
import os
from collections import OrderedDict
import re
import threading
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..types import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer
from . import ops as _ops
from .spans import Span, by_kind, propose

NUMERAL_GATE = re.compile(r"\d")


@dataclass
class Slot:
    name: str
    kind: str
    ask: str
    optional: bool = False
    prefer: str = ""  # "with_time" | "with_zone" | "" (ranking hint only)


@dataclass
class Step:
    op: str
    args: List[Any]
    out: str


@dataclass
class Chain:
    name: str
    description: str
    slots: List[Slot]
    steps: List[Step]
    answer: Dict[str, Any]
    path: str = ""
    requires: List[str] = field(default_factory=list)


def load_chain(path: str) -> Chain:
    with open(path, "rb") as f:
        d = tomllib.load(f)
    slots = [Slot(**s) for s in d.get("slots", [])]
    steps = [Step(op=s["op"], args=list(s.get("args", [])), out=s["out"]) for s in d.get("steps", [])]
    for s in steps:
        if s.op not in _ops.OPS:
            raise ValueError(f"{path}: unknown op '{s.op}'")
    return Chain(name=d["name"], description=d["description"], slots=slots, steps=steps,
                 answer=d.get("answer", {}), path=path, requires=list(d.get("requires", [])))


def load_chains(directory: str) -> List[Chain]:
    return [load_chain(p) for p in sorted(glob.glob(os.path.join(directory, "*.toml")))]


@dataclass
class Trace:
    chain: Optional[str] = None
    route_probs: Dict[str, float] = field(default_factory=dict)
    bindings: Dict[str, str] = field(default_factory=dict)
    values: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    fallback: Optional[str] = None


def _fmt(v: Any) -> str:
    if isinstance(v, Span):
        if isinstance(v.value, datetime):
            return _fmt(_ops._dt(v))
        if isinstance(v.value, dict) and "unit" in v.value:
            n = v.value["n"]
            return f"{n:g} {v.value['unit'].replace('_', ' ')}"
        return v.text
    if v is None:
        return "unspecified"
    if isinstance(v, datetime):
        s = v.strftime("%A %d %B %Y %H:%M")
        if v.tzinfo is not None:
            s += " " + (v.tzname() or v.strftime("%z"))
        return s
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) >= 0.01 or v == 0 else f"{v:.4g}"
    if isinstance(v, list):
        return ", ".join(f"{k}: {_fmt(x)}" for k, x in v[:12])
    return str(v)


class ChainRunner:
    """Drives chains using a backend's plain Choice/Noul/Score primitives."""

    def __init__(self, backend, chains_dir: str, min_numerals: int = 2, mode: Optional[str] = None):
        self.backend = backend
        self.mode = (mode or os.environ.get("VON_CHAINS_MODE", "route")).lower()
        self.max_rounds = int(os.environ.get("VON_CHAINS_ROUNDS", "3"))
        self.max_facts = int(os.environ.get("VON_CHAINS_MAX_FACTS", "12"))
        self.max_calls = int(os.environ.get("VON_CHAINS_MAX_CALLS", "16"))  # model sub-decisions per item
        self.chains = load_chains(chains_dir)
        self.min_numerals = min_numerals
        self._local = threading.local()
        # Binding and execution read only the state, never the question, so a
        # multi-question request (17 clauses of one contract) computes its facts
        # once. Small LRU keyed by state text; entries are immutable tuples.
        self._facts_cache: "OrderedDict[str, tuple]" = OrderedDict()
        self._facts_cache_size = int(os.environ.get("VON_CHAINS_CACHE", "64"))
        self._cache_lock = threading.Lock()

    # --- recursion guard: sub-decisions must never re-enter the chain path --
    def active(self) -> bool:
        return bool(getattr(self._local, "busy", False))

    class _Budget(Exception):
        pass

    def _choice(self, state: str, instructions: str, criteria: Dict[str, str]) -> ChoiceAnswer:
        used = getattr(self._local, "calls", 0)
        if used >= self.max_calls:
            raise ChainRunner._Budget()
        self._local.calls = used + 1
        self._local.busy = True
        try:
            return self.backend.evaluate_choice("chain", state, Choice(type="choice", instructions=instructions, criteria=criteria))
        finally:
            self._local.busy = False

    def _ask_original(self, description: str, q: Any):
        self._local.busy = True
        try:
            if isinstance(q, Choice):
                return self.backend.evaluate_choice("chain", description, q)
            if isinstance(q, Noul):
                return self.backend.evaluate_noul("chain", description, q)
            return self.backend.evaluate_score("chain", description, q)
        finally:
            self._local.busy = False


    # ------------------------------------------------------------ pieces --
    def _bind(self, chain: Chain, state: str, spans: List[Span], tr: Trace,
              fixed: Optional[Dict[str, Span]] = None) -> Optional[Dict[str, Any]]:
        """Fill every slot; a Choice is only spent when a slot is ambiguous.
        `fixed` pins slots ahead of time (used to force derived values in)."""
        env: Dict[str, Any] = {"state": state}
        bound: set = set()
        for name, sp in (fixed or {}).items():
            env[name] = sp
            bound.add(sp.key())
            tr.bindings[name] = sp.text
        # cheap pre-check: any required slot with zero candidates kills the chain before a model call
        for slot in chain.slots:
            if not slot.optional and not by_kind(spans, slot.kind):
                tr.fallback = f"bind:{slot.name}:no-candidates"
                return None
        for slot in chain.slots:
            if fixed and slot.name in fixed:
                continue
            cands = [s for s in by_kind(spans, slot.kind) if s.key() not in bound]
            if slot.prefer == "with_time":
                cands = sorted(cands, key=lambda s: not s.meta.get("has_time", False))
            if slot.prefer == "with_zone":
                cands = sorted(cands, key=lambda s: s.meta.get("zone") is None)
            if slot.kind in ("number", "amount"):
                cands = sorted(cands, key=lambda s: not s.meta.get("rich", False))
            cands = cands[:12]
            if not cands:
                if slot.optional:
                    env[slot.name] = None
                    continue
                tr.fallback = f"bind:{slot.name}:no-candidates"
                return None
            if len(cands) == 1:
                chosen = cands[0]
            else:
                c = {f"s{i}": s.text for i, s in enumerate(cands)}
                ans = self._choice(state, slot.ask, c)
                chosen = cands[int(ans.choice[1:])] if ans.choice in c else cands[0]
            env[slot.name] = chosen
            bound.add(chosen.key())
            tr.bindings[slot.name] = chosen.text
        return env

    def _exec(self, chain: Chain, env: Dict[str, Any], tr: Trace) -> bool:
        try:
            for st in chain.steps:
                args = [env[a] if isinstance(a, str) and a in env else a for a in st.args]
                env[st.out] = _ops.OPS[st.op](*args)
                tr.values[st.out] = env[st.out] if not isinstance(env[st.out], list) else env[st.out][:12]
                if isinstance(env[st.out], list) and not env[st.out]:
                    raise ValueError(f"{st.op} produced an empty series")
        except Exception as e:  # noqa: BLE001 - any executor failure means "not our item"
            tr.fallback = f"exec:{type(e).__name__}:{e}"
            return False
        return True

    def _describe(self, chain: Chain, env: Dict[str, Any]) -> Optional[str]:
        try:
            return chain.answer.get("describe", "").format(**{k: _fmt(v) if k != "state" else "" for k, v in env.items()},
                                                         **{f"{k}_raw": v for k, v in env.items() if k != "state"})
        except (KeyError, ValueError, IndexError):
            return None

    def _facts_for(self, state: str) -> tuple:
        """State-only phase of bindall: propose, bind, execute, compose.
        Returns (facts, execs, trace_fragment); cached per state text."""
        with self._cache_lock:
            hit = self._facts_cache.get(state)
            if hit is not None:
                self._facts_cache.move_to_end(state)
                return hit
        facts: List[Tuple[str, str]] = []       # (chain, provenance-carrying description)
        execs: List[Tuple[str, Any, str]] = []  # (chain, computed value, description) for value chains
        frag: Dict[str, Dict[str, Any]] = {"route_probs": {}, "bindings": {}, "values": {}}
        seen: set = set()                        # (chain, bindings) already executed
        base_spans = propose(state)
        derived: List[Span] = []
        self._local.calls = 0
        for rnd in range(self.max_rounds):
            spans = base_spans + derived
            new_facts = 0
            if rnd == 0:
                jobs = [(chain, None) for chain in self.chains]
            else:
                # composition is forced, not hoped for: every derived value
                # gets pinned into every datetime slot it could fill
                fresh = [d for d in derived if d.meta.get("round") == rnd - 1]
                jobs = [(chain, {slot.name: d}) for chain in self.chains for slot in chain.slots
                        if slot.kind == "datetime" for d in fresh]
            exhausted = False
            for chain, fixed in jobs:
                sub = Trace()
                try:
                    env = self._bind(chain, state, spans, sub, fixed=fixed)
                except ChainRunner._Budget:
                    exhausted = True
                    break
                if env is None or not _sane_bindings(chain, env):
                    continue
                sig = (chain.name, tuple(sorted(sub.bindings.items())))
                if sig in seen:
                    continue
                seen.add(sig)
                if not self._exec(chain, env, sub):
                    continue
                for d in _derived_spans(chain, env, spans + derived):
                    d.meta["round"] = rnd
                    derived.append(d)
                desc = self._describe(chain, env)
                if desc is None:
                    continue
                prov = f"[{chain.name}: " + ", ".join(f"{k}={v}" for k, v in sub.bindings.items()) + "] "
                facts.append((chain.name, prov + desc))
                new_facts += 1
                frag["route_probs"][chain.name] = frag["route_probs"].get(chain.name, 0.0) + 1.0
                frag["bindings"].update({f"r{rnd}.{chain.name}.{k}": v for k, v in sub.bindings.items()})
                frag["values"][f"r{rnd}.{chain.name}"] = {k: v for k, v in env.items() if k != "state" and not isinstance(v, Span)}
                if chain.answer.get("mode") == "value":
                    execs.append((chain.name, env.get(chain.answer.get("value", "")), prov + desc))
                if len(facts) >= self.max_facts:
                    break
            if exhausted or new_facts == 0 or len(facts) >= self.max_facts:
                break
        out = (tuple(facts), tuple(execs), frag)
        with self._cache_lock:
            self._facts_cache[state] = out
            while len(self._facts_cache) > self._facts_cache_size:
                self._facts_cache.popitem(last=False)
        return out

    def _run_bindall(self, state: str, q: Any, tr: Trace) -> Tuple[Optional[Any], Trace]:
        """No routing. Every chain whose slots bind is executed; computed
        facts are appended to the state and become spans for the next round
        (fixpoint, bounded), so chains compose without composed definitions.
        A chain whose value lands on an option answers through the matcher;
        the model arbitrates only among grounded survivors, or reads the
        state plus facts when nothing matched. Nothing reads the question."""
        instructions = getattr(q, "instructions", "") or ""
        facts, execs, frag = self._facts_for(state)
        tr.route_probs.update(frag["route_probs"])
        tr.bindings.update(frag["bindings"])
        tr.values.update(frag["values"])
        matched: Dict[str, Tuple[str, str]] = {}  # option key -> (chain, description)
        if isinstance(q, Choice):
            for name, val, desc in execs:
                hit = _match_any(val, q.criteria)
                if hit and hit not in matched:
                    matched[hit] = (name, desc)
        if not facts:
            tr.fallback = "bindall:nothing-executed"
            return None, tr
        if isinstance(q, Choice) and len(matched) == 1:
            (hit, (name, desc)), = matched.items()
            tr.chain, tr.description = name, desc
            probs = {k: (0.9 if k == hit else 0.1 / max(1, len(q.criteria) - 1)) for k in q.criteria}
            return ChoiceAnswer(choice=hit, probabilities=probs, confidence=0.8), tr
        if isinstance(q, Choice) and len(matched) > 1:
            crit = {k: f"{q.criteria[k]} (computed: {d})" for k, (_, d) in matched.items()}
            self._local.calls = 0
            ans = self._choice(state, instructions, crit)
            pick = ans.choice if ans.choice in matched else next(iter(matched))
            tr.chain, tr.description = f"arbitrate:{matched[pick][0]}", matched[pick][1]
            probs = {k: ans.probabilities.get(k, 0.0) for k in q.criteria}
            return ChoiceAnswer(choice=pick, probabilities=probs, confidence=ans.confidence), tr
        # nothing landed on an option: the model reads the state plus every computed fact
        tr.chain = "ask:" + "+".join(sorted({n for n, _ in facts}))
        tr.description = self._facts_block(facts)
        return self._ask_original(state + "\n\n" + tr.description, q), tr

    @staticmethod
    def _facts_block(facts: List[Tuple[str, str]]) -> str:
        return "Computed facts (deterministic arithmetic over the dates and amounts above):\n" + "\n".join(d for _, d in facts)

    # --------------------------------------------------------------- run --
    def applicable(self, state: str) -> bool:
        if not self.chains or len(NUMERAL_GATE.findall(state)) < self.min_numerals:
            return False
        if self.mode != "bindall":
            return True
        # bindall fires on computable structure only: a lone date yields
        # weekday/leap-year trivia that measurably hurts unrelated questions
        sp = propose(state)
        n_dt = len(by_kind(sp, "datetime"))
        n_dur = len(by_kind(sp, "duration"))
        n_amt = len([x for x in by_kind(sp, "number") if x.meta.get("rich")])
        return n_dt >= 2 or (n_dt >= 1 and n_dur >= 1) or n_amt >= 2

    def run(self, state: str, q: Any) -> Tuple[Optional[Any], Trace]:
        tr = Trace()
        if not self.applicable(state):
            tr.fallback = "gate"
            return None, tr
        if self.mode == "bindall":
            return self._run_bindall(state, q, tr)
        instructions = getattr(q, "instructions", "") or ""

        # level 1: route
        crit = {c.name: c.description for c in self.chains}
        crit["none"] = "None of these computations answers the question; answer it directly from the text."
        route = self._choice(state, f"Which computation, if any, is needed to answer this question: {instructions}", crit)
        tr.route_probs = dict(route.probabilities)
        if route.choice == "none" or route.choice not in crit:
            tr.fallback = "route:none"
            return None, tr
        chain = next(c for c in self.chains if c.name == route.choice)
        tr.chain = chain.name

        spans = propose(state)
        env = self._bind(chain, state, spans, tr)
        if env is None:
            return None, tr
        if not self._exec(chain, env, tr):
            return None, tr
        # answer
        ans_cfg = chain.answer
        desc = self._describe(chain, env)
        if desc is None:
            tr.fallback = "describe:format"
            return None, tr
        tr.description = desc
        mode = ans_cfg.get("mode", "ask")

        if mode == "value" and isinstance(q, Choice):
            key = ans_cfg.get("value", "")
            val = env.get(key)
            if isinstance(val, (int, float)):
                hit = _match_value(val, q.criteria)
                if hit:
                    probs = {k: (0.9 if k == hit else 0.1 / max(1, len(q.criteria) - 1)) for k in q.criteria}
                    return ChoiceAnswer(choice=hit, probabilities=probs, confidence=0.8), tr
            # no exact match => fall through to asking
        prompt = f"{desc}\n\nOriginal question context: {instructions}" if instructions else desc
        return self._ask_original(prompt, q), tr


def _match_value(val: float, criteria: Dict[str, str]) -> Optional[str]:
    """Option whose text carries the computed value as a standalone number.

    Strict on purpose: the loose form (substring of '%.0f') matched every
    option containing a '0'. A match needs a token-bounded number equal to
    the value, a positive value, and exactly one option carrying it."""
    if val != val or val in (float("inf"), float("-inf")) or val <= 0:
        return None
    hits = []
    for k, d in criteria.items():
        hay = f"{k} {d}".replace("_", " ").replace(",", "")
        for tok in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.]|\.\d)", hay):
            try:
                x = float(tok)
            except ValueError:
                continue
            if abs(x - val) <= max(0.005, abs(val) * 1e-4) or (abs(x - round(val, 2)) <= 0.005):
                hits.append(k)
                break
    return hits[0] if len(hits) == 1 else None


def _sane_bindings(chain: Chain, env: Dict[str, Any]) -> bool:
    """Reject degenerate argument sets before spending an executor on them:
    two datetime slots bound to the same instant, or a zero duration."""
    dts = []
    for slot in chain.slots:
        v = env.get(slot.name)
        if v is None:
            continue
        if slot.kind == "datetime":
            try:
                dt = _ops._dt(v).replace(tzinfo=None)
            except TypeError:
                return False
            dts.append(dt)
        if slot.kind == "duration":
            try:
                if float(_ops._dur(v).get("n", 0)) == 0:
                    return False
            except (TypeError, ValueError):
                return False
    return len(dts) == len(set(dts))


def _derived_spans(chain: Chain, env: Dict[str, Any], existing: List[Span]) -> List[Span]:
    """Computed step outputs become typed spans for the next round. Only the
    value is carried, never prose, so nothing is re-parsed out of a sentence."""
    have = {s.key() for s in existing}
    out: List[Span] = []
    for st in chain.steps:
        v = env.get(st.out)
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, datetime):
            sp = Span(kind="datetime", text=_fmt(v), value=v.replace(tzinfo=None), start=-1, end=-1,
                      meta={"derived": chain.name, "has_time": bool(v.hour or v.minute),
                            "zone": ({"iana": str(v.tzinfo.key)} if getattr(v.tzinfo, "key", None) else None)})
        else:
            # numbers do not carry across rounds: a derived amount fed back
            # into an amount slot manufactures decoys (prorate a day count...)
            continue
        if sp.key() not in have:
            have.add(sp.key())
            out.append(sp)
    return out


def _match_any(val: Any, criteria: Dict[str, str]) -> Optional[str]:
    """Option whose text contains the computed value in any common rendering."""
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, float)):
        return _match_value(float(val), criteria)
    if isinstance(val, datetime):
        forms = {val.strftime(f) for f in ("%Y-%m-%d", "%d %B %Y", "%B %d, %Y", "%b %d", "%d %b", "%B %d", "%d %B")}
        if val.hour or val.minute:
            forms |= {val.strftime(f) for f in ("%H:%M", "%I:%M %p", "%-I:%M %p", "%-I %p")}
        # every form must be a date form; time forms only count together with the date
        date_forms = {f for f in forms if not any(ch in f for ch in (":", "AM", "PM"))}
        time_forms = forms - date_forms
        for k, d in criteria.items():
            hay = f"{k} {d}".replace("_", " ")
            if any(f in hay for f in date_forms) and (not time_forms or any(f in hay for f in time_forms)):
                return k
        return None
    sval = str(val).strip()
    if not sval:
        return None
    for k, d in criteria.items():
        if sval.lower() in f"{k} {d}".replace("_", " ").lower():
            return k
    return None

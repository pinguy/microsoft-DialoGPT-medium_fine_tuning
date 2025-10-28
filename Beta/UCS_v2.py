# Enhanced UCS (Unified Cognition System) v2 - Production Ready
# Major improvements:
# - Battle-tested HNSW index via hnswlib for real ANN performance
# - Smart index rebuild batching
# - Bloom filter for fast membership testing
# - Query result caching with TTL
# - Enhanced telemetry and observability
# - Improved calibration with isotonic regression
# - Better error boundaries and circuit breakers
#
# Usage:
# - Install dependencies: pip3 install numpy hnswlib sentence-transformers
# - Smoke:    python UCS_v2.py --mode smoke
# - API:      RUN_API=1 UCS_REQUIRE_AUTH=0 python UCS_v2.py
# - API auth: RUN_API=1 UCS_API_KEY="secret" python UCS_v2.py
# -----------------------------------------------------------------------------
from __future__ import annotations
import os
import re
import sys
import json
import math
import time
import uuid
import random
import logging
import hashlib
import threading
import queue
import shutil
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque, defaultdict, namedtuple
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from functools import lru_cache

# Soft imports for optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    class DummyNumpy:
        def zeros(self, *a, **k): return []
        def random(self, *a, **k): return []
        def normal(self, *a, **k): return []
        def dot(self, *a, **k): return 0
        def exp(self, *a, **k): return 0
        def sum(self, *a, **k): return 0
        def log(self, *a, **k): return 0
        def linalg(self): return None
        def argpartition(self, *a, **k): return []
        def argsort(self, *a, **k): return []
        def concatenate(self, *a, **k): return []
        def arange(self, *a, **k): return []
        def intersect1d(self, *a, **k): return []
        def std(self, *a, **k): return 0
        def array(self, *a, **k): return []
        def mean(self, *a, **k): return 0
        def sqrt(self, *a, **k): return 0
        def where(self, *a, **k): return []
    np = DummyNumpy()

try:
    import hnswlib
    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False

try:
    from fastapi import FastAPI, HTTPException, Request, Response, Depends, Header
    from pydantic import BaseModel, Field
    from fastapi.middleware.cors import CORSMiddleware
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    import uvicorn
except Exception:
    uvicorn = None

try:
    import gzip
    HAS_GZIP = True
except ImportError:
    HAS_GZIP = False

# Constants and types
DIM = 128
MAX_PROMPT_LEN = int(os.getenv("UCS_MAX_PROMPT_LEN", "20000"))
MAX_INGEST_ITEMS = int(os.getenv("UCS_MAX_INGEST_ITEMS", "2000"))
LOG_LEVEL = os.getenv("UCS_LOG_LEVEL", "INFO").upper()
API_KEY = os.getenv("UCS_API_KEY")
REQUIRE_AUTH = os.getenv("UCS_REQUIRE_AUTH", "1") == "1"
TRUST_XFF = os.getenv("UCS_TRUST_XFF", "0") == "1"
UCS_LOG_PROPOSAL_CONTENT = os.getenv("UCS_LOG_PROPOSAL_CONTENT", "0") == "1"
UCS_REDACT_REGEX = os.getenv("UCS_REDACT_REGEX")

logging.basicConfig(level=LOG_LEVEL, format='[%(levelname)s] %(message)s')
_logger = logging.getLogger(__name__)

# --- Helper Functions ---

def timed(fn, *args, **kwargs):
    """Decorator-like function to time another function."""
    start = time.time()
    result = fn(*args, **kwargs)
    end = time.time()
    return result, end - start

def normalize_vectors(X: np.ndarray) -> np.ndarray:
    """L2-normalize a batch of vectors safely."""
    if not HAS_NUMPY:
        return X
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return X / norms

# --- Simple Bloom Filter for Fast Membership Testing ---

class BloomFilter:
    """Space-efficient probabilistic membership tester."""
    def __init__(self, size: int = 10000, num_hashes: int = 3):
        self.size = size
        self.num_hashes = num_hashes
        self.bits = [False] * size
    
    def add(self, item: str):
        for i in range(self.num_hashes):
            h = int(hashlib.sha256(f"{item}{i}".encode()).hexdigest(), 16)
            self.bits[h % self.size] = True
    
    def __contains__(self, item: str) -> bool:
        for i in range(self.num_hashes):
            h = int(hashlib.sha256(f"{item}{i}".encode()).hexdigest(), 16)
            if not self.bits[h % self.size]:
                return False
        return True

# --- Query Result Cache with TTL ---

@dataclass
class CachedResult:
    result: Any
    timestamp: float
    ttl: float = 300.0  # 5 minutes default
    
    def is_valid(self) -> bool:
        return time.time() - self.timestamp < self.ttl

class QueryCache:
    """Thread-safe LRU cache with TTL for query results."""
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, CachedResult] = {}
        self.max_size = max_size
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                cached = self.cache[key]
                if cached.is_valid():
                    self.hits += 1
                    return cached.result
                else:
                    del self.cache[key]
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: float = 300.0):
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
                del self.cache[oldest_key]
            self.cache[key] = CachedResult(result=value, timestamp=time.time(), ttl=ttl)
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                "size": len(self.cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate
            }

# --- Circuit Breaker for Fault Tolerance ---

class CircuitBreaker:
    """Simple circuit breaker to prevent cascading failures."""
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
        self.lock = threading.RLock()
    
    def call(self, fn: Callable, *args, **kwargs) -> Tuple[bool, Any]:
        """Returns (success, result)."""
        with self.lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "half_open"
                    _logger.info("Circuit breaker entering half-open state")
                else:
                    return False, None
            
            try:
                result = fn(*args, **kwargs)
                if self.state == "half_open":
                    self.state = "closed"
                    self.failures = 0
                    _logger.info("Circuit breaker closed")
                return True, result
            except Exception as e:
                self.failures += 1
                self.last_failure_time = time.time()
                
                if self.failures >= self.failure_threshold:
                    self.state = "open"
                    _logger.error(f"Circuit breaker opened after {self.failures} failures")
                
                _logger.error(f"Circuit breaker caught error: {e}")
                return False, None

# --- Expert proposal types ---
@dataclass
class ExpertProposal:
    """A formal proposal by an expert to perform an action."""
    action: str
    content: Any
    score: float = 0.5
    origin: str = ""
    trust_score: float = 0.0
    supporting_mementos: List[Tuple[str, float]] = field(default_factory=list)
    pre_calib_score: float = 0.0

# --- TrustFlow V2 Data Structures ---
ExpertReputation = namedtuple("ExpertReputation", ["n", "reward_sum", "reward_sq", "last_seen", "ema_reward"])

# --- Isotonic Regression for Better Calibration ---

class IsotonicCalibrator:
    """Simple isotonic regression for score calibration."""
    def __init__(self):
        self.thresholds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.calibrated_scores = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.observations = defaultdict(list)
    
    def update(self, raw_score: float, actual_reward: float):
        """Update calibration model with observed reward."""
        bucket = min(len(self.thresholds) - 1, int(raw_score * len(self.thresholds)))
        self.observations[bucket].append(actual_reward)
        
        # Recompute calibrated scores periodically
        if sum(len(obs) for obs in self.observations.values()) % 100 == 0:
            self._recompute()
    
    def _recompute(self):
        """Recompute isotonic calibration."""
        for i, obs_list in self.observations.items():
            if obs_list:
                self.calibrated_scores[i] = sum(obs_list) / len(obs_list)
        
        # Ensure monotonicity
        for i in range(1, len(self.calibrated_scores)):
            if self.calibrated_scores[i] < self.calibrated_scores[i-1]:
                self.calibrated_scores[i] = self.calibrated_scores[i-1]
    
    def calibrate(self, raw_score: float) -> float:
        """Calibrate a raw score."""
        raw_score = max(0.0, min(1.0, raw_score))
        
        for i in range(len(self.thresholds) - 1):
            if raw_score < self.thresholds[i + 1]:
                # Linear interpolation between calibrated points
                t = (raw_score - self.thresholds[i]) / (self.thresholds[i + 1] - self.thresholds[i])
                return self.calibrated_scores[i] + t * (self.calibrated_scores[i + 1] - self.calibrated_scores[i])
        
        return self.calibrated_scores[-1]

# --- Core Expert Management ---
class ExpertManager:
    """Manages experts and their proposals."""
    def __init__(self, parent_system, policy=None):
        self.experts = {}
        self.history = defaultdict(list)
        self.policy = policy or self.default_policy
        self.parent_system = parent_system

    def register_expert(self, name: str, handler: Callable, expertise_tags: Optional[List[str]] = None, phase: str = "propose"):
        """Registers a new expert with a name, handler function, tags, and a phase."""
        self.experts[name] = {"handler": handler, "tags": expertise_tags or [], "phase": phase}
        logging.info(f"Registered expert: {name} with tags: {expertise_tags} in phase: {phase}")

    def propose(self, ctx: Dict[str, Any]) -> List[ExpertProposal]:
        """Gathers proposals from experts across three phases: map, propose, filter."""
        # 1) map phase: allow experts to modify the context
        for n, e in self.experts.items():
            if e["phase"] != "map":
                continue
            try:
                ctx = e["handler"](ctx) or ctx
            except Exception as ex:
                logging.error(f"[map] {n} failed: {ex}")
        
        # 2) propose phase: experts generate proposals
        proposals = []
        for name, e in self.experts.items():
            if e["phase"] != "propose":
                continue
            try:
                proposal_content = e["handler"](ctx)
                if proposal_content is not None:
                    action = (proposal_content.get("operation") or proposal_content.get("action") or "PROPOSE").upper()
                    proposal = ExpertProposal(action=action, content=proposal_content, origin=name)
                    if action == "RETRIEVE" and "retrieval" in ctx:
                        proposal.supporting_mementos = ctx["retrieval"]
                    proposals.append(proposal)
            except Exception as e:
                logging.error(f"[propose] {name} failed: {e}")
                continue

        # 3) filter phase: experts veto or re-weight proposals
        for n, e in self.experts.items():
            if e["phase"] != "filter":
                continue
            try:
                proposals = e["handler"]({"ctx": ctx, "proposals": proposals, "parent_system": self.parent_system}) or proposals
            except Exception as ex:
                logging.error(f"[filter] {n} failed: {ex}")
        
        return proposals

    def score_proposals(self, proposals: List[ExpertProposal], ctx: Dict[str, Any]) -> List[ExpertProposal]:
        """Applies a multi-axis TrustFlow scoring to a list of proposals."""
        
        ctx_tags = set(ctx.get("ctx_tags", []))
        
        for p in proposals:
            # Axis 1: Policy check (hard gate)
            ok, penalty = self.policy(p, ctx)
            if not ok:
                p.trust_score = -1.0
                p.pre_calib_score = -1.0
                continue
            
            # Axis 2: Reputation (bandit-based)
            reputation_score = self.parent_system._reputation_score(p.origin)
            
            # Axis 3: Evidence
            evidence_score = 0.0
            expert_tags = set(self.experts.get(p.origin, {}).get("tags", []))
            tag_match_count = len(expert_tags.intersection(ctx_tags))
            evidence_score += 0.2 * (tag_match_count / len(expert_tags) if expert_tags else 0.0)
            
            if p.action == "RETRIEVE" and p.supporting_mementos:
                retrieved_mementos = p.supporting_mementos
                retrieval_density = len(retrieved_mementos) / 10
                evidence_score += 0.5 * min(1.0, retrieval_density)
                
                if self.parent_system.vmem:
                    retrieved_ids = {mid for mid, _ in retrieved_mementos}
                    memento_scores = [self.parent_system.vmem.scores.get(mid, {}).get("b", 0) for mid in retrieved_ids]
                    if memento_scores:
                        evidence_score += 0.3 * (sum(s for s in memento_scores if s > 0) / len(memento_scores))

            # Axis 4: Recency & Risk
            recency_bonus = 0.05 if datetime.now().timestamp() - self.parent_system._reputations[p.origin].last_seen < 60*60*24 else 0.0
            
            risk_penalty = 0.0
            if p.action == "SUMMARIZE" and self.parent_system._novelty_score(p.supporting_mementos) < 0.1:
                risk_penalty = 0.3
            
            # Axis 5: Uncertainty
            uncertainty_penalty = 0.0
            if p.supporting_mementos and HAS_NUMPY:
                scores = [m[1] for m in p.supporting_mementos]
                std_dev = float(np.std(scores)) if len(scores) > 1 else 0
                uncertainty_penalty = min(0.5, std_dev)

            # Sum and calibrate
            pre_calib_score = (
                reputation_score * 0.4 +
                evidence_score * 0.3 +
                (p.score or 0.5) * 0.2 +
                recency_bonus -
                risk_penalty -
                uncertainty_penalty
            )
            p.pre_calib_score = pre_calib_score
            p.trust_score = self.parent_system._calibrate(pre_calib_score)

        return sorted(proposals, key=lambda p: p.trust_score, reverse=True)

    @staticmethod
    def default_policy(proposal, history):
        """A basic RulesEnforcer policy that applies penalties based on content."""
        content = str(proposal.content).lower()
        bad = any(w in content for w in ("abusive", "hateful", "illegal"))
        return (not bad, 1.0 if bad else 0.0)

# --- Expert Handlers ---

def retrieval_expert(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    query = ctx.get("prompt", "")
    if "retrieve" in query.lower() or "find" in query.lower():
        return {"operation": "RETRIEVE", "query": query}
    return None

def summarization_expert(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if len(ctx.get("prompt", "")) > 100 or "summarize" in ctx.get("prompt", "").lower():
        return {"operation": "SUMMARIZE", "text": ctx.get("prompt")}
    return None

def rehearsal_expert(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "rehearse" in ctx.get("prompt", "").lower() or "re-state" in ctx.get("prompt", "").lower():
        if "retrieval" in ctx and ctx["retrieval"]:
            most_cited_memento_id = ctx["retrieval"][0][0]
            if ctx.get("parent_system").vmem.mementos[most_cited_memento_id].get("is_gold"):
                return None
            return {"operation": "REHEARSE", "memento_id": most_cited_memento_id}
    return None

def longform_expert(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    history = [op["operation"] for op in ctx.get("history", [])]
    if "RETRIEVE" in history and "SUMMARIZE" in history:
        return {"operation": "GENERATE_LONGFORM", "source_mementos": ctx.get("retrieval", [])}
    return None

def meta_expert(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return {**ctx, "mode": "System-2"} if len(ctx.get("prompt","")) > 100 else {**ctx, "mode":"System-1"}

def router_expert(payload: Dict[str, Any]) -> List[ExpertProposal]:
    ctx, proposals = payload["ctx"], payload["proposals"]
    plan = ctx.get("plan", [])
    if not plan:
        return proposals
    keep = [p for p in proposals if p.action in plan or p.action in ("SET_PLAN", "SET_MODE")]
    return keep or proposals

def set_plan_expert(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = ctx.get("prompt", "")
    wants_retrieval = "retrieve" in prompt.lower() or "find" in prompt.lower()
    wants_summary   = len(prompt) > 100 or "summarize" in prompt.lower()
    if wants_retrieval and wants_summary:
        return {"operation":"SET_PLAN", "plan":["RETRIEVE","SUMMARIZE"]}
    if wants_retrieval: return {"operation":"SET_PLAN", "plan":["RETRIEVE"]}
    if wants_summary:   return {"operation":"SET_PLAN", "plan":["SUMMARIZE"]}
    return None

def self_attention_expert(payload: Dict[str, Any]) -> List[ExpertProposal]:
    ctx, proposals, parent_system = payload["ctx"], payload["proposals"], payload["parent_system"]
    expert_manager = parent_system.expert_manager
    ctx_tags = set(ctx.get("ctx_tags", []))
    
    if not ctx_tags:
        return proposals
        
    for p in proposals:
        expert_tags = set(expert_manager.experts.get(p.origin, {}).get("tags", []))
        tag_overlap = len(expert_tags.intersection(ctx_tags))
        if tag_overlap > 0:
            p.score += (tag_overlap / len(ctx_tags)) * 0.1

    return proposals

# --- Enhanced Memory and State Management ---
class VectorMemory:
    """Enhanced vector memory with HNSW index for production-grade ANN search."""
    def __init__(self, dim: int = DIM, use_advanced_search: bool = True):
        if not HAS_NUMPY:
            raise RuntimeError("NumPy is required for VectorMemory operations.")
        self.dim = dim
        self.graph = {}
        self.embeddings = {}
        self.mementos = {}
        self.scores = {}
        self.id_counter = 0
        self._memory_lock = threading.RLock()  # For embeddings/mementos
        self._index_lock = threading.RLock()  # Separate lock for HNSW ops
        
        # Advanced search components
        self.use_advanced_search = use_advanced_search and HAS_HNSWLIB
        self.hnsw_index = None
        self._id_to_label = {}  # Maps memento IDs to HNSW labels
        self._label_to_id = {}  # Reverse mapping
        self._next_label = 0
        self._rebuild_threshold = 500  # Rebuild after this many inserts
        
        # Bloom filter for fast membership testing
        self.bloom_filter = BloomFilter(size=100000, num_hashes=3)
        
        # Query cache
        self.query_cache = QueryCache(max_size=1000)
        
        # Circuit breaker for index operations
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30.0)

        # Background indexing thread
        self._index_thread = None
        self._index_queue = queue.Queue()
        self._should_stop_indexing = threading.Event()

        if self.use_advanced_search:
            self._init_hnsw_index()
            self._start_index_thread()

    def _init_hnsw_index(self):
        """Initialize HNSW index with optimal parameters."""
        if not HAS_HNSWLIB:
            _logger.warning("hnswlib not available, falling back to traditional search")
            self.use_advanced_search = False
            return
        
        try:
            self.hnsw_index = hnswlib.Index(space='cosine', dim=self.dim)
            # M=16 is a good balance between accuracy and memory
            # ef_construction=200 ensures good index quality
            self.hnsw_index.init_index(max_elements=100000, ef_construction=200, M=16)
            self.hnsw_index.set_ef(50)  # ef for search
            _logger.info("HNSW index initialized successfully")
        except Exception as e:
            _logger.error(f"Failed to initialize HNSW index: {e}")
            self.use_advanced_search = False

    def _start_index_thread(self):
        """Background thread for HNSW updates."""
        def indexer_loop():
            while not self._should_stop_indexing.is_set():
                try:
                    # Wait for batch or timeout
                    batch = []
                    deadline = time.time() + 5.0  # 5 second timeout
                    
                    while len(batch) < self._rebuild_threshold and time.time() < deadline:
                        try:
                            item = self._index_queue.get(timeout=0.1)
                            batch.append(item)
                        except queue.Empty:
                            continue
                    
                    if batch:
                        self._batch_add_to_hnsw_internal(batch)
                
                except Exception as e:
                    _logger.error(f"Indexer thread error: {e}")
        
        self._index_thread = threading.Thread(target=indexer_loop, daemon=True)
        self._index_thread.start()
        _logger.info("Started background indexer thread")

    def _batch_add_to_hnsw_internal(self, batch: List[Tuple[str, np.ndarray]]):
        """Called by background thread."""
        try:
            with self._index_lock:
                mids_to_add = [mid for mid, _ in batch]
                embs_to_add = np.array([emb for _, emb in batch])
                
                # Normalize embeddings
                embs_to_add = normalize_vectors(embs_to_add)
                
                # Assign labels and add to index
                labels = []
                for mid in mids_to_add:
                    label = self._next_label
                    self._id_to_label[mid] = label
                    self._label_to_id[label] = mid
                    labels.append(label)
                    self._next_label += 1
                
                # Resize index if needed
                current_size = self.hnsw_index.get_max_elements()
                if self._next_label >= current_size:
                    self.hnsw_index.resize_index(current_size * 2)
                
                self.hnsw_index.add_items(embs_to_add, labels)
                
                _logger.debug(f"Background indexed {len(labels)} items")
        except Exception as e:
            _logger.error(f"Background indexing failed: {e}")

    def __del__(self):
        """Clean shutdown of indexer thread."""
        if self._index_thread:
            self._should_stop_indexing.set()
            self._index_thread.join(timeout=5.0)

    def _safe_float(self, x, default=0.0):
        try:
            xf = float(x)
            if math.isnan(xf) or math.isinf(xf):
                return default
            return xf
        except Exception:
            return default

    def add_memento(self, mid: str, emb: np.ndarray, tags: List[str] = None, reliability: float = 0.5, 
                    content: str = "", is_gold: bool = False, source: str = "ingest"):
        """Adds a new memento to the memory."""
        with self._memory_lock:
            if mid in self.bloom_filter:
                if mid in self.embeddings:  # Confirm with actual check
                    return False
            
            self.embeddings[mid] = emb
            self.mementos[mid] = {
                "tags": tags or [], 
                "reliability": reliability, 
                "content": content, 
                "is_gold": is_gold, 
                "source": source
            }
            self.scores[mid] = {"w": np.zeros(self.dim), "b": 0.0}
            self.graph[mid] = []
            self.bloom_filter.add(mid)
            
            # Add to insert queue for background HNSW updates
            if self.use_advanced_search and self.hnsw_index is not None:
                self._index_queue.put((mid, emb))
            
            return True

    def update_memento_content(self, mid: str, content: str):
        """Updates the text content of an existing memento."""
        with self._memory_lock:
            if mid in self.mementos:
                self.mementos[mid]["content"] = content
                return True
            return False

    def add_edge(self, mid1: str, mid2: str, weight: float = 1.0, etype: str = "assoc"):
        """Adds a weighted, directed edge between two mementos."""
        with self._memory_lock:
            if mid1 in self.graph and mid2 in self.graph:
                self.graph[mid1].append({"to": mid2, "weight": weight, "type": etype})

    def _logistic(self, x: np.ndarray, w: np.ndarray, b: float) -> float:
        """Helper for logistic function with clamping."""
        z = float(np.dot(w, x) + b)
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + np.exp(-z))

    def retrieve(self, query_emb: np.ndarray, top_k: int = 5, ann_K: int = 50, 
                 use_advanced: bool = None, use_cache: bool = True) -> List[Tuple[str, float]]:
        """Enhanced retrieval using HNSW index or traditional search."""
        if use_advanced is None:
            use_advanced = self.use_advanced_search
        
        ann_K = max(1, min(int(ann_K), 1024))
        top_k = max(1, min(int(top_k), 100))

        if not self.embeddings:
            return []
        
        # Check cache first (no lock needed, QueryCache is thread-safe)
        if use_cache:
            cache_key = hashlib.sha256(query_emb.tobytes()).hexdigest()[:16]
            cached = self.query_cache.get(cache_key)
            if cached is not None:
                return cached[:top_k]
        
        mids = list(self.embeddings.keys())
        
        # Choose search method
        if use_advanced and self.hnsw_index is not None and len(mids) > 100:
            results = self._hnsw_retrieve(query_emb, top_k, ann_K, mids)
        else:
            results = self._traditional_retrieve(query_emb, top_k, ann_K, mids)
        
        # Cache results
        if use_cache:
            self.query_cache.put(cache_key, results, ttl=300.0)
        
        return results

    def _hnsw_retrieve(self, query_emb: np.ndarray, top_k: int, ann_K: int, mids: List[str]) -> List[Tuple[str, float]]:
        """HNSW-based retrieval with activation spreading."""
        try:
            # Normalize query
            query_norm = normalize_vectors(query_emb.reshape(1, -1)).squeeze()
            
            # HNSW search
            k_search = min(ann_K, len(self._id_to_label))
            with self._index_lock:
                labels, distances = self.hnsw_index.knn_query(query_norm, k=k_search)
            
            # Convert labels back to memento IDs
            candidates = []
            for label, dist in zip(labels[0], distances[0]):
                mid = self._label_to_id.get(label)
                if mid:
                    candidates.append((mid, dist))
            
            # Apply activation spreading on top candidates
            activation = defaultdict(float)
            for mid, dist in candidates[:min(20, len(candidates))]:
                activation[mid] += (1.0 - dist)  # Convert distance to similarity
                
                # Spread to neighbors
                with self._memory_lock:
                    for edge in self.graph.get(mid, []):
                        neighbor_mid = edge['to']
                        activation[neighbor_mid] += activation[mid] * edge['weight'] * 0.15
            
            # Re-rank with activation and learned scores
            final_scores = []
            with self._memory_lock:
                for mid in activation.keys():
                    if mid not in self.embeddings:
                        continue
                        
                    emb = self.embeddings[mid]
                    w = self.scores[mid]["w"]
                    b = self.scores[mid]["b"]
                    pred_score = self._logistic(emb, w, b)
                    
                    combined_score = (
                        self._safe_float(activation[mid], 0.0) * 0.5 +
                        self._safe_float(pred_score, 0.0) * 0.3 +
                        self._safe_float(self.mementos[mid]["reliability"], 0.0) * 0.2
                    )
                    final_scores.append((mid, combined_score))
            
            final_scores.sort(key=lambda x: x[1], reverse=True)
            return final_scores[:top_k]
            
        except Exception as e:
            _logger.error(f"HNSW retrieval failed: {e}, falling back to traditional")
            return self._traditional_retrieve(query_emb, top_k, ann_K, mids)

    def _traditional_retrieve(self, query_emb: np.ndarray, top_k: int, ann_K: int, mids: List[str]) -> List[Tuple[str, float]]:
        """Traditional retrieval with activation spreading."""
        with self._memory_lock:
            items = list(self.embeddings.items())
            mat = np.array([emb for _, emb in items])
            scores = mat.dot(query_emb)
            ranked_indices = np.argsort(scores)[::-1]
            
            # Activation spreading
            activation = defaultdict(float)
            for i in range(min(ann_K, len(ranked_indices))):
                mid = mids[ranked_indices[i]]
                activation[mid] += scores[ranked_indices[i]]
                
                # Spread to neighbors
                for edge in self.graph.get(mid, []):
                    neighbor_mid = edge['to']
                    activation[neighbor_mid] += activation[mid] * edge['weight'] * 0.1

            # Re-rank with activation and logistic score
            final_scores = []
            for mid, emb in self.embeddings.items():
                w = self.scores[mid]["w"]
                b = self.scores[mid]["b"]
                pred_score = self._logistic(emb, w, b)
                combined_score = (
                    self._safe_float(activation[mid], 0.0) * 0.5 +
                    self._safe_float(pred_score, 0.0) * 0.4 +
                    self._safe_float(self.mementos[mid]["reliability"], 0.0) * 0.1
                )
                final_scores.append((mid, combined_score))
            
            final_scores.sort(key=lambda x: x[1], reverse=True)
            return final_scores[:top_k]

    def feedback(self, rewards: Dict[str, float], lr: float = 0.01):
        """Applies online logistic regression feedback."""
        with self._memory_lock:
            for mid, reward in rewards.items():
                if mid not in self.embeddings:
                    continue
                    
                emb = self.embeddings[mid]
                w = self.scores[mid]["w"]
                b = self.scores[mid]["b"]
                
                # Predict and calculate error
                p = self._logistic(emb, w, b)
                err = reward - p
                
                # Dynamic learning rate based on error magnitude
                step = lr * (0.5 + 0.5 * (1 - abs(err)))
                
                # Update weights and bias
                w += step * err * emb
                b += step * err
                self.scores[mid]["w"] = w
                self.scores[mid]["b"] = b

    def evaluate_retrieval(self, test_queries: List[Tuple[np.ndarray, List[str]]]) -> Dict[str, float]:
        """
        test_queries: List of (query_embedding, ground_truth_ids)
        Returns: {recall@k, precision@k, mrr}
        """
        recalls = []
        precisions = []
        reciprocal_ranks = []
        
        for query_emb, ground_truth in test_queries:
            results = self.retrieve(query_emb, top_k=10, use_cache=False)
            retrieved_ids = [mid for mid, _ in results]
            
            # Recall@k
            relevant_retrieved = len(set(retrieved_ids) & set(ground_truth))
            recall = relevant_retrieved / len(ground_truth) if ground_truth else 0
            recalls.append(recall)
            
            # Precision@k
            precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0
            precisions.append(precision)
            
            # MRR (Mean Reciprocal Rank)
            for i, mid in enumerate(retrieved_ids):
                if mid in ground_truth:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return {
            "recall@10": np.mean(recalls),
            "precision@10": np.mean(precisions),
            "mrr": np.mean(reciprocal_ranks),
            "num_queries": len(test_queries)
        }

    def save_state(self, path: str):
        """Saves memory state to a file."""
        with self._memory_lock:
            data = {
                "schema": 3,
                "dim": self.dim,
                "use_advanced_search": self.use_advanced_search,
                "embeddings": {mid: emb.tolist() for mid, emb in self.embeddings.items()},
                "mementos": self.mementos,
                "scores": {mid: {"w": s["w"].tolist(), "b": s["b"]} for mid, s in self.scores.items()},
                "graph": self.graph,
                "id_counter": self.id_counter,
                "_id_to_label": self._id_to_label,
                "_next_label": self._next_label
            }
            
            opener = gzip.open if HAS_GZIP and path.endswith(".gz") else open
            with opener(path, 'wt', encoding='utf-8') as f:
                json.dump(data, f)
            
            # Save HNSW index separately
            if self.use_advanced_search and self.hnsw_index is not None:
                hnsw_path = path.replace('.json', '.hnsw').replace('.gz', '.hnsw')
                try:
                    self.hnsw_index.save_index(hnsw_path)
                    data["hnsw_index_path"] = hnsw_path
                    _logger.info(f"HNSW index saved to {hnsw_path}")
                except Exception as e:
                    _logger.warning(f"Failed to save HNSW index: {e}")
            
            _logger.info(f"Memory state saved to {path}")

    def save_state_binary(self, path: str, create_backup: bool = True):
        """Binary format for >100k mementos."""
        if create_backup and os.path.exists(path):
            backup_path = f"{path}.backup.{int(time.time())}"
            shutil.copy(path, backup_path)
            _logger.info(f"Created backup: {backup_path}")
        
        try:
            temp_path = f"{path}.tmp"
            with self._memory_lock:
                # Save embeddings as numpy memmap
                emb_path = temp_path.replace('.vmem', '_embeddings.npy')
                emb_matrix = np.array([self.embeddings[mid] for mid in sorted(self.embeddings.keys())])
                np.save(emb_path, emb_matrix)
                
                # Save metadata as pickle (much faster than JSON)
                metadata = {
                    "schema": 4,
                    "dim": self.dim,
                    "id_order": sorted(self.embeddings.keys()),  # Index into emb_matrix
                    "mementos": self.mementos,
                    "scores": {mid: {"w": s["w"].tolist(), "b": s["b"]} for mid, s in self.scores.items()},
                    "graph": self.graph,
                    "_id_to_label": self._id_to_label,
                    "_next_label": self._next_label
                }
                
                with open(temp_path, 'wb') as f:
                    pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Save HNSW index
                if self.hnsw_index:
                    hnsw_path = temp_path.replace('.vmem', '.hnsw')
                    with self._index_lock:
                        self.hnsw_index.save_index(hnsw_path)
            
            # Atomic rename
            os.replace(temp_path, path)
            _logger.info(f"Binary state saved: {path}")
        
        except Exception as e:
            _logger.error(f"Save failed: {e}")
            if os.path.exists(f"{path}.tmp"):
                os.remove(f"{path}.tmp")
            raise

    @classmethod
    def load_state(cls, path: str) -> Optional['VectorMemory']:
        """Loads memory state from a file."""
        try:
            opener = gzip.open if HAS_GZIP and path.endswith(".gz") else open
            with opener(path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            
            schema = data.get("schema", 0)
            if schema < 2:
                _logger.warning(f"Old memory schema {schema}, attempting best-effort load")

            dim = data.get("dim", DIM)
            use_advanced = data.get("use_advanced_search", True)
            mem = cls(dim=dim, use_advanced_search=use_advanced)
            
            mem.embeddings = {mid: np.array(emb) for mid, emb in data["embeddings"].items()}
            mem.mementos = data["mementos"]
            mem.scores = {mid: {"w": np.array(s["w"]), "b": s["b"]} for mid, s in data["scores"].items()}
            mem.graph = data["graph"]
            mem.id_counter = data["id_counter"]
            
            # Restore label mappings
            if "_id_to_label" in data:
                mem._id_to_label = {k: int(v) for k, v in data["_id_to_label"].items()}
                mem._label_to_id = {int(v): k for k, v in mem._id_to_label.items()}
                mem._next_label = data.get("_next_label", len(mem._id_to_label))
            
            # Restore bloom filter
            for mid in mem.embeddings.keys():
                mem.bloom_filter.add(mid)
            
            # Load HNSW index
            if "hnsw_index_path" in data and mem.use_advanced_search:
                hnsw_path = data["hnsw_index_path"]
                try:
                    if mem.hnsw_index is None:
                        mem._init_hnsw_index()
                    mem.hnsw_index.load_index(hnsw_path)
                    _logger.info("Loaded HNSW index")
                except Exception as e:
                    _logger.warning(f"Failed to load HNSW index: {e}, will rebuild")
                    # Rebuild from scratch
                    if mem._id_to_label:
                        mem._rebuild_hnsw_from_scratch()
            
            _logger.info(f"Memory state loaded from {path}")
            return mem
            
        except Exception as e:
            _logger.error(f"Failed to load memory state: {e}")
            return None

    @classmethod
    def load_state_binary(cls, path: str, try_backup: bool = True) -> Optional['VectorMemory']:
        """Load from binary format."""
        try:
            with open(path, 'rb') as f:
                metadata = pickle.load(f)
            
            mem = cls(dim=metadata["dim"], use_advanced_search=True)
            
            # Load embeddings from memmap
            emb_path = path.replace('.vmem', '_embeddings.npy')
            emb_matrix = np.load(emb_path)
            
            # Reconstruct embeddings dict
            for i, mid in enumerate(metadata["id_order"]):
                mem.embeddings[mid] = emb_matrix[i]
            
            mem.mementos = metadata["mementos"]
            mem.scores = {mid: {"w": np.array(s["w"]), "b": s["b"]}
                          for mid, s in metadata["scores"].items()}
            mem.graph = metadata["graph"]
            mem._id_to_label = metadata["_id_to_label"]
            mem._next_label = metadata["_next_label"]
            
            # Rebuild bloom filter
            for mid in mem.embeddings.keys():
                mem.bloom_filter.add(mid)
            
            # Load HNSW
            hnsw_path = path.replace('.vmem', '.hnsw')
            if os.path.exists(hnsw_path):
                mem._init_hnsw_index()
                mem.hnsw_index.load_index(hnsw_path)
            
            return mem
        except Exception as e:
            _logger.error(f"Failed to load {path}: {e}")
            
            if try_backup:
                # Find most recent backup
                backups = sorted([f for f in os.listdir(os.path.dirname(path) or '.')
                                  if f.startswith(os.path.basename(path) + '.backup.')])
                
                if backups:
                    backup_path = os.path.join(os.path.dirname(path) or '.', backups[-1])
                    _logger.info(f"Attempting recovery from backup: {backup_path}")
                    return cls.load_state_binary(backup_path, try_backup=False)
            
            return None

    def _rebuild_hnsw_from_scratch(self):
        """Rebuild HNSW index from existing embeddings."""
        if not self.use_advanced_search or not self._id_to_label:
            return
        
        try:
            _logger.info("Rebuild ing HNSW index from scratch...")
            
            # Collect all embeddings in label order
            embeddings_list = []
            labels_list = []
            
            with self._memory_lock:
                for mid, label in sorted(self._id_to_label.items(), key=lambda x: x[1]):
                    if mid in self.embeddings:
                        embeddings_list.append(self.embeddings[mid])
                        labels_list.append(label)
            
            if embeddings_list:
                embeddings_array = np.array(embeddings_list)
                embeddings_array = normalize_vectors(embeddings_array)
                
                # Reinitialize index with correct size
                self._init_hnsw_index()
                if self.hnsw_index.get_max_elements() < len(labels_list):
                    self.hnsw_index.resize_index(len(labels_list) * 2)
                
                with self._index_lock:
                    self.hnsw_index.add_items(embeddings_array, labels_list)
                _logger.info(f"HNSW index rebuilt with {len(labels_list)} items")
        
        except Exception as e:
            _logger.error(f"Failed to rebuild HNSW index: {e}")

class UnifiedCognitionSystem:
    """The main UCS class, coordinating memory and blackboard."""
    def __init__(self, mem: Optional[VectorMemory] = None, dim: int = DIM, 
                 embed_fn=None, use_advanced_search: bool = True, embed_model: str = None):
        self._dim = dim
        self.vmem = mem
        self.embed_fn = embed_fn or self._embed_query
        self.expert_manager = ExpertManager(parent_system=self)
        self.telemetry = defaultdict(list)
        self._is_init = False
        self._session_memory = defaultdict(list)
        self._telemetry_cap = 50_000
        self._lock = threading.RLock()
        self._reputations = defaultdict(
            lambda: ExpertReputation(n=0, reward_sum=0, reward_sq=0, last_seen=0, ema_reward=0.5)
        )
        self.use_advanced_search = use_advanced_search
        self.calibrator = IsotonicCalibrator()
        self.metrics = defaultdict(int)
        self._embed_model = None
        
        if embed_model:
            try:
                from sentence_transformers import SentenceTransformer
                self._embed_model = SentenceTransformer(embed_model)
                self._dim = self._embed_model.get_sentence_embedding_dimension()
                _logger.info(f"Loaded embedding model: {embed_model} (dim={self._dim})")
            except ImportError:
                _logger.warning("sentence-transformers not installed, using deterministic embeddings")

    def _reputation_score(self, expert_name: str, c: float = 0.5) -> float:
        """UCB1-based reputation score."""
        rep = self._reputations[expert_name]
        if rep.n == 0:
            return rep.ema_reward + 0.1
        
        total_n = sum(r.n for r in self._reputations.values())
        exploration_bonus = c * math.sqrt(math.log(total_n + 1) / rep.n)
        return rep.reward_sum / rep.n + exploration_bonus

    def _update_reputation(self, expert_name: str, reward: float):
        """Updates expert reputation."""
        with self._lock:
            rep = self._reputations[expert_name]
            n = rep.n + 1
            reward_sum = rep.reward_sum + reward
            reward_sq = rep.reward_sq + reward**2
            last_seen = time.time()
            alpha = 0.1
            ema_reward = alpha * reward + (1-alpha) * rep.ema_reward
            self._reputations[expert_name] = ExpertReputation(n, reward_sum, reward_sq, last_seen, ema_reward)
    
    def _calibrate(self, raw_score: float) -> float:
        """Calibrate using isotonic regression."""
        return self.calibrator.calibrate(raw_score)

    def _ensure_memory(self):
        """Initializes vector memory if needed."""
        with self._lock:
            if self.vmem is None:
                if not HAS_NUMPY:
                    raise RuntimeError("Vector memory unavailable without NumPy")
                self.vmem = VectorMemory(self._dim, use_advanced_search=self.use_advanced_search)
            assert self._dim == self.vmem.dim, "Dimension mismatch"

    def _sanitize_prompt(self, prompt: str) -> Tuple[str, bool]:
        """Strips expert-override strings."""
        original_prompt = prompt
        sanitization_regex = re.compile(r'\b(set_plan|set_mode)\b', re.IGNORECASE)
        sanitized_prompt = sanitization_regex.sub('', prompt).strip()
        was_sanitized = sanitized_prompt != original_prompt
        return sanitized_prompt, was_sanitized

    def initialize_experts(self):
        """Initializes and registers experts."""
        if self._is_init:
            return
        
        self.expert_manager.register_expert(
            name="RetrievalExpert", handler=retrieval_expert,
            expertise_tags=["memory", "retrieval"], phase="propose"
        )
        self.expert_manager.register_expert(
            name="SummarizationExpert", handler=summarization_expert,
            expertise_tags=["language", "synthesis"], phase="propose"
        )
        self.expert_manager.register_expert(
            name="RehearsalExpert", handler=rehearsal_expert,
            expertise_tags=["memory", "synthesis"], phase="propose"
        )
        self.expert_manager.register_expert(
            name="LongformExpert", handler=longform_expert,
            expertise_tags=["language", "synthesis"], phase="propose"
        )
        self.expert_manager.register_expert(
            name="MetaExpert", handler=meta_expert,
            expertise_tags=["control"], phase="map"
        )
        self.expert_manager.register_expert(
            name="SetPlanExpert", handler=set_plan_expert,
            expertise_tags=["control", "routing"], phase="propose"
        )
        self.expert_manager.register_expert(
            name="RouterExpert", handler=router_expert,
            expertise_tags=["control", "routing"], phase="filter"
        )
        self.expert_manager.register_expert(
            name="SelfAttentionExpert", handler=self_attention_expert,
            expertise_tags=["control", "attention"], phase="filter"
        )
        self._is_init = True

    def _embed_query(self, text: str):
        """Real embeddings if model loaded, else deterministic."""
        if self._embed_model:
            emb = self._embed_model.encode([text], normalize_embeddings=True)[0]
            return emb
        
        # Fallback to deterministic
        h = hashlib.sha256(text.encode()).digest()
        if not HAS_NUMPY:
            vals = [h[i % len(h)] for i in range(self._dim)]
            s = sum(vals) or 1
            return [v / s for v in vals]
        
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        v = rng.normal(size=(self._dim,))
        norm = np.linalg.norm(v)
        if norm == 0:
            return np.zeros(self._dim)
        return v / norm
    
    def _embed(self, text: str):
        return self.embed_fn(text)

    def _quick_summarize(self, text: str, max_sents: int = 3) -> str:
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return " ".join(sents[:max_sents])

    def _novelty_score(self, results):
        vals = [self.vmem._safe_float(score, 0.0) for _, score in (results or [])]
        if len(vals) < 2:
            return 0.0
        if not HAS_NUMPY:
            mu = sum(vals)/len(vals)
            var = sum((x-mu)**2 for x in vals)/len(vals)
            return var ** 0.5
        return float(np.std(vals))

    def benchmark_retrieval(self, num_queries: int = 100, dataset_size: int = 10000):
        """Benchmark retrieval performance."""
        if not HAS_NUMPY:
            _logger.warning("Benchmarking requires NumPy")
            return {}
        
        self._ensure_memory()
        
        # Add test data if needed
        current_size = len(self.vmem.embeddings)
        if current_size < dataset_size:
            _logger.info(f"Adding {dataset_size - current_size} test embeddings")
            for i in range(current_size, dataset_size):
                emb = self._embed(f"test_document_{i}")
                self.vmem.add_memento(f"test_{i}", emb, content=f"Test document {i}", source="benchmark")
        
        queries = [self._embed(f"query_{i}") for i in range(num_queries)]
        
        results = {}
        
        # Traditional retrieval
        start_time = time.time()
        for q in queries:
            self.vmem.retrieve(q, top_k=10, use_advanced=False, use_cache=False)
        traditional_time = (time.time() - start_time) / num_queries
        results["traditional_avg_ms"] = traditional_time * 1000
        
        # Advanced retrieval
        if self.use_advanced_search and HAS_HNSWLIB:
            start_time = time.time()
            for q in queries:
                self.vmem.retrieve(q, top_k=10, use_advanced=True, use_cache=False)
            advanced_time = (time.time() - start_time) / num_queries
            results["advanced_avg_ms"] = advanced_time * 1000
            results["speedup"] = traditional_time / advanced_time if advanced_time > 0 else 0
        
        results["dataset_size"] = len(self.vmem.embeddings)
        results["has_hnsw"] = self.vmem.hnsw_index is not None
        
        _logger.info(f"Benchmark results: {results}")
        return results
    
    def run(self, prompt: str, actions: Optional[List[str]] = None, iters: int = 5, 
            session_id: Optional[str] = None) -> Dict[str, Any]:
        """Runs the blackboard loop."""
        self.initialize_experts()

        sanitized_prompt, was_sanitized = self._sanitize_prompt(prompt)
        prompt = sanitized_prompt
        
        if len(prompt) > MAX_PROMPT_LEN:
            return {"prompt": prompt[:256] + "...", "error": "Prompt too large", 
                    "history": [], "executed_ops": []}
        
        session_id = session_id or str(uuid.uuid4())
        
        # Deep copy blackboard for thread safety
        with self._lock:
            self._session_memory[session_id].append({"t": time.time(), "prompt": prompt})
            self._session_memory[session_id] = [e for e in self._session_memory[session_id] 
                                                 if time.time()-e["t"] < 900]
            session_snapshot = list(self._session_memory[session_id])
            
            blackboard = {
                "prompt": prompt, "history": [], "session_id": session_id,
                "executed_ops": [], "plan": [], "audit_sanitized": was_sanitized,
                "parent_system": self
            }
            if actions:
                blackboard["plan"] = actions
        
        blackboard["session_recent"] = len(session_snapshot)
        
        deadline = time.time() + float(os.getenv("UCS_RUN_DEADLINE_S", "5"))
        timed_out = False

        for i in range(iters):
            if time.time() > deadline:
                _logger.warning("Run deadline exceeded")
                blackboard["history"].append({"operation":"DEADLINE_EXCEEDED"})
                timed_out = True
                break
            
            # Refresh context tags
            ctx_tags = set()
            for item in blackboard.get("history", []):
                if isinstance(item, dict) and "retrieval" in item:
                    if self.vmem:
                        with self.vmem._memory_lock:
                            for mid, _ in item["retrieval"]:
                                ctx_tags.update(self.vmem.mementos.get(mid, {}).get("tags", []))
            blackboard["ctx_tags"] = sorted(list(ctx_tags))

            _logger.debug(f"--- Iteration {i+1}/{iters} ---")
            
            # Gather and score proposals
            proposals = self.expert_manager.propose(blackboard)
            scored_proposals = self.expert_manager.score_proposals(proposals, blackboard)
            
            # Log telemetry
            for prop in scored_proposals:
                log_entry = {
                    "expert": prop.origin,
                    "action": prop.action,
                    "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                    "trust_score": prop.trust_score,
                    "pre_calib_score": prop.pre_calib_score,
                    "timestamp": datetime.now().isoformat()
                }
                if UCS_LOG_PROPOSAL_CONTENT:
                    content_str = str(prop.content)
                    if UCS_REDACT_REGEX:
                        content_str = re.sub(UCS_REDACT_REGEX, "[REDACTED]", content_str)
                    log_entry["proposal_content"] = content_str

                if prop.action == "RETRIEVE":
                    retrieved_ids = [r[0] for r in blackboard.get("retrieval", [])]
                    if retrieved_ids:
                        log_entry["retrieved_ids"] = retrieved_ids
                
                self.telemetry["proposals"].append(log_entry)
            
            if len(self.telemetry["proposals"]) > self._telemetry_cap:
                del self.telemetry["proposals"][: len(self.telemetry["proposals"]) // 4]
            
            # Select winning proposal
            winning_proposal = max(scored_proposals, key=lambda p: p.trust_score, default=None)
            
            if winning_proposal and winning_proposal.trust_score > 0.01:
                _logger.info(f"Winning: {winning_proposal.action} (score: {winning_proposal.trust_score:.2f})")
                
                op = winning_proposal.action
                payload = winning_proposal.content
                blackboard["executed_ops"].append(op)
                
                if op == "SET_PLAN":
                    blackboard["plan"] = payload.get("plan", [])
                    blackboard["history"].append({"operation": "SET_PLAN", "plan": blackboard["plan"]})

                elif op == "RETRIEVE":
                    success, result = self.vmem.circuit_breaker.call(
                        self._execute_retrieve, blackboard, winning_proposal
                    )
                    if success:
                        self._update_reputation(winning_proposal.origin, reward=1.0)
                    else:
                        self._update_reputation(winning_proposal.origin, reward=0.0)

                elif op == "SUMMARIZE":
                    txt = payload.get("text") or blackboard.get("prompt", "")
                    summary = self._quick_summarize(txt)
                    blackboard["summary"] = summary
                    blackboard["history"].append({"operation":"SUMMARIZE", "summary": summary})
                    self._update_reputation(winning_proposal.origin, reward=1.0)
                
                elif op == "GENERATE_LONGFORM":
                    retrieved_content = " ".join([
                        self.vmem.mementos.get(mid, {}).get("content", "") 
                        for mid, _ in payload.get("source_mementos", [])
                    ])
                    longform_text = f"Longform: {self._quick_summarize(retrieved_content, 5)}"
                    blackboard["longform_output"] = longform_text
                    blackboard["history"].append({"operation": "GENERATE_LONGFORM", "output": longform_text})
                    self._update_reputation(winning_proposal.origin, reward=1.0)

                elif op == "REHEARSE":
                    try:
                        self._ensure_memory()
                        memento_id = payload.get("memento_id")
                        with self.vmem._memory_lock:
                            if memento_id and memento_id in self.vmem.mementos:
                                original_content = self.vmem.mementos[memento_id]["content"]
                                new_summary = self._quick_summarize(original_content)
                                self.vmem.update_memento_content(memento_id, new_summary)
                                blackboard["history"].append({
                                    "operation":"REHEARSE", 
                                    "memento_id": memento_id, 
                                    "summary": new_summary
                                })
                                self._update_reputation(winning_proposal.origin, reward=1.0)
                            else:
                                blackboard["history"].append({"operation":"REHEARSE_FAILED", "error": "Not found"})
                                self._update_reputation(winning_proposal.origin, reward=0.0)
                    except Exception as e:
                        _logger.error(f"Rehearsal failed: {e}")
                        blackboard["history"].append({"operation":"REHEARSE_FAILED", "error": str(e)})
                        self._update_reputation(winning_proposal.origin, reward=0.0)
                
                elif op == "SET_MODE":
                    blackboard["mode"] = payload.get("mode", "System-1")
                    blackboard["history"].append({"operation": "SET_MODE", "mode": blackboard["mode"]})
                
                else:
                    blackboard["history"].append(payload)

            else:
                _logger.warning("No viable proposal found")
                break
        
        # Update calibrator with outcomes
        if "retrieval" in blackboard:
            # Simplified feedback: retrieval success = 1.0
            for prop in scored_proposals:
                if prop.action == "RETRIEVE":
                    self.calibrator.update(prop.pre_calib_score, 1.0)
        
        # Metrics
        blackboard["metrics"] = {
            "iters": i+1,
            "mode": blackboard.get("mode","System-1"),
            "telemetry_buffer": len(self.telemetry["proposals"]),
            "retrieved": len(blackboard.get("retrieval", [])),
            "timed_out": timed_out,
            "advanced_search": self.use_advanced_search,
            "cache_stats": self.vmem.query_cache.stats() if self.vmem else {}
        }
        
        return blackboard
    
    def _execute_retrieve(self, blackboard, winning_proposal):
        """Execute retrieval with error handling."""
        try:
            self._ensure_memory()
            qv = self._embed(blackboard["prompt"])
            results = self.vmem.retrieve(qv, top_k=5, ann_K=64)
            blackboard["retrieval"] = results
            blackboard["history"].append({"operation":"RETRIEVE", "retrieval": results})
            
            nov = self._novelty_score(results)
            if nov > 0.25:
                _logger.warning(f"Black Swan signal: novelty={nov:.2f}")
            
            return results
        except Exception as e:
            _logger.error(f"Retrieval failed: {e}")
            blackboard["history"].append({"operation":"RETRIEVE_FAILED", "error": str(e)})
            raise

# --- FastAPI endpoints ---
if HAS_FASTAPI and os.environ.get('RUN_API') == '1' and uvicorn:
    app = FastAPI(title="Enhanced UCS API v2", 
                  description="Production-ready Unified Cognition System with HNSW")
    ucs_instance = UnifiedCognitionSystem(use_advanced_search=True)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("UCS_CORS_ORIGINS", "").split(",") if os.getenv("UCS_CORS_ORIGINS") else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response
    
    # Rate limiting
    RUN_RATE_LIMIT = 5
    GENERAL_RATE_LIMIT = 20
    RATE_LIMIT_BUCKET = defaultdict(lambda: {"tokens": GENERAL_RATE_LIMIT, "last_refill": time.time()})
    
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        start_time = time.time()
        client_ip = request.client.host
        if TRUST_XFF:
            client_ip = request.headers.get("X-Forwarded-For", "").split(',')[0].strip() or client_ip

        bucket = RATE_LIMIT_BUCKET[client_ip]
        now = time.time()
        
        refill_rate = RUN_RATE_LIMIT / 60 if request.url.path == "/run_blackboard" else GENERAL_RATE_LIMIT / 60
        bucket["tokens"] = min(GENERAL_RATE_LIMIT, bucket["tokens"] + (now - bucket["last_refill"]) * refill_rate)
        bucket["last_refill"] = now
        
        if bucket["tokens"] < 1:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        bucket["tokens"] -= 1

        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Process-Time"] = str(time.time() - start_time)
        return response

    def get_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
        if REQUIRE_AUTH:
            if not API_KEY:
                raise HTTPException(status_code=500, detail="Server misconfiguration")
            if x_api_key != API_KEY:
                raise HTTPException(status_code=401, detail="Invalid API Key")
        return x_api_key

    class IngestItem(BaseModel):
        id: str
        text: str
        tags: List[str] = Field(default_factory=list)

    class EdgeItem(BaseModel):
        src: str
        dst: str
        weight: float = 1.0
        etype: str = "assoc"

    class FeedbackItem(BaseModel):
        id: str
        reward: float

    @app.get("/health")
    async def health():
        return {
            "ok": True,
            "numpy": HAS_NUMPY,
            "hnswlib": HAS_HNSWLIB,
            "telemetry_buffer": len(ucs_instance.telemetry.get("proposals", [])),
            "dim": ucs_instance._dim,
            "advanced_search": ucs_instance.use_advanced_search
        }

    @app.get("/health/detailed")
    async def detailed_health():
        """Comprehensive health check."""
        health = {
            "status": "healthy",
            "checks": {}
        }
        
        # Check memory integrity
        try:
            if ucs_instance.vmem:
                health["checks"]["memory"] = {
                    "status": "ok",
                    "mementos": len(ucs_instance.vmem.embeddings),
                    "cache_hit_rate": ucs_instance.vmem.query_cache.stats()["hit_rate"]
                }
        except Exception as e:
            health["status"] = "degraded"
            health["checks"]["memory"] = {"status": "error", "error": str(e)}
        
        # Check HNSW index
        try:
            if ucs_instance.vmem and ucs_instance.vmem.hnsw_index:
                test_query = np.random.rand(ucs_instance._dim)
                ucs_instance.vmem.hnsw_index.knn_query(test_query, k=1)
                health["checks"]["hnsw_index"] = {"status": "ok"}
        except Exception as e:
            health["status"] = "degraded"
            health["checks"]["hnsw_index"] = {"status": "error", "error": str(e)}
        
        # Check circuit breaker state
        if ucs_instance.vmem:
            cb_state = ucs_instance.vmem.circuit_breaker.state
            health["checks"]["circuit_breaker"] = {"status": cb_state}
            if cb_state == "open":
                health["status"] = "unhealthy"
        
        return health

    @app.get("/version")
    async def version():
        return {
            "name": "enhanced_ucs_v2",
            "spec": 3,
            "dim": DIM,
            "features": ["hnsw_index", "query_cache", "circuit_breaker", "isotonic_calibration"]
        }

    @app.post("/run_blackboard", dependencies=[Depends(get_api_key)])
    async def run_blackboard(data: dict):
        prompt = (data.get("prompt") or "").strip()
        actions = data.get("actions")
        if len(prompt) > MAX_PROMPT_LEN:
            raise HTTPException(status_code=413, detail="Prompt too large")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        result = ucs_instance.run(prompt, actions)
        return {"result": result}
        
    @app.get("/telemetry", dependencies=[Depends(get_api_key)])
    async def get_telemetry():
        return ucs_instance.telemetry
        
    @app.post("/replay_query", dependencies=[Depends(get_api_key)])
    async def replay_query(data: dict):
        prompt = (data.get("prompt") or "").strip()
        past_proposals = [p for p in ucs_instance.telemetry["proposals"]
                          if hashlib.sha256(prompt.encode()).hexdigest() == p["prompt_hash"]]
        
        return {"count": len(past_proposals), "replayed_proposals": past_proposals[-500:]}

    @app.post("/ingest", dependencies=[Depends(get_api_key)])
    async def ingest(items: List[IngestItem]):
        ucs_instance._ensure_memory()
        items = items[:MAX_INGEST_ITEMS]
        ingested_count = 0
        for it in items:
            mid = it.id
            if mid in ucs_instance.vmem.bloom_filter and mid in ucs_instance.vmem.embeddings:
                raise HTTPException(status_code=409, detail=f"memento '{mid}' already exists")
            txt = it.text
            tags = it.tags
            emb = ucs_instance._embed(txt)
            if ucs_instance.vmem.add_memento(mid, emb, tags=tags, reliability=0.6, content=txt):
                ingested_count += 1
        return {"ok": True, "count": ingested_count}

    @app.post("/ingest_edges", dependencies=[Depends(get_api_key)])
    async def ingest_edges(edges: List[EdgeItem]):
        ucs_instance._ensure_memory()
        ingested_count = 0
        for e in edges:
            if e.src == e.dst:
                continue
            if e.src in ucs_instance.vmem.graph and e.dst in ucs_instance.vmem.graph:
                ucs_instance.vmem.add_edge(e.src, e.dst, e.weight, e.etype)
                ingested_count += 1
        return {"ok": True, "count": ingested_count}

    @app.post("/feedback", dependencies=[Depends(get_api_key)])
    async def feedback(items: List[FeedbackItem]):
        ucs_instance._ensure_memory()
        rewards = {it.id: float(it.reward) for it in items}
        ucs_instance.vmem.feedback(rewards)
        return {"ok": True, "count": len(items)}

    @app.post("/save_state", dependencies=[Depends(get_api_key)])
    async def save_state(payload: dict):
        ucs_instance._ensure_memory()
        path = (payload.get("path") or "ucs_v2_state.json").strip()
        ucs_instance.vmem.save_state(path)
        return {"ok": True, "path": path}

    @app.post("/load_state", dependencies=[Depends(get_api_key)])
    async def load_state(payload: dict):
        path = (payload.get("path") or "ucs_v2_state.json").strip()
        mem = VectorMemory.load_state(path)
        if mem is None:
            raise HTTPException(status_code=404, detail="Failed to load state")
        with ucs_instance._lock:
            ucs_instance.vmem = mem
        return {"ok": True, "path": path, "dim": mem.dim, "count": len(mem.embeddings)}

    @app.post("/clear_memory", dependencies=[Depends(get_api_key)])
    async def clear_memory():
        """Danger: wipes all in-memory vectors/graph/scores."""
        ucs_instance._ensure_memory()
        with ucs_instance._lock:
            ucs_instance.vmem.embeddings.clear()
            ucs_instance.vmem.mementos.clear()
            ucs_instance.vmem.scores.clear()
            ucs_instance.vmem.graph.clear()
            ucs_instance.vmem.id_counter = 0
            ucs_instance.vmem._id_to_label.clear()
            ucs_instance.vmem._label_to_id.clear()
            ucs_instance.vmem._next_label = 0
            ucs_instance.vmem.bloom_filter = BloomFilter(size=100000, num_hashes=3)
            ucs_instance.vmem.query_cache.clear()
            if ucs_instance.vmem.use_advanced_search:
                ucs_instance.vmem._init_hnsw_index()
        return {"ok": True, "count": 0}

    @app.post("/benchmark", dependencies=[Depends(get_api_key)])
    async def benchmark_retrieval(data: dict = None):
        """Run retrieval benchmarks."""
        if data is None:
            data = {}
        num_queries = data.get("num_queries", 100)
        dataset_size = data.get("dataset_size", 10000)
        
        results = ucs_instance.benchmark_retrieval(num_queries, dataset_size)
        return {"ok": True, "benchmark_results": results}

    @app.get("/memory_stats", dependencies=[Depends(get_api_key)])
    async def memory_stats():
        """Get detailed memory statistics."""
        if ucs_instance.vmem is None:
            return {"ok": True, "memory_initialized": False}
        
        stats = {
            "ok": True,
            "memory_initialized": True,
            "total_mementos": len(ucs_instance.vmem.embeddings),
            "total_edges": sum(len(edges) for edges in ucs_instance.vmem.graph.values()),
            "dimension": ucs_instance.vmem.dim,
            "advanced_search_enabled": ucs_instance.vmem.use_advanced_search,
            "has_hnsw_index": ucs_instance.vmem.hnsw_index is not None,
            "insert_buffer_size": len(ucs_instance.vmem._insert_buffer),
            "cache_stats": ucs_instance.vmem.query_cache.stats(),
            "circuit_breaker_state": ucs_instance.vmem.circuit_breaker.state
        }
        
        # Tag statistics
        tag_counts = defaultdict(int)
        for mid, memento in ucs_instance.vmem.mementos.items():
            for tag in memento.get("tags", []):
                tag_counts[tag] += 1
        stats["top_tags"] = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return stats

    @app.get("/expert_reputations", dependencies=[Depends(get_api_key)])
    async def expert_reputations():
        """Get expert reputation scores."""
        reps = {}
        for name, rep in ucs_instance._reputations.items():
            reps[name] = {
                "observations": rep.n,
                "avg_reward": rep.reward_sum / rep.n if rep.n > 0 else 0,
                "ema_reward": rep.ema_reward,
                "ucb_score": ucs_instance._reputation_score(name),
                "last_seen": datetime.fromtimestamp(rep.last_seen).isoformat() if rep.last_seen > 0 else None
            }
        return {"ok": True, "reputations": reps}

    @app.post("/clear_cache", dependencies=[Depends(get_api_key)])
    async def clear_cache():
        """Clear query cache."""
        if ucs_instance.vmem:
            ucs_instance.vmem.query_cache.clear()
            return {"ok": True, "message": "Cache cleared"}
        return {"ok": False, "message": "Memory not initialized"}

# --- Testing and Benchmarking ---

def test_retrieval_quality(u: UnifiedCognitionSystem):
    u._ensure_memory()
    
    # Clear memory for this test
    u.vmem.embeddings.clear()
    u.vmem.mementos.clear()
    u.vmem.scores.clear()
    u.vmem.graph.clear()
    u.vmem.id_counter = 0
    u.vmem._id_to_label.clear()
    u.vmem._label_to_id.clear()
    u.vmem._next_label = 0
    u.vmem.bloom_filter = BloomFilter(size=100000, num_hashes=3)
    u.vmem.query_cache.clear()
    if u.vmem.use_advanced_search:
        u.vmem._init_hnsw_index()
    
    # Create synthetic ground truth
    test_queries = []
    topic_vectors = {}
    rng = np.random.default_rng(42)
    for topic in ["ml", "physics", "history"]:
        topic_vectors[topic] = rng.normal(size=u._dim)
        topic_vectors[topic] /= np.linalg.norm(topic_vectors[topic]) + 1e-12
        
        # Add 10 docs per topic
        topic_ids = []
        for i in range(10):
            mid = f"{topic}_{i}"
            noise = rng.normal(size=u._dim) * 0.05
            emb = topic_vectors[topic] + noise
            emb /= np.linalg.norm(emb) + 1e-12
            u.vmem.add_memento(mid, emb, tags=[topic])
            topic_ids.append(mid)
        
        # Query should retrieve docs from same topic
        query_emb = topic_vectors[topic].copy()
        test_queries.append((query_emb, topic_ids))
    
    metrics = u.vmem.evaluate_retrieval(test_queries)
    print(f"Retrieval quality: {metrics}")
    
    assert metrics["recall@10"] > 0.7, "Recall too low!"
    assert metrics["mrr"] > 0.5, "MRR too low!"

def run_enhanced_smoke_test():
    """Enhanced smoke test with all new features."""
    if not HAS_NUMPY:
        _logger.warning("NumPy not found. Skipping smoke test.")
        return
    
    _logger.info("--- Running Enhanced UCS v2 Smoke Test ---")
    
    u = UnifiedCognitionSystem(use_advanced_search=True)
    u._ensure_memory()

    _logger.info("Testing enhanced memory system...")
    rng = np.random.default_rng(42)
    
    # Add test data
    for i in range(2000):
        v = rng.normal(size=(DIM,))
        v = v/(np.linalg.norm(v)+1e-12)
        u.vmem.add_memento(
            mid=f"m{i}", 
            emb=v, 
            tags=["alpha"] if i%2==0 else ["beta"], 
            reliability=0.6 if i%3==0 else 0.5, 
            content=f"Content for memento {i} about {'alpha' if i%2==0 else 'beta'}.", 
            source="test"
        )
    
    _logger.info(f"Added {len(u.vmem.embeddings)} mementos")
    
    # Test HNSW index
    if HAS_HNSWLIB:
        _logger.info("Testing HNSW index...")
        q = rng.normal(size=(DIM,))
        q = q/(np.linalg.norm(q)+1e-12)
        
        # Traditional
        start = time.time()
        trad_results = u.vmem.retrieve(q, top_k=10, use_advanced=False, use_cache=False)
        trad_time = time.time() - start
        
        # Advanced
        start = time.time()
        hnsw_results = u.vmem.retrieve(q, top_k=10, use_advanced=True, use_cache=False)
        hnsw_time = time.time() - start
        
        _logger.info(f"Traditional: {trad_time*1000:.2f}ms, HNSW: {hnsw_time*1000:.2f}ms")
        _logger.info(f"Speedup: {trad_time/hnsw_time:.2f}x")
    
    # Test query cache
    _logger.info("Testing query cache...")
    q = rng.normal(size=(DIM,))
    q = q/(np.linalg.norm(q)+1e-12)
    
    start = time.time()
    u.vmem.retrieve(q, top_k=10, use_cache=True)
    first_time = time.time() - start
    
    start = time.time()
    u.vmem.retrieve(q, top_k=10, use_cache=True)
    cached_time = time.time() - start
    
    _logger.info(f"First: {first_time*1000:.2f}ms, Cached: {cached_time*1000:.2f}ms")
    _logger.info(f"Cache stats: {u.vmem.query_cache.stats()}")
    
    # Test blackboard
    _logger.info("Testing blackboard system...")
    bb = u.run(
        "please retrieve vector memory about alpha and also summarize this paragraph. " + 
        "Lorem ipsum dolor sit amet. " * 10, 
        iters=4
    )
    
    _logger.info(f"Blackboard executed {len(bb['executed_ops'])} operations")
    _logger.info(f"Cache hit rate: {bb['metrics']['cache_stats'].get('hit_rate', 0):.2%}")
    
    # Test save/load
    _logger.info("Testing save/load...")
    u.vmem.save_state("test_v2_state.json")
    loaded_mem = VectorMemory.load_state("test_v2_state.json")
    assert loaded_mem is not None, "Failed to load state"
    assert len(loaded_mem.embeddings) == len(u.vmem.embeddings), "Embedding count mismatch"
    _logger.info("Save/load test passed")
    
    # Cleanup
    import os
    for file in ["test_v2_state.json", "test_v2_state.hnsw"]:
        if os.path.exists(file):
            os.remove(file)
    
    # Run benchmark
    _logger.info("Running retrieval benchmark...")
    benchmark_results = u.benchmark_retrieval(num_queries=50, dataset_size=2000)
    _logger.info(f"Benchmark: {benchmark_results}")
    
    # Test retrieval quality
    _logger.info("Testing retrieval quality...")
    test_retrieval_quality(u)
    
    _logger.info("Enhanced UCS v2 smoke test completed successfully!")
    print("Enhanced UCS v2 OK")

# --- FastAPI endpoints ---
if HAS_FASTAPI and os.environ.get('RUN_API') == '1' and uvicorn:
    app = FastAPI(title="Enhanced UCS API v2", 
                  description="Production-ready Unified Cognition System with HNSW")
    ucs_instance = UnifiedCognitionSystem(use_advanced_search=True)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("UCS_CORS_ORIGINS", "").split(",") if os.getenv("UCS_CORS_ORIGINS") else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response
    
    # Rate limiting
    RUN_RATE_LIMIT = 5
    GENERAL_RATE_LIMIT = 20
    RATE_LIMIT_BUCKET = defaultdict(lambda: {"tokens": GENERAL_RATE_LIMIT, "last_refill": time.time()})
    
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        start_time = time.time()
        client_ip = request.client.host
        if TRUST_XFF:
            client_ip = request.headers.get("X-Forwarded-For", "").split(',')[0].strip() or client_ip

        bucket = RATE_LIMIT_BUCKET[client_ip]
        now = time.time()
        
        refill_rate = RUN_RATE_LIMIT / 60 if request.url.path == "/run_blackboard" else GENERAL_RATE_LIMIT / 60
        bucket["tokens"] = min(GENERAL_RATE_LIMIT, bucket["tokens"] + (now - bucket["last_refill"]) * refill_rate)
        bucket["last_refill"] = now
        
        if bucket["tokens"] < 1:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        bucket["tokens"] -= 1

        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Process-Time"] = str(time.time() - start_time)
        return response

    def get_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
        if REQUIRE_AUTH:
            if not API_KEY:
                raise HTTPException(status_code=500, detail="Server misconfiguration")
            if x_api_key != API_KEY:
                raise HTTPException(status_code=401, detail="Invalid API Key")
        return x_api_key

    class IngestItem(BaseModel):
        id: str
        text: str
        tags: List[str] = Field(default_factory=list)

    class EdgeItem(BaseModel):
        src: str
        dst: str
        weight: float = 1.0
        etype: str = "assoc"

    class FeedbackItem(BaseModel):
        id: str
        reward: float

    @app.get("/health")
    async def health():
        return {
            "ok": True,
            "numpy": HAS_NUMPY,
            "hnswlib": HAS_HNSWLIB,
            "telemetry_buffer": len(ucs_instance.telemetry.get("proposals", [])),
            "dim": ucs_instance._dim,
            "advanced_search": ucs_instance.use_advanced_search
        }

    @app.get("/health/detailed")
    async def detailed_health():
        """Comprehensive health check."""
        health = {
            "status": "healthy",
            "checks": {}
        }
        
        # Check memory integrity
        try:
            if ucs_instance.vmem:
                health["checks"]["memory"] = {
                    "status": "ok",
                    "mementos": len(ucs_instance.vmem.embeddings),
                    "cache_hit_rate": ucs_instance.vmem.query_cache.stats()["hit_rate"]
                }
        except Exception as e:
            health["status"] = "degraded"
            health["checks"]["memory"] = {"status": "error", "error": str(e)}
        
        # Check HNSW index
        try:
            if ucs_instance.vmem and ucs_instance.vmem.hnsw_index:
                test_query = np.random.rand(ucs_instance._dim)
                ucs_instance.vmem.hnsw_index.knn_query(test_query, k=1)
                health["checks"]["hnsw_index"] = {"status": "ok"}
        except Exception as e:
            health["status"] = "degraded"
            health["checks"]["hnsw_index"] = {"status": "error", "error": str(e)}
        
        # Check circuit breaker state
        if ucs_instance.vmem:
            cb_state = ucs_instance.vmem.circuit_breaker.state
            health["checks"]["circuit_breaker"] = {"status": cb_state}
            if cb_state == "open":
                health["status"] = "unhealthy"
        
        return health

    @app.get("/version")
    async def version():
        return {
            "name": "enhanced_ucs_v2",
            "spec": 3,
            "dim": DIM,
            "features": ["hnsw_index", "query_cache", "circuit_breaker", "isotonic_calibration"]
        }

    @app.post("/run_blackboard", dependencies=[Depends(get_api_key)])
    async def run_blackboard(data: dict):
        prompt = (data.get("prompt") or "").strip()
        actions = data.get("actions")
        if len(prompt) > MAX_PROMPT_LEN:
            raise HTTPException(status_code=413, detail="Prompt too large")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        result = ucs_instance.run(prompt, actions)
        return {"result": result}
        
    @app.get("/telemetry", dependencies=[Depends(get_api_key)])
    async def get_telemetry():
        return ucs_instance.telemetry
        
    @app.post("/replay_query", dependencies=[Depends(get_api_key)])
    async def replay_query(data: dict):
        prompt = (data.get("prompt") or "").strip()
        past_proposals = [p for p in ucs_instance.telemetry["proposals"]
                          if hashlib.sha256(prompt.encode()).hexdigest() == p["prompt_hash"]]
        
        return {"count": len(past_proposals), "replayed_proposals": past_proposals[-500:]}

    @app.post("/ingest", dependencies=[Depends(get_api_key)])
    async def ingest(items: List[IngestItem]):
        ucs_instance._ensure_memory()
        items = items[:MAX_INGEST_ITEMS]
        ingested_count = 0
        for it in items:
            mid = it.id
            if mid in ucs_instance.vmem.bloom_filter and mid in ucs_instance.vmem.embeddings:
                raise HTTPException(status_code=409, detail=f"memento '{mid}' already exists")
            txt = it.text
            tags = it.tags
            emb = ucs_instance._embed(txt)
            if ucs_instance.vmem.add_memento(mid, emb, tags=tags, reliability=0.6, content=txt):
                ingested_count += 1
        return {"ok": True, "count": ingested_count}

    @app.post("/ingest_edges", dependencies=[Depends(get_api_key)])
    async def ingest_edges(edges: List[EdgeItem]):
        ucs_instance._ensure_memory()
        ingested_count = 0
        for e in edges:
            if e.src == e.dst:
                continue
            if e.src in ucs_instance.vmem.graph and e.dst in ucs_instance.vmem.graph:
                ucs_instance.vmem.add_edge(e.src, e.dst, e.weight, e.etype)
                ingested_count += 1
        return {"ok": True, "count": ingested_count}

    @app.post("/feedback", dependencies=[Depends(get_api_key)])
    async def feedback(items: List[FeedbackItem]):
        ucs_instance._ensure_memory()
        rewards = {it.id: float(it.reward) for it in items}
        ucs_instance.vmem.feedback(rewards)
        return {"ok": True, "count": len(items)}

    @app.post("/save_state", dependencies=[Depends(get_api_key)])
    async def save_state(payload: dict):
        ucs_instance._ensure_memory()
        path = (payload.get("path") or "ucs_v2_state.json").strip()
        ucs_instance.vmem.save_state(path)
        return {"ok": True, "path": path}

    @app.post("/load_state", dependencies=[Depends(get_api_key)])
    async def load_state(payload: dict):
        path = (payload.get("path") or "ucs_v2_state.json").strip()
        mem = VectorMemory.load_state(path)
        if mem is None:
            raise HTTPException(status_code=404, detail="Failed to load state")
        with ucs_instance._lock:
            ucs_instance.vmem = mem
        return {"ok": True, "path": path, "dim": mem.dim, "count": len(mem.embeddings)}

    @app.post("/clear_memory", dependencies=[Depends(get_api_key)])
    async def clear_memory():
        """Danger: wipes all in-memory vectors/graph/scores."""
        ucs_instance._ensure_memory()
        with ucs_instance._lock:
            ucs_instance.vmem.embeddings.clear()
            ucs_instance.vmem.mementos.clear()
            ucs_instance.vmem.scores.clear()
            ucs_instance.vmem.graph.clear()
            ucs_instance.vmem.id_counter = 0
            ucs_instance.vmem._id_to_label.clear()
            ucs_instance.vmem._label_to_id.clear()
            ucs_instance.vmem._next_label = 0
            ucs_instance.vmem.bloom_filter = BloomFilter(size=100000, num_hashes=3)
            ucs_instance.vmem.query_cache.clear()
            if ucs_instance.vmem.use_advanced_search:
                ucs_instance.vmem._init_hnsw_index()
        return {"ok": True, "count": 0}

    @app.post("/benchmark", dependencies=[Depends(get_api_key)])
    async def benchmark_retrieval(data: dict = None):
        """Run retrieval benchmarks."""
        if data is None:
            data = {}
        num_queries = data.get("num_queries", 100)
        dataset_size = data.get("dataset_size", 10000)
        
        results = ucs_instance.benchmark_retrieval(num_queries, dataset_size)
        return {"ok": True, "benchmark_results": results}

    @app.get("/memory_stats", dependencies=[Depends(get_api_key)])
    async def memory_stats():
        """Get detailed memory statistics."""
        if ucs_instance.vmem is None:
            return {"ok": True, "memory_initialized": False}
        
        stats = {
            "ok": True,
            "memory_initialized": True,
            "total_mementos": len(ucs_instance.vmem.embeddings),
            "total_edges": sum(len(edges) for edges in ucs_instance.vmem.graph.values()),
            "dimension": ucs_instance.vmem.dim,
            "advanced_search_enabled": ucs_instance.vmem.use_advanced_search,
            "has_hnsw_index": ucs_instance.vmem.hnsw_index is not None,
            "insert_buffer_size": len(ucs_instance.vmem._insert_buffer),
            "cache_stats": ucs_instance.vmem.query_cache.stats(),
            "circuit_breaker_state": ucs_instance.vmem.circuit_breaker.state
        }
        
        # Tag statistics
        tag_counts = defaultdict(int)
        for mid, memento in ucs_instance.vmem.mementos.items():
            for tag in memento.get("tags", []):
                tag_counts[tag] += 1
        stats["top_tags"] = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return stats

    @app.get("/expert_reputations", dependencies=[Depends(get_api_key)])
    async def expert_reputations():
        """Get expert reputation scores."""
        reps = {}
        for name, rep in ucs_instance._reputations.items():
            reps[name] = {
                "observations": rep.n,
                "avg_reward": rep.reward_sum / rep.n if rep.n > 0 else 0,
                "ema_reward": rep.ema_reward,
                "ucb_score": ucs_instance._reputation_score(name),
                "last_seen": datetime.fromtimestamp(rep.last_seen).isoformat() if rep.last_seen > 0 else None
            }
        return {"ok": True, "reputations": reps}

    @app.post("/clear_cache", dependencies=[Depends(get_api_key)])
    async def clear_cache():
        """Clear query cache."""
        if ucs_instance.vmem:
            ucs_instance.vmem.query_cache.clear()
            return {"ok": True, "message": "Cache cleared"}
        return {"ok": False, "message": "Memory not initialized"}

# --- Testing and Benchmarking ---

def test_retrieval_quality(u: UnifiedCognitionSystem):
    u._ensure_memory()
    
    # Clear memory for this test
    u.vmem.embeddings.clear()
    u.vmem.mementos.clear()
    u.vmem.scores.clear()
    u.vmem.graph.clear()
    u.vmem.id_counter = 0
    u.vmem._id_to_label.clear()
    u.vmem._label_to_id.clear()
    u.vmem._next_label = 0
    u.vmem.bloom_filter = BloomFilter(size=100000, num_hashes=3)
    u.vmem.query_cache.clear()
    if u.vmem.use_advanced_search:
        u.vmem._init_hnsw_index()
    
    # Create synthetic ground truth
    test_queries = []
    topic_vectors = {}
    rng = np.random.default_rng(42)
    for topic in ["ml", "physics", "history"]:
        topic_vectors[topic] = rng.normal(size=u._dim)
        topic_vectors[topic] /= np.linalg.norm(topic_vectors[topic]) + 1e-12
        
        # Add 10 docs per topic
        topic_ids = []
        for i in range(10):
            mid = f"{topic}_{i}"
            noise = rng.normal(size=u._dim) * 0.05
            emb = topic_vectors[topic] + noise
            emb /= np.linalg.norm(emb) + 1e-12
            u.vmem.add_memento(mid, emb, tags=[topic])
            topic_ids.append(mid)
        
        # Query should retrieve docs from same topic
        query_emb = topic_vectors[topic].copy()
        test_queries.append((query_emb, topic_ids))
    
    metrics = u.vmem.evaluate_retrieval(test_queries)
    print(f"Retrieval quality: {metrics}")
    
    assert metrics["recall@10"] > 0.7, "Recall too low!"
    assert metrics["mrr"] > 0.5, "MRR too low!"

def run_enhanced_smoke_test():
    """Enhanced smoke test with all new features."""
    if not HAS_NUMPY:
        _logger.warning("NumPy not found. Skipping smoke test.")
        return
    
    _logger.info("--- Running Enhanced UCS v2 Smoke Test ---")
    
    u = UnifiedCognitionSystem(use_advanced_search=True)
    u._ensure_memory()

    _logger.info("Testing enhanced memory system...")
    rng = np.random.default_rng(42)
    
    # Add test data
    for i in range(2000):
        v = rng.normal(size=(DIM,))
        v = v/(np.linalg.norm(v)+1e-12)
        u.vmem.add_memento(
            mid=f"m{i}", 
            emb=v, 
            tags=["alpha"] if i%2==0 else ["beta"], 
            reliability=0.6 if i%3==0 else 0.5, 
            content=f"Content for memento {i} about {'alpha' if i%2==0 else 'beta'}.", 
            source="test"
        )
    
    _logger.info(f"Added {len(u.vmem.embeddings)} mementos")
    
    # Test HNSW index
    if HAS_HNSWLIB:
        _logger.info("Testing HNSW index...")
        q = rng.normal(size=(DIM,))
        q = q/(np.linalg.norm(q)+1e-12)
        
        # Traditional
        start = time.time()
        trad_results = u.vmem.retrieve(q, top_k=10, use_advanced=False, use_cache=False)
        trad_time = time.time() - start
        
        # Advanced
        start = time.time()
        hnsw_results = u.vmem.retrieve(q, top_k=10, use_advanced=True, use_cache=False)
        hnsw_time = time.time() - start
        
        _logger.info(f"Traditional: {trad_time*1000:.2f}ms, HNSW: {hnsw_time*1000:.2f}ms")
        _logger.info(f"Speedup: {trad_time/hnsw_time:.2f}x")
    
    # Test query cache
    _logger.info("Testing query cache...")
    q = rng.normal(size=(DIM,))
    q = q/(np.linalg.norm(q)+1e-12)
    
    start = time.time()
    u.vmem.retrieve(q, top_k=10, use_cache=True)
    first_time = time.time() - start
    
    start = time.time()
    u.vmem.retrieve(q, top_k=10, use_cache=True)
    cached_time = time.time() - start
    
    _logger.info(f"First: {first_time*1000:.2f}ms, Cached: {cached_time*1000:.2f}ms")
    _logger.info(f"Cache stats: {u.vmem.query_cache.stats()}")
    
    # Test blackboard
    _logger.info("Testing blackboard system...")
    bb = u.run(
        "please retrieve vector memory about alpha and also summarize this paragraph. " + 
        "Lorem ipsum dolor sit amet. " * 10, 
        iters=4
    )
    
    _logger.info(f"Blackboard executed {len(bb['executed_ops'])} operations")
    _logger.info(f"Cache hit rate: {bb['metrics']['cache_stats'].get('hit_rate', 0):.2%}")
    
    # Test save/load
    _logger.info("Testing save/load...")
    u.vmem.save_state("test_v2_state.json")
    loaded_mem = VectorMemory.load_state("test_v2_state.json")
    assert loaded_mem is not None, "Failed to load state"
    assert len(loaded_mem.embeddings) == len(u.vmem.embeddings), "Embedding count mismatch"
    _logger.info("Save/load test passed")
    
    # Cleanup
    import os
    for file in ["test_v2_state.json", "test_v2_state.hnsw"]:
        if os.path.exists(file):
            os.remove(file)
    
    # Run benchmark
    _logger.info("Running retrieval benchmark...")
    benchmark_results = u.benchmark_retrieval(num_queries=50, dataset_size=2000)
    _logger.info(f"Benchmark: {benchmark_results}")
    
    # Test retrieval quality
    _logger.info("Testing retrieval quality...")
    test_retrieval_quality(u)
    
    _logger.info("Enhanced UCS v2 smoke test completed successfully!")
    print("Enhanced UCS v2 OK")

if __name__ == "__main__":
    if HAS_FASTAPI and os.environ.get('RUN_API') == '1' and uvicorn:
        _logger.info("Starting Enhanced UCS v2 FastAPI server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        try:
            run_enhanced_smoke_test()
        except (AssertionError, RuntimeError) as e:
            print(str(e))
            sys.exit(1)

__all__ = [
    "VectorMemory", "UnifiedCognitionSystem", "ExpertManager", "ExpertProposal",
    "BloomFilter", "QueryCache", "CircuitBreaker", "IsotonicCalibrator",
    "normalize_vectors", "timed"
]

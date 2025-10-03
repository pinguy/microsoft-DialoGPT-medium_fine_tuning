# Enhanced UCS (Unified Cognition System) with Advanced Vector Search
# Merged from UCS.py and vector_search_benchmark.py
# - To plug real embeddings:
#    # from sentence_transformers import SentenceTransformer
#    # model = SentenceTransformer("all-MiniLM-L6-v2")
#    # u = UnifiedCognitionSystem(embed_fn=lambda t: model.encode([t])[0])
# - Smoke:    python enhanced_ucs.py
# - API:      RUN_API=1 UCS_REQUIRE_AUTH=0 UCS_TRUST_XFF=1 python enhanced_ucs.py
# - API with auth: RUN_API=1 UCS_API_KEY="mysecret" python enhanced_ucs.py
# - Example:  curl -X POST localhost:8000/run_blackboard -H 'content-type: application/json' \
#             -H 'X-API-Key: mysecret' -d '{"prompt":"please retrieve alpha and summarize this: ..."}'
# - Ingest:   curl -X POST localhost:8000/ingest -H 'content-type: application/json' \
#             -H 'X-API-Key: mysecret' -d '[{"id":"doc1","text":"alpha is fast","tags":["alpha"]}]'
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
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque, defaultdict, namedtuple
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

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
    np = DummyNumpy()

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
TELEMETRY_LIGHT = os.getenv("UCS_TELEMETRY_LIGHT", "0") == "1"
API_KEY = os.getenv("UCS_API_KEY")
REQUIRE_AUTH = os.getenv("UCS_REQUIRE_AUTH", "1") == "1"
TRUST_XFF = os.getenv("UCS_TRUST_XFF", "0") == "1"
UCS_LOG_PROPOSAL_CONTENT = os.getenv("UCS_LOG_PROPOSAL_CONTENT", "0") == "1"
UCS_REDACT_REGEX = os.getenv("UCS_REDACT_REGEX")

logging.basicConfig(level=LOG_LEVEL, format='[%(levelname)s] %(message)s')
_logger = logging.getLogger(__name__)

# --- Enhanced Vector Search Functions (from vector_search_benchmark.py) ---

def timed(fn, *args, **kwargs):
    """Decorator-like function to time another function."""
    start = time.time()
    result = fn(*args, **kwargs)
    end = time.time()
    return result, end - start

def normalize_vectors(X: np.ndarray) -> np.ndarray:
    """L2-normalize a batch of vectors."""
    if not HAS_NUMPY:
        return X
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def recall_at_k(truth: np.ndarray, pred: np.ndarray) -> float:
    """Calculates recall@k for a single query."""
    if not HAS_NUMPY or truth.size == 0:
        return 0.0
    return np.intersect1d(truth, pred).size / truth.size

def precision_at_k(truth: np.ndarray, pred: np.ndarray) -> float:
    """Calculates precision@k for a single query."""
    if not HAS_NUMPY or pred.size == 0:
        return 0.0
    return np.intersect1d(truth, pred).size / pred.size

def topk_cosine_batch(Q: np.ndarray, M: np.ndarray, k: int = 10, block: int = 200_000):
    """
    Q: (B, d) and M: (N, d) must be L2-normalized.
    Returns:
      idx: (B, k) int64
      dist: (B, k) float32  # 1 - cosine
    """
    if not HAS_NUMPY:
        return np.array([]), np.array([])
    
    B, N = Q.shape[0], M.shape[0]
    k = min(k, N)
    idx = np.full((B, k), -1, dtype=np.int64)
    scr = np.full((B, k), -np.inf, dtype=np.float32)

    for start in range(0, N, block):
        slab = M[start:start+block]
        S = slab @ Q.T
        for b in range(B):
            s = S[:, b]
            if s.size <= k:
                cand_idx = start + np.arange(s.size, dtype=np.int64)
                cand_scr = s
            else:
                part = np.argpartition(s, -k)[-k:]
                cand_idx = start + part
                cand_scr = s[part]
            merged_idx = np.concatenate([idx[b], cand_idx])
            merged_scr = np.concatenate([scr[b], cand_scr])
            keep = np.argpartition(merged_scr, -k)[-k:]
            idx[b] = merged_idx[keep]
            scr[b] = merged_scr[keep]

    order = np.argsort(-scr, axis=1)
    rows = np.arange(B)[:, None]
    return idx[rows, order], (1.0 - scr[rows, order])

# --- Advanced Index Classes ---

class PQIndex:
    """Product Quantization Index for efficient vector compression and search."""
    def __init__(self, m: int, ks: int, iters: int, d: int = None):
        self.m = m
        self.ks = ks
        self.iters = iters
        self.d = d
        self.dsub = d // m if d else None
        self.codes = None
        self.codebooks = None
        self.N = 0

    def fit(self, X: np.ndarray):
        if not HAS_NUMPY:
            raise RuntimeError("NumPy is required for PQIndex operations.")
        
        self.N, self.d = X.shape
        self.dsub = self.d // self.m
        
        # Initialize codebooks with k-means-like process
        self.codes = np.zeros((self.N, self.m), dtype=np.int32)
        self.codebooks = []
        
        for i in range(self.m):
            start_idx = i * self.dsub
            end_idx = (i + 1) * self.dsub
            sub_vectors = X[:, start_idx:end_idx]
            
            # Simple k-means initialization
            codebook = np.random.rand(self.ks, self.dsub).astype(np.float32)
            
            # Assign codes (simplified)
            for j in range(self.N):
                dists = np.linalg.norm(sub_vectors[j:j+1] - codebook, axis=1)
                self.codes[j, i] = np.argmin(dists)
            
            self.codebooks.append(codebook)

    def add(self, X: np.ndarray):
        if not HAS_NUMPY:
            return
        self.N += X.shape[0]

    def search(self, q: np.ndarray, k: int = 10):
        if not HAS_NUMPY or self.codes is None:
            return np.array([]), np.array([])
        
        # Asymmetric Distance Computation (ADC)
        distances = np.zeros(self.N)
        
        for i in range(self.m):
            start_idx = i * self.dsub
            end_idx = (i + 1) * self.dsub
            q_sub = q[start_idx:end_idx]
            
            # Distance from query sub-vector to all centroids
            centroid_dists = np.linalg.norm(self.codebooks[i] - q_sub, axis=1)
            
            # Add distances based on codes
            for j in range(self.N):
                distances[j] += centroid_dists[self.codes[j, i]]
        
        # Return top-k closest
        k = min(k, self.N)
        top_k_indices = np.argpartition(distances, k)[:k]
        top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]
        
        return top_k_indices, distances[top_k_indices]

class OPQ:
    """Optimized Product Quantization with learned rotation."""
    def __init__(self, pq: PQIndex, seed: int = 0, iters: int = 3):
        self.pq = pq
        self.R = None
        self.rng = np.random.default_rng(seed) if HAS_NUMPY else None
        self.iters = iters

    def _orthonormalize(self, W):
        if not HAS_NUMPY:
            return W
        # Orthonormal via SVD
        U, _, Vt = np.linalg.svd(W, full_matrices=False)
        return (U @ Vt).astype(np.float32)

    def fit(self, X: np.ndarray):
        if not HAS_NUMPY:
            return self
        
        X = X.astype(np.float32, copy=False)
        Xn = normalize_vectors(X)
        
        # Start with PCA components as initialization
        Xc = Xn - Xn.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xc[: min(50000, Xc.shape[0])], full_matrices=False)
        R = Vt
        
        for _ in range(self.iters):
            Xr = (Xn @ R.T).astype(np.float32)
            self.pq.fit(Xr)
            codes = self.pq.codes
            m, dsub = self.pq.m, self.pq.dsub
            Xq = np.empty_like(Xr)
            for s in range(m):
                C = self.pq.codebooks[s]
                Xq[:, s*dsub:(s+1)*dsub] = C[codes[:, s]]
            W = (Xq.T @ Xn)
            R = self._orthonormalize(W)
        
        self.R = R
        return self

    def add(self, X: np.ndarray):
        if not HAS_NUMPY:
            return
        X = X.astype(np.float32, copy=False)
        Xr = (normalize_vectors(X) @ self.R.T).astype(np.float32)
        self.pq.add(Xr)

    def search(self, q: np.ndarray, k: int = 10):
        if not HAS_NUMPY or self.R is None:
            return np.array([]), np.array([])
        
        q = q.astype(np.float32, copy=False)
        qr = (normalize_vectors(q.reshape(1, -1)) @ self.R.T).squeeze().astype(np.float32)
        return self.pq.search(qr, k=k)

def pq_save(pq: PQIndex, path: str):
    if not HAS_NUMPY:
        return
    np.savez_compressed(
        path,
        m=pq.m, ks=pq.ks, iters=pq.iters, d=pq.d, dsub=pq.dsub,
        codes=pq.codes,
        *[cb for cb in pq.codebooks]
    )

def pq_load(path: str) -> PQIndex:
    if not HAS_NUMPY:
        return PQIndex(8, 256, 10)
    
    z = np.load(path, allow_pickle=False)
    pq = PQIndex(int(z['m']), int(z['ks']), int(z['iters']))
    pq.d = int(z['d']); pq.dsub = int(z['dsub'])
    pq.codes = z['codes']
    cb = []
    for i in range(pq.m):
        cb.append(z[f'arr_{i}'].astype(np.float32))
    pq.codebooks = cb
    pq.N = pq.codes.shape[0]
    return pq

def opq_save(opq: OPQ, path: str):
    if not HAS_NUMPY:
        return
    np.savez_compressed(path, R=opq.R)

def opq_load(pq: PQIndex, path: str) -> OPQ:
    if not HAS_NUMPY:
        return OPQ(pq)
    
    z = np.load(path, allow_pickle=False)
    opq = OPQ(pq)
    opq.R = z['R'].astype(np.float32)
    return opq

# --- Expert proposal types ---
@dataclass
class ExpertProposal:
    """A formal proposal by an expert to perform an action."""
    action: str
    content: Any
    score: float = 0.5 # Base score before TrustFlow
    origin: str = "" # Expert name
    trust_score: float = 0.0 # Final TrustFlow score
    supporting_mementos: List[Tuple[str, float]] = field(default_factory=list)
    pre_calib_score: float = 0.0

# --- TrustFlow V2 Data Structures ---
ExpertReputation = namedtuple("ExpertReputation", ["n", "reward_sum", "reward_sq", "last_seen", "ema_reward"])

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
                    # Add supporting mementos for evidence scoring later
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
                # Patch: Pass the parent_system to the filter payload
                proposals = e["handler"]({"ctx": ctx, "proposals": proposals, "parent_system": self.parent_system}) or proposals
            except Exception as ex:
                logging.error(f"[filter] {n} failed: {ex}")
        
        return proposals

    def score_proposals(self, proposals: List[ExpertProposal], ctx: Dict[str, Any]) -> List[ExpertProposal]:
        """Applies a multi-axis TrustFlow scoring to a list of proposals."""
        
        # Calculate evidence metrics once
        ctx_tags = set(ctx.get("ctx_tags", []))
        
        for p in proposals:
            # Axis 1: Policy check (hard gate)
            ok, penalty = self.policy(p, ctx)
            if not ok:
                p.trust_score = -1.0 # Deny
                p.pre_calib_score = -1.0
                continue
            
            # Axis 2: Reputation (bandit-based)
            reputation_score = self.parent_system._reputation_score(p.origin)
            
            # Axis 3: Evidence
            evidence_score = 0.0
            expert_tags = set(self.experts.get(p.origin, {}).get("tags", []))
            tag_match_count = len(expert_tags.intersection(ctx_tags))
            evidence_score += 0.2 * (tag_match_count / len(expert_tags) if expert_tags else 0.0)
            
            # Retrieval-specific evidence - now uses the proposal's own mementos
            if p.action == "RETRIEVE" and p.supporting_mementos:
                retrieved_mementos = p.supporting_mementos
                # Retrieval density: how many results came from this op?
                retrieval_density = len(retrieved_mementos) / 10
                evidence_score += 0.5 * min(1.0, retrieval_density)
                
                # Memento quality
                if self.parent_system.vmem:
                    retrieved_ids = {mid for mid, _ in retrieved_mementos}
                    memento_scores = [self.parent_system.vmem.scores.get(mid, {}).get("b", 0) for mid in retrieved_ids]
                    if memento_scores:
                        evidence_score += 0.3 * (sum(s for s in memento_scores if s > 0) / len(memento_scores))

            # Axis 4: Recency & Risk
            recency_bonus = 0.05 if datetime.now().timestamp() - self.parent_system._reputations[p.origin].last_seen < 60*60*24 else 0.0
            
            risk_penalty = 0.0
            if p.action == "SUMMARIZE" and self.parent_system._novelty_score(p.supporting_mementos) < 0.1:
                # Penalize summarization with low-novelty evidence
                risk_penalty = 0.3
            
            # Axis 5: Uncertainty
            uncertainty_penalty = 0.0
            if p.supporting_mementos and HAS_NUMPY:
                scores = [m[1] for m in p.supporting_mementos]
                std_dev = np.std(scores) if len(scores) > 1 else 0
                uncertainty_penalty = min(0.5, std_dev)

            # Sum and calibrate
            pre_calib_score = (
                reputation_score * 0.4 +
                evidence_score * 0.3 +
                (p.score or 0.5) * 0.2 + # Base score
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

# --- Expert Handlers for the Blackboard ---

def retrieval_expert(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handler for the Retrieval Expert."""
    query = ctx.get("prompt", "")
    if "retrieve" in query.lower() or "find" in query.lower():
        return {"operation": "RETRIEVE", "query": query}
    return None

def summarization_expert(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handler for the Summarization Expert."""
    if len(ctx.get("prompt", "")) > 100 or "summarize" in ctx.get("prompt", "").lower():
        return {"operation": "SUMMARIZE", "text": ctx.get("prompt")}
    return None

def rehearsal_expert(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handler for the Rehearsal Expert."""
    if "rehearse" in ctx.get("prompt", "").lower() or "re-state" in ctx.get("prompt", "").lower():
        if "retrieval" in ctx and ctx["retrieval"]:
            most_cited_memento_id = ctx["retrieval"][0][0]
            # Check if the memento is "gold" and should not be rehearsed
            if ctx.get("parent_system").vmem.mementos[most_cited_memento_id].get("is_gold"):
                return None
            return {"operation": "REHEARSE", "memento_id": most_cited_memento_id}
    return None

def longform_expert(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handler for the Longform Expert."""
    # Check if a grounded retrieval and a summary have occurred
    history = [op["operation"] for op in ctx.get("history", [])]
    if "RETRIEVE" in history and "SUMMARIZE" in history:
        return {"operation": "GENERATE_LONGFORM", "source_mementos": ctx.get("retrieval", [])}
    return None

def noisy_expert(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A dummy expert for testing bandit-based reputation."""
    if random.random() < 0.5:
        return {"operation": "NOISY_OP", "content": "I am a noisy proposal"}
    return None

def meta_expert(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handler for the Meta Expert (map phase)."""
    return {**ctx, "mode": "System-2"} if len(ctx.get("prompt","")) > 100 else {**ctx, "mode":"System-1"}

def router_expert(payload: Dict[str, Any]) -> List[ExpertProposal]:
    """Handler for the Router Expert (filter phase)."""
    ctx, proposals = payload["ctx"], payload["proposals"]
    plan = ctx.get("plan", [])
    if not plan:
        return proposals
    # Only allow proposals that are part of the plan, plus control operations
    keep = [p for p in proposals if p.action in plan or p.action in ("SET_PLAN", "SET_MODE")]
    return keep or proposals

def set_plan_expert(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handler that sets a plan based on the prompt."""
    prompt = ctx.get("prompt", "")
    wants_retrieval = "retrieve" in prompt.lower() or "find" in prompt.lower()
    wants_summary   = len(prompt) > 100 or "summarize" in prompt.lower()
    if wants_retrieval and wants_summary:
        return {"operation":"SET_PLAN", "plan":["RETRIEVE","SUMMARIZE"]}
    if wants_retrieval: return {"operation":"SET_PLAN", "plan":["RETRIEVE"]}
    if wants_summary:   return {"operation":"SET_PLAN", "plan":["SUMMARIZE"]}
    return None

def self_attention_expert(payload: Dict[str, Any]) -> List[ExpertProposal]:
    """Filters and re-ranks proposals by boosting scores for experts with tags that align with the current context's tags."""
    ctx, proposals, parent_system = payload["ctx"], payload["proposals"], payload["parent_system"]
    expert_manager = parent_system.expert_manager
    ctx_tags = set(ctx.get("ctx_tags", []))
    
    if not ctx_tags:
        return proposals
        
    for p in proposals:
        expert_tags = set(expert_manager.experts.get(p.origin, {}).get("tags", []))
        tag_overlap = len(expert_tags.intersection(ctx_tags))
        if tag_overlap > 0:
            # Boost the base score for the TrustFlow calculation
            p.score += (tag_overlap / len(ctx_tags)) * 0.1 # Small boost

    return proposals

# --- Enhanced Memory and State Management ---
class VectorMemory:
    """Enhanced vector memory with advanced search capabilities."""
    def __init__(self, dim: int = DIM, use_advanced_search: bool = True):
        if not HAS_NUMPY:
            raise RuntimeError("NumPy is required for VectorMemory operations.")
        self.dim = dim
        self.graph = {} # adj list {mid: [neighbors]}
        self.embeddings = {} # {mid: vector}
        self.mementos = {} # {mid: metadata}
        self.scores = {} # {mid: score} - online logistic regression
        self.id_counter = 0
        self._lock = threading.RLock()
        
        # Advanced search components
        self.use_advanced_search = use_advanced_search
        self.pq_index = None
        self.opq_index = None
        self._embedding_matrix = None
        self._needs_rebuild = True

    def _safe_float(self, x, default=0.0):
        try:
            xf = float(x)
            if math.isnan(xf) or math.isinf(xf):
                return default
            return xf
        except Exception:
            return default

    def add_memento(self, mid: str, emb: np.ndarray, tags: List[str] = None, reliability: float = 0.5, content: str = "", is_gold: bool = False, source: str = "ingest"):
        """Adds a new memento to the memory."""
        with self._lock:
            if mid in self.embeddings:
                return False
            self.embeddings[mid] = emb
            self.mementos[mid] = {"tags": tags or [], "reliability": reliability, "content": content, "is_gold": is_gold, "source": source}
            self.scores[mid] = {"w": np.zeros(self.dim), "b": 0.0}
            self.graph[mid] = []
            self._needs_rebuild = True
            return True

    def update_memento_content(self, mid: str, content: str):
        """Updates the text content of an existing memento."""
        with self._lock:
            if mid in self.mementos:
                self.mementos[mid]["content"] = content
                return True
            return False

    def add_edge(self, mid1: str, mid2: str, weight: float = 1.0, etype: str = "assoc"):
        """Adds a weighted, directed edge between two mementos."""
        with self._lock:
            if mid1 in self.graph and mid2 in self.graph:
                self.graph[mid1].append({"to": mid2, "weight": weight, "type": etype})

    def _rebuild_advanced_indices(self):
        """Rebuilds the advanced search indices when needed."""
        if not self.use_advanced_search or not self._needs_rebuild or not self.embeddings:
            return
        
        with self._lock:
            try:
                # Build embedding matrix
                mids = list(self.embeddings.keys())
                embeddings_list = [self.embeddings[mid] for mid in mids]
                self._embedding_matrix = np.array(embeddings_list).astype(np.float32)
                self._embedding_matrix = normalize_vectors(self._embedding_matrix)
                
                # Build PQ index
                if len(embeddings_list) > 100:  # Only use PQ for larger datasets
                    self.pq_index = PQIndex(m=min(8, self.dim//4), ks=256, iters=5, d=self.dim)
                    self.pq_index.fit(self._embedding_matrix)
                    
                    # Build OPQ index for even better performance
                    if len(embeddings_list) > 1000:
                        pq_opq = PQIndex(m=min(8, self.dim//4), ks=256, iters=5, d=self.dim)
                        self.opq_index = OPQ(pq_opq, seed=42, iters=2)
                        self.opq_index.fit(self._embedding_matrix)
                
                self._needs_rebuild = False
                logging.info(f"Rebuilt advanced indices for {len(embeddings_list)} embeddings")
            
            except Exception as e:
                logging.error(f"Failed to rebuild advanced indices: {e}")
                self.pq_index = None
                self.opq_index = None

    def _logistic(self, x: np.ndarray, w: np.ndarray, b: float) -> float:
        """Helper for logistic function with clamping."""
        z = float(np.dot(w, x) + b)
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + np.exp(-z))

    def retrieve(self, query_emb: np.ndarray, top_k: int = 5, ann_K: int = 50, use_advanced: bool = None) -> List[Tuple[str, float]]:
        """Enhanced retrieval using both traditional and advanced search methods."""
        with self._lock:
            if use_advanced is None:
                use_advanced = self.use_advanced_search
            
            ann_K = max(1, min(int(ann_K), 1024))
            top_k = max(1, min(int(top_k), 100))

            if not self.embeddings:
                return []
            
            # Rebuild indices if needed
            if use_advanced:
                self._rebuild_advanced_indices()
            
            mids = list(self.embeddings.keys())
            
            # Choose search method based on dataset size and configuration
            if use_advanced and self._embedding_matrix is not None and len(mids) > 100:
                return self._advanced_retrieve(query_emb, top_k, ann_K, mids)
            else:
                return self._traditional_retrieve(query_emb, top_k, ann_K, mids)

    def _advanced_retrieve(self, query_emb: np.ndarray, top_k: int, ann_K: int, mids: List[str]) -> List[Tuple[str, float]]:
        """Advanced retrieval using PQ/OPQ indices."""
        try:
            # Normalize query
            query_norm = normalize_vectors(query_emb.reshape(1, -1)).squeeze()
            
            # Try OPQ first, then PQ, then fall back to exact search
            if self.opq_index is not None:
                indices, distances = self.opq_index.search(query_norm, k=min(top_k * 2, len(mids)))
                if len(indices) > 0:
                    # Re-rank with exact cosine similarity for top candidates
                    candidate_embeddings = self._embedding_matrix[indices]
                    exact_scores = candidate_embeddings @ query_norm
                    
                    # Sort by exact scores
                    sorted_indices = np.argsort(exact_scores)[::-1][:top_k]
                    final_indices = indices[sorted_indices]
                    final_scores = exact_scores[sorted_indices]
                    
                    return [(mids[idx], 1.0 - score) for idx, score in zip(final_indices, final_scores)]
            
            elif self.pq_index is not None:
                indices, distances = self.pq_index.search(query_norm, k=min(top_k * 2, len(mids)))
                if len(indices) > 0:
                    # Re-rank with exact cosine similarity
                    candidate_embeddings = self._embedding_matrix[indices]
                    exact_scores = candidate_embeddings @ query_norm
                    
                    sorted_indices = np.argsort(exact_scores)[::-1][:top_k]
                    final_indices = indices[sorted_indices]
                    final_scores = exact_scores[sorted_indices]
                    
                    return [(mids[idx], 1.0 - score) for idx, score in zip(final_indices, final_scores)]
            
            # Fall back to batch exact search
            query_batch = query_norm.reshape(1, -1)
            indices, distances = topk_cosine_batch(query_batch, self._embedding_matrix, k=top_k)
            
            if len(indices) > 0 and len(indices[0]) > 0:
                return [(mids[idx], dist) for idx, dist in zip(indices[0], distances[0])]
            
        except Exception as e:
            logging.error(f"Advanced retrieval failed: {e}")
        
        # Ultimate fallback
        return self._traditional_retrieve(query_emb, top_k, ann_K, mids)

    def _traditional_retrieve(self, query_emb: np.ndarray, top_k: int, ann_K: int, mids: List[str]) -> List[Tuple[str, float]]:
        """Traditional retrieval with activation spreading."""
        # This is the original UCS retrieval logic
        items = list(self.embeddings.items())
        mat = np.array([emb for _, emb in items])
        scores = mat.dot(query_emb)
        ranked_indices = np.argsort(scores)[::-1]
        
        # Activation spreading
        activation = defaultdict(float)
        for i in range(min(ann_K, len(ranked_indices))):
            mid = mids[ranked_indices[i]]
            activation[mid] += scores[ranked_indices[i]]
            # Spread activation to neighbors
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
        """Applies online logistic regression feedback with confidence calibration."""
        with self._lock:
            for mid, reward in rewards.items():
                if mid not in self.embeddings: continue
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

    def save_state(self, path: str):
        """Saves memory state to a file, with optional gzip compression."""
        data = {
            "schema": 2,  # Updated schema version
            "dim": self.dim,
            "use_advanced_search": self.use_advanced_search,
            "embeddings": {mid: emb.tolist() for mid, emb in self.embeddings.items()},
            "mementos": self.mementos,
            "scores": {mid: {"w": s["w"].tolist(), "b": s["b"]} for mid, s in self.scores.items()},
            "graph": self.graph,
            "id_counter": self.id_counter
        }
        
        # Save advanced indices if they exist
        if self.pq_index is not None:
            pq_path = path.replace('.json', '_pq.npz').replace('.gz', '_pq.npz')
            try:
                pq_save(self.pq_index, pq_path)
                data["pq_index_path"] = pq_path
            except Exception as e:
                logging.warning(f"Failed to save PQ index: {e}")
        
        if self.opq_index is not None:
            opq_path = path.replace('.json', '_opq.npz').replace('.gz', '_opq.npz')
            try:
                opq_save(self.opq_index, opq_path)
                data["opq_index_path"] = opq_path
            except Exception as e:
                logging.warning(f"Failed to save OPQ index: {e}")
        
        opener = gzip.open if HAS_GZIP and path.endswith(".gz") else open
        with opener(path, 'wt', encoding='utf-8') as f:
            json.dump(data, f)
        logging.info(f"Enhanced memory state saved to {path}")

    @classmethod
    def load_state(cls, path: str) -> Optional['VectorMemory']:
        """Loads memory state from a file, with optional gzip decompression."""
        try:
            opener = gzip.open if HAS_GZIP and path.endswith(".gz") else open
            with opener(path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            
            schema = data.get("schema", 0)
            if schema < 1:
                logging.warning(f"Old memory schema {schema}, attempting best-effort load")

            dim = data.get("dim", DIM)
            use_advanced = data.get("use_advanced_search", True)
            mem = cls(dim=dim, use_advanced_search=use_advanced)
            
            mem.embeddings = {mid: np.array(emb) for mid, emb in data["embeddings"].items()}
            mem.mementos = data["mementos"]
            mem.scores = {mid: {"w": np.array(s["w"]), "b": s["b"]} for mid, s in data["scores"].items()}
            mem.graph = data["graph"]
            mem.id_counter = data["id_counter"]
            
            # Load advanced indices if available
            if "pq_index_path" in data:
                try:
                    mem.pq_index = pq_load(data["pq_index_path"])
                    logging.info("Loaded PQ index")
                except Exception as e:
                    logging.warning(f"Failed to load PQ index: {e}")
            
            if "opq_index_path" in data and mem.pq_index is not None:
                try:
                    mem.opq_index = opq_load(mem.pq_index, data["opq_index_path"])
                    logging.info("Loaded OPQ index")
                except Exception as e:
                    logging.warning(f"Failed to load OPQ index: {e}")
            
            mem._needs_rebuild = False
            logging.info(f"Enhanced memory state loaded from {path}")
            return mem
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logging.error(f"Failed to load memory state: {e}")
            return None

class UnifiedCognitionSystem:
    """The main UCS class, coordinating memory and blackboard with enhanced vector search."""
    def __init__(self, mem: Optional[VectorMemory] = None, dim: int = DIM, embed_fn=None, use_advanced_search: bool = True):
        self._dim = dim
        self.vmem = mem  # may be None until first retrieval/ingest
        self.embed_fn = embed_fn or self._embed_query
        self.expert_manager = ExpertManager(parent_system=self)
        self.telemetry = defaultdict(list)
        self._is_init = False
        self._session_memory = defaultdict(list)
        self._telemetry_cap = 50_000
        self._lock = threading.RLock()
        self._reputations = defaultdict(lambda: ExpertReputation(n=0, reward_sum=0, reward_sq=0, last_seen=0, ema_reward=0.5))
        self.use_advanced_search = use_advanced_search

    def _reputation_score(self, expert_name: str, c: float = 0.5) -> float:
        """Calculates a bandit-based reputation score for an expert using UCB1."""
        rep = self._reputations[expert_name]
        if rep.n == 0:
            return rep.ema_reward + 0.1 # Exploration bonus
        
        # Upper Confidence Bound (UCB1)
        exploration_bonus = c * math.sqrt(math.log(sum(r.n for r in self._reputations.values()) + 1) / rep.n)
        return rep.reward_sum / rep.n + exploration_bonus

    def _update_reputation(self, expert_name: str, reward: float):
        """Updates an expert's reputation with a new reward."""
        with self._lock:
            rep = self._reputations[expert_name]
            n = rep.n + 1
            reward_sum = rep.reward_sum + reward
            reward_sq = rep.reward_sq + reward**2
            last_seen = time.time()
            # Simple Exponential Moving Average (EMA)
            alpha = 0.1
            ema_reward = alpha * reward + (1-alpha) * rep.ema_reward
            self._reputations[expert_name] = ExpertReputation(n, reward_sum, reward_sq, last_seen, ema_reward)
    
    def _calibrate(self, raw_score: float) -> float:
        """A placeholder for a future calibration layer (e.g., Platt scaling)."""
        return max(0.0, min(1.0, raw_score))

    def _ensure_memory(self):
        """Initializes vector memory if it hasn't been already."""
        with self._lock:
            if self.vmem is None:
                if not HAS_NUMPY:
                    raise RuntimeError("Vector memory unavailable without NumPy")
                self.vmem = VectorMemory(self._dim, use_advanced_search=self.use_advanced_search)
            assert self._dim == self.vmem.dim, "Dimension mismatch: UCS was initialized with a different dimension than the loaded memory."

    def _sanitize_prompt(self, prompt: str) -> Tuple[str, bool]:
        """Strips common expert-override strings from a prompt to prevent abuse."""
        original_prompt = prompt
        # Regex to find and replace expert-override strings, case-insensitive
        sanitization_regex = re.compile(r'\b(set_plan|set_mode)\b', re.IGNORECASE)
        sanitized_prompt = sanitization_regex.sub('', prompt).strip()
        
        was_sanitized = sanitized_prompt != original_prompt
        return sanitized_prompt, was_sanitized

    def initialize_experts(self):
        """Initializes and registers experts. Called once."""
        if self._is_init:
            return
        
        # System experts
        self.expert_manager.register_expert(
            name="RetrievalExpert",
            handler=retrieval_expert,
            expertise_tags=["memory", "retrieval"],
            phase="propose"
        )
        self.expert_manager.register_expert(
            name="SummarizationExpert",
            handler=summarization_expert,
            expertise_tags=["language", "synthesis"],
            phase="propose"
        )
        self.expert_manager.register_expert(
            name="RehearsalExpert",
            handler=rehearsal_expert,
            expertise_tags=["memory", "synthesis"],
            phase="propose"
        )
        self.expert_manager.register_expert(
            name="LongformExpert",
            handler=longform_expert,
            expertise_tags=["language", "synthesis"],
            phase="propose"
        )
        self.expert_manager.register_expert(
            name="NoisyExpert",
            handler=noisy_expert,
            expertise_tags=["test", "noise"],
            phase="propose"
        )
        self.expert_manager.register_expert(
            name="MetaExpert",
            handler=meta_expert,
            expertise_tags=["control"],
            phase="map"
        )
        self.expert_manager.register_expert(
            name="SetPlanExpert",
            handler=set_plan_expert,
            expertise_tags=["control", "routing"],
            phase="propose"
        )
        self.expert_manager.register_expert(
            name="RouterExpert",
            handler=router_expert,
            expertise_tags=["control", "routing"],
            phase="filter"
        )
        self.expert_manager.register_expert(
            name="SelfAttentionExpert",
            handler=self_attention_expert,
            expertise_tags=["control", "attention"],
            phase="filter"
        )
        self._is_init = True

    def _embed_query(self, text: str):
        # deterministic embeddings; works with or without NumPy
        h = hashlib.sha256(text.encode()).digest()
        if not HAS_NUMPY:
            # use configured dimension even if self.vmem is None
            vals = [h[i % len(h)] for i in range(self._dim)]
            s = sum(vals) or 1
            return [v / s for v in vals]
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        v = rng.normal(size=(self._dim if self.vmem is None else self.vmem.dim,))
        v = v / (np.linalg.norm(v) + 1e-12)
        return v
    
    def _embed(self, text: str):
        return self.embed_fn(text)

    def _quick_summarize(self, text: str, max_sents: int = 3) -> str:
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return " ".join(sents[:max_sents])

    def _novelty_score(self, results):
        vals = [self.vmem._safe_float(score, 0.0) for _, score in (results or [])]
        if len(vals) < 2: return 0.0
        if not HAS_NUMPY:
            mu = sum(vals)/len(vals)
            var = sum((x-mu)**2 for x in vals)/len(vals)
            return var ** 0.5
        mu = sum(vals)/len(vals)
        var = sum((x-mu)**2 for x in vals)/len(vals)
        return var ** 0.5

    def benchmark_retrieval(self, num_queries: int = 100, dataset_size: int = 10000):
        """Benchmark the retrieval performance with different methods."""
        if not HAS_NUMPY:
            logging.warning("Benchmarking requires NumPy")
            return {}
        
        self._ensure_memory()
        
        # Add test data if memory is empty
        if len(self.vmem.embeddings) < dataset_size:
            logging.info(f"Adding {dataset_size} test embeddings for benchmark")
            for i in range(dataset_size):
                emb = self._embed(f"test_document_{i}")
                self.vmem.add_memento(f"test_{i}", emb, content=f"Test document {i}", source="benchmark")
        
        # Generate test queries
        queries = [self._embed(f"query_{i}") for i in range(num_queries)]
        
        results = {}
        
        # Benchmark traditional retrieval
        start_time = time.time()
        for q in queries:
            self.vmem.retrieve(q, top_k=10, use_advanced=False)
        traditional_time = (time.time() - start_time) / num_queries
        results["traditional_avg_ms"] = traditional_time * 1000
        
        # Benchmark advanced retrieval
        if self.use_advanced_search:
            start_time = time.time()
            for q in queries:
                self.vmem.retrieve(q, top_k=10, use_advanced=True)
            advanced_time = (time.time() - start_time) / num_queries
            results["advanced_avg_ms"] = advanced_time * 1000
            results["speedup"] = traditional_time / advanced_time if advanced_time > 0 else 0
        
        results["dataset_size"] = len(self.vmem.embeddings)
        results["has_pq"] = self.vmem.pq_index is not None
        results["has_opq"] = self.vmem.opq_index is not None
        
        logging.info(f"Retrieval benchmark results: {results}")
        return results
    
    def run(self, prompt: str, actions: Optional[List[str]] = None, iters: int = 5, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Runs the blackboard loop."""
        with self._lock:
            self.initialize_experts()

            # Apply prompt sanitization
            sanitized_prompt, was_sanitized = self._sanitize_prompt(prompt)
            prompt = sanitized_prompt
            
            if len(prompt) > MAX_PROMPT_LEN:
                return {"prompt": prompt[:256] + "...", "error": "Prompt too large", "history": [], "executed_ops": []}
            
            session_id = session_id or str(uuid.uuid4())
            self._session_memory[session_id].append({"t": time.time(), "prompt": prompt})
            self._session_memory[session_id] = [e for e in self._session_memory[session_id] if time.time()-e["t"] < 900]

            blackboard = {"prompt": prompt, "history": [], "session_id": session_id, "executed_ops": [], "plan": [], "audit_sanitized": was_sanitized, "parent_system": self}
            blackboard["session_recent"] = len(self._session_memory[session_id])
            
            if actions:
                blackboard["plan"] = actions
                
            deadline = time.time() + float(os.getenv("UCS_RUN_DEADLINE_S", "5"))
            timed_out = False

            for i in range(iters):
                if time.time() > deadline:
                    logging.warning("Run deadline exceeded.")
                    blackboard["history"].append({"operation":"DEADLINE_EXCEEDED"})
                    timed_out = True
                    break
                
                # REFRESH CONTEXT TAGS HERE
                ctx_tags = set()
                for item in blackboard.get("history", []):
                    if isinstance(item, dict) and "retrieval" in item:
                        if self.vmem:
                            for mid, _ in item["retrieval"]:
                                ctx_tags.update(self.vmem.mementos.get(mid, {}).get("tags", []))
                blackboard["ctx_tags"] = sorted(list(ctx_tags))

                logging.debug(f"--- Iteration {i+1}/{iters} ---")
                
                # 1. Experts propose actions and filter proposals
                proposals = self.expert_manager.propose(blackboard)
                logging.debug(f"Proposals gathered from {len(proposals)} experts.")

                # 2. Score proposals with TrustFlow
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
                
                # 3. Select the winning proposal
                winning_proposal = max(scored_proposals, key=lambda p: p.trust_score, default=None)
                
                if winning_proposal and winning_proposal.trust_score > 0.01:
                    logging.info(f"Winning proposal (score: {winning_proposal.trust_score:.2f}): {winning_proposal.content}")
                    
                    # Execute the winning operation
                    op = winning_proposal.action
                    payload = winning_proposal.content
                    blackboard["executed_ops"].append(op)
                    
                    if op == "SET_PLAN":
                        blackboard["plan"] = payload.get("plan", [])
                        blackboard["history"].append({"operation": "SET_PLAN", "plan": blackboard["plan"]})
                        logging.info(f"Plan set: {blackboard['plan']}")

                    elif op == "RETRIEVE":
                        try:
                            self._ensure_memory()
                            qv = self._embed(blackboard["prompt"])
                            results = self.vmem.retrieve(qv, top_k=5, ann_K=64)
                            blackboard["retrieval"] = results
                            blackboard["history"].append({"operation":"RETRIEVE", "retrieval": results})
                            
                            nov = self._novelty_score(results)
                            if nov > 0.25:
                                logging.warning("Black Swan signal: high novelty/dispersion")
                            
                            self._update_reputation(winning_proposal.origin, reward=1.0)
                        except RuntimeError as e:
                            logging.error(f"Retrieval failed: {e}")
                            blackboard["history"].append({"operation":"RETRIEVE_FAILED", "error": str(e)})
                            self._update_reputation(winning_proposal.origin, reward=0.0)

                    elif op == "SUMMARIZE":
                        txt = payload.get("text") or blackboard.get("prompt", "")
                        summary = self._quick_summarize(txt)
                        blackboard["summary"] = summary
                        blackboard["history"].append({"operation":"SUMMARIZE", "summary": summary})
                        self._update_reputation(winning_proposal.origin, reward=1.0)
                    
                    elif op == "GENERATE_LONGFORM":
                        # This is a mock for a full LLM call
                        retrieved_content = " ".join([self.vmem.mementos.get(mid, {}).get("content", "") for mid, _ in payload.get("source_mementos", [])])
                        longform_text = f"Generating long-form content based on: {self._quick_summarize(retrieved_content, 5)}"
                        blackboard["longform_output"] = longform_text
                        blackboard["history"].append({"operation": "GENERATE_LONGFORM", "output": longform_text})
                        self._update_reputation(winning_proposal.origin, reward=1.0)

                    elif op == "REHEARSE":
                        try:
                            self._ensure_memory()
                            memento_id = payload.get("memento_id")
                            if memento_id and memento_id in self.vmem.mementos:
                                original_content = self.vmem.mementos[memento_id]["content"]
                                new_summary = self._quick_summarize(original_content)
                                self.vmem.update_memento_content(memento_id, new_summary)
                                blackboard["history"].append({"operation":"REHEARSE", "memento_id": memento_id, "summary": new_summary})
                                self._update_reputation(winning_proposal.origin, reward=1.0)
                            else:
                                blackboard["history"].append({"operation":"REHEARSE_FAILED", "error": "Memento not found"})
                                self._update_reputation(winning_proposal.origin, reward=0.0)
                        except Exception as e:
                            logging.error(f"Rehearsal failed: {e}")
                            blackboard["history"].append({"operation":"REHEARSE_FAILED", "error": str(e)})
                            self._update_reputation(winning_proposal.origin, reward=0.0)
                    
                    elif op == "NOISY_OP":
                        self._update_reputation(winning_proposal.origin, reward=random.choice([0.0, 1.0]))
                        blackboard["history"].append({"operation": "NOISY_OP", "result": "done"})

                    elif op == "SET_MODE":
                        blackboard["mode"] = payload.get("mode", "System-1")
                        blackboard["history"].append({"operation": "SET_MODE", "mode": blackboard["mode"]})
                    
                    else:
                        blackboard["history"].append(payload)

                else:
                    logging.warning("No viable proposal found.")
                    break
                    
            # Add final metrics
            blackboard["metrics"] = {
                "iters": i+1,
                "mode": blackboard.get("mode","System-1"),
                "telemetry_buffer": len(self.telemetry["proposals"]),
                "retrieved": len(blackboard.get("retrieval", [])),
                "timed_out": timed_out,
                "advanced_search": self.use_advanced_search
            }
            
            return blackboard

# --- FastAPI endpoints for Enhanced UCS ---
if HAS_FASTAPI and os.environ.get('RUN_API') == '1' and uvicorn:
    app = FastAPI(title="Enhanced UCS API", description="Unified Cognition System with Advanced Vector Search")
    ucs_instance = UnifiedCognitionSystem(use_advanced_search=True)
    
    # Secure CORS configuration by default
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
        return response
    
    # Simple rate limiter (token bucket)
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
        
        # Determine the refill rate based on the path
        refill_rate = RUN_RATE_LIMIT / 60 if request.url.path == "/run_blackboard" else GENERAL_RATE_LIMIT / 60
        bucket["tokens"] = min(GENERAL_RATE_LIMIT, bucket["tokens"] + (now - bucket["last_refill"]) * refill_rate)
        bucket["last_refill"] = now
        
        if bucket["tokens"] < 1:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        bucket["tokens"] -= 1

        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    def get_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
        if REQUIRE_AUTH:
            if not API_KEY:
                raise HTTPException(status_code=500, detail="Server misconfiguration: API_KEY is required but not set.")
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
            "telemetry_buffer": len(ucs_instance.telemetry.get("proposals", [])),
            "dim": ucs_instance._dim,
            "advanced_search": ucs_instance.use_advanced_search
        }

    @app.get("/version")
    async def version():
        return {"name": "enhanced_ucs", "spec": 2, "dim": DIM, "features": ["advanced_search", "pq_index", "opq_index"]}

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
            if mid in ucs_instance.vmem.embeddings:
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
        path = (payload.get("path") or "enhanced_vmem_state.json").strip()
        ucs_instance.vmem.save_state(path)
        return {"ok": True, "path": path}

    @app.post("/load_state", dependencies=[Depends(get_api_key)])
    async def load_state(payload: dict):
        path = (payload.get("path") or "enhanced_vmem_state.json").strip()
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
            ucs_instance.vmem.pq_index = None
            ucs_instance.vmem.opq_index = None
            ucs_instance.vmem._needs_rebuild = True
        return {"ok": True, "count": 0}

    @app.post("/benchmark", dependencies=[Depends(get_api_key)])
    async def benchmark_retrieval(data: dict = None):
        """Run retrieval benchmarks comparing traditional vs advanced search."""
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
            "has_pq_index": ucs_instance.vmem.pq_index is not None,
            "has_opq_index": ucs_instance.vmem.opq_index is not None,
            "needs_rebuild": ucs_instance.vmem._needs_rebuild
        }
        
        # Add tag statistics
        tag_counts = defaultdict(int)
        for mid, memento in ucs_instance.vmem.mementos.items():
            for tag in memento.get("tags", []):
                tag_counts[tag] += 1
        stats["top_tags"] = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return stats

# --- Enhanced Testing and Benchmarking ---

def _benchmark_retrieve_enhanced(u: UnifiedCognitionSystem):
    """Enhanced benchmark that tests both traditional and advanced retrieval."""
    if not HAS_NUMPY:
        print("Benchmark requires NumPy")
        return
    
    t0 = time.perf_counter()
    rng = np.random.default_rng(7)
    q = rng.normal(size=(DIM,)); q /= (np.linalg.norm(q)+1e-12)
    
    # Traditional retrieval
    for _ in range(50):
        u.vmem.retrieve(q, top_k=5, ann_K=64, use_advanced=False)
    dt_traditional = time.perf_counter()-t0
    
    # Advanced retrieval
    t0 = time.perf_counter()
    for _ in range(50):
        u.vmem.retrieve(q, top_k=5, ann_K=64, use_advanced=True)
    dt_advanced = time.perf_counter()-t0
    
    print(f"Traditional retrieve x50 in {dt_traditional:.3f}s")
    print(f"Advanced retrieve x50 in {dt_advanced:.3f}s")
    if dt_advanced > 0:
        print(f"Speedup: {dt_traditional/dt_advanced:.2f}x")

def bench_one_enhanced(name, fn, *args, **kwargs):
    """Enhanced benchmark function with better error handling."""
    try:
        result, t = timed(fn, *args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 2:
            idx, dist = result[0], result[1]
            if hasattr(idx, '__len__') and len(idx) > 0:
                print(f"{name:25s}  time={t*1e3:7.1f} ms  results={len(idx)}")
            else:
                print(f"{name:25s}  time={t*1e3:7.1f} ms  results=0")
        else:
            print(f"{name:25s}  time={t*1e3:7.1f} ms  completed")
        return result
    except Exception as e:
        print(f"{name:25s}  FAILED: {e}")
        return None, 0

def test_enhanced_features():
    """Test the enhanced features of the merged system."""
    if not HAS_NUMPY:
        print("Enhanced features require NumPy")
        return
    
    print("Testing enhanced UCS features...")
    
    # Test advanced search initialization
    u = UnifiedCognitionSystem(use_advanced_search=True)
    u._ensure_memory()
    
    # Add test data
    for i in range(1000):
        emb = u._embed(f"test document {i}")
        u.vmem.add_memento(f"test_{i}", emb, tags=[f"tag_{i%10}"], content=f"Test document {i}")
    
    print(f"Added {len(u.vmem.embeddings)} test mementos")
    
    # Test retrieval with both methods
    query_emb = u._embed("test query")
    
    traditional_results = u.vmem.retrieve(query_emb, top_k=10, use_advanced=False)
    advanced_results = u.vmem.retrieve(query_emb, top_k=10, use_advanced=True)
    
    print(f"Traditional retrieval: {len(traditional_results)} results")
    print(f"Advanced retrieval: {len(advanced_results)} results")
    
    # Test PQ/OPQ indices
    print(f"PQ Index created: {u.vmem.pq_index is not None}")
    print(f"OPQ Index created: {u.vmem.opq_index is not None}")
    
    # Test save/load with enhanced features
    u.vmem.save_state("test_enhanced_state.json")
    loaded_mem = VectorMemory.load_state("test_enhanced_state.json")
    print(f"Save/load test: {'PASSED' if loaded_mem is not None else 'FAILED'}")
    
    # Cleanup
    import os
    for file in ["test_enhanced_state.json", "test_enhanced_state_pq.npz", "test_enhanced_state_opq.npz"]:
        if os.path.exists(file):
            os.remove(file)
    
    print("Enhanced features test completed")

def run_enhanced_smoke_test():
    """Enhanced smoke test that includes vector search benchmarking."""
    if not HAS_NUMPY:
        _logger.warning("NumPy not found. Skipping enhanced smoke test.")
        return
    
    np.random.seed(123)
    _logger.info("--- Running Enhanced UCS Smoke Test ---")
    
    # Test enhanced UCS
    u = UnifiedCognitionSystem(use_advanced_search=True)
    u._ensure_memory()

    _logger.info("Testing enhanced memory system...")
    rng = np.random.default_rng(42)
    
    # Add more test data for better benchmarking
    for i in range(5000):
        v = rng.normal(size=(DIM,)); v = v/(np.linalg.norm(v)+1e-12)
        u.vmem.add_memento(
            mid=f"m{i}", 
            emb=v, 
            tags=["alpha"] if i%2==0 else ["beta"], 
            reliability=0.6 if i%3==0 else 0.5, 
            content=f"This is the content for memento {i}. It talks about {'alpha' if i%2==0 else 'beta'}.", 
            source="test_ingest"
        )
    
    # Add random edges
    ids = list(u.vmem.graph.keys())
    for _ in range(1000):
        a, b = rng.choice(ids, size=2, replace=False)
        u.vmem.add_edge(a, b, weight=float(rng.uniform(0.3, 1.0)), etype=random.choice(["term","cite","people"]))

    # Enhanced benchmarking
    _benchmark_retrieve_enhanced(u)
    
    # Test vector search benchmarking
    print("\n--- Enhanced Vector Search Benchmark ---")
    q = rng.normal(size=(DIM,)); q = q/(np.linalg.norm(q)+1e-12)
    
    bench_one_enhanced("Traditional Retrieve", u.vmem.retrieve, q, 10, 64, False)
    bench_one_enhanced("Advanced Retrieve", u.vmem.retrieve, q, 10, 64, True)
    
    if u.vmem.pq_index:
        bench_one_enhanced("PQ Index Search", u.vmem.pq_index.search, q, 10)
    
    if u.vmem.opq_index:
        bench_one_enhanced("OPQ Index Search", u.vmem.opq_index.search, q, 10)
    
    # Test enhanced blackboard
    _logger.info("Testing enhanced blackboard system...")
    bb = u.run("please retrieve vector memory about alpha and also summarize this paragraph. " + "Lorem ipsum "+"dolor sit amet, consectetur adipiscing elit. "*5, iters=4)
    print("Enhanced blackboard state:", {k: v for k, v in bb.items() if k not in ["parent_system"]})
    
    # Run enhanced feature tests
    test_enhanced_features()
    
    # Run benchmark
    benchmark_results = u.benchmark_retrieval(num_queries=50, dataset_size=1000)
    print(f"Benchmark results: {benchmark_results}")
    
    _logger.info("Enhanced smoke test completed.")
    print("Enhanced UCS OK")

if __name__ == "__main__":
    if HAS_FASTAPI and os.environ.get('RUN_API') == '1' and uvicorn:
        _logger.info("Starting Enhanced UCS FastAPI server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        try:
            run_enhanced_smoke_test()
        except (AssertionError, RuntimeError) as e:
            print(str(e))
            sys.exit(1)

__all__ = [
    "VectorMemory", "UnifiedCognitionSystem", "ExpertManager", "ExpertProposal",
    "PQIndex", "OPQ", "topk_cosine_batch", "normalize_vectors", "pq_save", "pq_load", 
    "opq_save", "opq_load", "timed", "recall_at_k", "precision_at_k"
]

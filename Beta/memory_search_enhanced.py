#!/usr/bin/env python3
"""
Memory Explorer — Enhanced PLUS (streaming + metadata filters + performance improvements)

Key improvements over original:
- Fixed encoding issues (proper em-dash and ellipsis characters)
- Optimized conversation context retrieval with lazy indexing
- Consolidated metadata path extraction logic
- Fixed FuzzyMatch interface consistency
- Improved quality filtering with configurable fallback strategies
- Better diagnostic feedback for filter exclusions
- Enhanced error handling and validation
- Lower memory footprint with lazy loading
- Smarter deduplication suggestions

Examples:
  # Fuzzy search with 2-character tolerance
  python memory_search_enhanced.py memory.jsonl.gz --query "artifical intelligence" --fuzzy 2

  # Export results to JSON
  python memory_search_enhanced.py memory.jsonl.gz --query "machine learning" --export results.json

  # Filter by quality score and sort by it
  python memory_search_enhanced.py memory.jsonl.gz --query "AI" --min-quality 0.8 --sort quality

  # Advanced analytics
  python memory_search_enhanced.py memory.jsonl.gz --stats --author "assistant"
"""

from __future__ import annotations
import argparse
import gzip
import json
import os
import re
import sys
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Callable, Set, Union

try:
    import readline
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False

# ---------- Constants and Configuration ----------

CONVERSATION_ID_PATHS = [
    'source_metadata.user_msg.conversation_id',
    'source_metadata.assistant_msg.conversation_id',
    'metadata.conversation_id'
]

AUTHOR_PATHS = [
    'source_metadata.user_msg.author',
    'source_metadata.assistant_msg.author',
    'metadata.author'
]

NESTED_TS_PATHS = [
    'timestamp', 'time', 'ts', 'created_at', 'created', 'date',
    'source_metadata.user_msg.timestamp',
    'source_metadata.assistant_msg.timestamp', 
    'metadata.timestamp',
]

HIGHLIGHT_SCHEMES = {
    'ansi': ('\033[43m\033[30m', '\033[0m'),
    'html': ('<mark>', '</mark>'),
    'markdown': ('**', '**'),
    'none': ('', ''),
}

# ---------- Utility Functions ----------

def eprint(*args: Any, **kwargs: Any) -> None:
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)

def progress_bar(current: int, total: int, width: int = 50) -> str:
    """Generate a text-based progress bar."""
    if total == 0:
        return "[" + "=" * width + "]"
    filled = int(width * current / total)
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({100*current/total:.1f}%)"

def get_in(d: Any, path: str) -> Any:
    """Navigate nested dictionary using dot notation path."""
    cur = d
    for part in path.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

def get_first_in(obj: Dict[str, Any], paths: List[str]) -> Any:
    """Get the first non-None value from a list of paths."""
    for path in paths:
        value = get_in(obj, path)
        if value is not None:
            return value
    return None

def squeeze_whitespace(s: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    return re.sub(r"\s+", " ", s).strip()

class Timer:
    """Simple timer for performance measurement."""
    def __init__(self):
        self.start_time = time.time()
    
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    def reset(self) -> float:
        elapsed = self.elapsed()
        self.start_time = time.time()
        return elapsed

# ---------- String Matching Utilities ----------

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def fuzzy_match(text: str, pattern: str, threshold: float = 0.8) -> Optional[Tuple[int, int, float]]:
    """
    Find best fuzzy match of pattern in text.
    Returns (start, end, ratio) or None.
    """
    if not pattern:
        return None
    
    text_lower = text.lower()
    pattern_lower = pattern.lower()
    
    # Try exact substring first
    if pattern_lower in text_lower:
        idx = text_lower.index(pattern_lower)
        return (idx, idx + len(pattern), 1.0)
    
    # Sliding window fuzzy matching
    best_ratio = 0.0
    best_match = None
    pattern_len = len(pattern)
    
    for i in range(len(text) - pattern_len + 1):
        window = text_lower[i:i + pattern_len]
        ratio = SequenceMatcher(None, window, pattern_lower).ratio()
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_match = (i, i + pattern_len, ratio)
    
    return best_match

# ---------- Data Models ----------

@dataclass 
class FilterStats:
    """Statistics about why records were excluded."""
    idx_filtered: int = 0
    field_filtered: int = 0
    time_filtered: int = 0
    where_filtered: int = 0
    quality_filtered: int = 0
    
    def total_filtered(self) -> int:
        return (self.idx_filtered + self.field_filtered + self.time_filtered + 
                self.where_filtered + self.quality_filtered)
    
    def show(self) -> str:
        """Return a human-readable summary."""
        lines = []
        if self.idx_filtered: lines.append(f"  Index filter: {self.idx_filtered}")
        if self.field_filtered: lines.append(f"  Field filter: {self.field_filtered}")
        if self.time_filtered: lines.append(f"  Time filter: {self.time_filtered}")
        if self.where_filtered: lines.append(f"  Where filter: {self.where_filtered}")
        if self.quality_filtered: lines.append(f"  Quality filter: {self.quality_filtered}")
        return "\n".join(lines) if lines else "  (no filters applied)"

@dataclass 
class SearchStats:
    """Statistics collected during search operations."""
    total_records: int = 0
    filtered_records: int = 0
    matched_records: int = 0
    search_time: float = 0.0
    query_complexity: int = 0
    most_common_terms: List[Tuple[str, int]] = field(default_factory=list)
    temporal_distribution: Dict[str, int] = field(default_factory=dict)
    filter_stats: FilterStats = field(default_factory=FilterStats)

@dataclass
class Record:
    """Represents a single record from the memory file."""
    idx: int
    raw: Dict[str, Any]
    text: str
    timestamp: Optional[float] = None
    extra_timestamps: Optional[List[float]] = None
    _text_hash: Optional[int] = None
    
    def __post_init__(self):
        self._text_hash = hash(self.text)
    
    @property
    def text_hash(self) -> int:
        if self._text_hash is None:
            self._text_hash = hash(self.text)
        return self._text_hash

@dataclass
class SearchResult:
    """Represents a single search result with metadata."""
    record: Record
    match_score: float
    snippet: str
    metadata: Optional[Dict[str, Any]] = None

# ---------- Query Parsing ----------

TOKEN = re.compile(r"\(|\)|AND|OR|NOT|\"[^\"]+\"|'[^']+'|\S+")

@dataclass
class QueryNode:
    """AST node for parsed search queries."""
    op: str
    value: Optional[str] = None
    left: Optional['QueryNode'] = None
    right: Optional['QueryNode'] = None
    compiled_regex: Optional[re.Pattern] = None
    fuzzy_threshold: float = 0.8
    
    def __post_init__(self):
        if self.op in ('TERM', 'PHRASE') and self.value:
            try:
                flags = re.IGNORECASE if not hasattr(self, '_case_sensitive') or not self._case_sensitive else 0
                if self.op == 'TERM':
                    pattern = rf"(?<!\w){re.escape(self.value)}(?!\w)"
                else:
                    pattern = re.escape(self.value)
                self.compiled_regex = re.compile(pattern, flags)
            except re.error:
                pass

@lru_cache(maxsize=128)
def parse_query_cached(q: str) -> QueryNode:
    """Cached query parsing for repeated searches."""
    return parse_query(q)

def parse_query(q: str) -> QueryNode:
    """Parse search query into AST."""
    tokens = TOKEN.findall(q)
    if not tokens:
        return QueryNode('ALL')
    pos = 0

    def peek() -> Optional[str]:
        return tokens[pos] if pos < len(tokens) else None

    def eat(t: Optional[str] = None) -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError('Unexpected end of query')
        tok = tokens[pos]
        if t is not None and tok != t:
            raise ValueError(f'Expected {t}, got {tok}')
        pos += 1
        return tok

    def primary() -> QueryNode:
        tok = peek()
        if tok is None:
            return QueryNode('ALL')
        if tok == '(':
            eat('(')
            node = parse_or()
            if peek() != ')':
                raise ValueError('Missing )')
            eat(')')
            return node
        if tok == 'NOT':
            eat('NOT')
            return QueryNode('NOT', left=primary())
        if tok.startswith('"'):
            phrase = eat()
            return QueryNode('PHRASE', value=phrase.strip('"'))
        if tok.startswith("'"):
            phrase = eat()
            return QueryNode('PHRASE', value=phrase.strip("'"))
        return QueryNode('TERM', value=eat())

    def parse_and() -> QueryNode:
        left = primary()
        while True:
            tok = peek()
            if tok == 'AND':
                eat('AND')
                right = primary()
                left = QueryNode('AND', left=left, right=right)
                continue
            if tok is None or tok in (')', 'OR'):
                break
            right = primary()
            left = QueryNode('AND', left=left, right=right)
        return left

    def parse_or() -> QueryNode:
        left = parse_and()
        while True:
            tok = peek()
            if tok == 'OR':
                eat('OR')
                right = parse_and()
                left = QueryNode('OR', left=left, right=right)
            else:
                break
        return left

    ast = parse_or()
    if pos != len(tokens):
        raise ValueError('Trailing tokens')
    return ast

# ---------- Search Options ----------

@dataclass
class SearchOptions:
    """Configuration options for search operations."""
    case_sensitive: bool = False
    use_regex: bool = False
    fuzzy_search: bool = False
    fuzzy_distance: int = 2
    fuzzy_threshold: float = 0.8
    field_exists: Optional[str] = None
    t_from: Optional[float] = None
    t_to: Optional[float] = None
    max_results: int = 20
    max_matches: Optional[int] = None
    snippet_chars: int = 200
    full: bool = False
    where: List[WhereClause] = field(default_factory=list)
    print_fields: Optional[List[str]] = None
    idx_set: Optional[Set[int]] = None
    show_progress: bool = False
    fast_mode: bool = False
    collect_stats: bool = False
    export_format: Optional[str] = None
    highlight_style: str = 'ansi'
    deduplicate: bool = False
    quality_min: Optional[float] = None
    quality_max: Optional[float] = None
    quality_fallback: str = 'include'  # 'include', 'exclude', or 'infer'
    sort_by: Optional[str] = None
    dedupe_method: str = 'hash'
    dedupe_threshold: float = 0.9
    reverse_sort: bool = False

# ---------- FuzzyMatch Implementation ----------

class FuzzyMatch:
    """Mock match object for fuzzy search results, compatible with re.Match interface."""
    def __init__(self, start: int, end: int):
        self._start = start
        self._end = end
    
    def start(self) -> int:
        return self._start
    
    def end(self) -> int:
        return self._end
    
    def span(self) -> Tuple[int, int]:
        return (self._start, self._end)

# ---------- Where Clause Implementation ----------

@dataclass
class WhereClause:
    """Filter clause for metadata fields."""
    field: str
    op: str
    value: str
    test: Callable[[Any], bool] = lambda _: True

def compile_where(field: str, op: str, value: str) -> WhereClause:
    """Compile a where clause into a testable function."""
    def to_number(x: Any):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    
    if op == '=':
        return WhereClause(field, op, value, lambda v: str(v) == value)
    if op == '!=':
        return WhereClause(field, op, value, lambda v: str(v) != value)
    if op == '~=':
        rx = re.compile(value, re.IGNORECASE)
        return WhereClause(field, op, value, lambda v: isinstance(v, (str,int,float)) and bool(rx.search(str(v))))
    if op in ('>','>=','<','<='):
        rhs = to_number(value)
        if rhs is None:
            return WhereClause(field, op, value, lambda v: False)
        if op == '>':  return WhereClause(field, op, value, lambda v: (to_number(v) is not None) and to_number(v) >  rhs)
        if op == '>=': return WhereClause(field, op, value, lambda v: (to_number(v) is not None) and to_number(v) >= rhs)
        if op == '<':  return WhereClause(field, op, value, lambda v: (to_number(v) is not None) and to_number(v) <  rhs)
        if op == '<=': return WhereClause(field, op, value, lambda v: (to_number(v) is not None) and to_number(v) <= rhs)
    
    return WhereClause(field, op, value, lambda v: True)

def parse_where_expr(expr: str) -> WhereClause:
    """Parse a where expression string into a WhereClause."""
    for op in ('>=','<=','!=','~=','>','<','='):
        if op in expr:
            field, value = expr.split(op, 1)
            return compile_where(field.strip(), op, value.strip())
    field = expr.strip()
    return WhereClause(field, 'exists', '', lambda v: v is not None)

# ---------- Time Parsing ----------

def parse_time_any(v: Any) -> Optional[float]:
    """Parse various time formats into Unix timestamp."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace('Z','+00:00')).timestamp()
        except Exception:
            try:
                return float(v)
            except Exception:
                return None
    return None

def collect_timestamps(obj: Dict[str, Any]) -> List[float]:
    """Extract all timestamps from a record object."""
    out: List[float] = []
    for p in NESTED_TS_PATHS:
        v = get_in(obj, p) if '.' in p else obj.get(p)
        ts = parse_time_any(v)
        if ts is not None:
            out.append(ts)
    return out

# ---------- File Processing ----------

def try_json_loads(line: str) -> Optional[Dict[str, Any]]:
    """Safely attempt to parse JSON, returning None on failure."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None

def flatten(obj: Any, prefix: str = "", acc: Optional[List[str]] = None) -> List[str]:
    """Recursively flatten object into list of string values."""
    if acc is None:
        acc = []
    if obj is None:
        return acc
    if isinstance(obj, (str, int, float, bool)):
        acc.append(f"{prefix}{obj}")
        return acc
    if isinstance(obj, dict):
        for _, v in obj.items():
            flatten(v, prefix=prefix, acc=acc)
        return acc
    if isinstance(obj, (list, tuple)):
        for v in obj:
            flatten(v, prefix=prefix, acc=acc)
        return acc
    acc.append(f"{prefix}{obj}")
    return acc

def count_file_lines(path: str) -> int:
    """Fast line counting for progress bars."""
    try:
        if path.endswith('.gz'):
            with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
                return sum(1 for _ in f)
        else:
            with open(path, 'rt', encoding='utf-8', errors='ignore') as f:
                return sum(1 for _ in f)
    except Exception:
        return 0

def iter_memory_fast(path: str, show_progress: bool = False) -> Iterator[Record]:
    """Optimized memory iterator with optional progress tracking."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    
    total_lines = count_file_lines(path) if show_progress else 0
    
    fh: Iterable[str]
    if path.endswith('.gz'):
        fh = gzip.open(path, 'rt', encoding='utf-8', errors='ignore')
    else:
        fh = open(path, 'rt', encoding='utf-8', errors='ignore')
    
    idx = 0
    with fh as f:
        for line_no, line in enumerate(f):
            if show_progress and line_no % 1000 == 0:
                eprint(f"\r{progress_bar(line_no, total_lines)}", end='')
            
            if not line.strip():
                continue
            obj = try_json_loads(line)
            if obj is None or not isinstance(obj, dict):
                continue
            
            # Optimized text extraction
            pieces: List[str] = []
            for key in ('text','content','message','assistant','user','note'):
                if key in obj:
                    flatten(obj[key], acc=pieces)
            if not pieces:
                flatten(obj, acc=pieces)
            
            text = squeeze_whitespace(' '.join(map(str, pieces)))
            ts_list = collect_timestamps(obj)
            primary_ts = ts_list[0] if ts_list else None
            
            yield Record(idx=idx, raw=obj, text=text, timestamp=primary_ts, extra_timestamps=ts_list)
            idx += 1
    
    if show_progress:
        eprint("\r" + " " * 80 + "\r", end='')  # Clear progress bar

# ---------- Lazy Conversation Index ----------

class LazyConversationIndex:
    """
    Lazily builds conversation index as needed.
    Only loads records on-demand, reducing memory footprint.
    """
    def __init__(self, path: str):
        self.path = path
        self.index: Dict[str, List[Record]] = defaultdict(list)
        self.all_records: List[Record] = []
        self.fully_loaded = False
    
    def _ensure_loaded(self):
        """Load all records if not already loaded."""
        if not self.fully_loaded:
            eprint("Building conversation index (first time)...")
            self.all_records = list(iter_memory_fast(self.path, show_progress=False))
            for rec in self.all_records:
                conv_id = get_first_in(rec.raw, CONVERSATION_ID_PATHS)
                if conv_id:
                    self.index[str(conv_id)].append(rec)
            self.fully_loaded = True
            eprint(f"Indexed {len(self.all_records)} records, {len(self.index)} conversations")
    
    def get_context(self, target_record: Record, window_hours: int = 2) -> List[Record]:
        """Get conversation context for a record."""
        self._ensure_loaded()
        
        conv_id = get_first_in(target_record.raw, CONVERSATION_ID_PATHS)
        if not conv_id:
            return []
        
        conv_id = str(conv_id)
        candidates = self.index.get(conv_id, [])
        
        # Filter by time window
        rec_ts = target_record.timestamp or 0
        window_seconds = window_hours * 3600
        context = []
        
        for r in candidates:
            if abs((r.timestamp or 0) - rec_ts) < window_seconds:
                context.append(r)
        
        return sorted(context, key=lambda x: x.timestamp or 0)

# ---------- Temporal Analysis ----------

def analyze_temporal_patterns(results: List[SearchResult]) -> Dict[str, Any]:
    """Analyze when/how often topics appear."""
    timestamps = [r.record.timestamp for r in results if r.record.timestamp]
    if not timestamps:
        return {}
    
    # Group by hour of day, day of week, month, etc.
    hours = Counter(datetime.fromtimestamp(ts).hour for ts in timestamps)
    days = Counter(datetime.fromtimestamp(ts).strftime('%A') for ts in timestamps)
    months = Counter(datetime.fromtimestamp(ts).strftime('%Y-%m') for ts in timestamps)
    
    return {
        'peak_hours': hours.most_common(3),
        'peak_days': days.most_common(3),
        'monthly_distribution': dict(sorted(months.items())),
        'time_span_days': (max(timestamps) - min(timestamps)) / 86400 if len(timestamps) > 1 else 0
    }

def dedupe_by_similarity(results: List[SearchResult], threshold: float = 0.9) -> List[SearchResult]:
    """Remove near-duplicate results using semantic similarity scores."""
    unique = []
    seen_texts = []
    
    for result in results:
        is_dup = False
        for prev_text in seen_texts:
            if SequenceMatcher(None, result.record.text, prev_text).ratio() > threshold:
                is_dup = True
                break
        
        if not is_dup:
            unique.append(result)
            seen_texts.append(result.record.text)
    
    return unique

# ---------- Search Core ----------

def eval_query_optimized(node: QueryNode, text: str, opts: SearchOptions) -> Tuple[bool, Optional[Union[re.Match, FuzzyMatch]], float]:
    """Optimized query evaluation with fuzzy matching support."""
    if node.op == 'ALL':
        return True, None, 1.0
    
    if node.op in ('TERM', 'PHRASE'):
        if opts.fuzzy_search and node.value:
            # Fuzzy matching
            match_result = fuzzy_match(text, node.value, opts.fuzzy_threshold)
            if match_result:
                start, end, score = match_result
                return True, FuzzyMatch(start, end), score
            return False, None, 0.0
        
        elif opts.use_regex and node.value:
            # Regex matching
            flags = 0 if opts.case_sensitive else re.IGNORECASE
            try:
                pattern = node.value
                m = re.search(pattern, text, flags)
                return (m is not None, m, 1.0 if m else 0.0)
            except re.error:
                return False, None, 0.0
        
        elif node.compiled_regex:
            # Pre-compiled regex (faster)
            m = node.compiled_regex.search(text)
            return (m is not None, m, 1.0 if m else 0.0)
        
        elif node.value:
            # Standard term/phrase matching
            flags = 0 if opts.case_sensitive else re.IGNORECASE
            try:
                if node.op == 'TERM':
                    pattern = rf"(?<!\w){re.escape(node.value)}(?!\w)"
                else:
                    pattern = re.escape(node.value)
                m = re.search(pattern, text, flags)
                return (m is not None, m, 1.0 if m else 0.0)
            except re.error:
                return False, None, 0.0
    
    if node.op == 'NOT':
        ok, _, score = eval_query_optimized(node.left, text, opts)
        return (not ok, None, 1.0 - score if ok else 1.0)
    
    if node.op == 'AND':
        ok_l, m_l, score_l = eval_query_optimized(node.left, text, opts)
        if not ok_l:
            return (False, None, 0.0)
        ok_r, m_r, score_r = eval_query_optimized(node.right, text, opts)
        return (ok_r, m_l or m_r, min(score_l, score_r))
    
    if node.op == 'OR':
        ok_l, m_l, score_l = eval_query_optimized(node.left, text, opts)
        ok_r, m_r, score_r = eval_query_optimized(node.right, text, opts)
        return (ok_l or ok_r, m_l or m_r, max(score_l, score_r) if (ok_l or ok_r) else 0.0)
    
    raise ValueError('Unknown operation')

def make_snippet_enhanced(s: str, m: Optional[Union[re.Match, FuzzyMatch]], 
                         size: int, highlight_style: str = 'ansi') -> str:
    """Enhanced snippet generation with configurable highlighting."""
    open_tag, close_tag = HIGHLIGHT_SCHEMES.get(highlight_style, HIGHLIGHT_SCHEMES['ansi'])
    
    if not m:
        return s[:size] + ('…' if len(s) > size else '')
    
    match_start = m.start()
    match_end = m.end()
    
    start = max(0, match_start - size//2)
    end = min(len(s), start + size)
    snippet = s[start:end]
    
    token = s[match_start:match_end]
    
    if token and highlight_style != 'none':
        # Adjust token position relative to snippet
        token_start_in_snippet = match_start - start
        token_end_in_snippet = match_end - start
        
        if 0 <= token_start_in_snippet < len(snippet):
            highlighted = (snippet[:token_start_in_snippet] + 
                         open_tag + token + close_tag + 
                         snippet[token_end_in_snippet:])
            snippet = highlighted
    
    return ('…' if start > 0 else '') + snippet + ('…' if end < len(s) else '')

def within_time(rec: Record, t_from: Optional[float], t_to: Optional[float]) -> bool:
    """Check if record falls within time range."""
    if t_from is None and t_to is None:
        return True
    if not rec.extra_timestamps:
        return False
    for ts in rec.extra_timestamps:
        if t_from is not None and ts < t_from:
            continue
        if t_to is not None and ts > t_to:
            continue
        return True
    return False

def passes_filters(rec: Record, opts: SearchOptions, filter_stats: Optional[FilterStats] = None) -> bool:
    """
    Check if record passes all filter criteria.
    Optionally tracks which filters caused exclusions.
    """
    if opts.idx_set is not None and rec.idx not in opts.idx_set:
        if filter_stats: filter_stats.idx_filtered += 1
        return False
    
    if opts.field_exists and opts.field_exists not in rec.raw:
        if filter_stats: filter_stats.field_filtered += 1
        return False
    
    if not within_time(rec, opts.t_from, opts.t_to):
        if filter_stats: filter_stats.time_filtered += 1
        return False
    
    if opts.where:
        for w in opts.where:
            v = get_in(rec.raw, w.field)
            if not w.test(v):
                if filter_stats: filter_stats.where_filtered += 1
                return False
    
    # Quality filtering with configurable fallback
    if opts.quality_min is not None or opts.quality_max is not None:
        quality_score = get_in(rec.raw, 'quality_metrics.quality_score')
        
        if quality_score is None:
            # Handle missing quality based on fallback strategy
            if opts.quality_fallback == 'exclude':
                if filter_stats: filter_stats.quality_filtered += 1
                return False
            elif opts.quality_fallback == 'infer':
                # Infer quality from text length and structure (simple heuristic)
                word_count = len(rec.text.split())
                if word_count < 10:
                    inferred_quality = 0.3
                elif word_count > 100:
                    inferred_quality = 0.7
                else:
                    inferred_quality = 0.5
                quality_score = inferred_quality
            # If we reach here with 'include' fallback and no quality_score, pass the filter
            elif opts.quality_fallback == 'include':
                return True
        
        # Now check quality bounds if we have a score
        if quality_score is not None:
            try:
                score = float(quality_score)
                if opts.quality_min is not None and score < opts.quality_min:
                    if filter_stats: filter_stats.quality_filtered += 1
                    return False
                if opts.quality_max is not None and score > opts.quality_max:
                    if filter_stats: filter_stats.quality_filtered += 1
                    return False
            except (ValueError, TypeError):
                if opts.quality_fallback == 'exclude':
                    if filter_stats: filter_stats.quality_filtered += 1
                    return False
            
    return True

# ---------- Index Parsing ----------

def parse_idx_spec(spec: str, around: int = 0) -> Set[int]:
    """Parse index specification string into set of indices."""
    out: Set[int] = set()
    spec = spec.strip()
    if not spec:
        return out
    parts = [p.strip() for p in spec.split(',') if p.strip()]
    for p in parts:
        if '-' in p:
            try:
                a, b = p.split('-', 1)
                a_i, b_i = int(a), int(b)
                lo, hi = (a_i, b_i) if a_i <= b_i else (b_i, a_i)
                for i in range(lo, hi + 1):
                    for j in range(i - around, i + around + 1):
                        if j >= 0:
                            out.add(j)
            except ValueError:
                continue
        else:
            try:
                i = int(p)
                for j in range(i - around, i + around + 1):
                    if j >= 0:
                        out.add(j)
            except ValueError:
                continue
    return out

# ---------- Output Formatting ----------

def format_result_with_metrics(result: SearchResult, opts: SearchOptions) -> str:
    """Formats a single search result for display, including quality metrics."""
    lines = []
    
    # Header line
    ts = f" @ {datetime.fromtimestamp(result.record.timestamp)}" if result.record.timestamp else ''
    score_str = f" (score: {result.match_score:.2f})" if opts.fuzzy_search else ""
    
    qm = result.record.raw.get('quality_metrics', {})
    quality_score = qm.get('quality_score')
    quality_str = f" (Quality: {quality_score:.2f})" if quality_score is not None else ""
    
    header = f'- #{result.record.idx}{ts}{score_str}{quality_str}'
    lines.append(header)
    
    # Metrics line
    sim = qm.get('semantic_similarity')
    den = qm.get('information_density')
    wc = qm.get('word_count')
    metrics_parts = []
    if sim is not None: metrics_parts.append(f"Similarity: {sim:.2f}")
    if den is not None: metrics_parts.append(f"Density: {den:.2f}")
    if wc is not None: metrics_parts.append(f"Words: {wc}")
    if metrics_parts:
        lines.append(f"  [{' | '.join(metrics_parts)}]")

    # Snippet line
    lines.append(f'  {result.snippet}')
    
    # Metadata line
    if result.metadata:
        meta_parts = []
        for k, v in result.metadata.items():
            meta_parts.append(f"{k}={json.dumps(v, ensure_ascii=False)}")
        lines.append(f'  meta: {"; ".join(meta_parts)}')
        
    return "\n".join(lines)

# ---------- Export Functions ----------

def export_results(results: List[SearchResult], format_type: str, output_path: str) -> None:
    """Export search results in various formats."""
    if format_type.lower() == 'json':
        export_data = []
        for result in results:
            export_data.append({
                'idx': result.record.idx,
                'text': result.record.text,
                'score': result.match_score,
                'snippet': result.snippet,
                'timestamp': result.record.timestamp,
                'metadata': result.metadata,
                'raw': result.record.raw
            })
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
    
    elif format_type.lower() == 'csv':
        import csv
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['idx', 'score', 'timestamp', 'text', 'snippet'])
            for result in results:
                writer.writerow([
                    result.record.idx,
                    result.match_score,
                    result.record.timestamp,
                    result.record.text.replace('\n', ' '),
                    result.snippet.replace('\n', ' ')
                ])
    
    elif format_type.lower() == 'html':
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Memory Search Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .result {{ margin-bottom: 20px; border: 1px solid #ccc; padding: 15px; }}
        .header {{ font-weight: bold; color: #333; }}
        .snippet {{ background-color: #f9f9f9; padding: 10px; margin-top: 10px; }}
        mark {{ background-color: yellow; }}
    </style>
</head>
<body>
    <h1>Memory Search Results ({len(results)} matches)</h1>
"""
        for result in results:
            timestamp = datetime.fromtimestamp(result.record.timestamp).strftime('%Y-%m-%d %H:%M:%S') if result.record.timestamp else 'N/A'
            html_content += f"""
    <div class="result">
        <div class="header">Record #{result.record.idx} (Score: {result.match_score:.2f}) - {timestamp}</div>
        <div class="snippet">{result.snippet}</div>
    </div>
"""
        html_content += "</body></html>"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

# ---------- Enhanced REPL ----------

class EnhancedREPL:
    """Enhanced interactive REPL for memory search."""
    
    def __init__(self, path: str):
        self.path = path
        self.opts = SearchOptions(where=[], print_fields=None)
        self.history: List[str] = []
        self.stats = SearchStats()
        self.last_results: List[SearchResult] = []
        self.lazy_index: Optional[LazyConversationIndex] = None
        self.proximity_distance: int = 50  # words between terms for proximity search
        
        if HAS_READLINE:
            readline.set_completer(self.completer)
            readline.parse_and_bind("tab: complete")
    
    def completer(self, text: str, state: int) -> Optional[str]:
        """Tab completion for REPL commands."""
        commands = [
            ':n', ':m', ':f', ':from', ':to', ':S', ':R', ':where', ':full',
            ':print', ':idx', ':around', ':fuzzy', ':export', ':stats', ':help', ':q',
            ':quality', ':context', ':temporal', ':analyze', ':dedupe_semantic',
            ':progress', ':highlight', ':dedupe', ':sort', ':reverse', ':quality_fallback',
            ':proximity', ':near', ':synonym'
        ]
        matches = [cmd for cmd in commands if cmd.startswith(text)]
        return matches[state] if state < len(matches) else None
    
    def show_help(self):
        """Display help text for available commands."""
        help_text = """
Enhanced Memory Search Commands:
  Basic Search:
    :n <N>           - Set max results
    :m <N>           - Set max matches  
    :f <field>       - Filter by field existence
    :from <time>     - Start time filter
    :to <time>       - End time filter
    :S               - Toggle case sensitivity
    :R               - Toggle regex mode
    :fuzzy <dist>    - Enable fuzzy search (distance)
  
  Advanced Patterns:
    :proximity <N>   - Set word distance for NEAR operator (default: 50)
    :near term1 term2 - Find term1 within N words of term2
    :synonym word1,word2,word3 - Search for any of these terms
  
  Filters & Metadata:
    :where <expr>    - Add metadata filter (use 'clear' to reset)
    :quality <min-max> - Filter by quality_score (e.g., 0.7-0.9)
    :quality_fallback <mode> - Set fallback for missing quality (include/exclude/infer)
    :idx <spec>      - Filter by indices
    :around <N>      - Context window around indices (preview blocks)
  
  Output & Display:
    :full            - Toggle full text display
    :print <fields>  - Show metadata fields
    :export <file>   - Export results (json/csv/html)
    :highlight <style> - Set highlight style (ansi/html/markdown/none)
    :sort <field>    - Sort results (score/timestamp/idx/quality)
    :reverse         - Toggle reverse sort order
  
  Analysis:
    :context <idx>   - Show conversation context for a result index
    :temporal        - Show temporal analysis of last search results
    :analyze         - Show domain/author analytics of last results
    :stats           - Toggle statistics collection
    :progress        - Toggle progress display
  
  Deduplication:
    :dedupe          - Toggle deduplication
    :dedupe_semantic <thresh> - Set semantic deduplication threshold (0.0-1.0)
  
  :help            - Show this help
  :q               - Quit

Search Query Syntax:
  - Basic: word1 word2 word3 (implicit AND)
  - Boolean: word1 AND word2, word1 OR word2, NOT word
  - Phrases: "exact phrase" or 'exact phrase'
  - Grouping: (word1 OR word2) AND word3
  - Regex mode (:R): Use full regex patterns
        """
        print(help_text)
    
    def run(self):
        """Main REPL loop."""
        print("Enhanced Memory Explorer REPL")
        print("Type :help for commands, or enter a search query")
        
        while True:
            try:
                line = input('mem> ').strip()
                self.history.append(line)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            
            if not line:
                continue
            
            if line in (':q', ':quit', ':exit'):
                break
            
            if line == ':help':
                self.show_help()
                continue
            
            if line.startswith(':'):
                self.handle_command(line)
                continue
            
            # Execute search
            self.execute_search(line)
    
    def handle_command(self, line: str):
        """Handle REPL commands."""
        parts = line.split(maxsplit=1)
        cmd = parts[0]
        arg = parts[1] if len(parts) > 1 else ""
        
        if cmd == ':n':
            try:
                self.opts.max_results = int(arg)
                print(f'max_results = {self.opts.max_results}')
            except ValueError:
                print('Usage: :n 50')
        
        elif cmd == ':m':
            try:
                self.opts.max_matches = int(arg) if arg else None
                print(f'max_matches = {self.opts.max_matches}')
            except ValueError:
                print('Usage: :m 200')
        
        elif cmd == ':f':
            self.opts.field_exists = arg.strip() or None
            print(f'field filter = {self.opts.field_exists}')
        
        elif cmd == ':from':
            try:
                self.opts.t_from = self.parse_time(arg)
                print(f'from = {self.opts.t_from} ({datetime.fromtimestamp(self.opts.t_from)})')
            except Exception as e:
                print(f'bad time: {e}')
        
        elif cmd == ':to':
            try:
                self.opts.t_to = self.parse_time(arg)
                print(f'to = {self.opts.t_to} ({datetime.fromtimestamp(self.opts.t_to)})')
            except Exception as e:
                print(f'bad time: {e}')
        
        elif cmd == ':S':
            self.opts.case_sensitive = not self.opts.case_sensitive
            print(f'case_sensitive = {self.opts.case_sensitive}')
        
        elif cmd == ':R':
            self.opts.use_regex = not self.opts.use_regex
            print(f'regex = {self.opts.use_regex}')
        
        elif cmd == ':fuzzy':
            if arg:
                try:
                    self.opts.fuzzy_distance = int(arg)
                    self.opts.fuzzy_search = True
                    print(f'fuzzy search enabled, distance = {self.opts.fuzzy_distance}')
                except ValueError:
                    print('Usage: :fuzzy 2')
            else:
                self.opts.fuzzy_search = not self.opts.fuzzy_search
                print(f'fuzzy search = {self.opts.fuzzy_search}')
        
        elif cmd == ':where':
            if not arg:
                if self.opts.where:
                    print('where filters:')
                    for f in self.opts.where:
                        print(f'  - {f.field} {f.op} {f.value}')
                else:
                    print('(no where filters)')
            elif arg == 'clear':
                self.opts.where = []
                print('where filters cleared')
            else:
                wc = parse_where_expr(arg)
                self.opts.where.append(wc)
                print(f'added where: {wc.field} {wc.op} {wc.value}')
        
        elif cmd == ':full':
            self.opts.full = not self.opts.full
            print(f'full text mode = {self.opts.full}')
        
        elif cmd == ':print':
            if not arg:
                self.opts.print_fields = None
                print('print fields cleared')
            else:
                self.opts.print_fields = [p.strip() for p in arg.split(',') if p.strip()]
                print('print fields =', ', '.join(self.opts.print_fields))
        
        elif cmd == ':idx':
            if not arg:
                self.opts.idx_set = None
                print('idx filter cleared')
            else:
                self.opts.idx_set = parse_idx_spec(arg, around=0)
                print(f'idx set size = {len(self.opts.idx_set)}')
        
        elif cmd == ':around':
            try:
                n = int(arg)
                if self.opts.idx_set:
                    base = list(self.opts.idx_set)
                    self.opts.idx_set = parse_idx_spec(','.join(map(str, base)), around=n)
                    print(f'expanded idx set around {n}, new size = {len(self.opts.idx_set)}')
                else:
                    print(f'around = {n} (use with :idx first)')
            except ValueError:
                print('Usage: :around 3')
        
        elif cmd == ':export':
            if not arg:
                print('Usage: :export results.json|results.csv|results.html')
            else:
                self.opts.export_format = arg
                print(f'export format set to: {arg}')
        
        elif cmd == ':stats':
            self.opts.collect_stats = not self.opts.collect_stats
            print(f'statistics collection = {self.opts.collect_stats}')
        
        elif cmd == ':progress':
            self.opts.show_progress = not self.opts.show_progress
            print(f'progress display = {self.opts.show_progress}')
        
        elif cmd == ':highlight':
            if arg in HIGHLIGHT_SCHEMES:
                self.opts.highlight_style = arg
                print(f'highlight style = {arg}')
            else:
                print(f'Available styles: {", ".join(HIGHLIGHT_SCHEMES.keys())}')
        
        elif cmd == ':dedupe':
            self.opts.deduplicate = not self.opts.deduplicate
            if self.opts.deduplicate:
                print(f'Deduplication enabled (method: {self.opts.dedupe_method})')
            else:
                print(f'Deduplication disabled')
        
        elif cmd == ':sort':
            if arg in ('score', 'timestamp', 'idx', 'quality', 'clear'):
                if arg == 'clear':
                    self.opts.sort_by = None
                    print('sorting cleared')
                else:
                    self.opts.sort_by = arg
                    print(f'sort by = {arg}')
            else:
                print('Usage: :sort score|timestamp|idx|quality|clear')
        
        elif cmd == ':reverse':
            self.opts.reverse_sort = not self.opts.reverse_sort
            print(f'reverse sort = {self.opts.reverse_sort}')

        elif cmd == ':quality':
            if '-' in arg:
                try:
                    min_q, max_q = map(float, arg.split('-'))
                    self.opts.quality_min = min_q
                    self.opts.quality_max = max_q
                    print(f'Quality filter set: {min_q:.2f} - {max_q:.2f}')
                except ValueError:
                    print('Usage: :quality 0.7-0.9')
            elif arg == 'clear':
                self.opts.quality_min = None
                self.opts.quality_max = None
                print('Quality filter cleared.')
            else:
                print('Usage: :quality 0.7-0.9 or :quality clear')
        
        elif cmd == ':quality_fallback':
            if arg in ('include', 'exclude', 'infer'):
                self.opts.quality_fallback = arg
                print(f'quality_fallback = {arg}')
                if arg == 'include':
                    print('  (Records without quality metrics will be included)')
                elif arg == 'exclude':
                    print('  (Records without quality metrics will be excluded)')
                else:
                    print('  (Quality will be inferred from text characteristics)')
            else:
                print('Usage: :quality_fallback include|exclude|infer')
        
        elif cmd == ':proximity':
            try:
                self.proximity_distance = int(arg)
                print(f'proximity distance = {self.proximity_distance} words')
            except ValueError:
                print('Usage: :proximity 50')
        
        elif cmd == ':near':
            # Proximity search: find term1 NEAR term2
            if not arg:
                print('Usage: :near term1 term2')
                return
            terms = arg.split()
            if len(terms) < 2:
                print('Usage: :near term1 term2')
                return
            
            # Build proximity regex pattern
            # Match term1...term2 OR term2...term1 within N words
            term1, term2 = terms[0], terms[1]
            word_pattern = r'\S+'
            proximity_pattern = (
                f"(?:{re.escape(term1)}"
                f"(?:\\s+{word_pattern}){{0,{self.proximity_distance}}}"
                f"\\s+{re.escape(term2)})|"
                f"(?:{re.escape(term2)}"
                f"(?:\\s+{word_pattern}){{0,{self.proximity_distance}}}"
                f"\\s+{re.escape(term1)})"
            )
            
            print(f"Searching for '{term1}' within {self.proximity_distance} words of '{term2}'...")
            self.opts.use_regex = True
            self.execute_search(proximity_pattern)
            self.opts.use_regex = False  # Reset after
        
        elif cmd == ':synonym':
            # Synonym search: match any of the provided terms
            if not arg:
                print('Usage: :synonym word1,word2,word3')
                return
            
            synonyms = [s.strip() for s in arg.split(',') if s.strip()]
            if not synonyms:
                print('Usage: :synonym word1,word2,word3')
                return
            
            # Build OR query
            query = ' OR '.join(synonyms)
            print(f"Searching for any of: {', '.join(synonyms)}")
            self.execute_search(query)

        elif cmd == ':context':
            if not self.last_results:
                print("Perform a search first to get results.")
                return
            try:
                res_idx = int(arg)
                target_result = next((r for r in self.last_results if r.record.idx == res_idx), None)
                if not target_result:
                    print(f"Result with index #{res_idx} not in last search.")
                    return

                print(f"Retrieving context for record #{res_idx}...")
                
                # Lazy load index
                if self.lazy_index is None:
                    self.lazy_index = LazyConversationIndex(self.path)
                
                context_records = self.lazy_index.get_context(target_result.record)
                
                if not context_records:
                    print("No conversation context found.")
                    return

                print("\n--- Conversation Context ---")
                for rec in context_records:
                    is_target = " <<< TARGET" if rec.idx == target_result.record.idx else ""
                    ts = f" @ {datetime.fromtimestamp(rec.timestamp)}" if rec.timestamp else ''
                    print(f'- #{rec.idx}{ts}{is_target}')
                    print(f"  {squeeze_whitespace(rec.text)[:200]}...")
                print("--- End Context ---")

            except (ValueError, IndexError):
                print("Usage: :context <result_index>")
        
        elif cmd == ':temporal':
            if not self.last_results:
                print("Perform a search first.")
                return
            analysis = analyze_temporal_patterns(self.last_results)
            print("\nTemporal Analysis of Last Search:")
            print(json.dumps(analysis, indent=2))
        
        elif cmd == ':analyze':
            if not self.last_results:
                print("Perform a search first.")
                return
            
            domains = Counter()
            authors = Counter()
            avg_quality_by_domain = defaultdict(list)
            
            for result in self.last_results:
                domain = get_in(result.record.raw, 'source_metadata.domain')
                author = get_first_in(result.record.raw, AUTHOR_PATHS)
                quality = get_in(result.record.raw, 'quality_metrics.quality_score')
                
                if domain:
                    domains[domain] += 1
                    if quality is not None:
                        try:
                            avg_quality_by_domain[domain].append(float(quality))
                        except (ValueError, TypeError):
                            pass
                if author:
                    authors[author] += 1
            
            print("\n--- Analytics from Last Search ---")
            print("\nTop Domains:", domains.most_common(5))
            print("Top Authors:", authors.most_common(5))
            print("\nAverage Quality by Domain:")
            for domain, scores in avg_quality_by_domain.items():
                if scores:
                    print(f"- {domain}: {sum(scores)/len(scores):.2f} (from {len(scores)} records)")
            print("--- End Analytics ---")

        elif cmd == ':dedupe_semantic':
            self.opts.dedupe_method = 'semantic'
            if arg:
                try:
                    self.opts.dedupe_threshold = float(arg)
                    print(f"Semantic deduplication enabled with threshold {self.opts.dedupe_threshold}")
                except ValueError:
                    print("Usage: :dedupe_semantic 0.9")
            else:
                print(f"Semantic deduplication enabled with default threshold {self.opts.dedupe_threshold}")

        else:
            print(f'Unknown command: {cmd}. Type :help for available commands.')
    
    def parse_time(self, s: str) -> float:
        """Parse time string into Unix timestamp."""
        try:
            return float(s)
        except ValueError:
            return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
    
    def execute_search(self, query: str):
        """Execute a search query with current options."""
        timer = Timer()
        
        try:
            if self.opts.use_regex:
                qnode = QueryNode('TERM', value=query) if query else QueryNode('ALL')
            else:
                qnode = parse_query_cached(query)
        except Exception as e:
            print(f'Query parse error: {e}')
            return
        
        results = []
        seen_hashes = set()
        matches_seen = 0
        
        self.stats = SearchStats()
        self.stats.query_complexity = self.calculate_query_complexity(qnode)
        
        for rec in iter_memory_fast(self.path, self.opts.show_progress):
            self.stats.total_records += 1
            
            if not passes_filters(rec, self.opts, self.stats.filter_stats):
                continue
            
            self.stats.filtered_records += 1
            
            if self.opts.deduplicate and self.opts.dedupe_method == 'hash':
                if rec.text_hash in seen_hashes:
                    continue
                seen_hashes.add(rec.text_hash)
            
            ok, m, score = eval_query_optimized(qnode, rec.text, self.opts)
            if ok:
                self.stats.matched_records += 1
                
                snippet = (rec.text if self.opts.full else 
                          make_snippet_enhanced(rec.text, m, self.opts.snippet_chars, self.opts.highlight_style))
                
                metadata = None
                if self.opts.print_fields:
                    metadata = {}
                    for p in self.opts.print_fields:
                        metadata[p] = get_in(rec.raw, p)
                
                results.append(SearchResult(rec, score, snippet, metadata))
                matches_seen += 1
                
                if self.opts.max_matches and matches_seen >= self.opts.max_matches:
                    break
        
        # Deduplication suggestion
        if len(results) > 50 and not self.opts.deduplicate:
            eprint(f"\n💡 Tip: Found {len(results)} results. Try :dedupe to remove duplicates")
        
        if self.opts.deduplicate and self.opts.dedupe_method == 'semantic':
            original_count = len(results)
            results = dedupe_by_similarity(results, self.opts.dedupe_threshold)
            if original_count != len(results):
                print(f"Semantic deduplication removed {original_count - len(results)} similar results.")
        
        # Sort results
        if self.opts.sort_by == 'score':
            results.sort(key=lambda r: r.match_score, reverse=not self.opts.reverse_sort)
        elif self.opts.sort_by == 'timestamp':
            results.sort(key=lambda r: r.record.timestamp or 0, reverse=not self.opts.reverse_sort)
        elif self.opts.sort_by == 'idx':
            results.sort(key=lambda r: r.record.idx, reverse=self.opts.reverse_sort)
        elif self.opts.sort_by == 'quality':
            results.sort(key=lambda r: get_in(r.record.raw, 'quality_metrics.quality_score') or 0.0, 
                         reverse=not self.opts.reverse_sort)
        
        self.last_results = results
        display_results = results[:self.opts.max_results]
        
        for result in display_results:
            print(format_result_with_metrics(result, self.opts))
        
        self.stats.search_time = timer.elapsed()
        
        if display_results:
            print(f'\nFound {len(results)} matches ({len(display_results)} shown) in {self.stats.search_time:.2f}s')
        else:
            print('(no matches)')
            # Show diagnostic info if filters were applied
            if self.stats.filter_stats.total_filtered() > 0:
                print("\nRecords excluded by filters:")
                print(self.stats.filter_stats.show())
        
        if self.opts.export_format and results:
            try:
                export_results(results, 
                             self.opts.export_format.split('.')[-1], 
                             self.opts.export_format)
                print(f'Results exported to {self.opts.export_format}')
            except Exception as e:
                print(f'Export failed: {e}')
        
        if self.opts.collect_stats:
            self.show_statistics()
    
    def calculate_query_complexity(self, node: QueryNode) -> int:
        """Calculate query complexity score."""
        if node.op == 'ALL':
            return 1
        if node.op in ('TERM', 'PHRASE'):
            return 2 + (len(node.value) if node.value else 0) // 10
        if node.op == 'NOT':
            return 1 + self.calculate_query_complexity(node.left)
        if node.op in ('AND', 'OR'):
            return 1 + self.calculate_query_complexity(node.left) + self.calculate_query_complexity(node.right)
        return 1
    
    def show_statistics(self):
        """Display search statistics."""
        print(f'\nSearch Statistics:')
        print(f'  Total records: {self.stats.total_records:,}')
        print(f'  Filtered records: {self.stats.filtered_records:,}')
        print(f'  Matched records: {self.stats.matched_records:,}')
        print(f'  Search time: {self.stats.search_time:.3f}s')
        print(f'  Query complexity: {self.stats.query_complexity}')
        if self.stats.filtered_records > 0:
            print(f'  Match rate: {100 * self.stats.matched_records / self.stats.filtered_records:.1f}%')
        if self.stats.filter_stats.total_filtered() > 0:
            print('\nFilter Exclusions:')
            print(self.stats.filter_stats.show())

# ---------- Main CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser(description='Enhanced Memory Search with fuzzy matching, analytics, and export')
    ap.add_argument('path', help='Path to memory.jsonl(.gz) file')
    ap.add_argument('-q', '--query', help='Search query')
    ap.add_argument('-r', '--regex', help='Regex pattern')
    ap.add_argument('-S', '--case-sensitive', action='store_true')
    ap.add_argument('-n', '--max-results', type=int, default=20)
    ap.add_argument('--snippet', type=int, default=200)
    ap.add_argument('--field', help='Filter by field existence')
    ap.add_argument('--from', dest='t_from', help='Start time filter')
    ap.add_argument('--to', dest='t_to', help='End time filter')
    ap.add_argument('--max-matches', type=int)
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--print', dest='print_fields', help='Comma-separated metadata paths to print per hit')
    ap.add_argument('--where', action='append', help='field[op]value; ops: =,!=,~=,>,>=,<,<=; dot-notation ok')
    ap.add_argument('--domain', help='Filter by domain')
    ap.add_argument('--author', help='Filter by author')
    ap.add_argument('--conversation', type=str, help='Filter by conversation ID')
    ap.add_argument('--min-quality', type=float, help='Minimum quality score')
    ap.add_argument('--max-quality', type=float, help='Maximum quality score')
    ap.add_argument('--quality-fallback', choices=['include', 'exclude', 'infer'], default='include',
                    help='How to handle records without quality metrics')
    ap.add_argument('--idx', help='indices: e.g. 97,111,200-210')
    ap.add_argument('--around', type=int, default=0, help='expand each --idx by N before/after')
    ap.add_argument('--fuzzy', type=int, help='Enable fuzzy search with max distance')
    ap.add_argument('--fuzzy-threshold', type=float, default=0.8, help='Fuzzy match threshold (0.0-1.0)')
    ap.add_argument('--export', help='Export results (filename.json|csv|html)')
    ap.add_argument('--highlight', choices=list(HIGHLIGHT_SCHEMES.keys()), default='ansi', help='Highlight style')
    ap.add_argument('--progress', action='store_true', help='Show progress bar')
    ap.add_argument('--fast', action='store_true', help='Fast mode optimizations')
    ap.add_argument('--stats', action='store_true', help='Show search statistics')
    ap.add_argument('--dedupe', action='store_true', help='Remove duplicate results')
    ap.add_argument('--sort', choices=['score', 'timestamp', 'idx', 'quality'], help='Sort results')
    ap.add_argument('--reverse', action='store_true', help='Reverse sort order')
    ap.add_argument('--repl', action='store_true', help='Enter interactive mode')

    args = ap.parse_args()

    # Quick validation
    try:
        next(iter_memory_fast(args.path))
    except StopIteration:
        eprint('File appears empty')
        sys.exit(1)
    except FileNotFoundError as e:
        eprint(str(e))
        sys.exit(1)

    # Enter REPL mode
    if args.repl:
        repl = EnhancedREPL(args.path)
        repl.run()
        return

    # Build search options
    idx_set = parse_idx_spec(args.idx, around=args.around) if args.idx else None
    
    opts = SearchOptions(
        case_sensitive=args.case_sensitive,
        use_regex=bool(args.regex),
        fuzzy_search=bool(args.fuzzy),
        fuzzy_distance=args.fuzzy or 2,
        fuzzy_threshold=args.fuzzy_threshold,
        field_exists=args.field,
        t_from=parse_time_any(args.t_from) if args.t_from else None,
        t_to=parse_time_any(args.t_to) if args.t_to else None,
        max_results=args.max_results,
        max_matches=args.max_matches,
        snippet_chars=args.snippet,
        full=args.full,
        where=[],
        print_fields=[p.strip() for p in args.print_fields.split(',')] if args.print_fields else None,
        idx_set=idx_set,
        show_progress=args.progress,
        fast_mode=args.fast,
        collect_stats=args.stats,
        export_format=args.export,
        highlight_style=args.highlight,
        deduplicate=args.dedupe,
        sort_by=args.sort,
        reverse_sort=args.reverse,
        quality_min=args.min_quality,
        quality_max=args.max_quality,
        quality_fallback=args.quality_fallback,
    )

    # Build where clauses
    where_list: List[WhereClause] = []
    if args.where:
        for expr in args.where:
            where_list.append(parse_where_expr(expr))
    if args.domain:
        where_list.append(compile_where('source_metadata.domain', '=', args.domain))
        where_list.append(compile_where('domain', '=', args.domain))
    if args.author:
        for path in AUTHOR_PATHS:
            where_list.append(compile_where(path, '=', args.author))
    if args.conversation:
        for path in CONVERSATION_ID_PATHS:
            where_list.append(compile_where(path, '=', str(args.conversation)))
    
    opts.where = where_list

    # Parse query
    try:
        if args.regex:
            qnode = QueryNode('TERM', value=args.regex) if args.regex else QueryNode('ALL')
        else:
            qnode = parse_query_cached(args.query or '')
    except Exception as e:
        eprint(f'Query parse error: {e}')
        sys.exit(2)

    # Execute search
    timer = Timer()
    results = []
    seen_hashes = set()
    stats = SearchStats()
    
    for rec in iter_memory_fast(args.path, opts.show_progress):
        stats.total_records += 1
        
        if not passes_filters(rec, opts, stats.filter_stats):
            continue
        
        stats.filtered_records += 1
        
        if opts.deduplicate:
            if rec.text_hash in seen_hashes:
                continue
            seen_hashes.add(rec.text_hash)
        
        ok, m, score = eval_query_optimized(qnode, rec.text, opts)
        if ok:
            stats.matched_records += 1
            
            snippet = (rec.text if opts.full else 
                      make_snippet_enhanced(rec.text, m, opts.snippet_chars, opts.highlight_style))
            
            metadata = None
            if opts.print_fields:
                metadata = {}
                for p in opts.print_fields:
                    metadata[p] = get_in(rec.raw, p)
            
            results.append(SearchResult(rec, score, snippet, metadata))
            
            if opts.max_matches and len(results) >= opts.max_matches:
                break

    stats.search_time = timer.elapsed()

    # Deduplication suggestion
    if len(results) > 50 and not opts.deduplicate:
        eprint(f"\nTip: Found {len(results)} results. Try --dedupe to remove duplicates")

    # Sort results
    if opts.sort_by == 'score':
        results.sort(key=lambda r: r.match_score, reverse=not opts.reverse_sort)
    elif opts.sort_by == 'timestamp':
        results.sort(key=lambda r: r.record.timestamp or 0, reverse=not opts.reverse_sort)
    elif opts.sort_by == 'idx':
        results.sort(key=lambda r: r.record.idx, reverse=opts.reverse_sort)
    elif opts.sort_by == 'quality':
        results.sort(key=lambda r: get_in(r.record.raw, 'quality_metrics.quality_score') or 0.0, 
                     reverse=not opts.reverse_sort)

    # Display results
    display_results = results[:opts.max_results]
    
    for result in display_results:
        print(format_result_with_metrics(result, opts))

    # Show summary
    if display_results:
        print(f'\nFound {len(results)} matches ({len(display_results)} shown) in {stats.search_time:.2f}s')
    else:
        print('(no matches)')
        # Show diagnostic info if filters were applied
        if stats.filter_stats.total_filtered() > 0:
            print("\nRecords excluded by filters:")
            print(stats.filter_stats.show())

    # Export results
    if opts.export_format and results:
        try:
            export_results(results, opts.export_format.split('.')[-1], opts.export_format)
            print(f'Results exported to {opts.export_format}')
        except Exception as e:
            eprint(f'Export failed: {e}')

    # Show statistics
    if opts.collect_stats:
        print(f'\nSearch Statistics:')
        print(f'  Total records: {stats.total_records:,}')
        print(f'  Filtered records: {stats.filtered_records:,}')
        print(f'  Matched records: {stats.matched_records:,}')
        print(f'  Search time: {stats.search_time:.3f}s')
        if stats.filtered_records > 0:
            print(f'  Match rate: {100 * stats.matched_records / stats.filtered_records:.1f}%')
        if stats.filter_stats.total_filtered() > 0:
            print('\nFilter Exclusions:')
            print(stats.filter_stats.show())

if __name__ == '__main__':
    main()

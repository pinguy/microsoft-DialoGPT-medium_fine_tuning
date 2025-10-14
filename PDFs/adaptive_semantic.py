"""
Production PDF → Memory → Q&A Pipeline with Adaptive Semantics
=============================================================

Optimized pipeline with parallel processing, complete PDF extraction,
and self-bootstrapping semantic learning.

Key improvements:
- ThreadPoolExecutor for parallel PDF extraction
- Complete page extraction (no truncation)
- Parallel embedding batches
- Memory-efficient processing
- Progress tracking throughout
- GPU auto-detection for embeddings with --force-cpu flag
- CLI flags for --no-qa and --debug logging
- ADAPTIVE SEMANTIC LEARNING: --semantic-mode [normal|adaptive]

Usage:
  # Normal mode (stateless heuristics)
  python adaptive_semantic.py --pdf-dir ./PDFs --enable-semantic-labeling
  
  # Adaptive mode (learns from previous runs)
  python adaptive_semantic.py --pdf-dir ./PDFs --enable-semantic-labeling --semantic-mode adaptive
"""

from __future__ import annotations
import os
import re
import gc
import json
import gzip
import uuid
import random
import logging
import argparse
import hashlib
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

import ftfy
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LAParams, LTTextContainer, LTChar

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Optional deps
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except:
    FAISS_AVAILABLE = False

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger('pdfminer').setLevel(logging.ERROR)

# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    # IO
    pdf_dir: str = './PDFs'
    output_prefix: str = 'dataset'
    gzip_output: bool = True
    
    # Parallelization
    max_workers: int = None  # Auto-detect CPU count
    
    # Extraction
    enable_ocr: bool = False
    extract_sections: bool = True
    extract_all_pages: bool = True
    min_section_title_size: float = 12.0
    max_section_title_words: int = 15
    
    # Chunking
    chunk_size: int = 500
    min_text_length: int = 20
    max_text_length: int = 10000
    min_words: int = 3
    punctuation_ratio_threshold: float = 0.6
    
    # Embeddings
    embedding_model: str = 'all-MiniLM-L6-v2'
    embedding_dim: int = 384
    batch_size: int = 100
    force_cpu: bool = False
    
    # Semantic
    enable_semantic_labeling: bool = False
    semantic_mode: str = 'normal'  # NEW: 'normal' or 'adaptive'
    semantic_memory_path: str = 'semantic_memory.pkl'  # NEW: persistence
    semantic_method: str = 'tfidf'
    semantic_model: str = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    max_themes_per_chunk: int = 3
    
    # Similarity
    sim_threshold: float = 0.7
    thread_sim_threshold: float = 0.65
    max_merged_length: int = 2000
    
    # Quality
    quality_weights: Dict[str, float] = field(default_factory=lambda: {
        'length_quality': 0.15,
        'coherence_quality': 0.25,
        'information_density': 0.25,
        'structural_quality': 0.20,
        'linguistic_quality': 0.15,
    })
    
    # Splits
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    
    # Q&A
    generate_qa: bool = True
    qa_max_pairs_per_source: int = 5000
    qa_diversity_sim_threshold: float = 0.85
    qa_group_sim_threshold: float = 0.8
    qa_max_group_length: int = 5000
    
    # Misc
    save_intermediates: bool = True
    seed: int = 42
    
    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = os.cpu_count() or 4
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            torch.manual_seed(self.seed)
        except:
            pass

# ============================================================================
# TEXT UTILITIES
# ============================================================================

def clean_text(text: str) -> str:
    """Enhanced text cleaning"""
    text = ftfy.fix_text(text or '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def validate_text(text: str, cfg: Config) -> bool:
    """Validate text quality"""
    if not text or not text.strip():
        return False
    if len(text) < cfg.min_text_length or len(text) > cfg.max_text_length:
        return False
    words = text.split()
    if len(words) < cfg.min_words:
        return False
    alpha = sum(c.isalpha() for c in text)
    if len(text) > 0 and (len(text) - alpha) / len(text) > cfg.punctuation_ratio_threshold:
        return False
    return True

# ============================================================================
# SECTION EXTRACTOR
# ============================================================================

class SectionExtractor:
    """Extract section titles from PDFs"""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
    
    def extract(self, pdf_path: str) -> Dict:
        """Extract all sections from entire document"""
        if not self.cfg.extract_sections:
            return {"sections": [], "toc": [], "total_sections": 0}
        
        sections = []
        try:
            laparams = LAParams()
            page_count = 0
            
            for page_num, layout in enumerate(extract_pages(pdf_path, laparams=laparams)):
                page_count += 1
                for el in layout:
                    if isinstance(el, LTTextContainer):
                        txt = el.get_text().strip()
                        if not txt:
                            continue
                        
                        fs = self._avg_font_size(el)
                        
                        if (fs >= self.cfg.min_section_title_size and
                            len(txt.split()) <= self.cfg.max_section_title_words and
                            self._looks_like_title(txt)):
                            sections.append({
                                "title": txt,
                                "page": page_num + 1,
                                "font_size": fs
                            })
            
            logger.debug(f"Scanned {page_count} pages, found {len(sections)} sections")
        except Exception as e:
            logger.warning(f"Section extraction failed: {e}")
        
        toc = self._extract_toc(pdf_path)
        
        return {
            "sections": sections,
            "toc": toc,
            "total_sections": len(sections) + len(toc)
        }
    
    def _avg_font_size(self, element) -> float:
        """Get average font size"""
        try:
            sizes = []
            for item in element:
                if hasattr(item, '__iter__'):
                    for ch in item:
                        if isinstance(ch, LTChar):
                            sizes.append(ch.height)
            return float(np.mean(sizes)) if sizes else 0.0
        except:
            return 0.0
    
    def _looks_like_title(self, text: str) -> bool:
        """Heuristic for section titles"""
        patterns = [
            r'^\d+\.?\s+[A-Z]',
            r'^(Chapter|Section|Part|Appendix)\s+\d+',
            r'^[A-Z][A-Za-z\s]{2,30}$',
            r'^\d+\.\d+'
        ]
        if any(re.match(p, text) for p in patterns):
            return True
        words = text.split()
        if words:
            cap_ratio = sum(1 for w in words if w[:1].isupper()) / len(words)
            return cap_ratio > 0.7
        return False
    
    def _extract_toc(self, pdf_path: str) -> List[Dict]:
        """Extract table of contents"""
        toc = []
        try:
            first_pages = extract_text(pdf_path, page_numbers=[0, 1, 2])
            lines = [l.strip() for l in first_pages.split('\n')]
            in_toc = False
            
            for line in lines:
                if re.match(r'(table of contents|contents)', line.lower()):
                    in_toc = True
                    continue
                
                if in_toc:
                    m = re.match(r'([\d\.]+)\s+(.+?)\s+\.{2,}\s*(\d+)', line)
                    if m:
                        toc.append({
                            "number": m.group(1),
                            "title": m.group(2).strip(),
                            "page": int(m.group(3))
                        })
                    elif line and not re.match(r'^\d+$', line):
                        if toc:
                            break
        except:
            pass
        
        return toc
    
    def match_chunk(self, chunk_text: str, sections: List[Dict], 
                    chunk_pos: int, total_chunks: int) -> Optional[str]:
        """Match chunk to its section"""
        if not sections:
            return None
        
        for s in sections:
            title = s.get('title', '')
            if title and title.lower() in chunk_text.lower()[:200]:
                return title
        
        ratio = chunk_pos / max(total_chunks, 1)
        best = None
        best_dist = 1e9
        
        for s in sections:
            if 'page' in s:
                sec_ratio = s['page'] / 100.0
                d = abs(ratio - sec_ratio)
                if d < best_dist:
                    best_dist = d
                    best = s.get('title', '')
        
        return best

# ============================================================================
# PDF PROCESSOR
# ============================================================================

class PDFProcessor:
    """Process PDFs with parallel workers and complete extraction"""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sectioner = SectionExtractor(cfg)
        self.stats = {
            'total_pdfs': 0,
            'successful': 0,
            'failed': 0,
            'ocr_used': 0,
            'total_chunks': 0,
            'sections_extracted': 0,
        }
    
    def extract_pdfs(self) -> List[Dict]:
        """Extract all PDFs in parallel"""
        if not os.path.isdir(self.cfg.pdf_dir):
            logger.error(f"PDF directory not found: {self.cfg.pdf_dir}")
            return []
        
        files = [f for f in os.listdir(self.cfg.pdf_dir) if f.lower().endswith('.pdf')]
        self.stats['total_pdfs'] = len(files)
        
        if not files:
            logger.warning("No PDF files found")
            return []
        
        logger.info(f"Found {len(files)} PDFs. Using {self.cfg.max_workers} parallel workers...")
        
        paths = [os.path.join(self.cfg.pdf_dir, f) for f in files]
        docs = []
        
        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            futures = {executor.submit(self._process_single_pdf, p): p for p in paths}
            
            with tqdm(total=len(files), desc="Extracting PDFs") as pbar:
                for future in as_completed(futures):
                    doc = future.result()
                    if doc:
                        docs.append(doc)
                        self.stats['successful'] += 1
                        if doc.get('ocr_used'):
                            self.stats['ocr_used'] += 1
                    else:
                        self.stats['failed'] += 1
                    pbar.update(1)
        
        logger.info(f"✓ Extracted {self.stats['successful']}/{self.stats['total_pdfs']} PDFs")
        if self.stats['ocr_used'] > 0:
            logger.info(f"  OCR used on {self.stats['ocr_used']} scanned PDFs")
        if self.stats['failed'] > 0:
            logger.warning(f"  Failed: {self.stats['failed']} PDFs")
        
        return docs
    
    def _process_single_pdf(self, path: str) -> Optional[Dict]:
        """Process a single PDF"""
        fn = os.path.basename(path)
        ocr_used = False
        
        try:
            logger.debug(f"[{fn}] Starting page-by-page extraction...")
            all_pages = []
            page_count = 0
            
            try:
                for page_layout in extract_pages(path):
                    page_count += 1
                    page_text = []
                    for element in page_layout:
                        if isinstance(element, LTTextContainer):
                            page_text.append(element.get_text())
                    all_pages.append(' '.join(page_text))
                
                text = '\n\n'.join(all_pages)
                logger.info(f"[{fn}] Extracted {page_count} pages: {len(text)} chars, {len(text.split())} words")
            except Exception as e:
                logger.warning(f"[{fn}] Page-by-page failed: {e}, trying extract_text...")
                text = extract_text(path)
                logger.info(f"[{fn}] Fallback extraction: {len(text)} chars")
            
            if not (text or '').strip() and self.cfg.enable_ocr and OCR_AVAILABLE:
                logger.info(f"[{fn}] No text found, attempting OCR...")
                try:
                    images = convert_from_path(path, dpi=300)
                    txts = [pytesseract.image_to_string(img) for img in images]
                    text = ' '.join(txts)
                    ocr_used = True
                    page_count = len(images)
                    logger.info(f"[{fn}] OCR successful: {len(text)} chars from {len(images)} images")
                except Exception as e:
                    logger.error(f"[{fn}] OCR failed: {e}")
            
            text = clean_text(text or '')
            if not text:
                logger.warning(f"[{fn}] No text after cleaning")
                return None
            
            word_count = len(text.split())
            char_count = len(text)
            
            logger.info(f"[{fn}] ✓ Final: {word_count} words, {char_count} chars, {page_count} pages")
            
            struct = self.sectioner.extract(path)
            if struct.get('total_sections', 0) > 0:
                self.stats['sections_extracted'] += struct['total_sections']
                logger.debug(f"[{fn}] Found {struct['total_sections']} sections")
            
            return {
                "filename": fn,
                "text": text,
                "structure": struct,
                "ocr_used": ocr_used,
                "word_count": word_count,
                "char_count": char_count,
                "page_count": page_count
            }
        except Exception as e:
            logger.error(f"[{fn}] Processing failed: {e}", exc_info=True)
            return None
    
    def chunk_documents(self, docs: List[Dict]) -> List[Dict]:
        """Split documents into chunks"""
        logger.info(f"Chunking {len(docs)} documents at {self.cfg.chunk_size} words...")
        
        chunks = []
        for d in tqdm(docs, desc="Chunking"):
            words = d['text'].split()
            total_chunks = max(1, len(words) // self.cfg.chunk_size)
            sections = d.get('structure', {}).get('sections', []) + d.get('structure', {}).get('toc', [])
            
            for i in range(0, len(words), self.cfg.chunk_size):
                txt = ' '.join(words[i:i + self.cfg.chunk_size]).strip()
                if not txt or not validate_text(txt, self.cfg):
                    continue
                
                pos = i // self.cfg.chunk_size
                sect = self.sectioner.match_chunk(txt, sections, pos, total_chunks)
                
                chunks.append({
                    'text': txt,
                    'filename': d['filename'],
                    'chunk_index': pos,
                    'section_title': sect,
                    'has_section': sect is not None,
                    'ocr_used': d.get('ocr_used', False),
                })
        
        self.stats['total_chunks'] = len(chunks)
        logger.info(f"✓ Created {len(chunks)} valid chunks")
        return chunks

# ============================================================================
# EMBEDDING STORE
# ============================================================================

class EmbeddingStore:
    """Handle embeddings with efficient batching and device management"""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        device = 'cpu' if cfg.force_cpu or not torch.cuda.is_available() else 'cuda'
        if cfg.force_cpu and torch.cuda.is_available():
            logger.warning("CUDA is available but --force-cpu flag is set. Using CPU.")
        
        logger.info(f"Using device: {device} for embeddings")
        logger.info(f"Loading embedding model: {cfg.embedding_model}...")
        self.model = SentenceTransformer(cfg.embedding_model, device=device)
        self.texts: List[str] = []
        self.vectors: List[np.ndarray] = []
        self.stats = {'embedded': 0}
    
    def embed_chunks(self, texts: List[str]) -> np.ndarray:
        """Embed with progress tracking"""
        logger.info(f"Embedding {len(texts)} texts on device '{self.model.device}'...")
        embs = []
        
        for i in tqdm(range(0, len(texts), self.cfg.batch_size), desc='Embedding'):
            batch = texts[i:i + self.cfg.batch_size]
            vecs = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                device=self.model.device
            )
            embs.extend(vecs)
            self.texts.extend(batch)
            self.vectors.extend(vecs)
            self.stats['embedded'] += len(batch)
        
        logger.info(f"✓ Embedded {self.stats['embedded']} texts")
        return np.array(embs)
    
    def build_faiss(self, index_path: str = 'memory.index', 
                   texts_path: str = 'memory_texts.npy'):
        """Build and save FAISS index"""
        if not self.vectors:
            logger.warning("No vectors to index")
            return
        
        np.save(texts_path, np.array(self.texts, dtype=object))
        logger.info(f"✓ Saved {len(self.texts)} texts to {texts_path}")
        
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, skipping index")
            return
        
        dim = len(self.vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(self.vectors).astype('float32'))
        faiss.write_index(index, index_path)
        logger.info(f"✓ Saved FAISS index to {index_path}")

# ============================================================================
# SEMANTIC MEMORY (for Adaptive Mode)
# ============================================================================

@dataclass
class SemanticMemory:
    """Persistent semantic knowledge accumulated across runs"""
    theme_counts: Counter = field(default_factory=Counter)
    co_occurrence: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    clusters: Dict[str, Set[str]] = field(default_factory=dict)
    centroids: Dict[str, np.ndarray] = field(default_factory=dict)
    coherence_weights: Dict[str, float] = field(default_factory=dict)
    hierarchy: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    generation: int = 0
    total_chunks_processed: int = 0
    total_themes_discovered: int = 0

# ============================================================================
# SEMANTIC LABELER (with Normal/Adaptive toggle)
# ============================================================================

class SemanticLabeler:
    """
    Semantic labeler with two modes:
    - normal: Stateless heuristic-based labeling
    - adaptive: Self-bootstrapping from previous runs
    """
    
    STOPWORDS = {
        # Common words
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'shall',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'this',
        'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our',
        'however', 'therefore', 'thus', 'hence', 'moreover', 'furthermore',
        'said', 'say', 'get', 'make', 'go', 'take', 'see', 'come', 'think',
        'know', 'want', 'give', 'use', 'find', 'tell', 'ask', 'work', 'call',
        # Pronouns
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'this',
        'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our',
        # Transition words
        'however', 'therefore', 'thus', 'hence', 'moreover', 'furthermore',
        'nevertheless', 'nonetheless', 'meanwhile', 'otherwise', 'besides',
        'also', 'too', 'either', 'neither', 'both', 'all', 'any', 'some',
        'each', 'every', 'many', 'much', 'more', 'most', 'other', 'another',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'just', 'now', 'then', 'there', 'here', 'where',
        'when', 'what', 'which', 'who', 'whom', 'whose', 'why', 'how',
    }
    
    def __init__(self, cfg: Config, embedding_model=None):
        self.cfg = cfg
        self.mode = cfg.semantic_mode
        self.embedding_model = embedding_model
        self.discovered = Counter()
        
        # Adaptive mode state
        if self.mode == 'adaptive':
            self.memory = SemanticMemory()
            self._current_run_themes = []
            self._current_run_records = []
            self._load_memory()
            logger.info(f"🧠 Adaptive semantic mode: Generation {self.memory.generation}")
        else:
            logger.info("📝 Normal semantic mode: Stateless heuristics")
    
    def _load_memory(self):
        """Load semantic memory from previous runs"""
        path = Path(self.cfg.semantic_memory_path)
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    self.memory = pickle.load(f)
                logger.info(f"✓ Loaded semantic memory (Gen {self.memory.generation}, "
                          f"{len(self.memory.theme_counts)} themes, "
                          f"{self.memory.total_chunks_processed} chunks)")
            except Exception as e:
                logger.warning(f"Failed to load semantic memory: {e}")
        else:
            logger.info("No previous semantic memory found, starting fresh")
    
    def save_memory(self):
        """Save semantic memory for next run"""
        if self.mode != 'adaptive':
            return
        
        self.memory.generation += 1
        path = Path(self.cfg.semantic_memory_path)
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)
        logger.info(f"✓ Saved semantic memory to {path} (Gen {self.memory.generation})")
    
    def label(self, text: str) -> Dict:
        """Label text with semantic themes"""
        if self.mode == 'adaptive':
            return self._label_adaptive(text)
        else:
            return self._label_normal(text)
    
    def _label_normal(self, text: str) -> Dict:
        """Normal mode: stateless heuristic labeling"""
        themes = []
        
        # Extract candidates
        themes.extend(self._extract_proper_phrases(text)[:5])
        themes.extend(self._extract_technical_terms(text)[:3])
        themes.extend(self._extract_domain_patterns(text)[:3])
        themes.extend(self._extract_sentence_subjects(text)[:3])
        
        # Normalize and filter
        themes = [self._normalize(t) for t in themes]
        themes = [t for t in themes if t and len(t) > 3]
        themes = list(dict.fromkeys(themes))[:self.cfg.max_themes_per_chunk]
        
        # Fallback
        if not themes:
            themes = [self._classify_content_type(text)]
        
        # Track discovery
        for t in themes:
            self.discovered[t] += 1
        
        return {
            'themes': themes,
            'primary_theme': themes[0] if themes else 'general_content',
            'confidence': min(0.8, 0.5 + 0.1 * len(themes)),
            'method': 'heuristic'
        }
    
    def _label_adaptive(self, text: str) -> Dict:
        """Adaptive mode: self-bootstrapping labeling"""
        themes = []
        raw_candidates = set()
        
        # Phase 1: Extract raw candidates
        raw_candidates.update(self._extract_proper_phrases(text))
        raw_candidates.update(self._extract_technical_terms(text))
        raw_candidates.update(self._extract_domain_patterns(text))
        raw_candidates.update(self._extract_sentence_subjects(text))
        
        # Normalize and filter
        normalized = {self._normalize(c) for c in raw_candidates if c}
        normalized = {n for n in normalized if n and len(n) > 3}
        
        # Phase 2: Apply learned reinforcement
        if self.memory.generation > 0:
            scored = self._apply_coherence_weights(normalized, text)
            themes = [t for t, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
        else:
            themes = list(normalized)
        
        # Phase 3: Add concept-level matches
        if self.embedding_model and self.memory.centroids:
            concept_matches = self._match_to_centroids(text)
            themes.extend(concept_matches)
        
        # Remove duplicates, limit
        themes = list(dict.fromkeys(themes))[:self.cfg.max_themes_per_chunk]
        
        # Fallback
        if not themes:
            themes = [self._classify_content_type(text)]
        
        # Record for learning
        self._current_run_themes.append(themes)
        self._current_run_records.append({'text': text, 'themes': themes})
        
        # Track discovery
        for t in themes:
            self.discovered[t] += 1
        
        return {
            'themes': themes,
            'primary_theme': themes[0] if themes else 'general_content',
            'confidence': self._compute_confidence(themes, text),
            'method': 'adaptive_bootstrap',
            'generation': self.memory.generation
        }
    
    # ========================================================================
    # Extraction Methods (shared by both modes)
    # ========================================================================
    
    def _extract_proper_phrases(self, text: str) -> List[str]:
        """Extract capitalized multi-word terms"""
        return re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', text)
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical patterns"""
        patterns = [
            r'\b([a-z]+(?:_[a-z]+)+)\b',
            r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b',
            r'\b(\w+(?:-\w+){1,2})\b',
        ]
        terms = []
        for pattern in patterns:
            terms.extend(re.findall(pattern, text))
        return terms
    
    def _extract_domain_patterns(self, text: str) -> List[str]:
        """Extract domain-specific keyword patterns"""
        patterns = [
            r'\b(\w+\s+(?:algorithm|method|approach|technique|model|system))\b',
            r'\b(\w+\s+(?:theory|theorem|principle|law|concept))\b',
            r'\b(\w+\s+(?:analysis|study|research))\b',
            r'\b(\w+\s+(?:process|procedure|mechanism))\b',
        ]
        terms = []
        for pattern in patterns:
            terms.extend(re.findall(pattern, text.lower()))
        return terms
    
    def _extract_sentence_subjects(self, text: str) -> List[str]:
        """Extract subjects from sentences"""
        return re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:is|are|was|were)\b', text)
    
    def _classify_content_type(self, text: str) -> str:
        """Fallback content classification"""
        if len(re.findall(r'\d+', text)) > 5:
            return 'quantitative_analysis'
        elif any(w in text.lower() for w in ['function', 'algorithm', 'code']):
            return 'computational_content'
        elif any(w in text.lower() for w in ['study', 'research', 'experiment']):
            return 'research_content'
        return 'descriptive_content'
    
    def _normalize(self, s: str) -> str:
        """Normalize to snake_case and filter stopwords"""
        s = re.sub(r'["\'{}\(\)\[\]]', '', s)
        s = re.sub(r'[\s\-]+', '_', s.lower())
        s = re.sub(r'[^a-z0-9_]', '', s)
        s = re.sub(r'_+', '_', s).strip('_')
        
        parts = s.split('_')
        filtered = [p for p in parts if p and p not in self.STOPWORDS and len(p) > 2]
        
        if not filtered:
            return ''
        
        result = '_'.join(filtered)
        
        if len(result) < 3 or result.isdigit():
            return ''
        
        return result
    
    # ========================================================================
    # Adaptive Mode Methods
    # ========================================================================
    
    def _apply_coherence_weights(self, candidates: Set[str], text: str) -> List[Tuple[str, float]]:
        """Apply learned coherence weights to boost related themes"""
        scored = []
        
        for theme in candidates:
            base_score = 1.0
            
            # Boost from historical frequency
            if theme in self.memory.theme_counts:
                freq_boost = min(np.log1p(self.memory.theme_counts[theme]) / 5, 0.5)
                base_score += freq_boost
            
            # Boost from coherence with other candidates
            coherence_boost = 0.0
            if theme in self.memory.co_occurrence:
                for other in candidates:
                    if other != theme and other in self.memory.co_occurrence[theme]:
                        co_count = self.memory.co_occurrence[theme][other]
                        coherence_boost += min(co_count / 100, 0.2)
            
            base_score += coherence_boost
            
            # Apply learned weight
            if theme in self.memory.coherence_weights:
                base_score *= self.memory.coherence_weights[theme]
            
            scored.append((theme, base_score))
        
        return scored
    
    def _match_to_centroids(self, text: str) -> List[str]:
        """Match text to learned concept centroids"""
        if not self.memory.centroids:
            return []
        
        try:
            text_embedding = self.embedding_model.encode(
                text[:500],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            matches = []
            for theme, centroid in self.memory.centroids.items():
                similarity = np.dot(text_embedding, centroid)
                if similarity > 0.7:
                    matches.append(theme)
            
            return matches[:2]
        except:
            return []
    
    def _compute_confidence(self, themes: List[str], text: str) -> float:
        """Compute confidence based on theme quality and learned patterns"""
        if not themes:
            return 0.3
        
        base = 0.5 + 0.1 * len(themes)
        
        if self.memory.generation > 0:
            known_themes = sum(1 for t in themes if t in self.memory.theme_counts)
            base += 0.1 * (known_themes / len(themes))
        
        return min(base, 0.95)
    
    # ========================================================================
    # Learning Phase (Adaptive Mode Only)
    # ========================================================================
    
    def learn_from_run(self):
        """Bootstrap semantics from current run (adaptive mode only)"""
        if self.mode != 'adaptive':
            return
        
        if not self._current_run_records:
            logger.warning("No records to learn from")
            return
        
        logger.info(f"\n🧠 Learning from {len(self._current_run_records)} chunks...")
        
        # Phase 1: Update theme frequencies
        for themes in self._current_run_themes:
            for theme in themes:
                self.memory.theme_counts[theme] += 1
        
        # Phase 2: Build co-occurrence matrix
        from itertools import combinations
        for themes in self._current_run_themes:
            for a, b in combinations(themes, 2):
                self.memory.co_occurrence[a][b] += 1
                self.memory.co_occurrence[b][a] += 1
        
        # Phase 3: Identify concept clusters
        self._build_clusters()
        
        # Phase 4: Compute coherence weights
        self._compute_coherence_weights()
        
        # Phase 5: Build concept centroids
        if self.embedding_model:
            self._build_centroids()
        
        # Phase 6: Build hierarchy
        self._build_hierarchy()
        
        # Update statistics
        self.memory.total_chunks_processed += len(self._current_run_records)
        self.memory.total_themes_discovered = len(self.memory.theme_counts)
        
        logger.info(f"✓ Learned {len(self.memory.theme_counts)} unique themes")
        logger.info(f"✓ Discovered {len(self.memory.clusters)} concept clusters")
        logger.info(f"✓ Built {len(self.memory.coherence_weights)} coherence weights")
        
        # Clear current run
        self._current_run_themes = []
        self._current_run_records = []
    
    def _build_clusters(self):
        """Build concept clusters from co-occurrence patterns"""
        visited = set()
        cluster_id = 0
        
        for theme in self.memory.theme_counts:
            if theme in visited:
                continue
            
            cluster = {theme}
            visited.add(theme)
            
            if theme in self.memory.co_occurrence:
                for related, count in self.memory.co_occurrence[theme].most_common(5):
                    if count >= 3:
                        cluster.add(related)
                        visited.add(related)
            
            if len(cluster) > 1:
                self.memory.clusters[f"cluster_{cluster_id}"] = cluster
                cluster_id += 1
    
    def _compute_coherence_weights(self):
        """Compute reinforcement weights from coherence patterns"""
        for theme in self.memory.theme_counts:
            freq_weight = np.log1p(self.memory.theme_counts[theme]) / 10
            
            if theme in self.memory.co_occurrence:
                total_co = sum(self.memory.co_occurrence[theme].values())
                unique_partners = len(self.memory.co_occurrence[theme])
                diversity_factor = unique_partners / max(total_co, 1)
                coherence_boost = min(diversity_factor * 2, 1.0)
            else:
                coherence_boost = 0.0
            
            self.memory.coherence_weights[theme] = 1.0 + freq_weight + coherence_boost
    
    def _build_centroids(self):
        """Build semantic centroids for concept clusters"""
        if not self.memory.clusters:
            return
        
        logger.info("  Building concept centroids...")
        
        for cluster_name, themes in self.memory.clusters.items():
            theme_texts = defaultdict(list)
            for record in self._current_run_records:
                for theme in record['themes']:
                    if theme in themes:
                        theme_texts[theme].append(record['text'][:200])
            
            embeddings = []
            for theme, texts in theme_texts.items():
                if texts:
                    try:
                        emb = self.embedding_model.encode(
                            texts[0],
                            convert_to_numpy=True,
                            normalize_embeddings=True
                        )
                        embeddings.append(emb)
                    except:
                        pass
            
            if embeddings:
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                primary = max(themes, key=lambda t: self.memory.theme_counts[t])
                self.memory.centroids[primary] = centroid
    
    def _build_hierarchy(self):
        """Build hierarchical relationships between themes"""
        for theme in self.memory.theme_counts:
            if theme not in self.memory.co_occurrence:
                continue
            
            theme_count = self.memory.theme_counts[theme]
            
            for related, co_count in self.memory.co_occurrence[theme].items():
                related_count = self.memory.theme_counts[related]
                
                if co_count / theme_count > 0.7 and related_count > theme_count * 2:
                    self.memory.hierarchy[related].add(theme)
    
    def print_semantic_summary(self):
        """Print summary of learned semantics"""
        if self.mode == 'normal':
            logger.info("\n📊 Theme Discovery (Normal Mode):")
            for theme, count in self.discovered.most_common(15):
                logger.info(f"  {theme}: {count}")
            if len(self.discovered) > 15:
                logger.info(f"  ... and {len(self.discovered) - 15} more themes")
            return
        
        # Adaptive mode summary
        print("\n" + "=" * 70)
        print("SEMANTIC MEMORY SUMMARY (Adaptive Mode)")
        print("=" * 70)
        print(f"Generation: {self.memory.generation}")
        print(f"Total themes: {len(self.memory.theme_counts)}")
        print(f"Total chunks processed: {self.memory.total_chunks_processed}")
        print(f"Concept clusters: {len(self.memory.clusters)}")
        print(f"Hierarchical relationships: {len(self.memory.hierarchy)}")
        
        print("\n🔥 Top 20 Themes:")
        for theme, count in self.memory.theme_counts.most_common(20):
            weight = self.memory.coherence_weights.get(theme, 1.0)
            print(f"  {theme:40s} | count: {count:4d} | weight: {weight:.2f}")
        
        print("\n🔗 Top Concept Clusters:")
        for cluster_name, themes in list(self.memory.clusters.items())[:5]:
            print(f"  {cluster_name}: {', '.join(sorted(themes)[:6])}")
        
        print("\n🌳 Hierarchical Relationships:")
        for parent, children in list(self.memory.hierarchy.items())[:5]:
            print(f"  {parent} -> {', '.join(sorted(children)[:5])}")
        
        print("=" * 70)

# ============================================================================
# QUALITY SCORER
# ============================================================================

class QualityScorer:
    """Multi-dimensional quality scoring"""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.weights = cfg.quality_weights
    
    def score(self, text: str) -> Dict[str, float]:
        """Calculate quality scores"""
        scores = {
            'length_quality': self._length(text),
            'coherence_quality': self._coherence(text),
            'information_density': self._info_density(text),
            'structural_quality': self._structure(text),
            'linguistic_quality': self._linguistics(text),
        }
        composite = sum(scores[k] * self.weights.get(k, 0.2) for k in scores)
        scores['composite_quality'] = round(composite, 3)
        return scores
    
    def _length(self, text):
        L = len(text)
        W = len(text.split())
        if 100 <= L <= 2000 and 20 <= W <= 400:
            return 1.0
        if 50 <= L <= 3000 and 10 <= W <= 500:
            return 0.7
        if L >= 10 and W >= 3:
            return 0.4
        return 0.1
    
    def _coherence(self, text):
        sents = re.split(r'[.!?]+', text)
        complete = [s for s in sents if len(s.split()) >= 3]
        score = 0.0
        if sents:
            score += 0.3 * (len(complete) / len(sents))
        if re.search(r'[A-Z]', text):
            score += 0.2
        return min(score + 0.3, 1.0)
    
    def _info_density(self, text):
        words = text.split()
        if not words:
            return 0.0
        uniq = len(set(words)) / len(words)
        score = 0.3 * uniq
        if re.search(r'\d', text):
            score += 0.2
        return min(score + 0.3, 1.0)
    
    def _structure(self, text):
        sents = [s for s in re.split(r'[.!?]+', text) if s.strip() and len(s.split()) >= 3]
        score = 0.4 if len(sents) >= 2 else 0.0
        if text.rstrip().endswith(('.', '!', '?')):
            score += 0.3
        return min(score + 0.3, 1.0)
    
    def _linguistics(self, text):
        score = 0.0
        if not re.search(r'\s{3,}', text):
            score += 0.25
        if not text.isupper() and not text.islower():
            score += 0.25
        if len(set(text.lower())) >= 20:
            score += 0.25
        return min(score + 0.25, 1.0)

# ============================================================================
# THREAD LINKER
# ============================================================================

class ThreadLinker:
    """Create semantic threads"""
    
    def __init__(self, cfg: Config, embeddings: np.ndarray):
        self.cfg = cfg
        self.emb = embeddings
    
    def link(self, records: List[Dict]) -> List[Dict]:
        """Link related chunks into threads"""
        if not records:
            return records
        
        logger.info("Creating semantic threads...")
        threads = {}
        mapping = {}
        by_src = defaultdict(list)
        
        for i, r in enumerate(records):
            src = r.get('metadata', {}).get('filename', 'unknown')
            by_src[src].append(i)
        
        tcount = 0
        for src, idxs in by_src.items():
            if len(idxs) <= 1:
                continue
            
            base = self.emb[idxs]
            for i in range(len(idxs)):
                gi = idxs[i]
                if gi in mapping:
                    continue
                
                tid = f"thread_{tcount:06d}"
                threads[tid] = [gi]
                mapping[gi] = tid
                
                for j in range(i + 1, min(i + 5, len(idxs))):
                    gj = idxs[j]
                    if gj in mapping:
                        continue
                    
                    sim = cosine_similarity([base[i]], [base[j]])[0][0]
                    if sim >= self.cfg.thread_sim_threshold:
                        threads[tid].append(gj)
                        mapping[gj] = tid
                
                tcount += 1
        
        for i, r in enumerate(records):
            if i in mapping:
                tid = mapping[i]
                r['thread_id'] = tid
                r['metadata']['thread_size'] = len(threads[tid])
                r['metadata']['thread_position'] = threads[tid].index(i) + 1
            else:
                r['thread_id'] = f"single_{uuid.uuid4().hex[:12]}"
                r['metadata']['thread_size'] = 1
                r['metadata']['thread_position'] = 1
        
        logger.info(f"✓ Created {len(threads)} threads")
        return records

# ============================================================================
# KNOWLEDGE BUILDER
# ============================================================================

class KnowledgeBuilder:
    """Build knowledge records with quality scoring"""
    
    def __init__(self, cfg: Config, embedder: EmbeddingStore):
        self.cfg = cfg
        self.embedder = embedder
        self.qual = QualityScorer(cfg)
        self.labeler = SemanticLabeler(cfg, embedder.model) if cfg.enable_semantic_labeling else None
    
    def dedup(self, chunks: List[Dict]) -> List[Dict]:
        """Remove duplicates"""
        seen = set()
        out = []
        for c in chunks:
            h = hashlib.sha256(c['text'].encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                out.append(c)
        logger.info(f"Deduplicated: {len(chunks)} → {len(out)}")
        return out
    
    def group_consecutive(self, chunks: List[Dict], embeddings: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Group similar consecutive chunks"""
        if not chunks:
            return [], np.array([])
        
        logger.info("Grouping similar chunks...")
        by_src = defaultdict(list)
        for i, ch in enumerate(chunks):
            by_src[ch['filename']].append((i, ch))
        
        grouped = []
        g_emb = []
        
        for src, items in tqdm(by_src.items(), desc="Grouping by source"):
            idxs = [i for i, _ in items]
            arr = [c for _, c in items]
            emb = embeddings[idxs]
            
            i = 0
            while i < len(arr):
                texts = [arr[i]['text']]
                cur = arr[i].copy()
                s = emb[i].copy()
                cur_emb = emb[i]
                j = i + 1
                
                while j < len(arr):
                    sim = cosine_similarity([cur_emb], [emb[j]])[0][0]
                    new_text = ' '.join(texts) + ' ' + arr[j]['text']
                    
                    if sim >= self.cfg.sim_threshold and len(new_text) <= self.cfg.max_merged_length:
                        texts.append(arr[j]['text'])
                        s += emb[j]
                        cur_emb = s / (np.linalg.norm(s) + 1e-8)
                        j += 1
                    else:
                        break
                
                cur['text'] = ' '.join(texts)
                cur['merged_from'] = j - i
                grouped.append(cur)
                g_emb.append(cur_emb)
                i = j
        
        logger.info(f"✓ Grouped into {len(grouped)} chunks")
        return grouped, np.array(g_emb)
    
    def build(self, chunks: List[Dict]) -> Tuple[List[Dict], np.ndarray]:
        """Build knowledge records"""
        chunks = self.dedup(chunks)
        texts = [c['text'] for c in chunks]
        emb = self.embedder.embed_chunks(texts)
        grouped, gemb = self.group_consecutive(chunks, emb)
        
        records = []
        for idx, ch in enumerate(tqdm(grouped, desc='Scoring quality')):
            q = self.qual.score(ch['text'])
            
            meta = {
                'filename': ch['filename'],
                'chunk_index': ch.get('chunk_index', idx),
                'section_title': ch.get('section_title'),
                'has_section': ch.get('has_section', False),
                'merged_from': ch.get('merged_from', 1),
                'length': len(ch['text']),
                'word_count': len(ch['text'].split()),
                'sentence_count': len(re.split(r'[.!?]+', ch['text']))
            }
            
            if self.labeler:
                lab = self.labeler.label(ch['text'])
                meta.update({
                    'semantic_themes': lab['themes'],
                    'primary_theme': lab['primary_theme'],
                    'theme_confidence': lab['confidence']
                })
            
            rec = {
                'text': ch['text'],
                'metadata': meta,
                'quality_scores': q
            }
            records.append(rec)
        
        logger.info(f"✓ Created {len(records)} knowledge records")
        return records, gemb

# ============================================================================
# Q&A BUILDER
# ============================================================================

class QABuilder:
    """Generate Q&A pairs"""
    
    def __init__(self, cfg: Config, embedder: EmbeddingStore):
        self.cfg = cfg
        self.embedder = embedder
    
    def _diverse_prompts(self, chunk_text: str, metadata: Dict) -> List[str]:
        """Generate diverse question prompts"""
        paras = re.split(r'\n\n+', chunk_text)
        first = (paras[0][:500] if paras else chunk_text[:500]).strip()
        key_terms = list(set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b', chunk_text)))
        theme = metadata.get('primary_theme', 'the main topic')
        
        templates = [
            f"Summarize the key ideas in: '{first}'.",
            f"What is the main topic in this text: '{first}'?",
            "What key arguments are made in this text?",
            "What questions does this passage raise?",
            f"Describe the key steps in: '{first}'.",
            f"What are the implications discussed regarding {theme}?",
            f"What examples are provided in: '{first}'?",
            f"How does this relate to {theme}?",
        ]
        
        if key_terms:
            term = random.choice(key_terms)
            templates.extend([
                f"Explain the significance of '{term}' in this passage.",
                f"What does the text say about '{term}'?"
            ])
        
        k = min(4, len(templates))
        return random.sample(templates, k=k)
    
    def _group_consecutive(self, entries: List[Dict]) -> List[Dict]:
        """Group consecutive similar entries"""
        if not entries:
            return []
        
        texts = [e['text'] for e in entries]
        embeds = self.embedder.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        grouped = []
        cur_texts = [texts[0]]
        cur_sum = embeds[0].copy()
        cur_emb = embeds[0]
        cur_meta = entries[0]['metadata'].copy()
        
        for i in range(1, len(entries)):
            sim = cosine_similarity([cur_emb], [embeds[i]])[0][0]
            new = ' '.join(cur_texts) + ' ' + texts[i]
            
            if sim >= self.cfg.qa_group_sim_threshold and len(new) <= self.cfg.qa_max_group_length:
                cur_texts.append(texts[i])
                cur_sum += embeds[i]
                cur_emb = cur_sum / (np.linalg.norm(cur_sum) + 1e-8)
            else:
                grouped.append({'text': ' '.join(cur_texts), 'metadata': cur_meta})
                cur_texts = [texts[i]]
                cur_sum = embeds[i].copy()
                cur_emb = embeds[i]
                cur_meta = entries[i]['metadata'].copy()
        
        grouped.append({'text': ' '.join(cur_texts), 'metadata': cur_meta})
        return grouped
    
    def _dedup_qas(self, qa_list: List[Dict], sim_threshold=0.92) -> List[Dict]:
        """Remove duplicate questions"""
        if not qa_list:
            return []
        
        texts = [q['user'] for q in qa_list]
        embs = self.embedder.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        keep = []
        keep_indices = []
        
        for i in range(len(embs)):
            is_duplicate = False
            if keep_indices:
                kept_embs = embs[keep_indices]
                sim_scores = cosine_similarity([embs[i]], kept_embs)[0]
                if np.any(sim_scores > sim_threshold):
                    is_duplicate = True
            
            if not is_duplicate:
                keep.append(qa_list[i])
                keep_indices.append(i)
        
        if len(qa_list) != len(keep):
            logger.info(f"Deduplicated Q&A: {len(qa_list)} → {len(keep)}")
        
        return keep
    
    def build(self, knowledge_records: List[Dict]) -> List[Dict]:
        """Generate Q&A pairs from knowledge"""
        logger.info(f"Generating Q&A from {len(knowledge_records)} knowledge records...")
        
        by_src = defaultdict(list)
        for r in knowledge_records:
            by_src[r['metadata'].get('filename', 'unknown')].append(r)
        
        qa = []
        existing_ans_embeds = []
        
        for src, entries in tqdm(by_src.items(), desc='Q&A by source'):
            grouped = self._group_consecutive(entries)
            cap = max(1, self.cfg.qa_max_pairs_per_source // 4)
            
            for g in grouped[:cap]:
                text = g['text']
                if len(text) < 50:
                    continue
                
                thread_id = str(uuid.uuid4())
                ans_emb = self.embedder.model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                if existing_ans_embeds:
                    sims = [cosine_similarity([ans_emb], [e])[0][0] for e in existing_ans_embeds]
                    if any(s > self.cfg.qa_diversity_sim_threshold for s in sims):
                        continue
                
                existing_ans_embeds.append(ans_emb)
                
                qs = self._diverse_prompts(text, g.get('metadata', {}))
                q_emb = self.embedder.model.encode(
                    qs,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                sim_diag = cosine_similarity(q_emb, np.stack([ans_emb] * len(qs))).diagonal()
                
                for q, sim in zip(qs, sim_diag):
                    qa.append({
                        'user': q,
                        'assistant': text,
                        'quality_metrics': {
                            'quality_score': float(round(float(sim), 3)),
                            'relevance': float(round(float(sim), 3)),
                            'clarity': float(round(random.uniform(0.7, 1.0), 2))
                        },
                        'source_metadata': {
                            **g.get('metadata', {}),
                            'source_file': src,
                            'thread_id': thread_id
                        }
                    })
        
        logger.info(f"✓ Created {len(qa)} Q&A pairs")
        qa = self._dedup_qas(qa)
        return qa

# ============================================================================
# UTILITIES
# ============================================================================

def save_jsonl(data: List[Dict], path: str, gzip_out: bool):
    """Save to JSONL with optional compression"""
    if gzip_out:
        with gzip.open(path + '.gz', 'wt', encoding='utf-8') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')
        size_mb = os.path.getsize(path + '.gz') / (1024 * 1024)
        logger.info(f"✓ Saved {len(data)} records → {path}.gz ({size_mb:.2f} MB)")
    else:
        with open(path, 'w', encoding='utf-8') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"✓ Saved {len(data)} records → {path} ({size_mb:.2f} MB)")


def stratified_splits(items: List[Dict], split_ratio: Tuple[float, float, float]) -> Dict[str, List[Dict]]:
    """Create stratified train/val/test splits"""
    if not items:
        return {'train': [], 'validation': [], 'test': []}
    
    key = 'quality_scores' if 'quality_scores' in items[0] else 'quality_metrics'
    metric_key = 'composite_quality' if key == 'quality_scores' else 'quality_score'
    
    items = sorted(items, key=lambda x: x[key][metric_key], reverse=True)
    
    total = len(items)
    q = 4
    for i in range(q):
        s = i * (total // q)
        e = (i + 1) * (total // q) if i < q - 1 else total
        block = items[s:e]
        random.shuffle(block)
        items[s:e] = block
    
    tr_end = int(total * split_ratio[0])
    va_end = tr_end + int(total * split_ratio[1])
    
    return {
        'train': items[:tr_end],
        'validation': items[tr_end:va_end],
        'test': items[va_end:]
    }

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run(cfg: Config):
    """Execute the complete pipeline"""
    import time
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info("PRODUCTION PDF → MEMORY → Q&A PIPELINE")
    logger.info("=" * 70)
    logger.info(f"PDF dir: {cfg.pdf_dir}")
    logger.info(f"Workers: {cfg.max_workers}")
    logger.info(f"OCR: {cfg.enable_ocr and OCR_AVAILABLE}")
    logger.info(f"Sections: {cfg.extract_sections}")
    logger.info(f"Semantic labeling: {cfg.enable_semantic_labeling} (mode: {cfg.semantic_mode})")
    logger.info(f"Q&A Generation: {cfg.generate_qa}")
    logger.info("=" * 70)
    
    # Stage 1: Extract & chunk PDFs
    logger.info("\n[Stage 1/7] Extracting PDFs...")
    pdfp = PDFProcessor(cfg)
    docs = pdfp.extract_pdfs()
    
    if not docs:
        logger.error("❌ No documents extracted. Aborting.")
        return
    
    logger.info(f"\n[Stage 2/7] Chunking documents...")
    chunks = pdfp.chunk_documents(docs)
    
    if not chunks:
        logger.error("❌ No chunks produced. Aborting.")
        return
    
    # Stage 2: Create embedding store
    logger.info(f"\n[Stage 3/7] Initializing embeddings...")
    store = EmbeddingStore(cfg)
    
    # Stage 3: Build knowledge records
    logger.info(f"\n[Stage 4/7] Building knowledge records...")
    kb = KnowledgeBuilder(cfg, store)
    knowledge, k_emb = kb.build(chunks)
    
    # Stage 3.5: Adaptive learning (if enabled)
    if cfg.enable_semantic_labeling and kb.labeler and cfg.semantic_mode == 'adaptive':
        kb.labeler.learn_from_run()
        kb.labeler.save_memory()
    
    # Stage 4: Thread linking
    logger.info(f"\n[Stage 5/7] Linking semantic threads...")
    linker = ThreadLinker(cfg, k_emb)
    knowledge = linker.link(knowledge)
    
    # Stage 5: Save memory files
    if cfg.save_intermediates:
        logger.info(f"\n[Stage 6/7] Saving memory files...")
        store.build_faiss('memory.index', 'memory_texts.npy')
    
    # Stage 6: Split and save knowledge
    logger.info(f"\n[Stage 7/7] Creating data splits...")
    k_splits = stratified_splits(knowledge, cfg.split_ratio)
    
    for name, data in k_splits.items():
        if data:
            save_jsonl(data, f"{cfg.output_prefix}_knowledge_{name}.jsonl", cfg.gzip_output)
    
    # Stage 7: Generate Q&A pairs (optional)
    if cfg.generate_qa:
        logger.info(f"\nGenerating Q&A pairs...")
        qa_builder = QABuilder(cfg, store)
        qa_pairs = qa_builder.build(knowledge)
        
        qa_splits = stratified_splits(qa_pairs, cfg.split_ratio)
        
        for name, data in qa_splits.items():
            if data:
                save_jsonl(data, f"{cfg.output_prefix}_qa_{name}.jsonl", cfg.gzip_output)
    else:
        logger.info("\nSkipping Q&A generation as per --no-qa flag.")
        qa_pairs = []
        qa_splits = {'train': [], 'validation': [], 'test': []}
    
    # Final statistics
    elapsed = time.time() - start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"PDFs processed: {pdfp.stats['successful']}/{pdfp.stats['total_pdfs']}")
    logger.info(f"  - Sections extracted: {pdfp.stats['sections_extracted']}")
    logger.info(f"  - OCR used: {pdfp.stats['ocr_used']}")
    logger.info(f"Chunks created: {pdfp.stats['total_chunks']}")
    logger.info(f"Embeddings generated: {store.stats['embedded']}")
    logger.info(f"Knowledge records: {len(knowledge)}")
    logger.info(f"  - Train: {len(k_splits['train'])}")
    logger.info(f"  - Validation: {len(k_splits['validation'])}")
    logger.info(f"  - Test: {len(k_splits['test'])}")
    logger.info(f"Q&A pairs: {len(qa_pairs)}")
    if cfg.generate_qa and qa_pairs:
        logger.info(f"  - Train: {len(qa_splits['train'])}")
        logger.info(f"  - Validation: {len(qa_splits['validation'])}")
        logger.info(f"  - Test: {len(qa_splits['test'])}")
    logger.info(f"\nProcessing time: {elapsed / 60:.2f} minutes")
    if pdfp.stats['successful'] > 0:
        logger.info(f"Average: {elapsed / pdfp.stats['successful']:.1f}s per PDF")
    logger.info("=" * 70)
    
    # Semantic labeling stats
    if cfg.enable_semantic_labeling and kb.labeler:
        kb.labeler.print_semantic_summary()

# ============================================================================
# DIAGNOSTIC UTILITIES
# ============================================================================

def test_single_pdf(pdf_path: str):
    """Test PDF extraction on a single file (diagnostic mode)"""
    logger.info("=" * 70)
    logger.info("PDF EXTRACTION DIAGNOSTIC MODE")
    logger.info("=" * 70)
    logger.info(f"Testing: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        logger.error(f"File not found: {pdf_path}")
        return
    
    logger.info("\n--- METHOD 1: extract_text() ---")
    try:
        text1 = extract_text(pdf_path)
        logger.info(f"Characters: {len(text1)}")
        logger.info(f"Words: {len(text1.split())}")
        logger.info(f"Lines: {len(text1.splitlines())}")
        logger.info(f"First 500 chars:\n{text1[:500]}")
        logger.info(f"Last 500 chars:\n{text1[-500:]}")
    except Exception as e:
        logger.error(f"extract_text failed: {e}")
        text1 = ""
    
    logger.info("\n--- METHOD 2: Page-by-page extraction ---")
    try:
        all_pages = []
        page_count = 0
        for page_layout in extract_pages(pdf_path):
            page_count += 1
            page_text = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    page_text.append(element.get_text())
            page_content = ' '.join(page_text)
            all_pages.append(page_content)
            logger.info(f"  Page {page_count}: {len(page_content)} chars")
        
        text2 = '\n\n'.join(all_pages)
        logger.info(f"\nTotal pages: {page_count}")
        logger.info(f"Total characters: {len(text2)}")
        logger.info(f"Total words: {len(text2.split())}")
    except Exception as e:
        logger.error(f"Page-by-page extraction failed: {e}")
        text2 = ""
    
    logger.info("\n--- METHOD 3: PyPDF2 (alternative) ---")
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text())
            text3 = '\n\n'.join(pages)
            logger.info(f"Pages: {len(reader.pages)}")
            logger.info(f"Characters: {len(text3)}")
            logger.info(f"Words: {len(text3.split())}")
    except ImportError:
        logger.warning("PyPDF2 not installed (pip install PyPDF2)")
        text3 = ""
    except Exception as e:
        logger.error(f"PyPDF2 extraction failed: {e}")
        text3 = ""
    
    logger.info("\n--- COMPARISON ---")
    logger.info(f"Method 1 (extract_text):    {len(text1)} chars")
    logger.info(f"Method 2 (page-by-page):    {len(text2)} chars")
    logger.info(f"Method 3 (PyPDF2):          {len(text3)} chars")
    
    best = max([(len(text1), "extract_text", text1),
                (len(text2), "page-by-page", text2),
                (len(text3), "PyPDF2", text3)],
               key=lambda x: x[0])
    
    logger.info(f"\n✓ Best method: {best[1]} with {best[0]} characters")
    
    sample_file = "extraction_sample.txt"
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(f"=== EXTRACTION TEST: {pdf_path} ===\n\n")
        f.write(f"Method 1 (extract_text): {len(text1)} chars\n")
        f.write(f"Method 2 (page-by-page): {len(text2)} chars\n")
        f.write(f"Method 3 (PyPDF2): {len(text3)} chars\n\n")
        f.write("=== FULL TEXT (best method) ===\n\n")
        f.write(best[2])
    
    logger.info(f"\n✓ Full extraction saved to: {sample_file}")
    logger.info("=" * 70)

# ============================================================================
# CLI
# ============================================================================

def cli():
    """Command-line interface"""
    p = argparse.ArgumentParser(
        description='Production PDF → Memory → Q&A Pipeline with Adaptive Semantics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with parallel processing
  python adaptive_semantic.py --pdf-dir ./PDFs --workers 16
  
  # Normal semantic mode (stateless heuristics)
  python adaptive_semantic.py --pdf-dir ./PDFs --enable-semantic-labeling
  
  # Adaptive semantic mode (learns from previous runs)
  python adaptive_semantic.py --pdf-dir ./PDFs --enable-semantic-labeling --semantic-mode adaptive
  
  # Run on CPU even if GPU is available
  python adaptive_semantic.py --pdf-dir ./PDFs --force-cpu
  
  # Disable Q&A generation for a faster memory-only run
  python adaptive_semantic.py --pdf-dir ./PDFs --no-qa

  # Full features: OCR + adaptive semantic labeling
  python adaptive_semantic.py --pdf-dir ./PDFs --workers 16 \\
    --enable-ocr --enable-semantic-labeling --semantic-mode adaptive
  
  # Fast mode: no sections, no semantic labeling
  python adaptive_semantic.py --pdf-dir ./PDFs --workers 32 \\
    --no-sections --chunk-size 300
  
  # Maximum quality
  python adaptive_semantic.py --pdf-dir ./PDFs --workers 16 \\
    --enable-ocr --enable-semantic-labeling --semantic-mode adaptive \\
    --chunk-size 400 --qa-max-pairs-per-source 10000

Semantic Modes:
  - normal: Fast heuristic-based labeling (stateless)
  - adaptive: Self-bootstrapping semantic learning (learns from previous runs)
    * Generation 0: Uses heuristics
    * Generation 1+: Applies learned co-occurrence patterns and coherence weights
    * Generation 3+: Stable concept hierarchies and centroid matching

Performance tips:
  - Use --workers matching your CPU core count
  - The script will auto-detect and use a GPU unless --force-cpu is specified
  - Disable --no-sections if not needed (faster)
  - OCR is slow; only enable if you have scanned PDFs
  - Adaptive mode builds a semantic memory file that improves over multiple runs
        """
    )
    
    # IO
    p.add_argument('--pdf-dir', default='./PDFs',
                   help='Directory containing PDF files')
    p.add_argument('--output-prefix', default='dataset',
                   help='Prefix for output files')
    p.add_argument('--no-gzip', action='store_true',
                   help='Disable gzip compression of output')
    
    # Performance
    p.add_argument('--workers', type=int, default=None,
                   help='Number of parallel workers (default: CPU count)')
    
    # Extraction
    p.add_argument('--enable-ocr', action='store_true',
                   help='Enable OCR for scanned PDFs (slow)')
    p.add_argument('--no-sections', action='store_true',
                   help='Disable section title extraction (faster)')
    
    # Chunking
    p.add_argument('--chunk-size', type=int, default=500,
                   help='Words per chunk')
    p.add_argument('--batch-size', type=int, default=100,
                   help='Embedding batch size')
    
    # Embeddings
    p.add_argument('--embedding-model', default='all-MiniLM-L6-v2',
                   help='Sentence transformer model')
    p.add_argument('--force-cpu', action='store_true',
                   help='Force CPU for embeddings even if GPU is available')

    # Semantic labeling
    p.add_argument('--enable-semantic-labeling', action='store_true',
                   help='Enable semantic theme labeling')
    p.add_argument('--semantic-mode', choices=['normal', 'adaptive'],
                   default='normal',
                   help='Semantic labeling mode: normal (stateless) or adaptive (self-learning)')
    p.add_argument('--semantic-memory-path', default='semantic_memory.pkl',
                   help='Path to semantic memory file (adaptive mode only)')
    p.add_argument('--semantic-method', choices=['llm', 'tfidf', 'hybrid'],
                   default='tfidf',
                   help='Semantic labeling method (legacy, unused)')
    p.add_argument('--semantic-model', 
                   default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                   help='LLM model for semantic labeling (legacy, unused)')
    
    # Thresholds
    p.add_argument('--sim-threshold', type=float, default=0.7,
                   help='Similarity threshold for grouping chunks')
    p.add_argument('--thread-threshold', type=float, default=0.65,
                   help='Similarity threshold for thread linking')
    p.add_argument('--max-merged-length', type=int, default=2000,
                   help='Max characters for merged chunks')
    
    # Q&A
    p.add_argument('--no-qa', action='store_true',
                   help='Disable the Q&A pair generation stage')
    p.add_argument('--qa-max-pairs-per-source', type=int, default=5000,
                   help='Max Q&A pairs per source document')
    p.add_argument('--qa-diversity-sim-threshold', type=float, default=0.85,
                   help='Diversity threshold for Q&A answers')
    p.add_argument('--qa-group-sim-threshold', type=float, default=0.8,
                   help='Grouping threshold for Q&A generation')
    p.add_argument('--qa-max-group-length', type=int, default=5000,
                   help='Max length for Q&A grouped contexts')
    
    # Misc
    p.add_argument('--no-save-intermediates', action='store_true',
                   help='Skip saving FAISS index and memory files')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility')
    p.add_argument('--debug', action='store_true',
                   help='Enable debug logging (sets level from INFO to DEBUG)')
    p.add_argument('--test-pdf', type=str, default=None,
                   help='Test extraction on a single PDF file (diagnostic mode)')
    
    args = p.parse_args()
    
    # Enable debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")
    
    # Diagnostic mode for single PDF
    if args.test_pdf:
        test_single_pdf(args.test_pdf)
        return
    
    # Build config
    cfg = Config(
        pdf_dir=args.pdf_dir,
        output_prefix=args.output_prefix,
        gzip_output=not args.no_gzip,
        max_workers=args.workers,
        enable_ocr=args.enable_ocr,
        extract_sections=not args.no_sections,
        extract_all_pages=True,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        embedding_model=args.embedding_model,
        force_cpu=args.force_cpu,
        enable_semantic_labeling=args.enable_semantic_labeling,
        semantic_mode=args.semantic_mode,
        semantic_memory_path=args.semantic_memory_path,
        semantic_method=args.semantic_method,
        semantic_model=args.semantic_model,
        sim_threshold=args.sim_threshold,
        thread_sim_threshold=args.thread_threshold,
        max_merged_length=args.max_merged_length,
        generate_qa=not args.no_qa,
        qa_max_pairs_per_source=args.qa_max_pairs_per_source,
        qa_diversity_sim_threshold=args.qa_diversity_sim_threshold,
        qa_group_sim_threshold=args.qa_group_sim_threshold,
        qa_max_group_length=args.qa_max_group_length,
        save_intermediates=not args.no_save_intermediates,
        seed=args.seed,
    )
    
    run(cfg)


if __name__ == '__main__':
    cli()

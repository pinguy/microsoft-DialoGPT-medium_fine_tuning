"""
Enhanced PDF Processing Pipeline with Semantic Analysis
Converts PDFs → Embeddings → Semantically Labeled Training Data → Compressed JSONL

Features:
- Section/chapter title extraction from PDF structure
- Semantic theme labeling using LLM, TF-IDF, or hybrid approach
- Thread linking for conversation-style interactions
- Dynamic quality scoring with multiple metrics

Usage:
    python pdf_pipeline.py --pdf-dir ./PDFs --output-prefix dataset --semantic-method tfidf
"""

import os
import json
import gzip
import re
import random
import logging
import argparse
import hashlib
import uuid
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import ftfy
import numpy as np
import faiss
import torch
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LAParams, LTTextContainer, LTChar
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional OCR support
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress pdfminer font warnings
logging.getLogger('pdfminer').setLevel(logging.ERROR)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    # PDF Processing
    pdf_dir: str = "./PDFs"
    enable_ocr: bool = True
    max_workers: int = None
    
    # Embedding Configuration
    embedding_model: str = 'all-MiniLM-L6-v2'
    embedding_dim: int = 384
    chunk_size: int = 500
    batch_size: int = 100
    
    # Semantic Labeling
    enable_semantic_labeling: bool = False
    semantic_method: str = "llm"  # "llm", "tfidf", or "hybrid"
    semantic_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    max_themes_per_chunk: int = 3
    semantic_batch_size: int = 8
    use_json_output: bool = True
    
    # Section Extraction
    extract_sections: bool = True
    min_section_title_size: float = 12.0
    max_section_title_words: int = 15
    
    # Text Quality
    min_text_length: int = 20
    max_text_length: int = 10000
    min_words: int = 3
    min_chunk_length: int = 10
    punctuation_ratio_threshold: float = 0.6
    
    # Dynamic Quality Scoring
    quality_weights: Dict[str, float] = field(default_factory=lambda: {
        'length_quality': 0.15,
        'coherence_quality': 0.25,
        'information_density': 0.25,
        'structural_quality': 0.20,
        'linguistic_quality': 0.15
    })
    
    # Similarity Merging & Threading
    sim_threshold: float = 0.7
    max_merged_length: int = 2000
    thread_sim_threshold: float = 0.65
    
    # Data Splits
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    
    # Output
    output_prefix: str = "dataset"
    save_intermediates: bool = False
    compress_output: bool = True
    
    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = os.cpu_count() or 4
        
        logger.info("=" * 70)
        logger.info("Pipeline Configuration:")
        logger.info(f"  PDF Directory: {self.pdf_dir}")
        logger.info(f"  Embedding Model: {self.embedding_model}")
        logger.info(f"  Chunk Size: {self.chunk_size} words")
        logger.info(f"  Parallel Workers: {self.max_workers}")
        logger.info(f"  OCR Enabled: {self.enable_ocr and OCR_AVAILABLE}")
        logger.info(f"  Section Extraction: {self.extract_sections}")
        logger.info(f"  Semantic Labeling: {self.enable_semantic_labeling}")
        if self.enable_semantic_labeling:
            logger.info(f"  Semantic Method: {self.semantic_method}")
            if self.semantic_method in ["llm", "hybrid"]:
                logger.info(f"  Semantic Model: {self.semantic_model}")
        logger.info("=" * 70)


# ============================================================================
# TEXT CLEANING AND VALIDATION
# ============================================================================

def clean_text(text: str) -> str:
    """Enhanced text cleaning with encoding fixes"""
    text = ftfy.fix_text(text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'\b"(\w+)"\b', r'\1', text)
    text = re.sub(r'"([^"]+[.,!?])"', r'\1', text)
    text = re.sub(r'"([A-Z][^"]*?)"(?=\s|$)', r'\1', text)
    text = re.sub(r'\\(["\'])', r'\1', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def validate_text(text: str, config: PipelineConfig) -> bool:
    """Validate text quality"""
    if not isinstance(text, str) or not text.strip():
        return False
    
    text = text.strip()
    
    if len(text) < config.min_text_length or len(text) > config.max_text_length:
        return False
    
    words = text.split()
    if len(words) < config.min_words:
        return False
    
    alpha_chars = sum(c.isalpha() for c in text)
    if len(text) > 0:
        non_alpha_ratio = (len(text) - alpha_chars) / len(text)
        if non_alpha_ratio > config.punctuation_ratio_threshold:
            return False
    
    low_quality = [
        re.compile(r'^[\s\-_=]{10,}$'),
        re.compile(r'^\d+\s*$'),
        re.compile(r'^[^\w\s]{5,}$'),
    ]
    
    return not any(pattern.match(text) for pattern in low_quality)


# ============================================================================
# SEMANTIC LABELING
# ============================================================================

class SemanticLabeler:
    """Labels chunks with dynamic semantic themes using LLM, TF-IDF, or hybrid"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.discovered_themes = set()
        self.theme_clusters = defaultdict(list)
        self.theme_frequencies = Counter()
        
        self.tfidf_vectorizer = None
        self.tfidf_fitted = False
        
        if config.enable_semantic_labeling and config.semantic_method in ["llm", "hybrid"]:
            self._load_model()
        elif config.enable_semantic_labeling and config.semantic_method == "tfidf":
            logger.info("Using TF-IDF method for semantic labeling (no LLM required)")
    
    def _load_model(self):
        """Load the semantic labeling model"""
        try:
            logger.info(f"Loading semantic model: {self.config.semantic_model}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.semantic_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.semantic_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Semantic model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            logger.warning("Falling back to TF-IDF method")
            self.config.semantic_method = "tfidf"
    
    def label_chunk(self, text: str, chunk_idx: Optional[int] = None) -> Dict:
        """Label a single chunk based on configured method"""
        if not self.config.enable_semantic_labeling:
            return {
                "themes": ["general_content"], 
                "confidence": 0.0, 
                "primary_theme": "general_content",
                "theme_keywords": [],
                "method": "none"
            }
        
        if self.config.semantic_method == "tfidf":
            return self._label_tfidf(text, chunk_idx)
        elif self.config.semantic_method == "llm":
            return self._label_llm(text, chunk_idx)
        elif self.config.semantic_method == "hybrid":
            return self._label_hybrid(text, chunk_idx)
        else:
            return self._label_tfidf(text, chunk_idx)
    
    def _label_llm(self, text: str, chunk_idx: Optional[int] = None) -> Dict:
        """Label using LLM"""
        if not self.model:
            return self._label_tfidf(text, chunk_idx)
        
        try:
            if self.config.use_json_output:
                prompt_text = text[:600]
                prompt = '<think>\nAnalyze the following text and identify 1-3 concise semantic themes.\n\nText: ' + prompt_text + '\n\n</think>\n\nOutput as JSON with themes array:\n'
            else:
                prompt_text = text[:600]
                prompt = '<think>\nAnalyze text and identify themes.\n\nText: ' + prompt_text + '\n\n</think>\n\nThemes:'
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.4,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            normalized_themes = []
            keywords = []
            
            if self.config.use_json_output:
                try:
                    json_match = re.search(r'\{[^}]*"themes"[^}]*\}', response)
                    if json_match:
                        parsed = json.loads(json_match.group(0))
                        raw_themes = parsed.get('themes', [])
                    else:
                        raise json.JSONDecodeError("No JSON", "", 0)
                except:
                    themes_part = response.split("JSON:")[-1].strip()
                    themes_part = re.sub(r'[:\(\)\[\]{}"]', '', themes_part)
                    raw_themes = [t.strip() for t in re.split(r'[,;\n]', themes_part) if t.strip()]
            else:
                if "Themes" in response:
                    themes_part = response.split("Themes")[-1]
                else:
                    themes_part = response[len(prompt):].strip()
                themes_part = re.sub(r'[:\(\)\[\]]', '', themes_part).strip()
                raw_themes = [t.strip() for t in re.split(r'[,;\n]', themes_part) if t.strip()]
            
            for theme in raw_themes[:self.config.max_themes_per_chunk]:
                clean_theme = self._normalize_theme(theme)
                if clean_theme and len(clean_theme.split('_')) <= 5:
                    normalized_themes.append(clean_theme)
                    keywords.extend(clean_theme.split('_'))
                    self.discovered_themes.add(clean_theme)
                    self.theme_frequencies[clean_theme] += 1
                    if chunk_idx is not None:
                        self.theme_clusters[clean_theme].append(chunk_idx)
            
            if not normalized_themes:
                return self._label_tfidf(text, chunk_idx)
            
            return {
                "themes": normalized_themes,
                "primary_theme": normalized_themes[0],
                "confidence": min(0.90, 0.6 + 0.1 * len(normalized_themes)),
                "theme_keywords": list(set(keywords))[:10],
                "method": "llm"
            }
        except Exception as e:
            logger.warning(f"LLM labeling failed: {e}")
            return self._label_tfidf(text, chunk_idx)
    
    def _label_tfidf(self, text: str, chunk_idx: Optional[int] = None) -> Dict:
        """Label using TF-IDF"""
        if not self.tfidf_fitted or not self.tfidf_vectorizer:
            themes = self._extract_key_concepts(text)
            return {
                "themes": themes,
                "primary_theme": themes[0] if themes else "general_content",
                "confidence": 0.5,
                "theme_keywords": themes,
                "method": "tfidf_heuristic"
            }
        
        try:
            vec = self.tfidf_vectorizer.transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            scores = vec.toarray()[0]
            top_indices = scores.argsort()[-15:][::-1]
            top_terms = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            themes = []
            keywords = []
            
            for term in top_terms[:self.config.max_themes_per_chunk * 2]:
                normalized = self._normalize_theme(term)
                if normalized and len(normalized) > 3 and normalized not in themes:
                    themes.append(normalized)
                    keywords.append(term)
                    self.discovered_themes.add(normalized)
                    self.theme_frequencies[normalized] += 1
                    if chunk_idx is not None:
                        self.theme_clusters[normalized].append(chunk_idx)
                    if len(themes) >= self.config.max_themes_per_chunk:
                        break
            
            if not themes:
                themes = self._extract_key_concepts(text)
            
            return {
                "themes": themes[:self.config.max_themes_per_chunk],
                "primary_theme": themes[0] if themes else "general_content",
                "confidence": 0.7,
                "theme_keywords": keywords[:10],
                "method": "tfidf"
            }
        except Exception as e:
            logger.warning(f"TF-IDF failed: {e}")
            themes = self._extract_key_concepts(text)
            return {
                "themes": themes,
                "primary_theme": themes[0] if themes else "general_content",
                "confidence": 0.4,
                "theme_keywords": [],
                "method": "tfidf_fallback"
            }
    
    def _label_hybrid(self, text: str, chunk_idx: Optional[int] = None) -> Dict:
        """Combine LLM and TF-IDF"""
        llm_result = self._label_llm(text, chunk_idx=None)
        tfidf_result = self._label_tfidf(text, chunk_idx=None)
        
        combined_themes = []
        seen = set()
        
        if llm_result['confidence'] > 0.7:
            for theme in llm_result['themes']:
                if theme not in seen:
                    combined_themes.append(theme)
                    seen.add(theme)
        
        for theme in tfidf_result['themes']:
            if theme not in seen and len(combined_themes) < self.config.max_themes_per_chunk:
                combined_themes.append(theme)
                seen.add(theme)
        
        for theme in combined_themes:
            self.discovered_themes.add(theme)
            self.theme_frequencies[theme] += 1
            if chunk_idx is not None:
                self.theme_clusters[theme].append(chunk_idx)
        
        all_keywords = list(set(
            llm_result.get('theme_keywords', []) + 
            tfidf_result.get('theme_keywords', [])
        ))[:10]
        
        return {
            "themes": combined_themes[:self.config.max_themes_per_chunk],
            "primary_theme": combined_themes[0] if combined_themes else "general_content",
            "confidence": (llm_result['confidence'] + tfidf_result['confidence']) / 2,
            "theme_keywords": all_keywords,
            "method": "hybrid"
        }
    
    def _normalize_theme(self, theme: str) -> str:
        """Normalize to snake_case"""
        theme = re.sub(r'["\'\(\)\[\]{}]', '', theme)
        theme = re.sub(r'[\s\-]+', '_', theme.lower())
        theme = re.sub(r'[^a-z0-9_]', '', theme)
        theme = re.sub(r'_+', '_', theme).strip('_')
        filler = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        parts = [p for p in theme.split('_') if p not in filler]
        return '_'.join(parts) if parts else ''
    
    def _fit_tfidf(self, texts: List[str]):
        """Fit TF-IDF vectorizer"""
        try:
            logger.info("Fitting TF-IDF vectorizer...")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 3),
                stop_words='english',
                min_df=2,
                max_df=0.8
            )
            self.tfidf_vectorizer.fit(texts)
            self.tfidf_fitted = True
            logger.info("TF-IDF vectorizer fitted")
        except Exception as e:
            logger.warning(f"TF-IDF fitting failed: {e}")
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Basic heuristic extraction"""
        concepts = []
        phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', text)
        for phrase in phrases[:3]:
            normalized = self._normalize_theme(phrase)
            if normalized and len(normalized) > 3:
                concepts.append(normalized)
        
        patterns = {
            'algorithm': r'\b(\w+\s+algorithm)\b',
            'analysis': r'\b(\w+\s+analysis)\b',
            'method': r'\b(\w+\s+method)\b',
            'system': r'\b(\w+\s+system)\b',
            'model': r'\b(\w+\s+model)\b',
        }
        
        for base, pattern in patterns.items():
            matches = re.findall(pattern, text.lower())
            if matches:
                normalized = self._normalize_theme(matches[0])
                if normalized:
                    concepts.append(normalized)
        
        if not concepts:
            if len(re.findall(r'\d+', text)) > 5:
                concepts.append('quantitative_data')
            elif len(re.findall(r'[.!?]', text)) / max(len(text.split()), 1) > 0.1:
                concepts.append('descriptive_text')
            else:
                concepts.append('general_content')
        
        return concepts[:self.config.max_themes_per_chunk]
    
    def batch_label(self, chunks: List[str]) -> List[Dict]:
        """Label multiple chunks"""
        if not self.config.enable_semantic_labeling:
            return [{"themes": ["general_content"], "primary_theme": "general_content", 
                    "confidence": 0.5, "theme_keywords": [], "method": "none"}] * len(chunks)
        
        if self.config.semantic_method in ["tfidf", "hybrid"] or not self.model:
            self._fit_tfidf(chunks)
        
        method_display = {
            "llm": "LLM-based",
            "tfidf": "TF-IDF-based",
            "hybrid": "Hybrid (LLM + TF-IDF)"
        }
        
        logger.info(f"Discovering themes using {method_display.get(self.config.semantic_method, 'unknown')}...")
        results = []
        
        for idx, chunk in enumerate(tqdm(chunks, desc="Theme extraction")):
            results.append(self.label_chunk(chunk, chunk_idx=idx))
        
        logger.info(f"\nDiscovered {len(self.discovered_themes)} unique themes")
        top_themes = self.theme_frequencies.most_common(20)
        for theme, count in top_themes:
            percentage = (count / len(chunks)) * 100
            logger.info(f"  {theme}: {count} ({percentage:.1f}%)")
        
        if len(self.discovered_themes) > 20:
            logger.info(f"  ... and {len(self.discovered_themes) - 20} more")
        
        method_counts = Counter(r['method'] for r in results)
        logger.info("\nMethod usage:")
        for method, count in method_counts.items():
            percentage = (count / len(results)) * 100
            logger.info(f"  {method}: {count} ({percentage:.1f}%)")
        
        return results
    
    def get_theme_statistics(self) -> Dict:
        """Get theme statistics"""
        return {
            'total_unique_themes': len(self.discovered_themes),
            'theme_frequencies': dict(self.theme_frequencies.most_common(50)),
            'top_themes': [t for t, _ in self.theme_frequencies.most_common(10)],
            'theme_clusters': {k: len(v) for k, v in self.theme_clusters.items()},
            'semantic_method': self.config.semantic_method
        }


# ============================================================================
# SECTION EXTRACTOR
# ============================================================================

class SectionExtractor:
    """Extracts section titles from PDFs"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def extract_structure(self, pdf_path: str) -> Dict:
        """Extract document structure"""
        if not self.config.extract_sections:
            return {"sections": [], "toc": [], "total_sections": 0}
        
        try:
            sections = []
            laparams = LAParams()
            
            for page_num, page_layout in enumerate(extract_pages(pdf_path, laparams=laparams)):
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        text = element.get_text().strip()
                        if not text:
                            continue
                        
                        font_size = self._get_avg_font_size(element)
                        
                        if (font_size >= self.config.min_section_title_size and 
                            len(text.split()) <= self.config.max_section_title_words and
                            self._looks_like_section_title(text)):
                            
                            sections.append({
                                "title": text,
                                "page": page_num + 1,
                                "font_size": font_size
                            })
            
            toc_sections = self._extract_from_toc(pdf_path)
            
            return {
                "sections": sections,
                "toc": toc_sections,
                "total_sections": len(sections) + len(toc_sections)
            }
        except:
            return {"sections": [], "toc": [], "total_sections": 0}
    
    def _get_avg_font_size(self, element) -> float:
        """Get average font size"""
        try:
            font_sizes = []
            for item in element:
                if hasattr(item, '__iter__'):
                    for char in item:
                        if isinstance(char, LTChar):
                            font_sizes.append(char.height)
            return np.mean(font_sizes) if font_sizes else 0.0
        except:
            return 0.0
    
    def _looks_like_section_title(self, text: str) -> bool:
        """Check if text looks like section title"""
        patterns = [
            r'^\d+\.?\s+[A-Z]',
            r'^(Chapter|Section|Part)\s+\d+',
            r'^[A-Z][A-Za-z\s]{2,30}$',
            r'^\d+\.\d+',
        ]
        
        if any(re.match(p, text) for p in patterns):
            return True
        
        words = text.split()
        if words:
            cap_ratio = sum(1 for w in words if w[0].isupper()) / len(words)
            return cap_ratio > 0.7
        
        return False
    
    def _extract_from_toc(self, pdf_path: str) -> List[Dict]:
        """Extract table of contents"""
        toc = []
        try:
            text = extract_text(pdf_path, page_numbers=[0, 1, 2])
            lines = text.split('\n')
            in_toc = False
            
            for line in lines:
                line = line.strip()
                if re.match(r'(table of contents|contents)', line.lower()):
                    in_toc = True
                    continue
                
                if in_toc:
                    match = re.match(r'([\d\.]+)\s+(.+?)\s+\.{2,}\s*(\d+)', line)
                    if match:
                        toc.append({
                            "number": match.group(1),
                            "title": match.group(2).strip(),
                            "page": int(match.group(3))
                        })
                    elif line and not re.match(r'^\d+$', line):
                        if len(toc) > 0:
                            break
        except:
            pass
        
        return toc
    
    def match_chunk_to_section(self, chunk_text: str, sections: List[Dict], 
                               chunk_position: int, total_chunks: int) -> Optional[str]:
        """Match chunk to section"""
        if not sections:
            return None
        
        for section in sections:
            title = section.get('title', '')
            if title and title.lower() in chunk_text.lower()[:200]:
                return title
        
        chunk_ratio = chunk_position / max(total_chunks, 1)
        best_section = None
        min_distance = float('inf')
        
        for section in sections:
            if 'page' in section:
                section_ratio = section['page'] / 100
                distance = abs(chunk_ratio - section_ratio)
                if distance < min_distance:
                    min_distance = distance
                    best_section = section.get('title', '')
        
        return best_section


# ============================================================================
# QUALITY SCORER
# ============================================================================

class DynamicQualityScorer:
    """Multi-dimensional quality scoring"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.weights = config.quality_weights
    
    def score_chunk(self, text: str, metadata: Dict = None) -> Dict:
        """Calculate quality scores"""
        scores = {
            'length_quality': self._score_length(text),
            'coherence_quality': self._score_coherence(text),
            'information_density': self._score_information_density(text),
            'structural_quality': self._score_structure(text),
            'linguistic_quality': self._score_linguistics(text)
        }
        
        composite = sum(scores[k] * self.weights.get(k, 0.2) for k in scores.keys())
        scores['composite_quality'] = round(composite, 3)
        
        return scores
    
    def _score_length(self, text: str) -> float:
        """Score length"""
        length = len(text)
        word_count = len(text.split())
        
        if 100 <= length <= 2000 and 20 <= word_count <= 400:
            return 1.0
        elif 50 <= length <= 3000 and 10 <= word_count <= 500:
            return 0.7
        elif length >= 10 and word_count >= 3:
            return 0.4
        return 0.1
    
    def _score_coherence(self, text: str) -> float:
        """Score coherence"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        complete = [s for s in sentences if len(s.split()) >= 3]
        if sentences:
            score += 0.3 * (len(complete) / len(sentences))
        
        if re.search(r'[A-Z]', text):
            score += 0.2
        
        punct_count = len(re.findall(r'[.!?,;:]', text))
        word_count = len(text.split())
        if word_count > 0:
            punct_ratio = punct_count / word_count
            if 0.05 <= punct_ratio <= 0.3:
                score += 0.3
        
        transitions = ['however', 'therefore', 'moreover', 'furthermore']
        if any(t in text.lower() for t in transitions):
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_information_density(self, text: str) -> float:
        """Score information density"""
        score = 0.0
        words = text.split()
        
        if not words:
            return 0.0
        
        unique_ratio = len(set(words)) / len(words)
        score += 0.3 * unique_ratio
        
        avg_word_len = np.mean([len(w) for w in words])
        if 4 <= avg_word_len <= 7:
            score += 0.3
        elif 3 <= avg_word_len <= 8:
            score += 0.15
        
        if re.search(r'\d', text):
            score += 0.2
        
        specific_terms = re.findall(r'(?<!^)(?<!\.\s)[A-Z][a-z]+', text)
        if len(specific_terms) > 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_structure(self, text: str) -> float:
        """Score structure"""
        score = 0.0
        
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s for s in sentences if s.strip() and len(s.split()) >= 3]
        if len(valid_sentences) >= 2:
            score += 0.4
        
        if not re.match(r'^\s*[-•*]\s', text, re.MULTILINE):
            score += 0.3
        
        if text.rstrip().endswith(('.', '!', '?', '"', "'")):
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_linguistics(self, text: str) -> float:
        """Score linguistics"""
        score = 0.0
        
        if not re.search(r'\s{3,}', text):
            score += 0.25
        
        if not re.search(r'[.!?]{3,}', text):
            score += 0.25
        
        if not text.isupper() and not text.islower():
            score += 0.25
        
        unique_chars = len(set(text.lower()))
        if unique_chars >= 20:
            score += 0.25
        
        return score


# ============================================================================
# THREAD CREATOR
# ============================================================================

class ThreadCreator:
    """Creates semantic threads linking related chunks"""
    
    def __init__(self, config: PipelineConfig, embeddings: np.ndarray):
        self.config = config
        self.embeddings = embeddings
    
    def create_threads(self, chunks: List[Dict]) -> List[Dict]:
        """Link chunks into semantic threads"""
        if len(chunks) == 0:
            return chunks
        
        logger.info("Creating semantic threads...")
        
        threads = {}
        chunk_to_thread = {}
        
        by_source = defaultdict(list)
        for idx, chunk in enumerate(chunks):
            source = chunk.get('metadata', {}).get('filename', 'unknown')
            by_source[source].append(idx)
        
        thread_counter = 0
        
        for source, indices in by_source.items():
            if len(indices) <= 1:
                continue
            
            source_embeds = self.embeddings[indices]
            
            for i in range(len(indices)):
                idx_i = indices[i]
                
                if idx_i in chunk_to_thread:
                    continue
                
                thread_id = f"thread_{thread_counter:06d}"
                threads[thread_id] = [idx_i]
                chunk_to_thread[idx_i] = thread_id
                
                for j in range(i + 1, min(i + 5, len(indices))):
                    idx_j = indices[j]
                    
                    if idx_j in chunk_to_thread:
                        continue
                    
                    sim = cosine_similarity(
                        [source_embeds[i]], 
                        [source_embeds[j]]
                    )[0][0]
                    
                    if sim >= self.config.thread_sim_threshold:
                        threads[thread_id].append(idx_j)
                        chunk_to_thread[idx_j] = thread_id
                
                thread_counter += 1
        
        for idx, chunk in enumerate(chunks):
            if idx in chunk_to_thread:
                thread_id = chunk_to_thread[idx]
                chunk['thread_id'] = thread_id
                chunk['metadata']['thread_size'] = len(threads[thread_id])
                chunk['metadata']['thread_position'] = threads[thread_id].index(idx) + 1
            else:
                chunk['thread_id'] = f"single_{uuid.uuid4().hex[:12]}"
                chunk['metadata']['thread_size'] = 1
                chunk['metadata']['thread_position'] = 1
        
        logger.info(f"Created {len(threads)} semantic threads")
        return chunks


# ============================================================================
# PDF PROCESSOR
# ============================================================================

class PDFProcessor:
    """Handles PDF extraction with parallel processing and OCR"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.section_extractor = SectionExtractor(config)
        self.stats = {
            'total_pdfs': 0,
            'successful': 0,
            'failed': 0,
            'ocr_used': 0,
            'total_chunks': 0,
            'sections_extracted': 0
        }
    
    def extract_pdfs(self) -> List[Dict]:
        """Extract text from PDFs using parallel processing"""
        logger.info(f"Scanning PDF directory: {self.config.pdf_dir}")
        
        if not os.path.exists(self.config.pdf_dir):
            logger.error(f"PDF directory not found: {self.config.pdf_dir}")
            return []
        
        pdf_files = [f for f in os.listdir(self.config.pdf_dir) if f.endswith('.pdf')]
        self.stats['total_pdfs'] = len(pdf_files)
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        pdf_paths = [os.path.join(self.config.pdf_dir, f) for f in pdf_files]
        
        documents = []
        logger.info(f"Using {self.config.max_workers} parallel workers")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(self._process_single_pdf, path) for path in pdf_paths]
            
            for future in tqdm(as_completed(futures), total=len(pdf_paths), desc="Extracting PDFs"):
                result = future.result()
                if result:
                    documents.append(result)
                    self.stats['successful'] += 1
                    if result.get('ocr_used', False):
                        self.stats['ocr_used'] += 1
                else:
                    self.stats['failed'] += 1
        
        logger.info(f"Extracted {self.stats['successful']} PDFs successfully")
        if self.stats['ocr_used'] > 0:
            logger.info(f"OCR used on {self.stats['ocr_used']} scanned PDFs")
        
        return documents
    
    def _process_single_pdf(self, path: str) -> Optional[Dict]:
        """Process a single PDF"""
        filename = os.path.basename(path)
        ocr_used = False
        
        try:
            text = extract_text(path)
            
            if not text.strip() and self.config.enable_ocr and OCR_AVAILABLE:
                try:
                    images = convert_from_path(path, dpi=300)
                    ocr_texts = []
                    for img in images:
                        ocr_texts.append(pytesseract.image_to_string(img))
                    text = ' '.join(ocr_texts)
                    ocr_used = True
                except:
                    return None
            
            if not text.strip():
                return None
            
            structure = self.section_extractor.extract_structure(path)
            
            return {
                "filename": filename,
                "text": clean_text(text),
                "structure": structure,
                "ocr_used": ocr_used
            }
        except:
            return None
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Split documents into chunks"""
        logger.info(f"Chunking documents (size: {self.config.chunk_size} words)...")
        
        chunks = []
        for doc in documents:
            text = doc['text']
            filename = doc['filename']
            structure = doc.get('structure', {})
            sections = structure.get('sections', []) + structure.get('toc', [])
            ocr_used = doc.get('ocr_used', False)
            
            if structure.get('total_sections', 0) > 0:
                self.stats['sections_extracted'] += structure['total_sections']
            
            words = text.split()
            
            for i in range(0, len(words), self.config.chunk_size):
                chunk_text = " ".join(words[i:i + self.config.chunk_size])
                
                if not chunk_text.strip() or not validate_text(chunk_text, self.config):
                    continue
                
                chunk_position = i // self.config.chunk_size
                total_chunks = len(words) // self.config.chunk_size
                section_title = self.section_extractor.match_chunk_to_section(
                    chunk_text, sections, chunk_position, total_chunks
                )
                
                chunks.append({
                    'text': chunk_text,
                    'filename': filename,
                    'chunk_index': chunk_position,
                    'section_title': section_title,
                    'has_section': section_title is not None,
                    'ocr_used': ocr_used
                })
        
        self.stats['total_chunks'] = len(chunks)
        logger.info(f"Created {len(chunks)} valid chunks")
        return chunks


# ============================================================================
# EMBEDDING PROCESSOR
# ============================================================================

class EmbeddingProcessor:
    """Handles embedding generation"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        logger.info(f"Loading embedding model: {config.embedding_model}...")
        self.model = SentenceTransformer(config.embedding_model)
        
        self.memory = []
        self.memory_vectors = []
        
        self.stats = {'total_embedded': 0, 'batch_count': 0}
    
    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        """Embed all chunks"""
        logger.info(f"Embedding {len(chunks)} chunks...")
        
        texts = [chunk['text'] for chunk in chunks]
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.config.batch_size), desc="Embedding"):
            batch = texts[i:i + self.config.batch_size]
            embeddings = self._batch_embed(batch)
            all_embeddings.extend(embeddings)
            
            for text, emb in zip(batch, embeddings):
                self.memory.append(text)
                self.memory_vectors.append(emb)
        
        logger.info(f"Embedded {self.stats['total_embedded']} chunks")
        return np.array(all_embeddings)
    
    def _batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch"""
        try:
            embeddings = self.model.encode(
                texts, 
                convert_to_numpy=True, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            self.stats['total_embedded'] += len(texts)
            self.stats['batch_count'] += 1
            return embeddings
        except:
            return [np.zeros(self.config.embedding_dim) for _ in texts]
    
    def build_index(self, save_path: Optional[str] = None):
        """Build FAISS index"""
        if not self.memory_vectors:
            return
        
        logger.info("Building FAISS index...")
        dim = len(self.memory_vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(self.memory_vectors).astype('float32'))
        
        if save_path:
            faiss.write_index(index, save_path)
    
    def get_memory(self) -> np.ndarray:
        """Return memory texts"""
        return np.array(self.memory, dtype=object)


# ============================================================================
# DATASET CREATOR
# ============================================================================

class DatasetCreator:
    """Creates high-quality training datasets"""
    
    def __init__(self, config: PipelineConfig, embedding_model: SentenceTransformer):
        self.config = config
        self.embedding_model = embedding_model
        self.quality_scorer = DynamicQualityScorer(config)
        self.semantic_labeler = SemanticLabeler(config)
        
        self.stats = {
            'raw_chunks': 0,
            'after_dedup': 0,
            'after_quality': 0,
            'grouped_chunks': 0,
            'with_sections': 0,
            'with_themes': 0
        }
    
    def deduplicate(self, chunks: List[Dict]) -> List[Dict]:
        """Remove duplicates"""
        seen = set()
        deduped = []
        for chunk in chunks:
            text_hash = hashlib.sha256(chunk['text'].encode()).hexdigest()
            if text_hash not in seen:
                seen.add(text_hash)
                deduped.append(chunk)
        
        logger.info(f"Deduplicated: {len(chunks)} -> {len(deduped)}")
        return deduped
    
    def group_similar_chunks(self, chunks: List[Dict], embeddings: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Group similar consecutive chunks"""
        if not chunks:
            return [], np.array([])
        
        logger.info("Grouping similar chunks...")
        
        by_source = defaultdict(list)
        for idx, chunk in enumerate(chunks):
            source = chunk['filename']
            by_source[source].append((idx, chunk))
        
        grouped_chunks = []
        grouped_embeddings = []
        
        for source, source_items in by_source.items():
            indices = [idx for idx, _ in source_items]
            source_chunks = [chunk for _, chunk in source_items]
            source_embeds = embeddings[indices]
            
            i = 0
            while i < len(source_chunks):
                current_texts = [source_chunks[i]['text']]
                current_chunk = source_chunks[i].copy()
                current_embed_sum = source_embeds[i].copy()
                current_embed = source_embeds[i]
                
                j = i + 1
                while j < len(source_chunks):
                    sim = cosine_similarity([current_embed], [source_embeds[j]])[0][0]
                    new_text = ' '.join(current_texts) + ' ' + source_chunks[j]['text']
                    
                    if sim >= self.config.sim_threshold and len(new_text) <= self.config.max_merged_length:
                        current_texts.append(source_chunks[j]['text'])
                        current_embed_sum += source_embeds[j]
                        current_embed = current_embed_sum / np.linalg.norm(current_embed_sum)
                        j += 1
                    else:
                        break
                
                current_chunk['text'] = ' '.join(current_texts)
                current_chunk['merged_from'] = j - i
                grouped_chunks.append(current_chunk)
                grouped_embeddings.append(current_embed)
                
                i = j
        
        self.stats['grouped_chunks'] = len(grouped_chunks)
        logger.info(f"Grouped into {len(grouped_chunks)} chunks")
        return grouped_chunks, np.array(grouped_embeddings)
    
    def create_records(self, chunks: List[Dict], embeddings: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Create enriched training records"""
        logger.info(f"Creating records from {len(chunks)} chunks...")
        self.stats['raw_chunks'] = len(chunks)
        
        deduped = self.deduplicate(chunks)
        self.stats['after_dedup'] = len(deduped)
        
        grouped, grouped_embeds = self.group_similar_chunks(deduped, embeddings)
        
        if self.config.enable_semantic_labeling:
            texts = [c['text'] for c in grouped]
            semantic_labels = self.semantic_labeler.batch_label(texts)
            for chunk, labels in zip(grouped, semantic_labels):
                chunk['semantic_labels'] = labels
                if labels['themes']:
                    self.stats['with_themes'] += 1
        
        records = []
        for idx, chunk in enumerate(tqdm(grouped, desc="Scoring quality")):
            text = chunk['text']
            
            if len(text) < self.config.min_chunk_length:
                continue
            
            quality_scores = self.quality_scorer.score_chunk(text)
            
            record = {
                'text': text,
                'thread_id': str(uuid.uuid4()),
                'quality_scores': quality_scores,
                'metadata': {
                    'filename': chunk['filename'],
                    'chunk_index': chunk.get('chunk_index', idx),
                    'section_title': chunk.get('section_title'),
                    'has_section': chunk.get('has_section', False),
                    'merged_from': chunk.get('merged_from', 1),
                    'semantic_themes': chunk.get('semantic_labels', {}).get('themes', []),
                    'primary_theme': chunk.get('semantic_labels', {}).get('primary_theme', 'general_content'),
                    'theme_confidence': chunk.get('semantic_labels', {}).get('confidence', 0.0),
                    'length': len(text),
                    'word_count': len(text.split()),
                    'sentence_count': len(re.split(r'[.!?]+', text)),
                }
            }
            
            records.append(record)
            
            if chunk.get('section_title'):
                self.stats['with_sections'] += 1
        
        self.stats['after_quality'] = len(records)
        logger.info(f"Created {len(records)} enriched records")
        logger.info(f"  With sections: {self.stats['with_sections']}")
        logger.info(f"  With themes: {self.stats['with_themes']}")
        
        return records, grouped_embeds[:len(records)]
    
    def create_splits(self, records: List[Dict]) -> Dict[str, List[Dict]]:
        """Create train/val/test splits"""
        logger.info("Creating data splits...")
        
        if not records:
            return {"train": [], "validation": [], "test": []}
        
        records.sort(key=lambda x: x['quality_scores']['composite_quality'], reverse=True)
        
        num_quartiles = 4
        total = len(records)
        for i in range(num_quartiles):
            start = i * (total // num_quartiles)
            end = (i + 1) * (total // num_quartiles) if i < num_quartiles - 1 else total
            quartile = records[start:end]
            random.shuffle(quartile)
            records[start:end] = quartile
        
        train_end = int(total * self.config.split_ratio[0])
        val_end = train_end + int(total * self.config.split_ratio[1])
        
        splits = {
            "train": records[:train_end],
            "validation": records[train_end:val_end],
            "test": records[val_end:]
        }
        
        for name, data in splits.items():
            if data:
                avg_q = np.mean([r['quality_scores']['composite_quality'] for r in data])
                logger.info(f"  {name.capitalize()}: {len(data)} (quality: {avg_q:.3f})")
        
        return splits


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class Pipeline:
    """Complete pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.start_time = None
    
    def save_jsonl(self, data: List[Dict], filename: str, compress: bool = False):
        """Save to JSONL"""
        if compress:
            filename = filename + '.gz'
            with gzip.open(filename, 'wt', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
        else:
            with open(filename, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
        
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        logger.info(f"Saved {len(data)} records to {filename} ({size_mb:.2f} MB)")
    
    def run(self):
        """Execute pipeline"""
        import time
        self.start_time = time.time()
        
        logger.info("=" * 70)
        logger.info("STARTING PIPELINE")
        logger.info("=" * 70)
        
        pdf_proc = PDFProcessor(self.config)
        documents = pdf_proc.extract_pdfs()
        if not documents:
            logger.error("No documents extracted")
            return
        
        chunks = pdf_proc.chunk_documents(documents)
        if not chunks:
            logger.error("No chunks created")
            return
        
        embed_proc = EmbeddingProcessor(self.config)
        embeddings = embed_proc.embed_chunks(chunks)
        
        if self.config.save_intermediates:
            embed_proc.build_index('memory.index')
            np.save('memory_texts.npy', embed_proc.get_memory())
        
        dataset_creator = DatasetCreator(self.config, embed_proc.model)
        records, record_embeddings = dataset_creator.create_records(chunks, embeddings)
        
        if not records:
            logger.error("No records created")
            return
        
        thread_creator = ThreadCreator(self.config, record_embeddings)
        records = thread_creator.create_threads(records)
        
        splits = dataset_creator.create_splits(records)
        
        for split_name, data in splits.items():
            if data:
                filename = f"{self.config.output_prefix}_{split_name}.jsonl"
                self.save_jsonl(data, filename, self.config.compress_output)
        
        elapsed = time.time() - self.start_time
        self._print_stats(pdf_proc, embed_proc, dataset_creator, splits, elapsed)
    
    def _print_stats(self, pdf_proc, embed_proc, dataset_creator, splits, elapsed):
        """Print statistics"""
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"PDFs: {pdf_proc.stats['successful']}/{pdf_proc.stats['total_pdfs']}")
        logger.info(f"Sections: {pdf_proc.stats['sections_extracted']}")
        logger.info(f"Chunks: {pdf_proc.stats['total_chunks']}")
        logger.info(f"Embeddings: {embed_proc.stats['total_embedded']}")
        logger.info(f"After dedup: {dataset_creator.stats['after_dedup']}")
        logger.info(f"Final records: {dataset_creator.stats['after_quality']}")
        logger.info(f"Train: {len(splits['train'])}")
        logger.info(f"Val: {len(splits['validation'])}")
        logger.info(f"Test: {len(splits['test'])}")
        logger.info(f"Time: {elapsed/60:.2f} minutes")
        logger.info("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PDF Processing Pipeline')
    
    parser.add_argument('--pdf-dir', default='./PDFs')
    parser.add_argument('--output-prefix', default='dataset')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2')
    parser.add_argument('--semantic-model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
    parser.add_argument('--chunk-size', type=int, default=500)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--sim-threshold', type=float, default=0.7)
    parser.add_argument('--thread-threshold', type=float, default=0.65)
    parser.add_argument('--enable-semantic-labeling', action='store_true')
    parser.add_argument('--semantic-method', choices=['llm', 'tfidf', 'hybrid'], default='llm')
    parser.add_argument('--json-output', action='store_true')
    parser.add_argument('--enable-ocr', action='store_true')
    parser.add_argument('--no-sections', action='store_true')
    parser.add_argument('--no-compress', action='store_true')
    parser.add_argument('--save-intermediates', action='store_true')
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        pdf_dir=args.pdf_dir,
        output_prefix=args.output_prefix,
        embedding_model=args.embedding_model,
        semantic_model=args.semantic_model,
        chunk_size=args.chunk_size,
        max_workers=args.workers,
        sim_threshold=args.sim_threshold,
        thread_sim_threshold=args.thread_threshold,
        enable_semantic_labeling=args.enable_semantic_labeling,
        semantic_method=args.semantic_method,
        use_json_output=args.json_output,
        enable_ocr=args.enable_ocr,
        extract_sections=not args.no_sections,
        compress_output=not args.no_compress,
        save_intermediates=args.save_intermediates
    )
    
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()

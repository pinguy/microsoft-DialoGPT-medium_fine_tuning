"""
Enhanced PDF Processing Pipeline with Semantic Analysis
Converts PDFs → Embeddings → Semantically Labeled Training Data → Compressed JSONL

Features:
- Section/chapter title extraction from PDF structure
- Semantic theme labeling using DeepSeek-R1
- Thread linking for conversation-style interactions
- Dynamic quality scoring with multiple metrics

Usage:
    python pdf_pipeline.py --pdf-dir ./PDFs --output-prefix dataset --enable-semantic-labeling
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

import ftfy
import numpy as np
import faiss
import torch
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LAParams, LTTextContainer, LTChar
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

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


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    # PDF Processing
    pdf_dir: str = "./PDFs"
    
    # Embedding Configuration
    embedding_model: str = 'all-MiniLM-L6-v2'
    embedding_dim: int = 384
    chunk_size: int = 500  # words per chunk
    batch_size: int = 100
    
    # Semantic Labeling
    enable_semantic_labeling: bool = False
    semantic_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    max_themes_per_chunk: int = 3
    semantic_batch_size: int = 8
    
    # Section Extraction
    extract_sections: bool = True
    min_section_title_size: float = 12.0  # Font size threshold
    max_section_title_words: int = 15
    
    # Text Quality
    min_text_length: int = 20
    max_text_length: int = 10000
    min_words: int = 3
    min_chunk_length: int = 10
    punctuation_ratio_threshold: float = 0.6
    
    # Dynamic Quality Scoring
    quality_dimensions: List[str] = field(default_factory=lambda: [
        'length_quality', 'coherence_quality', 'information_density',
        'structural_quality', 'linguistic_quality'
    ])
    
    # Similarity Merging & Threading
    sim_threshold: float = 0.7
    max_merged_length: int = 2000
    thread_sim_threshold: float = 0.65  # For linking chunks into threads
    
    # Data Splits
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    
    # Output
    output_prefix: str = "dataset"
    save_intermediates: bool = False
    compress_output: bool = True
    
    def __post_init__(self):
        logger.info("📋 Pipeline Configuration:")
        logger.info(f"  PDF Directory: {self.pdf_dir}")
        logger.info(f"  Embedding Model: {self.embedding_model}")
        logger.info(f"  Chunk Size: {self.chunk_size} words")
        logger.info(f"  Section Extraction: {self.extract_sections}")
        logger.info(f"  Semantic Labeling: {self.enable_semantic_labeling}")
        if self.enable_semantic_labeling:
            logger.info(f"  Semantic Model: {self.semantic_model}")


# ============================================================================
# SEMANTIC LABELING
# ============================================================================

class SemanticLabeler:
    """Labels chunks with dynamic semantic themes using DeepSeek-R1"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Dynamic theme discovery
        self.discovered_themes = set()
        self.theme_clusters = defaultdict(list)  # theme -> list of chunk indices
        self.theme_frequencies = Counter()
        
        if config.enable_semantic_labeling:
            self._load_model()
    
    def _load_model(self):
        """Load the semantic labeling model"""
        try:
            logger.info(f"🧠 Loading semantic model: {self.config.semantic_model}...")
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
            logger.info(f"✅ Semantic model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            self.config.enable_semantic_labeling = False
    
    def label_chunk(self, text: str, chunk_idx: Optional[int] = None) -> Dict:
        """Label a single chunk with dynamically generated themes"""
        if not self.config.enable_semantic_labeling or not self.model:
            return {
                "themes": ["general_content"], 
                "confidence": 0.0, 
                "primary_theme": "general_content",
                "theme_keywords": []
            }
        
        try:
            # Open-ended prompt for dynamic theme extraction
            prompt = f"""<think>
Analyze the following text and identify 1-3 concise semantic themes that best describe its content and purpose.

Guidelines:
- Generate SHORT, descriptive theme labels (2-4 words max)
- Use snake_case format (e.g., "machine_learning_theory", "data_preprocessing")
- Focus on the CORE concepts, not surface-level topics
- Be specific but generalizable
- Extract key domain terms or concepts

Text excerpt:
{text[:600]}

</think>

Themes (comma-separated, 1-3 themes):"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.4,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract themes from response (after the prompt)
            if "Themes" in response:
                themes_part = response.split("Themes")[-1]
                # Remove common separators and clean up
                themes_part = re.sub(r'[:\(\)]', '', themes_part).strip()
            else:
                themes_part = response[len(prompt):].strip()
            
            # Parse themes
            raw_themes = [t.strip() for t in re.split(r'[,;\n]', themes_part) if t.strip()]
            
            # Clean and normalize themes
            normalized_themes = []
            keywords = []
            
            for theme in raw_themes[:self.config.max_themes_per_chunk]:
                # Convert to snake_case and clean
                clean_theme = self._normalize_theme(theme)
                if clean_theme and len(clean_theme.split('_')) <= 5:  # Max 5 words
                    normalized_themes.append(clean_theme)
                    keywords.extend(clean_theme.split('_'))
                    
                    # Track discovered themes
                    self.discovered_themes.add(clean_theme)
                    self.theme_frequencies[clean_theme] += 1
                    
                    if chunk_idx is not None:
                        self.theme_clusters[clean_theme].append(chunk_idx)
            
            if not normalized_themes:
                # Fallback: extract key terms from text
                normalized_themes = self._extract_key_concepts(text)
            
            return {
                "themes": normalized_themes,
                "primary_theme": normalized_themes[0] if normalized_themes else "general_content",
                "confidence": min(0.85, 0.6 + 0.1 * len(normalized_themes)),
                "theme_keywords": list(set(keywords))[:10]
            }
            
        except Exception as e:
            logger.warning(f"Semantic labeling failed: {e}")
            fallback_themes = self._extract_key_concepts(text)
            return {
                "themes": fallback_themes, 
                "confidence": 0.3, 
                "primary_theme": fallback_themes[0] if fallback_themes else "general_content",
                "theme_keywords": []
            }
    
    def _normalize_theme(self, theme: str) -> str:
        """Normalize theme to snake_case format"""
        # Remove quotes, parentheses, special chars
        theme = re.sub(r'["\'\(\)\[\]{}]', '', theme)
        
        # Replace spaces and hyphens with underscores
        theme = re.sub(r'[\s\-]+', '_', theme.lower())
        
        # Remove non-alphanumeric (except underscore)
        theme = re.sub(r'[^a-z0-9_]', '', theme)
        
        # Remove leading/trailing underscores and collapse multiple
        theme = re.sub(r'_+', '_', theme).strip('_')
        
        # Remove common filler words
        filler_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        parts = [p for p in theme.split('_') if p not in filler_words]
        
        return '_'.join(parts) if parts else ''
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Fallback: extract key concepts from text using heuristics"""
        concepts = []
        
        # Extract capitalized multi-word phrases (likely important terms)
        phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', text)
        for phrase in phrases[:3]:
            normalized = self._normalize_theme(phrase)
            if normalized and len(normalized) > 3:
                concepts.append(normalized)
        
        # Extract domain-specific patterns
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
        
        # If still empty, use generic based on text characteristics
        if not concepts:
            if len(re.findall(r'\d+', text)) > 5:
                concepts.append('quantitative_data')
            elif len(re.findall(r'[.!?]', text)) / max(len(text.split()), 1) > 0.1:
                concepts.append('descriptive_text')
            else:
                concepts.append('general_content')
        
        return concepts[:self.config.max_themes_per_chunk]
    
    def batch_label(self, chunks: List[str]) -> List[Dict]:
        """Label multiple chunks and discover theme taxonomy"""
        if not self.config.enable_semantic_labeling:
            return [{
                "themes": ["general_content"], 
                "primary_theme": "general_content", 
                "confidence": 0.5,
                "theme_keywords": []
            }] * len(chunks)
        
        logger.info(f"🏷️  Discovering semantic themes from {len(chunks)} chunks...")
        results = []
        
        for idx, chunk in enumerate(tqdm(chunks, desc="Dynamic theme extraction")):
            results.append(self.label_chunk(chunk, chunk_idx=idx))
        
        # Log discovered theme taxonomy
        logger.info(f"\n🔍 Discovered Theme Taxonomy ({len(self.discovered_themes)} unique themes):")
        
        # Show top themes by frequency
        top_themes = self.theme_frequencies.most_common(20)
        for theme, count in top_themes:
            percentage = (count / len(chunks)) * 100
            logger.info(f"   • {theme}: {count} chunks ({percentage:.1f}%)")
        
        if len(self.discovered_themes) > 20:
            logger.info(f"   ... and {len(self.discovered_themes) - 20} more themes")
        
        return results
    
    def get_theme_statistics(self) -> Dict:
        """Get statistics about discovered themes"""
        return {
            'total_unique_themes': len(self.discovered_themes),
            'theme_frequencies': dict(self.theme_frequencies.most_common(50)),
            'top_themes': [t for t, _ in self.theme_frequencies.most_common(10)],
            'theme_clusters': {k: len(v) for k, v in self.theme_clusters.items()},
            'avg_themes_per_chunk': np.mean([count for count in self.theme_frequencies.values()]) if self.theme_frequencies else 0
        }


# ============================================================================
# SECTION EXTRACTOR
# ============================================================================

class SectionExtractor:
    """Extracts section titles and structure from PDFs"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def extract_structure(self, pdf_path: str) -> Dict[str, any]:
        """Extract document structure including sections"""
        if not self.config.extract_sections:
            return {"sections": [], "toc": []}
        
        try:
            sections = []
            laparams = LAParams()
            
            for page_num, page_layout in enumerate(extract_pages(pdf_path, laparams=laparams)):
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        text = element.get_text().strip()
                        if not text:
                            continue
                        
                        # Detect section titles by font size
                        font_size = self._get_avg_font_size(element)
                        
                        if (font_size >= self.config.min_section_title_size and 
                            len(text.split()) <= self.config.max_section_title_words and
                            self._looks_like_section_title(text)):
                            
                            sections.append({
                                "title": text,
                                "page": page_num + 1,
                                "font_size": font_size,
                                "position": element.bbox[1]  # y-coordinate
                            })
            
            # Also try to extract from TOC patterns
            toc_sections = self._extract_from_toc(pdf_path)
            
            return {
                "sections": sections,
                "toc": toc_sections,
                "total_sections": len(sections) + len(toc_sections)
            }
            
        except Exception as e:
            logger.warning(f"Section extraction failed for {pdf_path}: {e}")
            return {"sections": [], "toc": [], "total_sections": 0}
    
    def _get_avg_font_size(self, element) -> float:
        """Get average font size from text element"""
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
        """Heuristic to detect section titles"""
        # Common section patterns
        patterns = [
            r'^\d+\.?\s+[A-Z]',  # "1. Introduction" or "1 Introduction"
            r'^(Chapter|Section|Part)\s+\d+',
            r'^[A-Z][A-Za-z\s]{2,30}$',  # Capitalized phrase
            r'^\d+\.\d+',  # "1.1 Subsection"
        ]
        
        if any(re.match(p, text) for p in patterns):
            return True
        
        # Check if mostly capitalized (like a title)
        words = text.split()
        if words:
            cap_ratio = sum(1 for w in words if w[0].isupper()) / len(words)
            return cap_ratio > 0.7
        
        return False
    
    def _extract_from_toc(self, pdf_path: str) -> List[Dict]:
        """Try to extract table of contents"""
        toc = []
        try:
            text = extract_text(pdf_path, page_numbers=[0, 1, 2])  # Check first few pages
            
            # Look for TOC patterns
            lines = text.split('\n')
            in_toc = False
            
            for line in lines:
                line = line.strip()
                if re.match(r'(table of contents|contents)', line.lower()):
                    in_toc = True
                    continue
                
                if in_toc:
                    # Match patterns like "1.2 Section Name ... 15"
                    match = re.match(r'([\d\.]+)\s+(.+?)\s+\.{2,}\s*(\d+)', line)
                    if match:
                        toc.append({
                            "number": match.group(1),
                            "title": match.group(2).strip(),
                            "page": int(match.group(3))
                        })
                    elif line and not re.match(r'^\d+$', line):
                        # Break if we hit non-TOC content
                        if len(toc) > 0:
                            break
        except:
            pass
        
        return toc
    
    def match_chunk_to_section(self, chunk_text: str, sections: List[Dict], 
                               chunk_position: int, total_chunks: int) -> Optional[str]:
        """Match a chunk to its most likely section"""
        if not sections:
            return None
        
        # Simple heuristic: match based on position in document
        chunk_ratio = chunk_position / max(total_chunks, 1)
        
        # Find section closest to this position
        best_section = None
        min_distance = float('inf')
        
        for section in sections:
            # Estimate section position in document
            if 'page' in section:
                # Rough estimate (this could be improved with better page tracking)
                section_ratio = section['page'] / 100  # Assume ~100 pages
                distance = abs(chunk_ratio - section_ratio)
                
                if distance < min_distance:
                    min_distance = distance
                    best_section = section.get('title', '')
        
        # Also check if section title appears in chunk
        for section in sections:
            title = section.get('title', '')
            if title and title.lower() in chunk_text.lower()[:200]:
                return title
        
        return best_section


# ============================================================================
# QUALITY SCORER
# ============================================================================

class DynamicQualityScorer:
    """Multi-dimensional quality scoring system"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def score_chunk(self, text: str, metadata: Dict = None) -> Dict:
        """Calculate comprehensive quality scores"""
        scores = {}
        
        # 1. Length Quality (0-1)
        scores['length_quality'] = self._score_length(text)
        
        # 2. Coherence Quality (0-1)
        scores['coherence_quality'] = self._score_coherence(text)
        
        # 3. Information Density (0-1)
        scores['information_density'] = self._score_information_density(text)
        
        # 4. Structural Quality (0-1)
        scores['structural_quality'] = self._score_structure(text)
        
        # 5. Linguistic Quality (0-1)
        scores['linguistic_quality'] = self._score_linguistics(text)
        
        # Composite score (weighted average)
        weights = {
            'length_quality': 0.15,
            'coherence_quality': 0.25,
            'information_density': 0.25,
            'structural_quality': 0.20,
            'linguistic_quality': 0.15
        }
        
        composite = sum(scores[k] * weights[k] for k in scores.keys())
        scores['composite_quality'] = round(composite, 3)
        
        return scores
    
    def _score_length(self, text: str) -> float:
        """Score based on optimal length"""
        length = len(text)
        word_count = len(text.split())
        
        # Optimal range: 100-2000 chars, 20-400 words
        if 100 <= length <= 2000 and 20 <= word_count <= 400:
            return 1.0
        elif 50 <= length <= 3000 and 10 <= word_count <= 500:
            return 0.7
        elif length >= 10 and word_count >= 3:
            return 0.4
        return 0.1
    
    def _score_coherence(self, text: str) -> float:
        """Score based on textual coherence indicators"""
        score = 0.0
        
        # Complete sentences
        sentences = re.split(r'[.!?]+', text)
        complete_sentences = [s for s in sentences if len(s.split()) >= 3]
        if sentences:
            score += 0.3 * (len(complete_sentences) / len(sentences))
        
        # Proper capitalization
        if re.search(r'[A-Z]', text):
            score += 0.2
        
        # Balanced punctuation
        punct_count = len(re.findall(r'[.!?,;:]', text))
        word_count = len(text.split())
        if word_count > 0:
            punct_ratio = punct_count / word_count
            if 0.05 <= punct_ratio <= 0.3:
                score += 0.3
        
        # Transition words (indicators of coherence)
        transitions = ['however', 'therefore', 'moreover', 'furthermore', 'consequently', 
                      'thus', 'hence', 'additionally', 'meanwhile', 'nevertheless']
        if any(t in text.lower() for t in transitions):
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_information_density(self, text: str) -> float:
        """Score based on information content"""
        score = 0.0
        words = text.split()
        
        if not words:
            return 0.0
        
        # Unique word ratio
        unique_ratio = len(set(words)) / len(words)
        score += 0.3 * unique_ratio
        
        # Average word length (longer words = more specific info)
        avg_word_len = np.mean([len(w) for w in words])
        if 4 <= avg_word_len <= 7:
            score += 0.3
        elif 3 <= avg_word_len <= 8:
            score += 0.15
        
        # Presence of numbers/data
        if re.search(r'\d', text):
            score += 0.2
        
        # Presence of technical/specific terms (capitalized words beyond sentence starts)
        specific_terms = re.findall(r'(?<!^)(?<!\.\s)[A-Z][a-z]+', text)
        if len(specific_terms) > 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_structure(self, text: str) -> float:
        """Score based on structural elements"""
        score = 0.0
        
        # Paragraph-like structure (multiple sentences)
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s for s in sentences if s.strip() and len(s.split()) >= 3]
        if len(valid_sentences) >= 2:
            score += 0.4
        
        # Not just a list/bullet points
        if not re.match(r'^\s*[-•*]\s', text, re.MULTILINE):
            score += 0.3
        
        # Has proper ending
        if text.rstrip().endswith(('.', '!', '?', '"', "'")):
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_linguistics(self, text: str) -> float:
        """Score based on linguistic quality"""
        score = 0.0
        
        # Proper spacing
        if not re.search(r'\s{3,}', text):
            score += 0.25
        
        # No excessive punctuation
        if not re.search(r'[.!?]{3,}', text):
            score += 0.25
        
        # Mixed case (not all caps)
        if not text.isupper() and not text.islower():
            score += 0.25
        
        # Character diversity
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
        
        logger.info("🪢 Creating semantic threads...")
        
        # Build similarity matrix
        n = len(self.embeddings)
        threads = {}  # thread_id -> list of chunk indices
        chunk_to_thread = {}  # chunk index -> thread_id
        
        # Group chunks by source first
        by_source = defaultdict(list)
        for idx, chunk in enumerate(chunks):
            source = chunk.get('metadata', {}).get('filename', 'unknown')
            by_source[source].append(idx)
        
        thread_counter = 0
        
        # Create threads within each source
        for source, indices in by_source.items():
            if len(indices) <= 1:
                continue
            
            # Get embeddings for this source
            source_embeds = self.embeddings[indices]
            
            # Calculate pairwise similarities
            for i in range(len(indices)):
                idx_i = indices[i]
                
                # Skip if already in a thread
                if idx_i in chunk_to_thread:
                    continue
                
                # Start new thread
                thread_id = f"thread_{thread_counter:06d}"
                threads[thread_id] = [idx_i]
                chunk_to_thread[idx_i] = thread_id
                
                # Find similar consecutive chunks
                for j in range(i + 1, min(i + 5, len(indices))):  # Look ahead 5 chunks max
                    idx_j = indices[j]
                    
                    if idx_j in chunk_to_thread:
                        continue
                    
                    # Calculate similarity
                    sim = cosine_similarity(
                        [source_embeds[i]], 
                        [source_embeds[j]]
                    )[0][0]
                    
                    if sim >= self.config.thread_sim_threshold:
                        threads[thread_id].append(idx_j)
                        chunk_to_thread[idx_j] = thread_id
                
                thread_counter += 1
        
        # Assign thread_ids to chunks
        for idx, chunk in enumerate(chunks):
            if idx in chunk_to_thread:
                thread_id = chunk_to_thread[idx]
                chunk['thread_id'] = thread_id
                chunk['metadata']['thread_size'] = len(threads[thread_id])
                chunk['metadata']['thread_position'] = threads[thread_id].index(idx) + 1
            else:
                # Standalone chunk
                chunk['thread_id'] = f"single_{uuid.uuid4().hex[:12]}"
                chunk['metadata']['thread_size'] = 1
                chunk['metadata']['thread_position'] = 1
        
        logger.info(f"✅ Created {len(threads)} semantic threads")
        return chunks


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
# PDF PROCESSOR
# ============================================================================

class PDFProcessor:
    """Handles PDF extraction and initial chunking with structure"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.section_extractor = SectionExtractor(config)
        self.stats = {
            'total_pdfs': 0,
            'successful': 0,
            'failed': 0,
            'total_chunks': 0,
            'sections_extracted': 0
        }
    
    def extract_pdfs(self) -> List[Dict]:
        """Extract text and structure from all PDFs"""
        logger.info(f"📂 Scanning PDF directory: {self.config.pdf_dir}")
        
        if not os.path.exists(self.config.pdf_dir):
            logger.error(f"PDF directory not found: {self.config.pdf_dir}")
            return []
        
        pdf_files = [f for f in os.listdir(self.config.pdf_dir) if f.endswith('.pdf')]
        self.stats['total_pdfs'] = len(pdf_files)
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        documents = []
        for filename in tqdm(pdf_files, desc="Extracting PDFs"):
            path = os.path.join(self.config.pdf_dir, filename)
            try:
                # Extract text
                text = extract_text(path)
                if not text.strip():
                    continue
                
                # Extract structure
                structure = self.section_extractor.extract_structure(path)
                self.stats['sections_extracted'] += structure['total_sections']
                
                documents.append({
                    "filename": filename,
                    "text": clean_text(text),
                    "structure": structure
                })
                self.stats['successful'] += 1
            except Exception as e:
                logger.warning(f"Failed to process {filename}: {e}")
                self.stats['failed'] += 1
        
        logger.info(f"✅ Extracted {self.stats['successful']} PDFs with {self.stats['sections_extracted']} sections")
        return documents
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Split documents into chunks with metadata"""
        logger.info(f"✂️  Chunking documents (size: {self.config.chunk_size} words)...")
        
        chunks = []
        for doc in documents:
            text = doc['text']
            filename = doc['filename']
            structure = doc.get('structure', {})
            sections = structure.get('sections', []) + structure.get('toc', [])
            
            words = text.split()
            
            for i in range(0, len(words), self.config.chunk_size):
                chunk_text = " ".join(words[i:i + self.config.chunk_size])
                
                if not chunk_text.strip() or not validate_text(chunk_text, self.config):
                    continue
                
                # Match chunk to section
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
                    'has_section': section_title is not None
                })
        
        self.stats['total_chunks'] = len(chunks)
        logger.info(f"Created {len(chunks)} valid chunks")
        return chunks


# ============================================================================
# EMBEDDING PROCESSOR
# ============================================================================

class EmbeddingProcessor:
    """Handles embedding generation and FAISS indexing"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        logger.info(f"🧠 Loading embedding model: {config.embedding_model}...")
        self.model = SentenceTransformer(config.embedding_model)
        
        self.memory = []
        self.memory_vectors = []
        
        self.stats = {
            'total_embedded': 0,
            'batch_count': 0
        }
    
    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        """Embed all chunks in batches and return embeddings"""
        logger.info(f"🔢 Embedding {len(chunks)} chunks...")
        
        texts = [chunk['text'] for chunk in chunks]
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.config.batch_size), desc="Embedding"):
            batch = texts[i:i + self.config.batch_size]
            embeddings = self._batch_embed(batch)
            all_embeddings.extend(embeddings)
            
            # Store in memory
            for text, emb in zip(batch, embeddings):
                self.memory.append(text)
                self.memory_vectors.append(emb)
        
        logger.info(f"✅ Embedded {self.stats['total_embedded']} chunks in {self.stats['batch_count']} batches")
        return np.array(all_embeddings)
    
    def _batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts"""
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
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}")
            return [np.zeros(self.config.embedding_dim) for _ in texts]
    
    def build_index(self, save_path: Optional[str] = None) -> None:
        """Build FAISS index"""
        if not self.memory_vectors:
            logger.error("No vectors to index!")
            return
        
        logger.info("🗂️  Building FAISS index...")
        dim = len(self.memory_vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(self.memory_vectors).astype('float32'))
        
        if save_path:
            faiss.write_index(index, save_path)
            logger.info(f"💾 FAISS index saved to '{save_path}'")
    
    def get_memory(self) -> np.ndarray:
        """Return memory texts as numpy array"""
        return np.array(self.memory, dtype=object)


# ============================================================================
# DATASET CREATOR
# ============================================================================

class DatasetCreator:
    """Creates high-quality training datasets with semantic enrichment"""
    
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
        """Remove duplicate chunks"""
        seen = set()
        deduped = []
        for chunk in chunks:
            text_hash = hashlib.sha256(chunk['text'].encode()).hexdigest()
            if text_hash not in seen:
                seen.add(text_hash)
                deduped.append(chunk)
        
        logger.info(f"Deduplicated: {len(chunks)} → {len(deduped)} chunks")
        return deduped
    
    def group_similar_chunks(self, chunks: List[Dict], embeddings: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Group similar consecutive chunks within same source"""
        if not chunks:
            return [], np.array([])
        
        logger.info("🔗 Grouping similar consecutive chunks...")
        
        # Group by source
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
                # Start new group
                current_texts = [source_chunks[i]['text']]
                current_chunk = source_chunks[i].copy()
                current_embed_sum = source_embeds[i].copy()
                current_embed = source_embeds[i]
                
                j = i + 1
                while j < len(source_chunks):
                    # Check similarity
                    sim = cosine_similarity([current_embed], [source_embeds[j]])[0][0]
                    new_text = ' '.join(current_texts) + ' ' + source_chunks[j]['text']
                    
                    if sim >= self.config.sim_threshold and len(new_text) <= self.config.max_merged_length:
                        current_texts.append(source_chunks[j]['text'])
                        current_embed_sum += source_embeds[j]
                        current_embed = current_embed_sum / np.linalg.norm(current_embed_sum)
                        j += 1
                    else:
                        break
                
                # Save grouped chunk
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
        logger.info(f"🔨 Creating enriched records from {len(chunks)} chunks...")
        self.stats['raw_chunks'] = len(chunks)
        
        # Deduplicate
        deduped = self.deduplicate(chunks)
        self.stats['after_dedup'] = len(deduped)
        
        # Group similar chunks
        grouped, grouped_embeds = self.group_similar_chunks(deduped, embeddings)
        
        # Semantic labeling
        if self.config.enable_semantic_labeling:
            texts = [c['text'] for c in grouped]
            semantic_labels = self.semantic_labeler.batch_label(texts)
            for chunk, labels in zip(grouped, semantic_labels):
                chunk['semantic_labels'] = labels
                if labels['themes']:
                    self.stats['with_themes'] += 1
        
        # Create final records with quality scoring
        records = []
        for idx, chunk in enumerate(tqdm(grouped, desc="Scoring quality")):
            text = chunk['text']
            
            # Skip very short chunks
            if len(text) < self.config.min_chunk_length:
                continue
            
            # Calculate quality scores
            quality_scores = self.quality_scorer.score_chunk(text)
            
            # Build comprehensive record
            record = {
                'text': text,
                'thread_id': str(uuid.uuid4()),  # Will be updated by ThreadCreator
                'quality_scores': quality_scores,
                'metadata': {
                    'filename': chunk['filename'],
                    'chunk_index': chunk.get('chunk_index', idx),
                    'section_title': chunk.get('section_title'),
                    'has_section': chunk.get('has_section', False),
                    'merged_from': chunk.get('merged_from', 1),
                    
                    # Semantic metadata
                    'semantic_themes': chunk.get('semantic_labels', {}).get('themes', []),
                    'primary_theme': chunk.get('semantic_labels', {}).get('primary_theme', 'general_content'),
                    'theme_confidence': chunk.get('semantic_labels', {}).get('confidence', 0.0),
                    
                    # Text statistics
                    'length': len(text),
                    'word_count': len(text.split()),
                    'sentence_count': len(re.split(r'[.!?]+', text)),
                    'unique_word_ratio': len(set(text.lower().split())) / max(len(text.split()), 1),
                    'avg_word_length': round(np.mean([len(w) for w in text.split()]), 2) if text.split() else 0,
                    
                    # Content indicators
                    'has_numbers': bool(re.search(r'\d', text)),
                    'has_special_chars': bool(re.search(r'[#$%&*@]', text)),
                    'capitalized_terms': len(re.findall(r'\b[A-Z][a-z]+\b', text)),
                }
            }
            
            records.append(record)
            
            if chunk.get('section_title'):
                self.stats['with_sections'] += 1
        
        self.stats['after_quality'] = len(records)
        logger.info(f"✅ Created {len(records)} enriched records")
        logger.info(f"   - With section titles: {self.stats['with_sections']}")
        logger.info(f"   - With semantic themes: {self.stats['with_themes']}")
        
        return records, grouped_embeds[:len(records)]
    
    def create_splits(self, records: List[Dict]) -> Dict[str, List[Dict]]:
        """Create stratified train/val/test splits"""
        logger.info(f"✂️  Creating data splits...")
        
        if not records:
            return {"train": [], "validation": [], "test": []}
        
        # Sort by composite quality
        records.sort(
            key=lambda x: x['quality_scores']['composite_quality'], 
            reverse=True
        )
        
        # Stratified shuffle by quartiles
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
        
        # Print statistics per split
        for name, data in splits.items():
            if data:
                avg_q = np.mean([r['quality_scores']['composite_quality'] for r in data])
                with_sections = sum(1 for r in data if r['metadata'].get('section_title'))
                with_themes = sum(1 for r in data if r['metadata'].get('semantic_themes'))
                
                logger.info(f"  {name.capitalize()}: {len(data)} records")
                logger.info(f"    - Avg quality: {avg_q:.3f}")
                logger.info(f"    - With sections: {with_sections} ({100*with_sections/len(data):.1f}%)")
                if self.config.enable_semantic_labeling:
                    logger.info(f"    - With themes: {with_themes} ({100*with_themes/len(data):.1f}%)")
        
        return splits


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class Pipeline:
    """Complete enhanced PDF to training data pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.start_time = None
    
    def save_jsonl(self, data: List[Dict], filename: str, compress: bool = False) -> None:
        """Save data to JSONL (optionally compressed)"""
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
        logger.info(f"💾 Saved {len(data)} records to {filename} ({size_mb:.2f} MB)")
    
    def run(self) -> None:
        """Execute complete pipeline"""
        import time
        self.start_time = time.time()
        
        logger.info("="*70)
        logger.info("🚀 ENHANCED PDF PROCESSING PIPELINE")
        logger.info("="*70)
        
        # Step 1: Extract PDFs with structure
        pdf_proc = PDFProcessor(self.config)
        documents = pdf_proc.extract_pdfs()
        if not documents:
            logger.error("No documents extracted. Exiting.")
            return
        
        chunks = pdf_proc.chunk_documents(documents)
        if not chunks:
            logger.error("No valid chunks created. Exiting.")
            return
        
        # Step 2: Generate embeddings
        embed_proc = EmbeddingProcessor(self.config)
        embeddings = embed_proc.embed_chunks(chunks)
        
        if self.config.save_intermediates:
            embed_proc.build_index('memory.index')
            np.save('memory_texts.npy', embed_proc.get_memory())
            logger.info("💾 Saved intermediates: memory.index, memory_texts.npy")
        
        # Step 3: Create enriched dataset
        dataset_creator = DatasetCreator(self.config, embed_proc.model)
        records, record_embeddings = dataset_creator.create_records(chunks, embeddings)
        
        if not records:
            logger.error("No records created. Exiting.")
            return
        
        # Step 4: Create semantic threads
        thread_creator = ThreadCreator(self.config, record_embeddings)
        records = thread_creator.create_threads(records)
        
        # Step 5: Create splits and save
        splits = dataset_creator.create_splits(records)
        
        for split_name, data in splits.items():
            if data:
                filename = f"{self.config.output_prefix}_{split_name}.jsonl"
                self.save_jsonl(data, filename, self.config.compress_output)
        
        # Print final statistics
        elapsed = time.time() - self.start_time
        self._print_final_stats(pdf_proc, embed_proc, dataset_creator, splits, elapsed)
    
    def _print_final_stats(self, pdf_proc, embed_proc, dataset_creator, splits, elapsed):
        """Print comprehensive final statistics"""
        logger.info("\n" + "="*70)
        logger.info("📊 PIPELINE COMPLETE - FINAL STATISTICS")
        logger.info("="*70)
        logger.info(f"PDFs processed: {pdf_proc.stats['successful']}/{pdf_proc.stats['total_pdfs']}")
        logger.info(f"Sections extracted: {pdf_proc.stats['sections_extracted']}")
        logger.info(f"Initial chunks: {pdf_proc.stats['total_chunks']}")
        logger.info(f"Embeddings generated: {embed_proc.stats['total_embedded']}")
        logger.info(f"After deduplication: {dataset_creator.stats['after_dedup']}")
        logger.info(f"After grouping: {dataset_creator.stats['grouped_chunks']}")
        logger.info(f"Final records: {dataset_creator.stats['after_quality']}")
        logger.info(f"  - With sections: {dataset_creator.stats['with_sections']}")
        
        if self.config.enable_semantic_labeling:
            logger.info(f"  - With semantic themes: {dataset_creator.stats['with_themes']}")
            
            # Get and save theme statistics
            theme_stats = dataset_creator.semantic_labeler.get_theme_statistics()
            logger.info(f"\n🏷️  Theme Discovery Results:")
            logger.info(f"  - Unique themes discovered: {theme_stats['total_unique_themes']}")
            logger.info(f"  - Top 5 themes: {', '.join(theme_stats['top_themes'][:5])}")
            
            # Save theme taxonomy to file
            taxonomy_file = f"{self.config.output_prefix}_theme_taxonomy.json"
            with open(taxonomy_file, 'w', encoding='utf-8') as f:
                json.dump(theme_stats, f, indent=2, ensure_ascii=False)
            logger.info(f"  - Theme taxonomy saved to: {taxonomy_file}")
        
        logger.info(f"\nData splits:")
        logger.info(f"  Train: {len(splits['train'])}")
        logger.info(f"  Validation: {len(splits['validation'])}")
        logger.info(f"  Test: {len(splits['test'])}")
        logger.info(f"\nProcessing time: {elapsed/60:.2f} minutes")
        logger.info("="*70)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced PDF Processing Pipeline with Semantic Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python pdf_pipeline.py --pdf-dir ./PDFs --output-prefix dataset
  
  # With semantic labeling
  python pdf_pipeline.py --pdf-dir ./PDFs --enable-semantic-labeling
  
  # Custom configuration
  python pdf_pipeline.py --pdf-dir ./docs --chunk-size 300 --no-compress
  
  # Disable section extraction
  python pdf_pipeline.py --pdf-dir ./PDFs --no-sections
        """
    )
    
    parser.add_argument('--pdf-dir', default='./PDFs', 
                       help='Directory containing PDFs')
    parser.add_argument('--output-prefix', default='dataset', 
                       help='Output filename prefix')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2', 
                       help='Sentence transformer model for embeddings')
    parser.add_argument('--semantic-model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                       help='Model for semantic labeling')
    parser.add_argument('--chunk-size', type=int, default=500, 
                       help='Words per chunk')
    parser.add_argument('--sim-threshold', type=float, default=0.7,
                       help='Similarity threshold for grouping chunks')
    parser.add_argument('--thread-threshold', type=float, default=0.65,
                       help='Similarity threshold for creating threads')
    parser.add_argument('--enable-semantic-labeling', action='store_true',
                       help='Enable semantic theme labeling (requires GPU)')
    parser.add_argument('--no-sections', action='store_true',
                       help='Disable section title extraction')
    parser.add_argument('--no-compress', action='store_true', 
                       help='Disable gzip compression')
    parser.add_argument('--save-intermediates', action='store_true', 
                       help='Save intermediate files')
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        pdf_dir=args.pdf_dir,
        output_prefix=args.output_prefix,
        embedding_model=args.embedding_model,
        semantic_model=args.semantic_model,
        chunk_size=args.chunk_size,
        sim_threshold=args.sim_threshold,
        thread_sim_threshold=args.thread_threshold,
        enable_semantic_labeling=args.enable_semantic_labeling,
        extract_sections=not args.no_sections,
        compress_output=not args.no_compress,
        save_intermediates=args.save_intermediates
    )
    
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()

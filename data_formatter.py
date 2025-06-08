import numpy as np
import json
import ftfy
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any, Set
import yaml
import random
import hashlib
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import gc
import psutil
from functools import lru_cache # For caching embeddings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataQualityConfig:
    """Configuration for data quality filtering"""
    def __init__(self, config_path: Optional[str] = None):
        # Default configuration
        self.min_text_length = 10
        self.max_text_length = 2000
        self.min_words = 3
        self.max_similarity_threshold = 0.95  # Remove near-duplicates across all data
        self.min_semantic_similarity = 0.1   # Minimum relevance between user/assistant (Q&A)
        self.max_semantic_similarity = 0.95  # Avoid identical responses for Q&A
        self.min_length_ratio = 0.1          # assistant/user length ratio for Q&A
        self.max_length_ratio = 10.0
        self.punctuation_ratio_threshold = 0.7 # Max ratio of non-alphanumeric chars
        self.quality_score_threshold = 0.46   # Minimum score for a pair to be included
        
        # New parameters for enhanced processing
        self.max_pairs_per_source = float('inf') # Changed to infinity
        self.diversity_threshold = 0.85      # Threshold for semantic diversity within dataset
        self.context_window_size = 3         # Number of surrounding chunks to consider for context
        
        # Performance optimization parameters
        self.batch_size = 64                 # Batch size for embedding computation
        self.max_workers = min(mp.cpu_count(), 8)  # Number of worker processes
        self.embedding_cache_size = 50000    # Max embeddings to cache
        
        if config_path and Path(config_path).exists():
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                for key, value in config_dict.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                    else:
                        logger.warning(f"Unknown config key: {key}")
            logger.info(f"Loaded quality configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")

def safe_unicode_escape(text: str) -> str:
    try:
        return text.encode('utf-8').decode('unicode_escape')
    except UnicodeDecodeError:
        # Try to manually replace malformed unicode escapes instead of failing
        text = re.sub(r'\\u[0-9A-Fa-f]{0,3}[^0-9A-Fa-f]', '', text)  # Kill malformed ones
        try:
            return text.encode('utf-8').decode('unicode_escape')
        except UnicodeDecodeError:
            return text  # Give up and return as-is

def clean_text(text: str) -> str:
    text = ftfy.fix_encoding(text)
    text = ftfy.fix_text(text)

    # Normalize stray whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip zero-width and invisible characters
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)

    # Fix line-break hyphens safely
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)

    # Collapse all newlines into single spaces
    text = re.sub(r'\s*\n\s*', ' ', text)

    # REMOVE UNWANTED QUOTES AROUND SINGLE WORDS OR PHRASES
    # This targets the specific pattern of quotes around single words like "trapped"
    # or phrases like "Certificate of Premedical College Work,"
    
    # Remove quotes around single words (handles the "trapped" case)
    text = re.sub(r'\b"(\w+)"\b', r'\1', text)
    
    # Remove quotes around phrases that end with punctuation (handles the certificate cases)
    text = re.sub(r'"([^"]+[.,!?])"', r'\1', text)
    
    # Remove quotes around phrases without ending punctuation
    text = re.sub(r'"([A-Z][^"]*?)"(?=\s|$)', r'\1', text)
    
    # Clean up any remaining escaped quotes from JSON artifacts
    text = re.sub(r'\\(["\'])', r'\1', text)
    
    # Handle any remaining double-escaped quotes
    while '\\\"' in text or '\\\'' in text:
        text = text.replace('\\\"', '"')
        text = text.replace('\\\'', "'")
    
    # Fix markdown + quote mismatches: *", "* etc.
    text = re.sub(r'\*+"', '"', text)     # remove redundant *"
    text = re.sub(r'"\*+', '"', text)     # remove redundant "*
    text = re.sub(r'\*\s*"', ' *"', text)
    text = re.sub(r'"\s*\*', '"* ', text)

    # Collapse accidental punctuation before or after quotes
    text = re.sub(r'([!?.,]){2,}["\']', r'\1"', text)

    # Normalize multi-spaces
    text = re.sub(r' {2,}', ' ', text)

    # FINAL CLEANUP: Remove any stray quote marks that don't serve a purpose
    # This is more aggressive - only use if you want to remove most quotes
    # text = re.sub(r'(?<!\w)"(?!\w)|(?<!\w)"(?=\w)|(?<=\w)"(?!\w)', '', text)
    
    return text.strip()

def _clean_text_batch(batch: List[str]) -> List[str]:
    return [clean_text(text) for text in batch]

def _validate_text_batch(texts: List[str], config: DataQualityConfig) -> List[bool]:
    """Validate texts in batch"""
    results = []
    low_quality_patterns = [
        re.compile(r'^[\s\-_=]{10,}$'),
        re.compile(r'^\d+\s*$'),
        re.compile(r'^[^\w\s]{5,}$'),
    ]
    
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            results.append(False)
            continue
            
        text = text.strip()
        if len(text) < config.min_text_length or len(text) > config.max_text_length:
            results.append(False)
            continue
        
        words = text.split()
        if len(words) < config.min_words:
            results.append(False)
            continue
            
        # Check character ratio
        alpha_chars = sum(c.isalpha() for c in text)
        if len(text) > 0 and (len(text) - alpha_chars) / len(text) > config.punctuation_ratio_threshold:
            results.append(False)
            continue
        
        # Check low quality patterns
        is_low_quality = any(pattern.match(text) for pattern in low_quality_patterns)
        results.append(not is_low_quality)
    
    return results

class OptimizedDataProcessor:
    def __init__(self, config: Optional[DataQualityConfig] = None, use_semantic_filtering: bool = True):
        self.config = config or DataQualityConfig()
        self.use_semantic_filtering = use_semantic_filtering
        self.model: Optional[SentenceTransformer] = None
        if self.use_semantic_filtering:
            self._load_embedding_model()
        self.text_hashes: Set[str] = set()
        self.domain_keywords = self._load_domain_keywords()
        
        if self.use_semantic_filtering:
            self._get_single_embedding = lru_cache(maxsize=self.config.embedding_cache_size)(self.__get_single_embedding_uncached)

    def _load_embedding_model(self) -> None:
        """Load the sentence transformer model for semantic filtering"""
        logger.info("Loading SentenceTransformer model for semantic filtering...")
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            self.use_semantic_filtering = False
            logger.warning("Disabling semantic filtering due to model loading failure.")

    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """Load domain-specific keywords with expanded coverage for interdisciplinary academic works"""
        return {
        'neuroscience': [
            'neural', 'brain', 'cognitive', 'neuron', 'synapse', 'computational', 
            'cortex', 'hippocampus', 'plasticity', 'neurotransmitter', 'consciousness',
            'perception', 'memory', 'learning', 'behavior', 'neurological', 'cerebral',
            'dendrite', 'axon', 'dopamine', 'serotonin', 'neuroplasticity', 'cognition',
            'neural network', 'brain imaging', 'fMRI', 'EEG', 'neuropsychology'
        ],
        
        'physics': [
            'quantum', 'field', 'particle', 'wave', 'energy', 'force', 'relativity', 
            'thermodynamics', 'electromagnetic', 'mechanics', 'entropy', 'cosmology',
            'universe', 'spacetime', 'gravity', 'photon', 'electron', 'atom', 'nuclear',
            'string theory', 'dark matter', 'black hole', 'big bang', 'physics',
            'theoretical physics', 'experimental', 'statistical mechanics', 'optics'
        ],
        
        'mathematics': [
            'theorem', 'proof', 'equation', 'algorithm', 'logic', 'function', 
            'derivative', 'integral', 'matrix', 'topology', 'geometry', 'calculus',
            'algebra', 'analysis', 'probability', 'statistics', 'number theory',
            'differential', 'mathematical', 'computation', 'set theory', 'graph theory',
            'combinatorics', 'optimization', 'linear algebra', 'abstract algebra'
        ],
        
        'computer_science': [
            'algorithm', 'computation', 'data', 'programming', 'artificial', 
            'database', 'network', 'security', 'software', 'hardware', 'computing',
            'machine learning', 'AI', 'artificial intelligence', 'cybernetics',
            'information theory', 'digital', 'computer', 'processor', 'memory',
            'operating system', 'distributed', 'parallel', 'complexity theory'
        ],
        
        'philosophy': [
            'consciousness', 'knowledge', 'reality', 'existence', 'ethics', 'logic', 
            'metaphysics', 'epistemology', 'ontology', 'phenomenology', 'philosophy',
            'philosophical', 'moral', 'virtue', 'truth', 'meaning', 'being',
            'mind', 'soul', 'free will', 'determinism', 'rationalism', 'empiricism',
            'existentialism', 'pragmatism', 'idealism', 'materialism', 'dualism'
        ],
        
        'biography': [
            'life', 'born', 'career', 'achievement', 'contribution', 'influence', 
            'biography', 'autobiography', 'memoir', 'legacy', 'childhood', 'education',
            'family', 'personal', 'death', 'accomplishment', 'pioneer', 'revolutionary',
            'inventor', 'scientist', 'philosopher', 'mathematician', 'physicist',
            'journey', 'story', 'experiences', 'letters', 'correspondence'
        ],
        
        'law': [
            'legal', 'court', 'statute', 'case', 'jurisdiction', 'precedent', 
            'constitutional', 'contract', 'tort', 'criminal', 'civil', 'justice',
            'law', 'lawyer', 'attorney', 'judge', 'trial', 'evidence', 'witness',
            'litigation', 'appeal', 'Supreme Court', 'federal', 'state law',
            'regulation', 'compliance', 'legal system', 'jurisprudence'
        ],
        
        'economics': [
            'economic', 'economy', 'market', 'trade', 'money', 'finance', 'banking',
            'investment', 'capitalism', 'socialism', 'wealth', 'poverty', 'inequality',
            'GDP', 'inflation', 'recession', 'supply', 'demand', 'price', 'cost',
            'profit', 'business', 'industry', 'labor', 'employment', 'game theory'
        ],
        
        'literature': [
            'novel', 'story', 'narrative', 'character', 'plot', 'theme', 'literary',
            'author', 'writer', 'poetry', 'poem', 'prose', 'fiction', 'non-fiction',
            'criticism', 'analysis', 'interpretation', 'symbolism', 'metaphor',
            'style', 'genre', 'classic', 'contemporary', 'modernism', 'postmodern'
        ],
        
        'psychology': [
            'psychological', 'behavior', 'mental', 'emotion', 'personality', 'therapy',
            'psychoanalysis', 'cognitive psychology', 'social psychology', 'development',
            'learning', 'memory', 'perception', 'motivation', 'unconscious',
            'conscious', 'psyche', 'psychiatry', 'clinical', 'experimental psychology'
        ],
        
        'history': [
            'historical', 'history', 'ancient', 'medieval', 'modern', 'contemporary',
            'civilization', 'culture', 'society', 'political', 'revolution', 'war',
            'empire', 'dynasty', 'century', 'era', 'period', 'timeline', 'chronology',
            'archaeological', 'primary source', 'secondary source', 'historiography'
        ],
        
        'science_general': [
            'scientific', 'research', 'experiment', 'hypothesis', 'theory', 'evidence',
            'method', 'empirical', 'observation', 'measurement', 'data', 'analysis',
            'discovery', 'innovation', 'technology', 'laboratory', 'peer review',
            'publication', 'journal', 'academic', 'scholar', 'study'
        ],
        
        'astronomy': [
            'astronomical', 'telescope', 'star', 'planet', 'galaxy', 'solar system',
            'constellation', 'nebula', 'comet', 'asteroid', 'spacecraft', 'NASA',
            'observatory', 'celestial', 'cosmic', 'interstellar', 'planetary',
            'astrophysics', 'cosmology', 'universe', 'space exploration'
        ],
        
        'biology': [
            'biological', 'evolution', 'species', 'organism', 'cell', 'DNA', 'gene',
            'genetics', 'natural selection', 'adaptation', 'ecosystem', 'ecology',
            'molecular biology', 'biochemistry', 'physiology', 'anatomy', 'taxonomy',
            'biodiversity', 'population', 'habitat', 'reproduction', 'inheritance'
        ],
        
        'chemistry': [
            'chemical', 'molecule', 'atom', 'element', 'compound', 'reaction',
            'bond', 'catalyst', 'organic chemistry', 'inorganic chemistry',
            'physical chemistry', 'analytical chemistry', 'laboratory', 'synthesis',
            'periodic table', 'ion', 'acid', 'base', 'solution', 'crystallography'
        ]
    }

    def _detect_domain_batch(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
        """Detect domains for multiple texts at once"""
        domains = []
        for text, metadata in zip(texts, metadatas):
            text_lower = text.lower()
            filename = metadata.get('filename', '').lower()
            
            domain_scores = defaultdict(int)
            for domain, keywords in self.domain_keywords.items():
                for keyword in keywords:
                    domain_scores[domain] += text_lower.count(keyword)
                    domain_scores[domain] += filename.count(keyword) * 2
            
            if domain_scores:
                domains.append(max(domain_scores, key=domain_scores.get))
            else:
                domains.append('general')
        return domains

    def __get_single_embedding_uncached(self, text: str) -> np.ndarray:
        """Helper to compute a single embedding (uncached version)"""
        if not self.use_semantic_filtering or not self.model:
            return np.array([])
        
        try:
            embedding = self.model.encode(
                text, 
                convert_to_numpy=True, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embedding
        except Exception as e:
            logger.warning(f"Failed to compute embedding for text: {text[:50]}... Error: {e}")
            return np.zeros(384) # Return zeros for all-MiniLM-L6-v2 dimension

    def _compute_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings in batches, utilizing cache for individual texts"""
        if not self.use_semantic_filtering or not self.model:
            return np.array([])
        
        cached_embeddings = []
        texts_to_encode = []
        original_indices = [] # To reorder embeddings correctly
        
        # Initialize the unified progress bar with dynamic_ncols and mininterval
        progress_bar = tqdm(total=len(texts), desc="Embedding All Texts", dynamic_ncols=True, mininterval=0.5)

        for i, text in enumerate(texts):
            try:
                embedding = self._get_single_embedding(text)
                if embedding.size > 0:
                    cached_embeddings.append((i, embedding))
                    progress_bar.update(1)  # Update for cache hit
                else:
                    texts_to_encode.append(text)
                    original_indices.append(i)
            except TypeError:
                texts_to_encode.append(text)
                original_indices.append(i)
            except Exception as e:
                logger.warning(f"Error accessing cache for text: {text[:50]}... Error: {e}")
                texts_to_encode.append(text)
                original_indices.append(i)
        
        new_embeddings = []
        if texts_to_encode:
            logger.info(f"Encoding {len(texts_to_encode)} new texts (not in cache)...") 
            batch_size = self.config.batch_size
            for i in range(0, len(texts_to_encode), batch_size):
                batch = texts_to_encode[i:i + batch_size]
                try:
                    batch_embeddings = self.model.encode(
                        batch, 
                        convert_to_numpy=True, 
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        batch_size=len(batch)
                    )
                    for j, emb in enumerate(batch_embeddings):
                        original_idx = original_indices[i + j]
                        new_embeddings.append((original_idx, emb))
                        progress_bar.update(1)  # Update for each newly encoded embedding
                except Exception as e:
                    logger.warning(f"Failed to compute embeddings for batch {i//batch_size}: {e}")
                    for j in range(len(batch)):
                        original_idx = original_indices[i+j]
                        new_embeddings.append((original_idx, np.zeros(384)))
                        progress_bar.update(1) # Still update even on failure
        
        progress_bar.close() # Close the progress bar
        
        # Combine cached and newly computed embeddings, ensuring correct order
        all_embeddings_with_indices = sorted(cached_embeddings + new_embeddings, key=lambda x: x[0])
        final_embeddings = np.vstack([emb for idx, emb in all_embeddings_with_indices]) if all_embeddings_with_indices else np.array([])
        
        return final_embeddings

    def _efficient_deduplication(self, embeddings: np.ndarray, threshold: float = None) -> List[int]:
        """Efficient semantic deduplication using vectorized operations"""
        if embeddings.size == 0:
            return list(range(len(embeddings)))
        
        threshold = threshold or self.config.max_similarity_threshold
        keep_indices = []
        
        # Process in chunks to manage memory
        chunk_size = 1000
        
        for i in range(0, len(embeddings), chunk_size):
            end_idx = min(i + chunk_size, len(embeddings))
            current_chunk = embeddings[i:end_idx]
            
            # Only compare with previously kept embeddings (or all if first chunk)
            if keep_indices:
                kept_embeddings = embeddings[keep_indices]
                similarities = cosine_similarity(current_chunk, kept_embeddings)
                max_similarities = np.max(similarities, axis=1)
                
                # Keep only items below threshold
                chunk_keep = [j for j, sim in enumerate(max_similarities) if sim < threshold]
                keep_indices.extend([i + j for j in chunk_keep])
            else:
                # First chunk - perform internal deduplication
                if len(current_chunk) > 1:
                    internal_similarities = cosine_similarity(current_chunk)
                    np.fill_diagonal(internal_similarities, 0) # Don't compare item to itself
                    
                    # Use a boolean mask for efficient removal
                    to_remove_in_chunk = set()
                    for r in range(current_chunk.shape[0]):
                        for c in range(r + 1, current_chunk.shape[0]): # Only check upper triangle
                            if internal_similarities[r, c] >= threshold:
                                to_remove_in_chunk.add(c) # Mark the second item in the pair for removal
                    
                    for j in range(current_chunk.shape[0]):
                        if j not in to_remove_in_chunk:
                            keep_indices.append(i + j)
                else: # Single item in chunk
                    keep_indices.extend(list(range(i, end_idx)))
        
        return keep_indices

    def deduplicate_and_clean_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimized deduplication and cleaning with parallel processing"""
        logger.info(f"Starting optimized deduplication and cleaning of {len(entries)} entries...")
        
        # Extract texts for batch processing
        texts = [entry.get('text', '') for entry in entries]
        metadatas = [entry.get('metadata', {}) for entry in entries]
        
        # Parallel text cleaning
        logger.info("Cleaning texts in parallel...")
        chunk_size = len(texts) // self.config.max_workers + 1
        text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        cleaned_texts_flat = []
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            for chunk in tqdm(executor.map(_clean_text_batch, text_chunks), total=len(text_chunks), desc="Cleaning Chunks"):
                cleaned_texts_flat.extend(chunk)
        
        # Parallel text validation
        logger.info("Validating texts in parallel...")
        config_for_workers = self.config
        validate_func = partial(_validate_text_batch, config=config_for_workers)
        
        validations_flat = []
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            for chunk in tqdm(executor.map(validate_func, text_chunks), total=len(text_chunks), desc="Validating Chunks"):
                validations_flat.extend(chunk)
        
        # Filter valid entries
        valid_entries = []
        valid_texts = []
        valid_metadatas = []
        
        current_text_hashes = set()
        
        for i, (entry, cleaned_text, is_valid) in enumerate(zip(entries, cleaned_texts_flat, validations_flat)):
            if not is_valid or not cleaned_text:
                continue
            
            # Hash-based deduplication
            text_hash = hashlib.md5(cleaned_text.encode('utf-8')).hexdigest()
            if text_hash in current_text_hashes:
                continue
            current_text_hashes.add(text_hash)
            
            entry['cleaned_text'] = cleaned_text
            valid_entries.append(entry)
            valid_texts.append(cleaned_text)
            valid_metadatas.append(metadatas[i])
        
        logger.info(f"After basic filtering (length, patterns, exact duplicates): {len(valid_entries)} entries remain")
        
        if not self.use_semantic_filtering:
            # Add domains without semantic filtering
            domains = self._detect_domain_batch(valid_texts, valid_metadatas)
            for entry, domain in zip(valid_entries, domains):
                entry['domain'] = domain
            return valid_entries
        
        # Batch embedding computation
        logger.info("Computing embeddings in batches (with cache)...")
        embeddings = self._compute_embeddings_batch(valid_texts)
        
        if embeddings.size == 0 or len(embeddings) != len(valid_texts):
            logger.warning("No embeddings computed or mismatch in length. Returning entries without semantic filtering")
            domains = self._detect_domain_batch(valid_texts, valid_metadatas)
            for entry, domain in zip(valid_entries, domains):
                entry['domain'] = domain
            return valid_entries
        
        # Efficient semantic deduplication
        logger.info("Performing semantic deduplication...")
        keep_indices = self._efficient_deduplication(embeddings)
        
        # Filter entries and add domains
        final_entries = [valid_entries[i] for i in keep_indices]
        final_texts = [valid_texts[i] for i in keep_indices]
        final_metadatas = [valid_metadatas[i] for i in keep_indices]
        
        domains = self._detect_domain_batch(final_texts, final_metadatas)
        for entry, domain in zip(final_entries, domains):
            entry['domain'] = domain
        
        logger.info(f"Final count after semantic deduplication: {len(final_entries)} entries")
        return final_entries

    def _assess_pair_quality_batch(self, user_texts: List[str], assistant_texts: List[str]) -> List[Dict[str, Any]]:
        """Assess quality for multiple pairs at once"""
        results = []
        
        # Compute embeddings in batch if semantic filtering is enabled
        all_texts = user_texts + assistant_texts
        if self.use_semantic_filtering:
            # Use _compute_embeddings_batch which handles caching
            all_embeddings = self._compute_embeddings_batch(all_texts)
            user_embeddings = all_embeddings[:len(user_texts)]
            assistant_embeddings = all_embeddings[len(user_texts):]
        else:
            user_embeddings = assistant_embeddings = None
        
        for i, (user_text, assistant_text) in enumerate(zip(user_texts, assistant_texts)):
            metrics = {
                "user_len": len(user_text),
                "assistant_len": len(assistant_text),
                "user_words": len(user_text.split()),
                "assistant_words": len(assistant_text.split()),
                "semantic_similarity": 0.0,
                "length_ratio": 0.0,
                "quality_score": 0.0,
                "readability_score": 0.0,
                "information_density": 0.0
            }

            # Length ratio
            if metrics["user_len"] > 0:
                metrics["length_ratio"] = metrics["assistant_len"] / metrics["user_len"]

            # Semantic similarity
            if self.use_semantic_filtering and user_embeddings is not None and user_embeddings.size > 0:
                try:
                    similarity = cosine_similarity(
                        [user_embeddings[i]], 
                        [assistant_embeddings[i]]
                    )[0][0]
                    metrics["semantic_similarity"] = float(similarity)
                except Exception as e:
                    logger.warning(f"Error computing semantic similarity for pair {i}: {e}")
                    metrics["semantic_similarity"] = 0.0

            # Readability score (simplified Flesch-Kincaid concept)
            sentences = len(re.split(r'[.!?]+', assistant_text))
            avg_sentence_len = metrics["assistant_words"] / max(sentences, 1)
            metrics["readability_score"] = min(1.0, 20 / max(avg_sentence_len, 1)) # Normalize to 0-1, higher is better

            # Information density (simplified lexical diversity)
            words = assistant_text.lower().split()
            unique_words = len(set(words))
            metrics["information_density"] = unique_words / max(len(words), 1)

            # Calculate composite quality score
            score = 1.0

            # Penalize if length ratio is outside desirable range
            if not (self.config.min_length_ratio <= metrics["length_ratio"] <= self.config.max_length_ratio):
                score *= 0.5 # Significant penalty

            # Penalize if semantic similarity is outside desirable range (too low or too high)
            if not (self.config.min_semantic_similarity <= metrics["semantic_similarity"] <= self.config.max_semantic_similarity):
                score *= 0.7 # Moderate penalty

            # Incorporate readability and information density (weighted)
            score *= (0.7 + 0.3 * metrics["readability_score"]) # Readability is 30% of remaining score
            score *= (0.8 + 0.2 * metrics["information_density"]) # Information density is 20% of remaining score

            metrics["quality_score"] = score
            results.append(metrics)
        
        return results

    def create_conversational_pairs(self, convo_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimized conversational pair creation"""
        pairs = []
        
        # Group messages by conversation ID
        conversations: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for entry in convo_entries:
            convo_id = entry['metadata'].get('conversation_id')
            if convo_id is not None:
                conversations[convo_id].append(entry)
        
        logger.info(f"Processing {len(conversations)} conversations in batches...")
        
        all_pairs_from_conversations = []
        
        for convo_id, msgs in tqdm(conversations.items(), desc="Creating conversational pairs"):
            msgs.sort(key=lambda x: x['metadata'].get('timestamp') or 0)
            
            current_user_msg = None
            conversation_context = []
            
            conversation_batch_user_texts = []
            conversation_batch_assistant_texts = []
            conversation_batch_metadata = []
            
            for msg in msgs:
                author = msg['metadata'].get('author')
                text = msg.get('cleaned_text')
                
                if not text:
                    continue
                
                if author == 'user':
                    current_user_msg = msg
                    context_text = ""
                    if len(conversation_context) > 0:
                        context_text = " ".join(conversation_context[-self.config.context_window_size:])
                    current_user_msg['context'] = context_text
                    
                elif author == 'assistant' and current_user_msg:
                    user_text_with_context = current_user_msg['cleaned_text']
                    if current_user_msg.get('context'):
                        user_text_with_context = f"Context: {current_user_msg['context']}\n\nUser: {user_text_with_context}"
                    
                    conversation_batch_user_texts.append(user_text_with_context)
                    conversation_batch_assistant_texts.append(text)
                    conversation_batch_metadata.append({
                        'user_msg': current_user_msg['metadata'],
                        'assistant_msg': msg['metadata'],
                        'source_file': current_user_msg['metadata'].get('source_file', 'conversation'),
                        'domain': current_user_msg.get('domain', 'general')
                    })
                    
                    conversation_context.append(f"User: {current_user_msg['cleaned_text']}")
                    conversation_context.append(f"Assistant: {text}")
                    current_user_msg = None
                else:
                    current_user_msg = None

            if conversation_batch_user_texts:
                quality_metrics_batch = self._assess_pair_quality_batch(
                    conversation_batch_user_texts, conversation_batch_assistant_texts
                )
                
                for user_text, assistant_text, metadata, quality_metrics in zip(
                    conversation_batch_user_texts, conversation_batch_assistant_texts, 
                    conversation_batch_metadata, quality_metrics_batch
                ):
                    if quality_metrics["quality_score"] >= self.config.quality_score_threshold:
                        all_pairs_from_conversations.append({
                            'user': user_text,
                            'assistant': assistant_text,
                            'quality_metrics': quality_metrics,
                            'source_metadata': metadata
                        })
        
        logger.info(f"Created {len(all_pairs_from_conversations)} conversational pairs")
        return all_pairs_from_conversations

    def generate_diverse_questions(self, chunk_text: str, metadata: Dict[str, Any], domain: str) -> List[str]:
        """Generate multiple diverse questions for a given chunk based on domain"""
        domain_templates = {
            'neuroscience': [
                "How does this relate to brain function and neural processing?",
                "What computational principles or neural algorithms are discussed here?",
                "Explain the neural mechanisms and pathways described:",
                "What cognitive processes and behavioral outcomes are involved?",
                "How does this contribute to our understanding of consciousness?",
                "What are the implications for neuroplasticity and learning?",
                "Describe the relationship between structure and function in this context:",
                "How do these findings relate to computational neuroscience models?"
            ],
            
            'physics': [
                "What physical principles and fundamental laws are explained here?",
                "How does this relate to quantum mechanics, relativity, or field theory?",
                "Explain the mathematical formulation and theoretical framework:",
                "What are the experimental implications and testable predictions?",
                "How does this contribute to our understanding of the universe?",
                "What role does symmetry and conservation laws play here?",
                "Describe the connection between theory and observation:",
                "How does this relate to cosmology or particle physics?"
            ],
            
            'mathematics': [
                "Prove or rigorously explain this mathematical concept:",
                "What is the significance and applications of this theorem/result?",
                "How is this mathematical framework applied in practice?",
                "Walk through the mathematical reasoning and logical structure:",
                "What are the underlying assumptions and their validity?",
                "How does this relate to other areas of mathematics?",
                "Explain the geometric or algebraic intuition behind this:",
                "What are the computational aspects and algorithmic implementations?"
            ],
            
            'computer_science': [
                "Explain the algorithm and data structures described in this text:",
                "How does this relate to computational complexity and efficiency?",
                "What are the theoretical foundations and practical applications?",
                "Discuss the artificial intelligence and machine learning concepts:",
                "How does this contribute to information theory or cybernetics?",
                "What are the security and privacy implications?",
                "Describe the software engineering principles involved:",
                "How does this relate to distributed systems or parallel computing?"
            ],
            
            'philosophy': [
                "What philosophical argument and logical structure is presented here?",
                "How does this relate to consciousness, knowledge, or reality?",
                "What are the ethical implications and moral considerations?",
                "Critically analyze this position and its assumptions:",
                "How does this contribute to metaphysical or epistemological debates?",
                "What are the phenomenological aspects of this discussion?",
                "Compare this view with other philosophical traditions:",
                "What are the practical implications for human existence?"
            ],
            
            'biography': [
                "What were the key contributions and revolutionary insights of this person?",
                "How did their work influence and transform their field?",
                "What personal and professional challenges did they overcome?",
                "Describe their lasting impact on science, society, or culture:",
                "What was their intellectual journey and development process?",
                "How did their personal life intersect with their professional work?",
                "What lessons can we learn from their approach to problems?",
                "How did they collaborate with or influence other notable figures?"
            ],
            
            'law': [
                "Explain the legal precedent and its broader implications:",
                "What is the constitutional interpretation and statutory analysis?",
                "How does this court case affect future legal reasoning?",
                "Summarize the jurisdictional issues and procedural aspects:",
                "What are the policy implications of this legal principle?",
                "How does this relate to civil rights or constitutional law?",
                "Explain the balance between competing legal interests:",
                "What is the historical context and evolution of this legal concept?"
            ],
            
            'economics': [
                "What economic principles and market mechanisms are discussed?",
                "How does this relate to game theory and strategic behavior?",
                "Explain the mathematical models and their assumptions:",
                "What are the policy implications and real-world applications?",
                "How does this contribute to understanding of wealth and inequality?",
                "Describe the relationship between individual and collective behavior:",
                "What are the microeconomic and macroeconomic connections?",
                "How does this relate to behavioral economics and decision-making?"
            ],
            
            'literature': [
                "What literary techniques and narrative structures are employed?",
                "How do themes and symbolism contribute to the overall meaning?",
                "Analyze the character development and psychological depth:",
                "What is the cultural and historical context of this work?",
                "How does this relate to broader literary movements or genres?",
                "What social commentary or critique is being presented?",
                "Explain the stylistic choices and their artistic effects:",
                "How does this work influence or reflect contemporary thought?"
            ],
            
            'psychology': [
                "What psychological theories and behavioral patterns are described?",
                "How does this relate to cognitive processes and mental functions?",
                "Explain the experimental methodology and research findings:",
                "What are the clinical applications and therapeutic implications?",
                "How does this contribute to understanding of human development?",
                "Describe the relationship between psychology and neuroscience:",
                "What are the social and cultural factors influencing behavior?",
                "How does this relate to personality theory and individual differences?"
            ],
            
            'history': [
                "What historical context and chronological developments are presented?",
                "How do these events relate to broader historical patterns?",
                "What were the causes and consequences of these developments?",
                "Analyze the primary sources and historical evidence:",
                "How does this period influence contemporary society?",
                "What role did key historical figures play in these events?",
                "Describe the political, social, and economic factors involved:",
                "How does this historical analysis challenge or confirm existing narratives?"
            ],
            
            'science_general': [
                "What scientific methods and research approaches are employed?",
                "How does this contribute to the advancement of scientific knowledge?",
                "Explain the experimental design and data interpretation:",
                "What are the broader implications for scientific understanding?",
                "How does this research connect multiple scientific disciplines?",
                "What technological applications emerge from this work?",
                "Describe the peer review process and scientific validation:",
                "How does this challenge or extend existing scientific paradigms?"
            ],
            
            'astronomy': [
                "What astronomical phenomena and celestial mechanics are described?",
                "How does this relate to our understanding of the cosmos?",
                "Explain the observational techniques and instrumentation used:",
                "What are the implications for cosmology and the universe's evolution?",
                "How does this contribute to the search for extraterrestrial life?",
                "Describe the relationship between theory and astronomical observation:",
                "What role does this play in space exploration and technology?",
                "How does this connect to fundamental physics and the laws of nature?"
            ],
            
            'biology': [
                "What biological processes and evolutionary mechanisms are discussed?",
                "How does this relate to genetics, molecular biology, or ecology?",
                "Explain the experimental approaches and biological techniques:",
                "What are the implications for understanding life and evolution?",
                "How does this contribute to medical or biotechnological applications?",
                "Describe the relationship between structure and function in living systems:",
                "What environmental and ecological factors are involved?",
                "How does this research impact conservation and biodiversity efforts?"
            ],
            
            'chemistry': [
                "What chemical reactions and molecular processes are described?",
                "How does this relate to atomic structure and chemical bonding?",
                "Explain the experimental procedures and analytical techniques:",
                "What are the industrial or pharmaceutical applications?",
                "How does this contribute to materials science and nanotechnology?",
                "Describe the thermodynamic and kinetic aspects:",
                "What safety and environmental considerations are involved?",
                "How does this research advance our understanding of matter and energy?"
            ]
        }
        
        # Enhanced general templates for interdisciplinary works
        general_templates = [
            "Summarize the key insights and central arguments from this text:",
            "What are the main ideas and how do they connect to broader themes?",
            "Provide detailed analysis of the concepts presented in this passage:",
            "What primary information and supporting evidence is conveyed here?",
            "What are the theoretical and practical implications of this information?",
            "How does this contribute to interdisciplinary understanding?",
            "What questions does this text raise for future investigation?",
            "Compare and contrast the different perspectives presented:",
            "What methodological approaches are discussed or implied?",
            "How does this text challenge conventional wisdom or established views?"
        ]
        
        # Get domain-specific templates, fallback to general if domain not found
        templates = domain_templates.get(domain, general_templates)
        
        # Combine with general templates for more diversity
        all_templates = list(set(templates + general_templates))
        
        # Select 3-5 questions depending on availability
        num_questions = min(5, max(3, len(all_templates) // 3))
        selected = random.sample(all_templates, num_questions)
        
        # Enhanced context prefix with more metadata
        context_prefix = ""
        if 'filename' in metadata:
            # Clean filename for better readability
            clean_filename = metadata['filename'].replace('.pdf', '').replace('_', ' ')
            context_prefix = f"From '{clean_filename}': "
        if 'page' in metadata:
            context_prefix += f"(Page {metadata['page']}) "
        if 'chapter' in metadata:
            context_prefix += f"[Chapter: {metadata['chapter']}] "
        
        return [context_prefix + template for template in selected]

    def create_pdf_qa_pairs(self, pdf_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimized PDF Q&A pair creation with batch processing"""
        qa_pairs = []
        logger.info(f"Generating Q&A pairs from {len(pdf_entries)} PDF chunks in batches...")
        
        by_source = defaultdict(list)
        for entry in pdf_entries:
            source = entry['metadata'].get('filename', 'unknown_pdf_source')
            by_source[source].append(entry)
        
        for source, entries in tqdm(by_source.items(), desc="Processing PDF sources"):
            logger.info(f"Processing {len(entries)} chunks from {source}")
            
            max_entries_for_source = min(len(entries), self.config.max_pairs_per_source // 3)
            selected_entries = random.sample(entries, max_entries_for_source) if len(entries) > max_entries_for_source else entries
            
            batch_questions = []
            batch_answers = []
            batch_metadata = []
            
            for entry in selected_entries:
                chunk_text = entry.get('cleaned_text')
                metadata = entry.get('metadata', {})
                domain = entry.get('domain', 'general')
                
                if not chunk_text:
                    continue
                
                questions = self.generate_diverse_questions(chunk_text, metadata, domain) # Fixed line
                
                for question in questions:
                    batch_questions.append(question)
                    batch_answers.append(chunk_text)
                    batch_metadata.append({**metadata, 'domain': domain, 'source_file': source})
            
            if batch_questions:
                quality_metrics_batch = self._assess_pair_quality_batch(batch_questions, batch_answers)
                
                for question, answer, metadata, quality_metrics in zip(
                    batch_questions, batch_answers, batch_metadata, quality_metrics_batch
                ):
                    if quality_metrics["quality_score"] >= self.config.quality_score_threshold:
                        qa_pairs.append({
                            'user': question,
                            'assistant': answer,
                            'quality_metrics': quality_metrics,
                            'source_metadata': metadata
                        })
        
        logger.info(f"Created {len(qa_pairs)} PDF Q&A pairs")
        return qa_pairs

    def create_data_splits(self, all_pairs: List[Dict[str, Any]], split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Dict[str, List[Dict[str, Any]]]:
        """Creates stratified train, validation, and test splits based on source and quality."""
        logger.info(f"Splitting {len(all_pairs)} pairs into train/val/test...")
        
        if not all_pairs:
            return {"train": [], "validation": [], "test": []}

        all_pairs.sort(key=lambda x: x['quality_metrics']['quality_score'], reverse=True)
        
        num_quartiles = 4
        total_len = len(all_pairs)
        for i in range(num_quartiles):
            start_idx = i * (total_len // num_quartiles)
            end_idx = (i + 1) * (total_len // num_quartiles) if i < num_quartiles - 1 else total_len
            quartile = all_pairs[start_idx:end_idx]
            random.shuffle(quartile)
            all_pairs[start_idx:end_idx] = quartile

        total_pairs = len(all_pairs)
        train_end = int(total_pairs * split_ratio[0])
        val_end = train_end + int(total_pairs * split_ratio[1])

        train_data = all_pairs[:train_end]
        val_data = all_pairs[train_end:val_end]
        test_data = all_pairs[val_end:]

        logger.info(f"Splits: Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")
        
        for split_name, data in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
            if data:
                avg_quality = np.mean([pair['quality_metrics']['quality_score'] for pair in data])
                logger.info(f"{split_name} average quality: {avg_quality:.3f}")
        
        return {
            "train": train_data,
            "validation": val_data,
            "test": test_data
        }

    def save_datasets(self, splits: Dict[str, List[Dict[str, Any]]], output_dir: Path) -> None:
        """Saves the splits to .jsonl files with enhanced formatting options."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_summary = {
            'total_pairs': sum(len(data) for data in splits.values()),
            'splits': {name: len(data) for name, data in splits.items()},
            'quality_stats': {},
            'source_distribution': defaultdict(int),
            'domain_distribution': defaultdict(int)
        }
        
        for split_name, data in splits.items():
            if not data:
                continue
                
            qualities = [pair['quality_metrics']['quality_score'] for pair in data]
            metadata_summary['quality_stats'][split_name] = {
                'mean': float(np.mean(qualities)),
                'std': float(np.std(qualities)),
                'min': float(np.min(qualities)),
                'max': float(np.max(qualities))
            }
            
            for pair in data:
                source = pair.get('source_metadata', {}).get('source_file', 'conversation')
                domain = pair.get('source_metadata', {}).get('domain', 'general')
                metadata_summary['source_distribution'][source] += 1
                metadata_summary['domain_distribution'][domain] += 1
            
            output_path = output_dir / f"{split_name}.jsonl"
            with open(output_path, "w", encoding="utf-8") as f:
                for item in data:
                    formatted_line = (
                        f"<|user|>{item['user']}"
                        f"<|assistant|>{item['assistant']}"
                        f"<|endoftext|>"
                    )
                    f.write(json.dumps({"text": formatted_line}, ensure_ascii=False) + "\n")
            
            detailed_path = output_dir / f"{split_name}_detailed.jsonl"
            with open(detailed_path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            logger.info(f"Saved {len(data)} items to {output_path}")
        
        with open(output_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved dataset metadata to {output_dir / 'dataset_metadata.json'}")

def main():
    config = DataQualityConfig()
    processor = OptimizedDataProcessor(
        config=config,
        use_semantic_filtering=True
    )
    
    try:
        embedded_texts_raw = np.load("memory_texts.npy", allow_pickle=True)
        with open("memory_metadata.pkl", 'rb') as f:
            embedded_metadata_raw = pickle.load(f)
        logger.info(f"Loaded {len(embedded_texts_raw)} embedded texts and metadata.")
        
        all_entries_raw = [
            {'text': text, 'metadata': meta} 
            for text, meta in zip(embedded_texts_raw, embedded_metadata_raw)
        ]

    except FileNotFoundError:
        logger.error("memory_texts.npy or memory_metadata.pkl not found! Run batch_embedder.py first.")
        return
    except Exception as e:
        logger.error(f"Error loading embedded data: {e}")
        return
    
    cleaned_all_entries = processor.deduplicate_and_clean_entries(all_entries_raw)

    convo_entries = [e for e in cleaned_all_entries if e['metadata'].get('source') == 'conversation']
    pdf_entries = [e for e in cleaned_all_entries if e['metadata'].get('source') == 'pdf']
    
    logger.info(f"Data distribution: {len(convo_entries)} conversation entries, {len(pdf_entries)} PDF entries")
    
    conversational_pairs = processor.create_conversational_pairs(convo_entries)
    pdf_qa_pairs = processor.create_pdf_qa_pairs(pdf_entries)
    
    all_final_pairs = conversational_pairs + pdf_qa_pairs
    
    if not all_final_pairs:
        logger.error("No quality dialogue pairs created from any source! Consider adjusting quality thresholds or checking input data.")
        return
    
    splits = processor.create_data_splits(all_final_pairs)
    
    output_dir = Path("data_finetune")
    processor.save_datasets(splits, output_dir)
    
    total_pairs = len(all_final_pairs)
    avg_quality = np.mean([pair['quality_metrics']['quality_score'] for pair in all_final_pairs])
    
    logger.info(f"\nEnhanced data formatting complete!")
    logger.info(f"  Total quality pairs for fine-tuning: {total_pairs}")
    logger.info(f"  Average quality score: {avg_quality:.3f}")
    logger.info(f"  Conversational pairs: {len(conversational_pairs)}")
    logger.info(f"  PDF Q&A pairs: {len(pdf_qa_pairs)}")
    logger.info(f"  Generated training files in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()

import os
import json
import faiss
import numpy as np
import ftfy
import re
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- DATA QUALITY CONFIGURATION ---
@dataclass
class DataQualityConfig:
    """Configuration for text quality scoring and filtering"""
    min_text_length: int = 50
    max_text_length: int = 10000
    min_words: int = 5
    punctuation_ratio_threshold: float = 0.4
    enable_cleaning: bool = True
    enable_validation: bool = True
    
    def __post_init__(self):
        logger.info(f"📋 Data Quality Config: min_len={self.min_text_length}, "
                   f"max_len={self.max_text_length}, min_words={self.min_words}")


# --- TEXT CLEANING FUNCTIONS ---
def safe_unicode_escape(text: str) -> str:
    """Safely handle unicode escape sequences"""
    try:
        return text.encode('utf-8').decode('unicode_escape')
    except UnicodeDecodeError:
        # Remove malformed unicode escapes
        text = re.sub(r'\\u[0-9A-Fa-f]{0,3}[^0-9A-Fa-f]', '', text)
        try:
            return text.encode('utf-8').decode('unicode_escape')
        except UnicodeDecodeError:
            return text


def clean_text(text: str) -> str:
    """Deep clean text with encoding fixes and normalization"""
    # Fix encoding issues
    text = ftfy.fix_encoding(text)
    text = ftfy.fix_text(text)
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip zero-width and invisible characters
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    
    # Fix line-break hyphens
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    
    # Collapse newlines into spaces
    text = re.sub(r'\s*\n\s*', ' ', text)
    
    # Remove unwanted quotes around single words
    text = re.sub(r'\b"(\w+)"\b', r'\1', text)
    
    # Remove quotes around phrases with punctuation
    text = re.sub(r'"([^"]+[.,!?])"', r'\1', text)
    
    # Remove quotes around capitalized phrases
    text = re.sub(r'"([A-Z][^"]*?)"(?=\s|$)', r'\1', text)
    
    # Clean up escaped quotes
    text = re.sub(r'\\(["\'])', r'\1', text)
    
    # Handle double-escaped quotes
    while '\\\"' in text or '\\\'' in text:
        text = text.replace('\\\"', '"')
        text = text.replace('\\\'', "'")
    
    # Fix markdown + quote mismatches
    text = re.sub(r'\*+"', '"', text)
    text = re.sub(r'"\*+', '"', text)
    text = re.sub(r'\*\s*"', ' *"', text)
    text = re.sub(r'"\s*\*', '"* ', text)
    
    # Collapse accidental punctuation
    text = re.sub(r'([!?.,]){2,}["\']', r'\1"', text)
    
    # Normalize multi-spaces
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


def validate_text(text: str, config: DataQualityConfig) -> bool:
    """Validate text quality based on configuration"""
    if not isinstance(text, str) or not text.strip():
        return False
    
    text = text.strip()
    
    # Length checks
    if len(text) < config.min_text_length or len(text) > config.max_text_length:
        return False
    
    # Word count check
    words = text.split()
    if len(words) < config.min_words:
        return False
    
    # Character ratio check
    alpha_chars = sum(c.isalpha() for c in text)
    if len(text) > 0 and (len(text) - alpha_chars) / len(text) > config.punctuation_ratio_threshold:
        return False
    
    # Low quality pattern checks
    low_quality_patterns = [
        re.compile(r'^[\s\-_=]{10,}$'),
        re.compile(r'^\d+\s*$'),
        re.compile(r'^[^\w\s]{5,}$'),
    ]
    
    if any(pattern.match(text) for pattern in low_quality_patterns):
        return False
    
    return True


# --- BATCH EMBEDDER CLASS ---
class BatchEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', 
                 quality_config: Optional[DataQualityConfig] = None):
        logger.info(f"🧠 Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.quality_config = quality_config or DataQualityConfig()
        
        # Memory stores
        self.memory = []
        self.memory_vectors = []
        self.index = None
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'total_embedded': 0,
            'rejected_validation': 0,
            'rejected_cleaning': 0
        }
    
    def embed_and_store(self, text: str) -> bool:
        """Embed text and add to memory store"""
        try:
            embedding = self.model.encode([text])[0]
            self.memory.append(text)
            self.memory_vectors.append(embedding)
            self.stats['total_embedded'] += 1
            return True
        except Exception as e:
            logger.warning(f"Failed to embed text: {e}")
            return False
    
    def stream_pdf_json_chunks(self, json_path: str, chunk_size: int = 500):
        """Stream and chunk PDF JSON data"""
        with open(json_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        
        for doc in entries:
            text = doc.get("text", "")
            filename = doc.get("filename", "unknown.pdf")
            words = text.split()
            
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk.strip():
                    yield f"[PDF:{filename}] {chunk.strip()}", filename
    
    def process_chunk(self, chunk: str, source: str = "unknown") -> Optional[str]:
        """Process a single chunk with cleaning and validation"""
        self.stats['total_processed'] += 1
        
        # Clean text if enabled
        if self.quality_config.enable_cleaning:
            try:
                cleaned = clean_text(chunk)
            except Exception as e:
                logger.warning(f"Cleaning failed for chunk from {source}: {e}")
                self.stats['rejected_cleaning'] += 1
                return None
        else:
            cleaned = chunk
        
        # Validate text if enabled
        if self.quality_config.enable_validation:
            if not validate_text(cleaned, self.quality_config):
                self.stats['rejected_validation'] += 1
                return None
        
        return cleaned
    
    def preload_all(self, pdf_json_path: Optional[str] = None, 
                   batch_size: int = 100):
        """Load and embed all data sources"""
        if pdf_json_path and os.path.exists(pdf_json_path):
            logger.info(f"📥 Loading PDF chunks from: {pdf_json_path}")
            
            chunks_to_embed = []
            for chunk, filename in tqdm(self.stream_pdf_json_chunks(pdf_json_path), 
                                       desc="Processing PDFs"):
                processed = self.process_chunk(chunk, filename)
                if processed:
                    chunks_to_embed.append(processed)
                
                # Batch embed when we hit batch_size
                if len(chunks_to_embed) >= batch_size:
                    self._batch_embed(chunks_to_embed)
                    chunks_to_embed = []
            
            # Embed remaining chunks
            if chunks_to_embed:
                self._batch_embed(chunks_to_embed)
            
            self._print_stats()
        else:
            logger.warning(f"PDF JSON path not found: {pdf_json_path}")
    
    def _batch_embed(self, chunks: List[str]):
        """Embed a batch of chunks"""
        for chunk in chunks:
            self.embed_and_store(chunk)
    
    def build_faiss_index_and_save(self, index_path: str = "memory.index",
                                    texts_path: str = "memory_texts.npy"):
        """Build and save FAISS index"""
        if not self.memory_vectors:
            logger.error("No vectors to index!")
            return
        
        logger.info("🏗️ Building FAISS index...")
        dim = len(self.memory_vectors[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.memory_vectors).astype('float32'))
        
        faiss.write_index(self.index, index_path)
        logger.info(f"💾 FAISS index saved to '{index_path}'")
        
        np.save(texts_path, np.array(self.memory, dtype=object))
        logger.info(f"📚 Memory texts saved to '{texts_path}'")
    
    def _print_stats(self):
        """Print processing statistics"""
        logger.info("\n" + "="*60)
        logger.info("📊 PROCESSING STATISTICS")
        logger.info("="*60)
        logger.info(f"Total processed: {self.stats['total_processed']}")
        logger.info(f"Total embedded: {self.stats['total_embedded']}")
        logger.info(f"Rejected (validation): {self.stats['rejected_validation']}")
        logger.info(f"Rejected (cleaning): {self.stats['rejected_cleaning']}")
        
        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['total_embedded'] / self.stats['total_processed']) * 100
            logger.info(f"Success rate: {success_rate:.2f}%")
        logger.info("="*60 + "\n")


# --- MAIN ---
if __name__ == "__main__":
    logger.info("🚢 Starting enhanced memory embedding process...")
    
    # Configure data quality
    quality_config = DataQualityConfig(
        min_text_length=50,
        max_text_length=10000,
        min_words=5,
        punctuation_ratio_threshold=0.4,
        enable_cleaning=True,
        enable_validation=True
    )
    
    # Initialize embedder
    embedder = BatchEmbedder(
        model_name='all-MiniLM-L6-v2',
        quality_config=quality_config
    )
    
    # Process PDFs
    embedder.preload_all(pdf_json_path="pdf_texts.json")
    
    # Build and save index
    embedder.build_faiss_index_and_save()
    
    logger.info("✅ Embedding process complete!")

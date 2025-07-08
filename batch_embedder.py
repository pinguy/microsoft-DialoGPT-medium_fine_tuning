import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging
from typing import List, Generator, Optional, Dict, Any
from dataclasses import dataclass, field
import hashlib
import pickle
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
import time
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for the embedding process"""
    model_name: str = 'all-MiniLM-L6-v2'
    batch_size: int = 32
    chunk_size: int = 512
    max_chunk_overlap: int = 50
    min_text_length: int = 20
    max_text_length: int = 2000
    index_type: str = 'flat'
    use_gpu: bool = False
    save_incremental: bool = True
    deduplication: bool = True
    min_sentence_length: int = 5
    max_non_alpha_ratio: float = 0.5
    filter_common_patterns: bool = True
    num_cpu_threads: Optional[int] = None  # None = auto-detect
    enable_parallel_processing: bool = True
    parallel_workers: Optional[int] = None  # None = auto-detect

class ImprovedBatchEmbedder:
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config: EmbeddingConfig = config or EmbeddingConfig()
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[Any] = None
        self.memory_texts: List[str] = []
        self.memory_vectors: List[np.ndarray] = []
        self.memory_metadata: List[Dict[str, Any]] = []
        self.text_hashes: set = set()
        self.total_embedded: int = 0
        self.device: str = "cpu"
        
        self._setup_cpu_threading()
        self._load_model()
        self._setup_device()
    
    def _setup_cpu_threading(self) -> None:
        """Configure CPU threading for optimal performance"""
        if not self.config.use_gpu:
            if self.config.num_cpu_threads is None:
                num_threads = os.cpu_count()
                logger.info(f"Auto-detected {num_threads} CPU cores")
            else:
                num_threads = self.config.num_cpu_threads
                
            torch.set_num_threads(num_threads)
            os.environ['OMP_NUM_THREADS'] = str(num_threads)
            os.environ['MKL_NUM_THREADS'] = str(num_threads)
            os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
            
            logger.info(f"🔧 Configured CPU threading: {num_threads} threads")
    
    def _load_model(self) -> None:
        """Load the sentence transformer model"""
        try:
            logger.info(f"📥 Loading model: {self.config.model_name}")
            self.model = SentenceTransformer(self.config.model_name)
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {e}")
            raise e
    
    def _setup_device(self) -> None:
        """Setup device (CPU/GPU) for the model"""
        if self.config.use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            if self.model:
                self.model = self.model.to(self.device)
            logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.info("🔧 Using CPU for embeddings")
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts"""
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed batch: {e}")
            return np.array([])
    
    def _embed_batch_parallel(self, texts: List[str]) -> np.ndarray:
        """Embed batches with parallel processing"""
        if not texts or not self.config.enable_parallel_processing:
            return self._embed_batch(texts)
        
        if self.config.parallel_workers is None:
            num_workers = min(os.cpu_count(), 4)
        else:
            num_workers = self.config.parallel_workers
            
        if len(texts) < num_workers * 2:
            return self._embed_batch(texts)
        
        chunk_size = max(1, len(texts) // num_workers)
        text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        embeddings = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._embed_batch, chunk) for chunk in text_chunks]
            
            for future in tqdm(futures, desc="Parallel Embedding Chunks"):
                try:
                    chunk_embeddings = future.result()
                    if len(chunk_embeddings) > 0:
                        embeddings.extend(chunk_embeddings)
                except Exception as e:
                    logger.error(f"Parallel embedding chunk failed: {e}")
        
        return np.array(embeddings) if embeddings else np.array([])
    
    def _clean_and_validate_text(self, text: str) -> str:
        """Clean and validate text before processing"""
        if not isinstance(text, str):
            return ""
        
        text = text.strip()
        
        if len(text) < self.config.min_text_length:
            return ""
        
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]
        
        if len(text.split()) < self.config.min_sentence_length:
            return ""
            
        alpha_count = sum(1 for c in text if c.isalpha())
        if len(text) > 0 and alpha_count / len(text) < (1 - self.config.max_non_alpha_ratio):
            return ""
        
        if self.config.filter_common_patterns:
            if re.match(r'^[\s\d\W]*$', text):
                return ""
        
        return text
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        if not text:
            return []
        
        chunks = []
        words = text.split()
        
        if len(words) <= self.config.chunk_size:
            return [{
                'text': text,
                'metadata': {**metadata, 'chunk_id': 0, 'total_chunks': 1}
            }]
        
        for i in range(0, len(words), max(1, self.config.chunk_size - self.config.max_chunk_overlap)):
            chunk_words = words[i:i + self.config.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text.strip()) < self.config.min_text_length:
                continue
            
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_id': len(chunks),
                    'total_chunks': -1
                }
            })
        
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        return chunks
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text deduplication"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def add_texts_batch(self, text_data: List[Dict[str, Any]]) -> int:
        """Enhanced version with parallel processing option"""
        if not text_data:
            return 0
        
        processed_items = []
        for item in text_data:
            original_text = item.get('text', '')
            metadata = item.get('metadata', {})

            cleaned_text = self._clean_and_validate_text(original_text)
            
            if cleaned_text:
                chunks = self._chunk_text(cleaned_text, metadata)
                processed_items.extend(chunks)
            else:
                logger.debug(f"Skipping text due to cleaning/validation: '{original_text[:50]}...'")
        
        if not processed_items:
            return 0
        
        texts_to_embed = [item['text'] for item in processed_items]
        metadata_for_embed = [item['metadata'] for item in processed_items]

        if self.config.deduplication:
            unique_texts = []
            unique_metadata = []
            
            for text, meta in zip(texts_to_embed, metadata_for_embed):
                text_hash = self._get_text_hash(text)
                if text_hash not in self.text_hashes:
                    self.text_hashes.add(text_hash)
                    unique_texts.append(text)
                    unique_metadata.append(meta)
            
            texts_to_embed = unique_texts
            metadata_for_embed = unique_metadata

        if not texts_to_embed:
            return 0

        if self.config.enable_parallel_processing and not self.config.use_gpu:
            embeddings = self._embed_batch_parallel(texts_to_embed)
        else:
            embeddings = self._embed_batch(texts_to_embed)
        
        for i, emb in enumerate(embeddings):
            if emb is not None and not np.all(emb == 0):
                self.memory_texts.append(texts_to_embed[i])
                self.memory_vectors.append(emb)
                self.memory_metadata.append(metadata_for_embed[i])
        
        added_count = len(embeddings)
        self.total_embedded += added_count
        
        return added_count
    
    def stream_conversations(self, path: str, strict_mode: bool = False) -> Generator[Dict[str, Any], None, None]:
        """
        Enhanced conversation streaming with better Claude chat format support.
        Handles various conversation formats including Claude chat exports.
        """
        if not os.path.exists(path):
            logger.warning(f"Conversations file not found: {path}")
            return
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            logger.debug(f"Conversations data type: {type(data)}")
            
            # Handle different conversation formats
            conversations = []
            
            if isinstance(data, list):
                # Direct list of conversations or messages
                conversations = data
            elif isinstance(data, dict):
                # Check for various dictionary formats
                if "conversations" in data:
                    # Format: {"conversations": [...]}
                    conversations = data["conversations"]
                elif "mapping" in data:
                    # Single ChatGPT conversation
                    conversations = [data]
                elif "messages" in data:
                    # Single conversation with messages array
                    conversations = [data]
                elif all(isinstance(v, dict) for v in data.values()):
                    # Dictionary of conversations (keyed by ID)
                    conversations = list(data.values())
                elif all(isinstance(v, list) for v in data.values()):
                    # Dictionary where values are lists of messages
                    for conv_id, messages in data.items():
                        conversations.append({"messages": messages, "id": conv_id})
                else:
                    # Try to treat as single conversation
                    conversations = [data]
            
            for convo_idx, convo in enumerate(conversations):
                if not isinstance(convo, dict):
                    if strict_mode:
                        raise ValueError(f"Conversation entry at index {convo_idx} is not a dictionary.")
                    logger.debug(f"Skipping non-dict conversation entry at index {convo_idx}")
                    continue
                
                # Handle ChatGPT mapping format
                if "mapping" in convo:
                    self._process_mapping_format(convo, convo_idx, strict_mode)
                
                # Handle messages array format
                elif "messages" in convo:
                    messages = convo["messages"]
                    if isinstance(messages, list):
                        for msg_idx, msg in enumerate(messages):
                            try:
                                if isinstance(msg, dict):
                                    # Handle different message structures
                                    content = self._extract_message_content(msg)
                                    if content:
                                        author = self._extract_author(msg)
                                        timestamp = msg.get("timestamp") or msg.get("created_at") or msg.get("time")
                                        
                                        yield {
                                            'text': content,
                                            'metadata': {
                                                'source': 'conversation',
                                                'conversation_id': convo_idx,
                                                'message_id': msg_idx,
                                                'author': author,
                                                'timestamp': timestamp
                                            }
                                        }
                                elif strict_mode:
                                    raise ValueError(f"Message at index {msg_idx} is not a dictionary.")
                            except Exception as e:
                                if strict_mode:
                                    raise ValueError(f"Error processing message at index {msg_idx}: {e}") from e
                                logger.debug(f"Skipping message at index {msg_idx}: {e}")
                    else:
                        if strict_mode:
                            raise ValueError(f"'messages' in conversation {convo_idx} is not a list.")
                        logger.debug(f"'messages' in conversation {convo_idx} is not a list, skipping")
                
                # Handle Claude chat format (array of turn objects)
                elif isinstance(convo, list):
                    for turn_idx, turn in enumerate(convo):
                        if isinstance(turn, dict):
                            content = self._extract_message_content(turn)
                            if content:
                                author = self._extract_author(turn)
                                timestamp = turn.get("timestamp") or turn.get("created_at")
                                
                                yield {
                                    'text': content,
                                    'metadata': {
                                        'source': 'conversation',
                                        'conversation_id': convo_idx,
                                        'message_id': turn_idx,
                                        'author': author,
                                        'timestamp': timestamp
                                    }
                                }
                
                # Handle direct message format (single message object)
                else:
                    content = self._extract_message_content(convo)
                    if content:
                        author = self._extract_author(convo)
                        timestamp = convo.get("timestamp") or convo.get("created_at")
                        
                        yield {
                            'text': content,
                            'metadata': {
                                'source': 'conversation',
                                'conversation_id': convo_idx,
                                'message_id': 0,
                                'author': author,
                                'timestamp': timestamp
                            }
                        }
                            
        except json.JSONDecodeError as e:
            if strict_mode:
                raise ValueError(f"Invalid JSON in conversations file {path}: {e}") from e
            logger.error(f"Invalid JSON in conversations file {path}: {e}")
        except Exception as e:
            if strict_mode:
                raise ValueError(f"Error reading conversations file {path}: {e}") from e
            logger.error(f"Error reading conversations file {path}: {e}")
    
    def _process_mapping_format(self, convo: dict, convo_idx: int, strict_mode: bool) -> Generator[Dict[str, Any], None, None]:
        """Process ChatGPT mapping format"""
        mapping = convo["mapping"]
        for msg_id, msg_data in mapping.items():
            try:
                if not isinstance(msg_data, dict):
                    if strict_mode:
                        raise ValueError(f"Message data for ID {msg_id} is not a dictionary.")
                    continue
                    
                message = msg_data.get("message")
                if not message or not isinstance(message, dict):
                    if strict_mode:
                        raise ValueError(f"Message for ID {msg_id} is missing or not a dictionary.")
                    continue
                    
                content = self._extract_message_content(message)
                if content:
                    author = self._extract_author(message)
                    timestamp = message.get("create_time") or message.get("timestamp")
                    
                    yield {
                        'text': content,
                        'metadata': {
                            'source': 'conversation',
                            'conversation_id': convo_idx,
                            'message_id': msg_id,
                            'author': author,
                            'timestamp': timestamp
                        }
                    }
                    
            except Exception as e:
                if strict_mode:
                    raise ValueError(f"Error processing message ID {msg_id}: {e}") from e
                logger.debug(f"Skipping malformed message {msg_id}: {e}")
    
    def _extract_message_content(self, message: dict) -> str:
        """Extract content from various message formats"""
        if not isinstance(message, dict):
            return ""
        
        content = message.get("content") or message.get("text") or message.get("message")
        
        if isinstance(content, str):
            return content.strip()
        elif isinstance(content, dict):
            # Handle nested content structures
            if "parts" in content and isinstance(content["parts"], list):
                parts = content["parts"]
                if parts and isinstance(parts[0], str):
                    return parts[0].strip()
            elif "text" in content:
                return str(content["text"]).strip()
        elif isinstance(content, list):
            # Handle content as array
            if content and isinstance(content[0], str):
                return content[0].strip()
            elif content and isinstance(content[0], dict):
                # Handle array of content objects
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if text and isinstance(text, str):
                            return text.strip()
        
        return ""
    
    def _extract_author(self, message: dict) -> str:
        """Extract author/role from various message formats"""
        if not isinstance(message, dict):
            return "unknown"
        
        # Try different author field names
        author_info = message.get("author") or message.get("role") or message.get("sender")
        
        if isinstance(author_info, dict):
            role = author_info.get("role") or author_info.get("name") or "unknown"
        elif isinstance(author_info, str):
            role = author_info
        else:
            role = "unknown"
        
        # Normalize common role names
        role = role.lower()
        if role in ["human", "user"]:
            return "user"
        elif role in ["assistant", "ai", "claude"]:
            return "assistant"
        else:
            return role
    
    def stream_pdf_chunks(self, json_path: str, strict_mode: bool = False) -> Generator[Dict[str, Any], None, None]:
        """Stream PDF chunks from a JSON file"""
        if not os.path.exists(json_path):
            logger.warning(f"PDF JSON file not found: {json_path}")
            return
            
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            
            if not isinstance(entries, list):
                if strict_mode:
                    raise ValueError(f"Expected list of PDF entries, got {type(entries)}")
                logger.error(f"Expected list of PDF entries, got {type(entries)}")
                return
            
            for doc_idx, doc in enumerate(entries):
                if not isinstance(doc, dict):
                    if strict_mode:
                        raise ValueError(f"PDF entry at index {doc_idx} is not a dictionary.")
                    continue
                
                text = ""
                for text_field in ["text", "content", "total_text", "body"]:
                    if text_field in doc and isinstance(doc[text_field], str):
                        text = doc[text_field].strip()
                        break
                
                if not text:
                    if strict_mode:
                        raise ValueError(f"Document {doc_idx} has no valid text field.")
                    continue
                
                filename = doc.get("filename", f"document_{doc_idx}.pdf")
                
                source_info = {
                    'source': 'pdf',
                    'filename': filename,
                    'document_id': doc_idx
                }
                
                chunks = self._chunk_text(text, source_info)
                for chunk in chunks:
                    yield chunk
                    
        except json.JSONDecodeError as e:
            if strict_mode:
                raise ValueError(f"Invalid JSON in PDF file {json_path}: {e}") from e
            logger.error(f"Invalid JSON in PDF file {json_path}: {e}")
        except Exception as e:
            if strict_mode:
                raise ValueError(f"Error reading PDF file {json_path}: {e}") from e
            logger.error(f"Error reading PDF file {json_path}: {e}")
    
    def load_and_embed_all(self, 
                          convo_path: Optional[str] = None,
                          convo_path2: Optional[str] = None,
                          pdf_json_path: Optional[str] = None,
                          custom_data: Optional[List[Dict[str, Any]]] = None,
                          strict_mode: bool = False) -> None:
        """Load and embed all data sources"""
        
        logger.info("🟢 Starting comprehensive embedding process...")
        start_time = time.time()
        
        # Estimate total items for progress bar
        total_items_to_process = 0
        
        def estimate_convo_items(path):
            count = 0
            if path and os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        count = len(data)
                    elif isinstance(data, dict):
                        if "conversations" in data:
                            count = len(data["conversations"])
                        elif "mapping" in data:
                            count = len(data["mapping"])
                        elif "messages" in data:
                            count = len(data["messages"])
                        else:
                            count = sum(len(v) if isinstance(v, (list, dict)) else 1 for v in data.values())
                except Exception as e:
                    logger.warning(f"Could not estimate count for {path}: {e}")
            return count

        total_items_to_process += estimate_convo_items(convo_path)
        total_items_to_process += estimate_convo_items(convo_path2)

        if pdf_json_path and os.path.exists(pdf_json_path):
            try:
                with open(pdf_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    total_items_to_process += len(data)
            except Exception as e:
                logger.warning(f"Could not estimate count for PDF file: {e}")

        if custom_data:
            total_items_to_process += len(custom_data)

        pbar = tqdm(total=max(total_items_to_process, 1), desc="Total Embedding Progress", unit="items")
        
        def process_convo_file(path, pbar_instance):
            if path:
                logger.info(f"📥 Processing conversations from: {path}")
                batch = []
                try:
                    for msg_data in self.stream_conversations(path, strict_mode=strict_mode):
                        batch.append(msg_data)
                        if len(batch) >= self.config.batch_size:
                            added = self.add_texts_batch(batch)
                            pbar_instance.update(len(batch))
                            batch = []
                            if self.config.save_incremental and self.total_embedded % 1000 == 0:
                                self._save_checkpoint()
                    if batch:
                        added = self.add_texts_batch(batch)
                        pbar_instance.update(len(batch))
                    logger.info(f"✅ Finished processing conversations from {path}")
                except Exception as e:
                    logger.error(f"Error processing conversations from {path}: {e}")
                    if strict_mode:
                        raise e

        process_convo_file(convo_path, pbar)
        process_convo_file(convo_path2, pbar)
        
        if pdf_json_path:
            logger.info(f"📚 Processing PDF chunks from: {pdf_json_path}")
            batch = []
            
            try:
                for chunk_data in self.stream_pdf_chunks(pdf_json_path, strict_mode=strict_mode):
                    batch.append(chunk_data)
                    
                    if len(batch) >= self.config.batch_size:
                        added = self.add_texts_batch(batch)
                        pbar.update(len(batch))
                        batch = []
                        if self.config.save_incremental and self.total_embedded % 1000 == 0:
                            self._save_checkpoint()
                
                if batch:
                    added = self.add_texts_batch(batch)
                    pbar.update(len(batch))
                
                logger.info(f"✅ Finished processing PDF chunks")
                
            except Exception as e:
                logger.error(f"Error processing PDFs: {e}")
                if strict_mode:
                    raise e
        
        if custom_data:
            logger.info(f"🔧 Processing custom data: {len(custom_data)} items")
            try:
                for i in range(0, len(custom_data), self.config.batch_size):
                    batch = custom_data[i:i + self.config.batch_size]
                    added = self.add_texts_batch(batch)
                    pbar.update(len(batch))
                    if self.config.save_incremental and self.total_embedded % 1000 == 0:
                        self._save_checkpoint()
            except Exception as e:
                logger.error(f"Error processing custom data: {e}")
                if strict_mode:
                    raise e
        
        pbar.close()
        
        if self.total_embedded > 0:
            self.build_and_save_index()
        else:
            logger.warning("No data was embedded. Check your input files and structure.")
        
        elapsed = time.time() - start_time
        logger.info(f"🎉 Embedding complete! Total: {self.total_embedded} items in {elapsed:.2f}s")
        if elapsed > 0:
            logger.info(f"⚡ Rate: {self.total_embedded/elapsed:.2f} items/second")
    
    def _save_checkpoint(self) -> None:
        """Save incremental checkpoint"""
        checkpoint_path = "embedding_checkpoint.pkl"
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'memory_texts': self.memory_texts,
                    'memory_vectors': self.memory_vectors,
                    'memory_metadata': self.memory_metadata,
                    'text_hashes': self.text_hashes,
                    'total_embedded': self.total_embedded
                }, f)
            logger.debug(f"💾 Checkpoint saved: {self.total_embedded} items")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str = "embedding_checkpoint.pkl") -> bool:
        """Load from checkpoint"""
        if not os.path.exists(checkpoint_path):
            return False
        
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            
            self.memory_texts = data['memory_texts']
            self.memory_vectors = data['memory_vectors']
            self.memory_metadata = data['memory_metadata']
            self.text_hashes = data.get('text_hashes', set())
            self.total_embedded = data['total_embedded']
            
            logger.info(f"📥 Loaded checkpoint: {self.total_embedded} items")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def build_and_save_index(self, output_dir: str = ".") -> None:
        """Build and save FAISS index"""
        if not self.memory_vectors:
            logger.warning("No vectors to index!")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"🔨 Building FAISS index with {len(self.memory_vectors)} vectors...")
        
        vectors = np.array(self.memory_vectors).astype('float32')
        dimension = vectors.shape[1]
        
        if self.config.index_type == 'flat' or len(vectors) < 1000:
            self.index = faiss.IndexFlatL2(dimension)
        elif self.config.index_type == 'ivf':
            nlist = min(int(np.sqrt(len(vectors))), 100)
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
            self.index.train(vectors)
        elif self.config.index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_L2)
            self.index.hnsw.efConstruction = 40
        else:
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(vectors)
        
        # Save all components
        index_path = output_path / "memory.index"
        faiss.write_index(self.index, str(index_path))
        logger.info(f"💾 FAISS index saved to: {index_path}")
        
        texts_path = output_path / "memory_texts.npy"
        np.save(texts_path, np.array(self.memory_texts, dtype=object))
        logger.info(f"📚 Memory texts saved to: {texts_path}")
        
        metadata_path = output_path / "memory_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.memory_metadata, f)
        logger.info(f"🏷️ Metadata saved to: {metadata_path}")
        
        config_path = output_path / "embedding_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        logger.info(f"⚙️ Configuration saved to: {config_path}")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search the index for similar texts"""
        if not self.index:
            logger.error("Index not built! Call build_and_save_index() first or load a checkpoint.")
            return []
        
        query_vector = self.model.encode([query], normalize_embeddings=True)
        
        if self.config.index_type == 'hnsw' and hasattr(self.index.hnsw, 'efSearch'):
            self.index.hnsw.efSearch = max(k, self.index.hnsw.efConstruction)
        
        scores, indices = self.index.search(query_vector.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1 and idx < len(self.memory_texts):
                results.append({
                    'text': self.memory_texts[idx],
                    'metadata': self.memory_metadata[idx],
                    'similarity_score': float(score),
                    'rank': i + 1
                })
        
        return results

    def diagnose_json_files(self, convo_path: Optional[str] = None, convo_path2: Optional[str] = None, pdf_path: Optional[str] = None) -> None:
        """Diagnose JSON file structures for debugging"""
        logger.info("🔍 Diagnosing JSON file structures...")
        
        def _diagnose_single_convo_file(path):
            if not os.path.exists(path):
                logger.warning(f"Conversations file not found: {path}")
                return
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                logger.info(f"📄 Conversations file ({path}):")
                logger.info(f"  - Type: {type(data)}")
                logger.info(f"  - Length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
                
                if isinstance(data, list) and len(data) > 0:
                    first_item = data[0]
                    logger.info(f"  - First item type: {type(first_item)}")
                    if isinstance(first_item, dict):
                        logger.info(f"  - First item keys: {list(first_item.keys())}")
                        if 'mapping' in first_item:
                            logger.info("  - Appears to be ChatGPT 'mapping' format.")
                            if first_item['mapping']:
                                first_map_id = list(first_item['mapping'].keys())[0]
                                first_map_msg = first_item['mapping'][first_map_id]
                                logger.info(f"    - First mapping message keys: {list(first_map_msg.keys())}")
                                if 'message' in first_map_msg and isinstance(first_map_msg['message'], dict):
                                    author_role = first_map_msg['message'].get('author', {}).get('role', 'N/A')
                                    logger.info(f"      - 'message' content keys: {list(first_map_msg['message'].get('content', {}).keys())}")
                                    logger.info(f"      - Example author role: '{author_role}'")
                        elif 'messages' in first_item and isinstance(first_item['messages'], list):
                            logger.info("  - Appears to be 'messages' array format.")
                            if first_item['messages']:
                                logger.info(f"    - First message in 'messages' keys: {list(first_item['messages'][0].keys())}")
                                author_role = first_item['messages'][0].get('role', 'N/A')
                                logger.info(f"    - Example author role: '{author_role}'")
                
                elif isinstance(data, dict):
                    logger.info(f"  - Root keys: {list(data.keys())}")
                    if 'mapping' in data:
                        logger.info("  - Appears to be a single ChatGPT 'mapping' conversation at root.")
                        if data['mapping']:
                            first_map_id = list(data['mapping'].keys())[0]
                            first_map_msg = data['mapping'][first_map_id]
                            logger.info(f"    - First mapping message keys: {list(first_map_msg.keys())}")
                            if 'message' in first_map_msg and isinstance(first_map_msg['message'], dict):
                                author_role = first_map_msg['message'].get('author', {}).get('role', 'N/A')
                                logger.info(f"      - 'message' content keys: {list(first_map_msg['message'].get('content', {}).keys())}")
                                logger.info(f"      - Example author role: '{author_role}'")
            except Exception as e:
                logger.error(f"Error diagnosing conversations file {path}: {e}")

        _diagnose_single_convo_file(convo_path)
        _diagnose_single_convo_file(convo_path2) # Diagnose the new conversation file
        
        if pdf_path:
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF file not found: {pdf_path}")
            else:
                try:
                    with open(pdf_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    logger.info(f"📚 PDF file ({pdf_path}):")
                    logger.info(f"  - Type: {type(data)}")
                    logger.info(f"  - Length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
                    
                    if isinstance(data, list) and len(data) > 0:
                        first_item = data[0]
                        logger.info(f"  - First item type: {type(first_item)}")
                        if isinstance(first_item, dict):
                            logger.info(f"  - First item keys: {list(first_item.keys())}")
                            found_text_field = False
                            for text_field in ["text", "content", "total_text", "body"]:
                                if text_field in first_item:
                                    logger.info(f"  - Contains '{text_field}' field.")
                                    if isinstance(first_item[text_field], str):
                                        logger.info(f"    - '{text_field}' type: string, length: {len(first_item[text_field])}")
                                    else:
                                        logger.info(f"    - '{text_field}' type: {type(first_item[text_field])}")
                                    found_text_field = True
                            if not found_text_field:
                                logger.warning("  - No common text fields ('text', 'content', 'total_text', 'body') found in first item.")
                                
                except Exception as e:
                    logger.error(f"Error diagnosing PDF file {pdf_path}: {e}")

def main():
    """Main execution function with CPU/GPU optimization and progress indication"""
    config = EmbeddingConfig(
        batch_size=64,
        chunk_size=400,
        max_chunk_overlap=50,
        min_text_length=30,
        use_gpu=False,  # Set to True if you have a compatible GPU (CUDA)
        index_type='flat',
        deduplication=True,
        min_sentence_length=10,
        max_non_alpha_ratio=0.4,
        filter_common_patterns=True,
        num_cpu_threads=None,
        enable_parallel_processing=True,
        parallel_workers=None
    )
    
    embedder = ImprovedBatchEmbedder(config)
    
    # Diagnose both conversation files
    embedder.diagnose_json_files("conversations.json", "conversations2.json", "pdf_texts.json")
    
    if embedder.load_checkpoint():
        logger.info("Resuming from checkpoint...")
    
    embedder.load_and_embed_all(
        convo_path="conversations.json",
        convo_path2="conversations2.json", # Pass the new conversation file path
        pdf_json_path="pdf_texts.json",
        strict_mode=False
    )
    
    if embedder.total_embedded > 0:
        logger.info("\n🔍 Testing search functionality:")
        query_texts = [
            "What are the main findings of the study?",
            "Can you explain the concept of consciousness?",
            "Tell me about the project timeline and key milestones."
        ]
        for query in query_texts:
            print(f"\nQuery: '{query}'")
            results = embedder.search(query, k=3)
            if results:
                for result in results:
                    print(f"  Rank {result['rank']}: Score={result['similarity_score']:.4f}")
                    print(f"    Text: {result['text'][:150]}...")
                    print(f"    Metadata: {result['metadata']}")
            else:
                print("  No results found.")
    else:
        logger.warning("No data was embedded. Check your input files and their structure, or try running with strict_mode=True in load_and_embed_all for detailed errors.")

if __name__ == "__main__":
    main()


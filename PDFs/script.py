import os
import json
import re
import random
import logging
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Tuple, Any

# --- New Imports from your snippet ---
import ftfy
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial, lru_cache
import gc
import psutil
import uuid
import numpy as np

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Configuration ---
class Config:
    """A simple class to hold configuration parameters."""
    # Increased for more completeness
    max_pairs_per_source = 5000
    # Lowered slightly to retain more pairs
    quality_score_threshold = 0.25
    # The ratio for splitting the data into training, validation, and test sets.
    split_ratio = (0.8, 0.1, 0.1)
    # Input file paths from the user's upload.
    input_npy_path = 'memory_texts.npy'
    
    # --- New configuration options for embeddings ---
    use_semantic_filtering = True 
    model_name = 'all-MiniLM-L6-v2'
    embedding_dim = 384
    # Raised to merge less aggressively, reducing huge chunks/repeats
    sim_threshold = 0.8
    # New: Max length for merged groups to avoid truncation issues
    max_merged_length = 5000  # chars
    # New: Similarity threshold for pair diversity (skip too-similar pairs)
    diversity_sim_threshold = 0.9


# --- Main Data Processing Class ---
class DatasetProcessor:
    """
    A class to handle the entire process of loading data, generating Q&A pairs,
    and creating data splits.
    """
    def __init__(self, config: Config):
        """Initializes the processor with a configuration object."""
        self.config = config
        self.model = None
        # Load the embedding model on initialization if required.
        self._load_embedding_model()

    def _load_embedding_model(self):
        """Loads the sentence transformer model if semantic filtering is enabled."""
        if self.config.use_semantic_filtering:
            try:
                logger.info(f"Loading sentence transformer model: {self.config.model_name}...")
                self.model = SentenceTransformer(self.config.model_name)
                logger.info("Model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer model. Error: {e}")
                self.model = None
                self.config.use_semantic_filtering = False

    def __get_single_embedding_uncached(self, text: str) -> np.ndarray:
        """Helper to compute a single embedding (uncached version)"""
        if not self.config.use_semantic_filtering or not self.model:
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
            return np.zeros(self.config.embedding_dim)

    @staticmethod
    def parse_text_chunk(raw_entry: str) -> dict:
        """
        Parses a raw string from the numpy array to separate metadata from text.
        It looks for metadata in the format [KEY:Value].
        """
        metadata = {}
        text = raw_entry
        
        match = re.match(r'\[PDF:(?P<filename>[^\]]+)\]\s*', raw_entry)
        if match:
            filename = match.group('filename')
            metadata['filename'] = filename
            text = raw_entry[match.end():]
            
        cleaned_text = text.strip()
        return {'cleaned_text': cleaned_text, 'metadata': metadata, 'domain': 'general'} if cleaned_text else None

    def generate_diverse_questions(self, chunk_text: str, metadata: dict, domain: str) -> list[str]:
        """
        Generates a few diverse questions based on a text chunk, with better linking to the passage.
        """
        questions = []

        # Extract first paragraph or sentence for excerpt
        paragraphs = re.split(r'\n\n+', chunk_text)
        first_para = paragraphs[0][:500].strip() if paragraphs else chunk_text[:500].strip()
        if first_para:
            questions.append(f"What is the main topic discussed in this passage: '{first_para}'?")
            questions.append(f"Provide a detailed summary of the following text: '{first_para}'.")

        # Find potential key terms: capitalized words or phrases
        key_terms = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b', chunk_text)
        unique_terms = list(set(key_terms))
        if unique_terms:
            term = random.choice(unique_terms)
            questions.append(f"What does the text say about {term}?")
            questions.append(f"Explain the significance of {term} in this context.")

        # Additional varied templates (expanded for diversity)
        questions.append(f"Based on this passage, what are the implications for {domain}: '{first_para}'?")
        questions.append("What key arguments are made in this text?")
        questions.append(f"How is the concept introduced in '{first_para[:200]}' further developed?")
        questions.append(f"What examples or evidence are provided in: '{first_para}'?")
        questions.append(f"Describe the key steps or process outlined in this excerpt: '{first_para}'.")
        questions.append("What questions does this passage raise about the topic?")

        # Shuffle and select up to 4 for variety (increased from 3)
        random.shuffle(questions)
        return questions[:4]

    def _assess_pair_quality_batch(self, questions: list[str], answers: list[str]) -> list[dict]:
        """
        Assesses the quality of a batch of Q&A pairs using semantic similarity if available.
        """
        if not self.config.use_semantic_filtering or not self.model:
            # Fallback to mock
            quality_metrics_batch = []
            for _ in questions:
                score = random.uniform(0.6, 1.0)
                quality_metrics_batch.append({
                    "quality_score": float(round(score, 3)),
                    "relevance": float(round(random.uniform(0.7, 1.0), 2)),
                    "clarity": float(round(random.uniform(0.7, 1.0), 2))
                })
            return quality_metrics_batch

        # Compute embeddings
        q_embeds = self.model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
        a_embeds = self.model.encode(answers, convert_to_numpy=True, normalize_embeddings=True)
        sims = cosine_similarity(q_embeds, a_embeds).diagonal()

        quality_metrics_batch = []
        for sim in sims:
            quality_metrics_batch.append({
                "quality_score": float(round(sim, 3)),
                "relevance": float(round(sim, 2)),
                "clarity": float(round(random.uniform(0.7, 1.0), 2))  # Mock clarity
            })
        return quality_metrics_batch

    def group_similar_chunks(self, entries: List[dict], sim_threshold: float = 0.75) -> List[dict]:
        """
        Groups consecutive similar chunks using cosine similarity to handle multi-paragraph ideas.
        Now with max length to avoid huge merged chunks.
        """
        if not entries or not self.config.use_semantic_filtering or not self.model:
            return entries

        texts = [e['cleaned_text'] for e in entries if e.get('cleaned_text')]
        if not texts:
            return []

        embeds = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        grouped = []
        current_texts = [texts[0]]
        current_sum = embeds[0].copy()
        current_embed = embeds[0]
        current_group = entries[0].copy()

        for i in range(1, len(entries)):
            sim = cosine_similarity([current_embed], [embeds[i]])[0][0]
            new_text = ' '.join(current_texts) + ' ' + texts[i]
            if sim >= sim_threshold and len(new_text) <= self.config.max_merged_length:
                # Add to current group if under max length
                current_texts.append(texts[i])
                current_sum += embeds[i]
                current_embed = current_sum / np.linalg.norm(current_sum)
            else:
                # Save current group
                current_group['cleaned_text'] = ' '.join(current_texts)
                grouped.append(current_group)
                # Start new group
                current_texts = [texts[i]]
                current_sum = embeds[i].copy()
                current_embed = embeds[i]
                current_group = entries[i].copy()

        # Add the last group
        if current_texts:
            current_group['cleaned_text'] = ' '.join(current_texts)
            grouped.append(current_group)

        return grouped

    def create_qa_pairs(self, text_entries: list[dict]) -> list[dict]:
        """
        Optimized Q&A pair creation with batch processing, grouping, and improved prompts.
        """
        qa_pairs = []
        existing_answer_embeds = []  # For diversity check
        logger.info(f"Generating Q&A pairs from {len(text_entries)} text chunks...")

        by_source = defaultdict(list)
        for entry in text_entries:
            source = entry['metadata'].get('filename', 'unknown_source')
            by_source[source].append(entry)

        max_groups = self.config.max_pairs_per_source // 4  # Adjusted for up to 4 questions

        for source, entries in tqdm(by_source.items(), desc="Processing sources"):
            if not entries:
                continue

            # Group similar consecutive chunks
            grouped_entries = self.group_similar_chunks(entries, self.config.sim_threshold)

            # Process all groups (no random sample; cap if too many)
            selected_groups = grouped_entries[:max_groups]

            batch_questions, batch_answers, batch_metadata = [], [], []
            for entry in selected_groups:
                if not entry.get('cleaned_text') or len(entry['cleaned_text']) < 50:  # Skip short/empty
                    continue
                questions = self.generate_diverse_questions(
                    entry['cleaned_text'], entry.get('metadata', {}), entry.get('domain', 'general')
                )
                thread_id = str(uuid.uuid4())  # Assign thread_id per group for longform simulation
                answer_embed = self.__get_single_embedding_uncached(entry['cleaned_text'])
                # Diversity check: Skip if too similar to existing
                if any(cosine_similarity([answer_embed], [e])[0][0] > self.config.diversity_sim_threshold for e in existing_answer_embeds):
                    continue
                existing_answer_embeds.append(answer_embed)
                for question in questions:
                    batch_questions.append(question)
                    batch_answers.append(entry['cleaned_text'])
                    batch_metadata.append({**entry.get('metadata', {}), 'domain': entry.get('domain', 'general'), 'source_file': source, 'thread_id': thread_id})

            if batch_questions:
                quality_metrics_batch = self._assess_pair_quality_batch(batch_questions, batch_answers)
                for q, a, meta, quality in zip(batch_questions, batch_answers, batch_metadata, quality_metrics_batch):
                    if quality["quality_score"] >= self.config.quality_score_threshold:
                        qa_pairs.append({'user': q, 'assistant': a, 'quality_metrics': quality, 'source_metadata': meta})

        logger.info(f"Created {len(qa_pairs)} high-quality Q&A pairs.")
        return qa_pairs

    def deduplicate_entries(self, entries: List[dict]) -> List[dict]:
        """New: Deduplicate based on hash of cleaned_text to avoid repeats."""
        seen_hashes = set()
        deduped = []
        for entry in entries:
            if not entry:
                continue
            text_hash = hashlib.sha256(entry['cleaned_text'].encode()).hexdigest()
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                deduped.append(entry)
        logger.info(f"Deduplicated: {len(entries)} -> {len(deduped)} entries")
        return deduped

    def create_data_splits(self, all_pairs: list[dict]) -> dict[str, list[dict]]:
        """
        Creates stratified train, validation, and test splits based on quality.
        """
        logger.info(f"Splitting {len(all_pairs)} pairs into train/val/test...")
        if not all_pairs:
            return {"train": [], "validation": [], "test": []}

        all_pairs.sort(key=lambda x: x['quality_metrics']['quality_score'], reverse=True)
        
        num_quartiles = 4
        total_len = len(all_pairs)
        for i in range(num_quartiles):
            start = i * (total_len // num_quartiles)
            end = (i + 1) * (total_len // num_quartiles) if i < num_quartiles - 1 else total_len
            quartile = all_pairs[start:end]
            random.shuffle(quartile)
            all_pairs[start:end] = quartile

        train_end = int(total_len * self.config.split_ratio[0])
        val_end = train_end + int(total_len * self.config.split_ratio[1])
        splits = {"train": all_pairs[:train_end], "validation": all_pairs[train_end:val_end], "test": all_pairs[val_end:]}

        logger.info(f"Splits created: Train={len(splits['train'])}, Validation={len(splits['validation'])}, Test={len(splits['test'])}")
        for name, data in splits.items():
            if data:
                avg_quality = np.mean([p['quality_metrics']['quality_score'] for p in data])
                logger.info(f"{name.capitalize()} average quality: {avg_quality:.3f}")

        return splits

    @staticmethod
    def save_to_jsonl(data: list[dict], filename: str):
        """Saves a list of dictionaries to a JSONL file."""
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Successfully saved {len(data)} records to {filename}")

    def run(self):
        """Executes the full data processing pipeline."""
        try:
            raw_text_array = np.load(self.config.input_npy_path, allow_pickle=True)
            logger.info(f"Loaded {len(raw_text_array)} text chunks from {self.config.input_npy_path}")
            
            parsed_entries = [self.parse_text_chunk(entry) for entry in raw_text_array]
            parsed_entries = [e for e in parsed_entries if e]  # Filter None
            deduped_entries = self.deduplicate_entries(parsed_entries)
            all_qa_pairs = self.create_qa_pairs(deduped_entries)
            data_splits = self.create_data_splits(all_qa_pairs)

            for split_name, data in data_splits.items():
                if data:
                    self.save_to_jsonl(data, f"{split_name}.jsonl")
                    
        except FileNotFoundError:
            logger.error(f"Error: Input file not found at '{self.config.input_npy_path}'.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)


# --- Main Execution ---
def main():
    """Main function to run the data processing pipeline."""
    config = Config()
    processor = DatasetProcessor(config)
    processor.run()

if __name__ == "__main__":
    main()

import torch
import re
import os
import warnings
import gradio as gr
import numpy as np
import tempfile
import soundfile as sf
import wave
import json
import subprocess
import threading
import time
import webbrowser
import uuid
import gc
import asyncio
import concurrent.futures
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import logging
import multiprocessing
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from contextlib import contextmanager
import psutil
from pathlib import Path

# Try to import optional dependencies
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
    print("✅ Kokoro TTS library found")
except ImportError:
    KOKORO_AVAILABLE = False
    print("⚠️ Kokoro TTS not available. Install with: pip install kokoro>=0.9.4 soundfile")

try:
    import vosk
    VOSK_AVAILABLE = True
    print("✅ Vosk speech recognition found")
except ImportError:
    VOSK_AVAILABLE = False
    print("⚠️ Vosk not available. Install with: pip install vosk")

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Configuration
@dataclass
class Config:
    base_dir: str = "./dialogpt-finetuned/"
    coherence_threshold: float = 0.4
    use_coherence: bool = True
    vosk_model_path: str = "vosk-model-en-us-0.42-gigaspeech"
    server_port: int = 7860
    auto_open_browser: bool = True
    max_cache_size: int = 100
    max_response_length: int = 150
    tts_max_length: int = 200
    memory_cleanup_threshold: float = 0.8  # Clean memory when usage exceeds 80%

config = Config()

# Global variables
DEVICE = torch.device("cpu")
DEVICE_INFO = "CPU (default)"
DEVICE_DETAILS = {}

class PerformanceMonitor:
    """Monitor system performance and optimize accordingly"""
    
    def __init__(self):
        self.response_times = [] # Stores (timestamp, duration, method)
        self.memory_usage = []
        self.gpu_usage = []
        self.start_time = time.time()
        
    def log_response_time(self, duration: float, method: str):
        """Log response time with method"""
        self.response_times.append((time.time(), duration, method))
        # Keep only last 100 entries
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        stats = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'uptime': time.time() - self.start_time
        }
        
        if torch.cuda.is_available():
            try:
                stats['gpu_memory_used'] = torch.cuda.memory_allocated() / 1024**3  # GB
                stats['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
                stats['gpu_utilization'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            except:
                pass
                
        return stats
    
    def should_cleanup_memory(self) -> bool:
        """Check if memory cleanup is needed"""
        return psutil.virtual_memory().percent > config.memory_cleanup_threshold * 100

class EnhancedCache:
    """Intelligent caching with better eviction and hit rate tracking"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.access_times = {} # Stores timestamp of last access for LRU
        self.hit_count = 0
        self.miss_count = 0
        self.max_size = max_size
        
    def _normalize_key(self, key: str) -> str:
        """Normalize cache key for better hit rates (case-insensitive, whitespace-normalized)"""
        return re.sub(r'\s+', ' ', key.lower().strip())
    
    def get(self, key: str) -> Optional[str]:
        """Retrieve a cached response and update its access time."""
        normalized_key = self._normalize_key(key)
        if normalized_key in self.cache:
            self.access_times[normalized_key] = time.time() # Update last access time
            self.hit_count += 1
            return self.cache[normalized_key]
        
        self.miss_count += 1
        return None
    
    def put(self, key: str, value: str):
        """Cache a new response, evicting the least recently used if cache is full."""
        normalized_key = self._normalize_key(key)
        
        if len(self.cache) >= self.max_size and normalized_key not in self.cache:
            # Remove least recently used item
            oldest_key = min(self.access_times.keys(), 
                           key=lambda k: self.access_times.get(k, 0))
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[normalized_key] = value
        self.access_times[normalized_key] = time.time() # Set access time for new/updated entry
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics including size, hit rate, hits, and misses."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': f"{hit_rate:.2f}%",
            'hits': self.hit_count,
            'misses': self.miss_count
        }
    
    def clear(self):
        """Clear cache and reset stats."""
        self.cache.clear()
        self.access_times.clear()
        self.hit_count = 0
        self.miss_count = 0

@contextmanager
def torch_inference_mode():
    """Context manager for optimized PyTorch inference, including autocasting for CUDA."""
    with torch.inference_mode():
        if DEVICE.type == 'cuda':
            with torch.cuda.amp.autocast(): # Automatic mixed precision for CUDA
                yield
        else:
            yield

def get_optimal_device_config() -> Tuple[torch.device, str, Dict]:
    """
    Intelligently detects and configures the optimal device (CPU/GPU) for PyTorch.
    Prioritizes modern GPUs, falls back to CPU if GPU is too old or encounters CUDA errors.
    """
    device = torch.device("cpu")
    device_info = "CPU (default)"
    details = {'cpu_cores': multiprocessing.cpu_count()}
    
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            compute_capability = f"{props.major}.{props.minor}"
            memory_gb = props.total_memory / (1024**3)
            
            details.update({
                'gpu_name': gpu_name,
                'compute_capability': compute_capability,
                'memory_gb': memory_gb,
                'multiprocessor_count': props.multi_processor_count
            })
            
            # Test CUDA functionality by performing a simple operation
            try:
                test_tensor = torch.randn(100, 100, device='cuda')
                _ = test_tensor @ test_tensor.T
                del test_tensor
                torch.cuda.empty_cache()
                
                # Use GPU if it's modern enough (compute capability >= 7.0 for Turing/Ampere/Ada)
                # or if it's Pascal (>=6.0) with sufficient memory (e.g., >= 4GB)
                if props.major >= 7 or (props.major >= 6 and memory_gb >= 4):
                    device = torch.device("cuda")
                    device_info = f"GPU: {gpu_name} ({compute_capability}, {memory_gb:.1f}GB)"
                    details['decision'] = "GPU selected - modern and sufficient memory"
                else:
                    device_info = f"CPU: {details['cpu_cores']} cores (GPU too old/limited for optimal use)"
                    details['decision'] = "CPU selected - GPU insufficient for optimal performance"
                    
            except Exception as e:
                # Fallback to CPU if CUDA test fails
                device_info = f"CPU: {details['cpu_cores']} cores (GPU CUDA failed: {str(e)})"
                details['decision'] = f"CPU selected - CUDA error: {str(e)}"
                
        except Exception as e:
            # Fallback to CPU if GPU info retrieval fails
            details['gpu_error'] = str(e)
            device_info = f"CPU: {details['cpu_cores']} cores (GPU detection error: {str(e)})"
            details['decision'] = "CPU selected - GPU detection error"
            
    return device, device_info, details

def optimize_torch_settings(device: torch.device, cpu_cores: int):
    """Applies optimal PyTorch settings based on the selected device."""
    if device.type == "cuda":
        print("🔧 Configuring GPU optimizations...")
        torch.backends.cudnn.benchmark = True # Enables cuDNN autotuner for faster convolutions
        torch.backends.cudnn.enabled = True   # Enables cuDNN
        # Enable memory efficient attention if available (e.g., for newer Transformers models)
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(True)
    else:
        print("🔧 Configuring CPU optimizations...")
        # Set optimal thread count for CPU operations, capping to avoid diminishing returns
        optimal_threads = max(1, min(cpu_cores - 1, 8))  # Cap at 8 for common CPU workloads
        torch.set_num_threads(optimal_threads)
        torch.set_num_interop_threads(optimal_threads)
        
        # Enable MKL-DNN for CPU if available
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True

class AsyncTTSProcessor:
    """Handles asynchronous Text-to-Speech generation using a thread pool."""
    
    def __init__(self, tts_pipeline):
        self.tts_pipeline = tts_pipeline
        # Using ThreadPoolExecutor to run TTS in a separate thread, preventing UI freeze
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2) 
        
    def generate_async(self, text: str, voice: str = 'af_heart', speed: float = 1.0) -> concurrent.futures.Future:
        """Submits a TTS generation task to the executor and returns a Future."""
        return self.executor.submit(self._generate_tts, text, voice, speed)
    
    def _generate_tts(self, text: str, voice: str, speed: float) -> Optional[str]:
        """
        Internal method to generate TTS audio. Runs in a separate thread.
        Returns the path to the temporary audio file.
        """
        # Do not attempt TTS if pipeline is not loaded or text is too long
        if not self.tts_pipeline or len(text) > config.tts_max_length:
            return None
            
        try:
            # Clean text for better TTS synthesis (remove non-standard characters, normalize whitespace)
            clean_text = re.sub(r'[^\w\s.,!?;:-]', '', text).strip()
            if not clean_text:
                return None
                
            # Generate audio using the Kokoro pipeline
            audio_gen = self.tts_pipeline(clean_text, voice=voice, speed=speed)
            audio_segment = next(audio_gen)[2] # Get the first (and usually only) audio segment
            
            # Ensure audio tensor is on CPU before converting to numpy and saving
            if hasattr(audio_segment, 'device') and audio_segment.device.type != 'cpu':
                audio_segment = audio_segment.cpu()
            
            # Save the audio to a temporary WAV file
            filename = f"/tmp/tts_{uuid.uuid4().hex[:8]}.wav" # Unique filename in /tmp
            sf.write(filename, audio_segment, 24000) # Sample rate 24000 Hz
            return filename
            
        except Exception as e:
            print(f"⚡ Async TTS Error: {e}")
            return None
    
    def shutdown(self):
        """Shuts down the thread pool executor, waiting for active tasks to complete."""
        self.executor.shutdown(wait=True)

class EnhancedVoiceTranscriber:
    """
    Handles voice transcription using Vosk, with enhanced audio preprocessing
    for better accuracy.
    """
    
    def __init__(self, model_path: str = config.vosk_model_path):
        self.model_path = model_path
        self.model = None
        self.recognizer = None
        self.load_model()
    
    def load_model(self) -> bool:
        """Loads the Vosk speech recognition model."""
        if not VOSK_AVAILABLE:
            print("❌ Vosk not available for transcription")
            return False
            
        if not Path(self.model_path).exists():
            print(f"❌ Vosk model not found at: {self.model_path}")
            print("Please download the Vosk model, e.g., from https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip")
            return False
            
        try:
            print(f"🔄 Loading Vosk model from {self.model_path}...")
            self.model = vosk.Model(self.model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, 16000) # 16kHz sample rate
            print("✅ Vosk model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load Vosk model: {e}")
            return False
    
    def transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribes an audio file to text, applying preprocessing."""
        if not self.model or not audio_file_path or not Path(audio_file_path).exists():
            return "❌ Transcription unavailable or file not found"
            
        temp_dir = tempfile.mkdtemp()
        processed_wav = Path(temp_dir) / "processed.wav"
        
        try:
            # Enhanced audio preprocessing using ffmpeg for better quality
            if not self._preprocess_audio(audio_file_path, str(processed_wav)):
                processed_wav = Path(audio_file_path) # Fallback to original if preprocessing fails
            
            # Transcribe the (processed) WAV file
            return self._transcribe_wav(str(processed_wav))
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return f"❌ Transcription failed: {str(e)}"
        finally:
            # Clean up the temporary directory
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"⚠️ Failed to clean up temp directory: {e}")
    
    def _preprocess_audio(self, input_path: str, output_path: str) -> bool:
        """
        Uses ffmpeg to convert, normalize, and apply audio filters for better
        transcription quality.
        """
        try:
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vn', '-acodec', 'pcm_s16le', # PCM signed 16-bit little-endian
                '-ar', '16000', '-ac', '1',    # 16kHz sample rate, mono channel
                # Corrected compand filter: gain stages separated by spaces, not commas
                '-af', 'highpass=f=200,lowpass=f=3400,volume=1.2,compand=0.3:0.8:-90/-60 -60/-40 -40/-30 -20/-20:6:0.05',
                output_path, '-y' # Overwrite output file if it exists
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"ffmpeg error: {result.stderr}")
            return result.returncode == 0
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Preprocessing with ffmpeg failed ({type(e).__name__}). Falling back to simple copy.")
            # Fallback to simple copy if ffmpeg is not found or times out
            try:
                import shutil
                shutil.copy2(input_path, output_path)
                return True
            except Exception as copy_e:
                print(f"Fallback copy failed: {copy_e}")
                return False
        except Exception as e:
            print(f"Unexpected preprocessing error: {e}")
            return False
    
    def _transcribe_wav(self, wav_path: str) -> str:
        """
        Performs the actual transcription of a WAV file using Vosk.
        Includes improved chunking for better accuracy.
        """
        try:
            wf = wave.open(wav_path, "rb")
            self.recognizer.Reset() # Reset recognizer state for a new transcription
            
            results = []
            chunk_size = 8000 # Larger chunks can improve accuracy for some models
            
            while True:
                data = wf.readframes(chunk_size)
                if len(data) == 0:
                    break
                    
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '').strip()
                    if text:
                        results.append(text)
            
            # Get the final result after processing all chunks
            final_result = json.loads(self.recognizer.FinalResult())
            final_text = final_result.get('text', '').strip()
            if final_text:
                results.append(final_text)
            
            wf.close()
            
            # Join all transcribed segments and clean up common artifacts
            full_text = ' '.join(results).strip()
            if full_text:
                full_text = re.sub(r'\b(um|uh|er|ah)\b', '', full_text, flags=re.IGNORECASE) # Remove filler words
                full_text = re.sub(r'\s+', ' ', full_text).strip() # Normalize whitespace
                return full_text
            else:
                return "❌ No speech detected in audio"
                
        except Exception as e:
            return f"❌ Transcription processing failed: {str(e)}"

class EnhancedChatBot:
    """
    The main chatbot class, integrating DialoGPT for text generation,
    Kokoro TTS for speech output, Vosk for speech input, and enhanced
    performance features like caching and monitoring.
    """
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.coherence_model = None
        self.tts_processor = None
        self.voice_transcriber = None
        self.response_cache = EnhancedCache(config.max_cache_size)
        self.performance_monitor = PerformanceMonitor()
        
        # Statistics tracking
        self.stats = {
            'total_responses': 0,
            'method_counts': {}, # Tracks usage of different generation methods
            'error_count': 0
        }
        
        # Pre-defined generation configurations for varied responses
        self.generation_configs = self._create_generation_configs()
    
    def _create_generation_configs(self) -> List[Dict]:
        """Defines different text generation configurations for the model."""
        return [
            {
                'name': 'balanced', # General purpose, good balance of creativity and coherence
                'max_new_tokens': 512,
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'repetition_penalty': 1.3,
                'no_repeat_ngram_size': 3,
            },
            {
                'name': 'creative', # More diverse and imaginative responses
                'max_new_tokens': 600,
                'do_sample': True,
                'temperature': 0.8,
                'top_p': 0.95,
                'top_k': 50,
                'repetition_penalty': 1.2,
                'no_repeat_ngram_size': 4,
            },
            {
                'name': 'focused', # Shorter, more direct and concise responses
                'max_new_tokens': 160,
                'do_sample': True,
                'temperature': 0.6,
                'top_p': 0.85,
                'top_k': 30,
                'repetition_penalty': 1.4,
                'no_repeat_ngram_size': 2,
            }
        ]
    
    def load_models(self) -> bool:
        """
        Loads all necessary models: DialoGPT, SentenceTransformer (for coherence),
        Kokoro TTS, and Vosk (for voice transcription).
        """
        try:
            print("🔄 Loading DialoGPT model and tokenizer...")
            checkpoint_path = self._find_latest_checkpoint(config.base_dir)
            
            # Load tokenizer and model from the latest checkpoint
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # Suppress warnings during model loading
                self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
                # Load model to the detected device, using float16 for CUDA if available
                self.model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
                    device_map={'': DEVICE} # Ensures model is loaded onto the correct device
                )
                
                # Set pad_token if not already defined (common for GPT-like models)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                self.model.eval() # Set model to evaluation mode
                print(f"✅ Model loaded on {DEVICE}")
            
            # Load coherence model if enabled
            if config.use_coherence:
                print("🔄 Loading coherence model...")
                self.coherence_model = SentenceTransformer(
                    "all-MiniLM-L6-v2", 
                    device=DEVICE.type # Ensure coherence model is also on the correct device
                )
                print("✅ Coherence model loaded")
            
            # Initialize TTS processor if Kokoro is available
            if KOKORO_AVAILABLE:
                print("🔄 Loading Kokoro TTS pipeline...")
                try:
                    tts_pipeline = KPipeline(lang_code='a') # 'a' for English
                    # Move TTS model to the detected device if it has a 'to' method
                    if hasattr(tts_pipeline, 'model') and hasattr(tts_pipeline.model, 'to'):
                        tts_pipeline.model.to(DEVICE)
                    self.tts_processor = AsyncTTSProcessor(tts_pipeline) # Wrap in async processor
                    print("✅ Kokoro TTS loaded with async processing")
                except Exception as e:
                    print(f"⚠️ Failed to load Kokoro TTS: {e}")
            
            # Initialize voice transcriber if Vosk is available
            if VOSK_AVAILABLE:
                print("🔄 Loading enhanced voice transcriber...")
                self.voice_transcriber = EnhancedVoiceTranscriber()
            
            self._pre_warm_model() # Pre-warm the main model for faster first response
            print("✅ All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            return False
    
    def _find_latest_checkpoint(self, base_dir: str) -> str:
        """Finds the path to the latest checkpoint directory within a base directory."""
        base_path = Path(base_dir)
        if not base_path.exists():
            raise FileNotFoundError(f"Directory not found: {base_dir}")
        
        checkpoints = []
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    num = int(item.name.split("-")[1])
                    checkpoints.append((num, item))
                except (IndexError, ValueError):
                    continue
        
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint directories found in {base_dir}")
        
        return str(max(checkpoints, key=lambda x: x[0])[1]) # Return path of the latest checkpoint
    
    def _pre_warm_model(self):
        """Performs a dummy inference run to 'warm up' the model, reducing first-response latency."""
        print("🔥 Pre-warming model...")
        dummy_input = "Hello, how are you?"
        
        with torch_inference_mode(): # Use the optimized inference context
            inputs = self.tokenizer(
                dummy_input, 
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(DEVICE)
            
            _ = self.model.generate(
                **inputs,
                max_new_tokens=10, # Generate a few tokens to warm up
                do_sample=False,  # Greedy decoding for speed
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        print("✅ Model pre-warmed")
    
    def generate_response_optimized(self, user_input: str) -> Tuple[str, str]:
        """
        Generates an optimized text response using the DialoGPT model.
        Includes caching, intelligent config selection, and robust post-processing.
        """
        start_time = time.perf_counter()
        
        # 1. Check cache first for immediate response
        cached_response = self.response_cache.get(user_input)
        if cached_response:
            duration = time.perf_counter() - start_time
            self.performance_monitor.log_response_time(duration, "cached")
            return cached_response, "cached"
        
        # 2. Select the best generation configuration based on input characteristics
        config_idx = self._select_generation_config(user_input)
        gen_config = self.generation_configs[config_idx]
        
        try:
            # 3. Format input for the model (DialoGPT specific format)
            formatted_input = f"<|user|>\n{user_input}\n<|assistant|>\n"
            
            with torch_inference_mode(): # Optimize for inference
                inputs = self.tokenizer(
                    formatted_input,
                    return_tensors="pt",
                    max_length=350, # Max input length
                    truncation=True,
                    padding=False # No padding needed for single input
                ).to(DEVICE)
                
                # 4. Generate response using the selected configuration
                outputs = self.model.generate(
                    **inputs,
                    **{k: v for k, v in gen_config.items() if k != 'name'}, # Pass all config params except 'name'
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True # Enable KV caching for faster subsequent token generation
                )
            
            # 5. Decode and post-process the generated tokens
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], # Decode only the newly generated tokens
                skip_special_tokens=True
            ).strip()
            
            response = self._post_process_response(response) # Apply robust cleaning
            
            if response and len(response) > 10: # Ensure response is meaningful
                self.response_cache.put(user_input, response) # Cache the successful response
                
                duration = time.perf_counter() - start_time
                method = f"optimized_{gen_config['name']}"
                self.performance_monitor.log_response_time(duration, method)
                
                return response, method
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            self.stats['error_count'] += 1 # Increment error count
        
        # 6. Fallback if generation fails or is problematic
        fallback = self._get_intelligent_fallback(user_input)
        duration = time.perf_counter() - start_time
        self.performance_monitor.log_response_time(duration, "fallback")
        
        return fallback, "fallback"
    
    def _select_generation_config(self, user_input: str) -> int:
        """
        Selects an appropriate generation configuration based on the user's input.
        This helps tailor the response style (e.g., more creative for questions).
        """
        input_length = len(user_input.split())
        
        # If the input is a question, lean towards a more creative response
        if any(word in user_input.lower() for word in ['why', 'how', 'what', 'explain', '?']):
            return 1  # Index for 'creative' config
        
        # For very short inputs, a more focused/concise response might be better
        if input_length < 5:
            return 2  # Index for 'focused' config
        
        # Default to the balanced configuration
        return 0  # Index for 'balanced' config
    
    def _post_process_response(self, response: str) -> str:
        """
        Cleans and formats the generated text response, removing artifacts
        and ensuring proper sentence endings.
        """
        if not response:
            return ""
        
        # Remove common conversational turn markers that might appear in raw generation
        patterns = [
            r'User:.*$', r'<\|user\|>.*$',
            r'Assistant:.*$', r'<\|assistant\|>.*$',
            r'--- User:.*$', r'--- Assistant:.*$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                response = response[:match.start()].strip() # Truncate at the first marker
                break
        
        # Clean up various formatting artifacts and normalize whitespace
        response = re.sub(r'<[^>]+>', '', response)  # Remove HTML-like tags (e.g., <|endoftext|>)
        response = re.sub(r'\*+', '', response)     # Remove multiple asterisks
        response = re.sub(r'#+\s*', '', response)   # Remove hash symbols used in markdown
        response = re.sub(r'\s+', ' ', response)    # Normalize multiple spaces to single space
        
        # Ensure the response ends with proper punctuation
        response = response.strip()
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response
    
    def _get_intelligent_fallback(self, user_input: str) -> str:
        """
        Provides a contextually appropriate fallback response if the main
        generation process fails or produces a problematic output.
        """
        fallbacks = {
            'question': [
                "That's a thoughtful question that deserves more consideration.",
                "I'd need to think about that more carefully to give you a good answer.",
                "That's an interesting question - let me reflect on it."
            ],
            'statement': [
                "That's an interesting perspective.",
                "I can see why you'd think about it that way.",
                "That's worth considering further."
            ],
            'greeting': [
                "Hello! How can I help you today?",
                "Hi there! What would you like to chat about?",
                "Hello! I'm here to help."
            ]
        }
        
        # Determine the category of the user's input to pick a relevant fallback
        if any(word in user_input.lower() for word in ['hello', 'hi', 'hey', 'greetings']):
            category = 'greeting'
        elif user_input.strip().endswith('?'):
            category = 'question'
        else:
            category = 'statement'
        
        return np.random.choice(fallbacks[category]) # Return a random fallback from the category
    
    def chat_response_parallel(self, user_input: str, history: List, enable_tts: bool, 
                             voice: str, speed: float) -> Tuple[List, str, Optional[str]]:
        """
        The main chat response function for the Gradio interface.
        It generates text and initiates TTS generation in parallel for responsiveness.
        """
        if not user_input.strip():
            return history, "", None
        
        start_time = time.perf_counter()
        
        # 1. Generate text response (this is the primary, potentially time-consuming step)
        response, method = self.generate_response_optimized(user_input)
        
        # 2. Start TTS generation asynchronously if enabled and response is within length limits
        tts_future = None
        if enable_tts and self.tts_processor and len(response) <= config.tts_max_length:
            tts_future = self.tts_processor.generate_async(response, voice, speed)
        
        # 3. Update chat history immediately for responsiveness
        history.append([user_input, response])
        
        # 4. Update internal statistics
        self.stats['total_responses'] += 1
        self.stats['method_counts'][method] = self.stats['method_counts'].get(method, 0) + 1
        
        # 5. Retrieve TTS result (wait for it if it was started)
        audio_file = None
        if tts_future:
            try:
                # Wait for TTS to complete with a timeout to prevent indefinite blocking
                audio_file = tts_future.result(timeout=5.0) 
            except concurrent.futures.TimeoutError:
                print("⚠️ TTS generation timed out for this response.")
            except Exception as e:
                print(f"⚠️ TTS generation failed: {e}")
        
        # 6. Perform memory cleanup if system memory usage is high
        if self.performance_monitor.should_cleanup_memory():
            self._cleanup_memory()
        
        duration = time.perf_counter() - start_time
        print(f"⚡ Response generated in {duration:.2f}s (Method: {method})")
        
        return history, "", audio_file
    
    def transcribe_voice_input(self, audio_file_path: str) -> str:
        """Transcribes an audio file using the enhanced voice transcriber."""
        if not self.voice_transcriber:
            return "❌ Voice transcription not available."
        
        if not audio_file_path:
            return "❌ No audio file provided for transcription."
        
        return self.voice_transcriber.transcribe_audio(audio_file_path)
    
    def _cleanup_memory(self):
        """
        Clears GPU cache (if applicable) and runs Python's garbage collector
        to free up memory.
        """
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache() # Clear PyTorch's CUDA memory cache
            torch.cuda.synchronize() # Wait for all CUDA operations to complete
        
        gc.collect() # Run Python's garbage collector
        print("🧹 Memory cleanup performed.")
    
    def get_comprehensive_stats(self) -> str:
        """
        Generates a comprehensive string report of system, performance,
        and chatbot-specific statistics.
        """
        system_stats = self.performance_monitor.get_system_stats()
        cache_stats = self.response_cache.get_stats()
        
        # Calculate average response time from the performance monitor's logs
        if self.performance_monitor.response_times:
            # Calculate average of the last 20 response times
            avg_time = np.mean([rt[1] for rt in self.performance_monitor.response_times[-20:]])
        else:
            avg_time = 0.0
        
        # Format method distribution statistics
        method_stats_str = ""
        if self.stats['method_counts']:
            method_stats_str = "\n".join([f"- {method}: {count} responses" 
                                          for method, count in self.stats['method_counts'].items()])
            method_stats_str = "\n\n**Generation Methods Used:**\n" + method_stats_str
        
        # Construct the full statistics string
        stats_report = f"""
📊 **Session Statistics:**
- Total responses: {self.stats['total_responses']}
- Average response time (last 20): {avg_time:.2f}s
- Errors encountered: {self.stats['error_count']}
- Device: {DEVICE_INFO}
- TTS Available: {'Yes' if self.tts_processor else 'No'}
- Voice Recognition: {'Yes' if self.voice_transcriber else 'No'}

---
💻 **System Metrics:**
- CPU Usage: {system_stats.get('cpu_percent', 0):.1f}%
- Memory Usage: {system_stats.get('memory_percent', 0):.1f}%
- Uptime: {system_stats.get('uptime', 0):.0f} seconds
"""
        if 'gpu_name' in DEVICE_DETAILS:
            stats_report += f"""
- GPU: {DEVICE_DETAILS['gpu_name']}
- GPU Memory Used: {system_stats.get('gpu_memory_used', 0):.2f}GB / {system_stats.get('gpu_memory_total', 0):.2f}GB
- GPU Utilization: {system_stats.get('gpu_utilization', 0):.1f}%
"""
        
        stats_report += f"""
---
📦 **Response Cache:**
- Cache Size: {cache_stats['size']} / {cache_stats['max_size']}
- Cache Hit Rate: {cache_stats['hit_rate']}
- Cache Hits: {cache_stats['hits']}
- Cache Misses: {cache_stats['misses']}
{method_stats_str}
        """
        return stats_report
    
    def clear_chat(self):
        """Clears chat history and resets all internal statistics and cache."""
        self.stats = { # Reset stats
            'total_responses': 0,
            'method_counts': {},
            'error_count': 0
        }
        self.response_cache.clear() # Clear the response cache
        self.performance_monitor = PerformanceMonitor() # Reset performance monitor
        print("🗑️ Chat history, stats, and cache cleared.")
        return [] # Return empty list for Gradio chat history

# Global initialization of device and chatbot
DEVICE, DEVICE_INFO, DEVICE_DETAILS = get_optimal_device_config()
optimize_torch_settings(DEVICE, DEVICE_DETAILS.get('cpu_cores', multiprocessing.cpu_count()))
chatbot = EnhancedChatBot()

def record_and_transcribe(audio_file_path):
    """Gradio function to transcribe an audio file and return the text."""
    if audio_file_path is None:
        return "No audio recorded."
    return chatbot.transcribe_voice_input(audio_file_path)

def process_voice_to_chat(audio_file_path, history, enable_tts, voice_selection, speed_control):
    """
    Gradio function to transcribe audio and then send the transcribed text
    to the chatbot for a response.
    """
    transcribed_text = record_and_transcribe(audio_file_path)
    if transcribed_text and not transcribed_text.startswith("❌") and transcribed_text != "No audio recorded.":
        # Use the parallel chat response function
        return chatbot.chat_response_parallel(transcribed_text, history, enable_tts, voice_selection, speed_control)
    else:
        # If transcription failed or no speech, return current history and empty input
        return history, "", None

def shutdown_server():
    """Shuts down the Gradio server gracefully."""
    print("🛑 Shutdown requested by user...")
    print("💭 Closing server in 2 seconds...")
    if chatbot.tts_processor:
        chatbot.tts_processor.shutdown() # Ensure TTS thread pool is shut down

    def delayed_shutdown():
        time.sleep(2)
        os._exit(0) # Force exit after delay

    threading.Thread(target=delayed_shutdown).start()
    return "🛑 Server shutting down..."

def create_gradio_interface():
    """Creates and returns the Gradio Blocks interface for the chatbot."""

    # Available voices for Kokoro TTS (if available)
    available_voices = [
        "af", "af_bella", "af_heart", "af_sky", "af_wave", "af_happy", "af_happy_2", "af_confused",
        "am", "am_adam", "am_michael", "bf", "bf_emma", "bf_isabella", "bm", "bm_george", "bm_lewis"
    ]

    # Custom CSS for a more polished look
    css = """
    .gradio-container {
        max-width: 1400px !important;
        font-family: 'Inter', sans-serif;
    }
    .chat-message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 10px;
    }
    .gradio-container .gr-button.primary {
        background-color: #4CAF50; /* Green */
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .gradio-container .gr-button.secondary {
        background-color: #f0f0f0; /* Light grey */
        color: #333;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .gradio-container .gr-button.stop {
        background-color: #f44336; /* Red */
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .gradio-container .gr-checkbox {
        margin-top: 10px;
    }
    .gradio-container .gr-slider {
        margin-top: 10px;
    }
    .gradio-container .gr-dropdown {
        margin-top: 10px;
    }
    """

    with gr.Blocks(css=css, title="Enhanced DialoGPT Chat with Voice I/O") as demo:
        gr.Markdown("# 🧠 Enhanced DialoGPT Chat with Voice Input/Output")
        gr.Markdown("An optimized chatbot featuring voice input (STT), voice output (TTS), and performance monitoring.")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot_interface = gr.Chatbot(
                    height=500,
                    label="Chat History",
                    show_label=True,
                    avatar_images=(None, "https://www.gravatar.com/avatar/?d=retro") # Placeholder for bot avatar
                )

                # Text input row
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Type your message here or use voice input below...",
                        label="Your Message",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send 💬", scale=1, variant="primary")

                # Voice input row
                with gr.Row():
                    with gr.Column(scale=2):
                        audio_input = gr.Microphone(
                            label="🎙️ Voice Input",
                            type="filepath", # Records audio to a temporary file
                            interactive=VOSK_AVAILABLE # Only interactive if Vosk is available
                        )
                    with gr.Column(scale=2):
                        transcribe_btn = gr.Button(
                            "🎤 Transcribe Voice → Text",
                            variant="secondary",
                            size="lg",
                            interactive=VOSK_AVAILABLE
                        )
                        voice_to_chat_btn = gr.Button(
                            "🗣️ Voice → Chat",
                            variant="primary",
                            size="lg",
                            interactive=VOSK_AVAILABLE
                        )

                with gr.Row():
                    enable_tts = gr.Checkbox(
                        label="Enable Text-to-Speech 🔊",
                        value=KOKORO_AVAILABLE,
                        interactive=KOKORO_AVAILABLE # Only interactive if Kokoro is available
                    )
                    clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")
                    shutdown_btn = gr.Button("Shutdown Server 🛑", variant="stop")

            with gr.Column(scale=1):
                gr.Markdown("### 🎵 Audio Output")
                audio_output = gr.Audio(
                    label="Generated Speech",
                    autoplay=True, # Automatically play the generated audio
                    show_label=True
                )

                if KOKORO_AVAILABLE:
                    gr.Markdown("### 🎙️ Voice Settings")
                    voice_selection = gr.Dropdown(
                        choices=available_voices,
                        value="af_heart", # Default voice
                        label="Voice",
                        info="Select voice for TTS"
                    )

                    speed_control = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speech Speed",
                        info="Adjust playback speed (0.5x to 2.0x)"
                    )
                else:
                    # Hide TTS controls if Kokoro is not available
                    voice_selection = gr.Dropdown(choices=["af_heart"], value="af_heart", visible=False)
                    speed_control = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, visible=False)

                gr.Markdown("### 📊 Statistics")
                stats_display = gr.Markdown(chatbot.get_comprehensive_stats()) # Initial stats display
                refresh_stats = gr.Button("Refresh Stats 📊", size="sm")
                
                gr.Markdown("### 🛑 Server Control")
                shutdown_status = gr.Markdown("Server running normally")

                gr.Markdown("### ℹ️ Information")
                tts_status = "✅ Kokoro TTS Available" if KOKORO_AVAILABLE else "❌ Install: pip install kokoro>=0.9.4 soundfile"
                vosk_status = "✅ Vosk STT Available" if VOSK_AVAILABLE else "❌ Install: pip install vosk"
                gr.Markdown(f"""
**Model Status:**
- Device: {DEVICE_INFO}
- TTS: {tts_status}
- Speech Recognition: {vosk_status}
- Coherence Check: {'✅ Enabled' if config.use_coherence else '❌ Disabled'}

**Tips:**
- Shorter, clearer messages often yield better responses.
- Enable TTS to hear the chatbot's replies.
- Monitor statistics to understand performance.
                """)

        # Event handlers for Gradio components
        def handle_chat(user_input_text, history, enable_tts_val, voice_val, speed_val):
            # Call the parallel chat response function
            return chatbot.chat_response_parallel(user_input_text, history, enable_tts_val, voice_val, speed_val)

        def handle_clear():
            # Clear chat history and reset stats
            return chatbot.clear_chat(), chatbot.get_comprehensive_stats()

        def handle_stats_refresh():
            # Refresh and display current statistics
            return chatbot.get_comprehensive_stats()

        def handle_shutdown():
            # Trigger server shutdown
            return shutdown_server()

        # Wire up events to functions
        send_btn.click(
            fn=handle_chat,
            inputs=[user_input, chatbot_interface, enable_tts, voice_selection, speed_control],
            outputs=[chatbot_interface, user_input, audio_output]
        )

        user_input.submit( # Allow submitting with Enter key
            fn=handle_chat,
            inputs=[user_input, chatbot_interface, enable_tts, voice_selection, speed_control],
            outputs=[chatbot_interface, user_input, audio_output]
        )
        
        clear_btn.click(
            fn=handle_clear,
            outputs=[chatbot_interface, stats_display]
        )

        refresh_stats.click(
            fn=handle_stats_refresh,
            outputs=[stats_display]
        )

        shutdown_btn.click(
            fn=handle_shutdown,
            outputs=[shutdown_status]
        )

        # Wire up the transcribe button
        transcribe_btn.click(
            fn=record_and_transcribe,
            inputs=[audio_input],
            outputs=[user_input] # Puts transcribed text into the user input box
        )

        # Wire up the voice_to_chat button
        voice_to_chat_btn.click(
            fn=process_voice_to_chat,
            inputs=[audio_input, chatbot_interface, enable_tts, voice_selection, speed_control],
            outputs=[chatbot_interface, user_input, audio_output]
        )

    return demo

def open_browser():
    """Opens the web browser to the Gradio interface after a short delay."""
    time.sleep(2) # Give the server a moment to start up
    webbrowser.open(f'http://localhost:{config.server_port}')
    print(f"🌐 Opened browser automatically to http://localhost:{config.server_port}")

def main():
    """Main function to initialize and run the Gradio application."""
    print("🚀 Starting Enhanced DialoGPT Chat with Voice I/O...")

    # Load models. If this fails, exit.
    if not chatbot.load_models():
        print("❌ Failed to initialize. Please check your model path and dependencies.")
        return

    # Create the Gradio interface
    demo = create_gradio_interface()

    print("\n✅ Ready! Starting web interface...")
    print(f"🌐 Access the chat at: http://localhost:{config.server_port}")

    if KOKORO_AVAILABLE:
        print("🔊 Kokoro TTS is available and enabled.")
        print("🎙️ Available voices: af_heart, af_bella, bf_emma, bm_george, and more!")
    else:
        print("⚠️ Kokoro TTS not available. Install with: pip install kokoro>=0.9.4 soundfile")
        print("🐧 On Ubuntu/Debian also run: apt-get install espeak-ng (for espeak-ng dependency)")

    if VOSK_AVAILABLE:
        print("🎤 Vosk Speech Recognition is available and enabled.")
    else:
        print("⚠️ Vosk not available. Install with: pip install vosk")

    # Open browser automatically in a separate thread to not block the main thread
    if config.auto_open_browser:
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True # Allow main program to exit even if thread is running
        browser_thread.start()

    try:
        # Launch the Gradio demo
        demo.launch(
            server_name="127.0.0.1", # Bind to localhost
            server_port=config.server_port,
            share=False, # Do not create a public share link
            inbrowser=False, # Browser opened manually by open_browser function
            show_error=False, # Reduce verbose error logging in UI
            quiet=True, # Reduce console logging from Gradio
            max_threads=1, # Limit concurrent requests to avoid overloading the model
            allowed_paths=['/tmp'] # Allow access to temporary audio files
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user (Ctrl+C).")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
    finally:
        if chatbot.tts_processor:
            chatbot.tts_processor.shutdown() # Ensure TTS thread pool is shut down on exit
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()



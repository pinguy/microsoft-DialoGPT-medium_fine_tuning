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
import concurrent.futures
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import multiprocessing
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
    base_dir: str = "./dialogpt-finetuned/"  # Path to the parent directory of your checkpoints (legacy name, but for Rhizome 1.5B)
    vosk_model_path: str = "vosk-model-en-us-0.42-gigaspeech"
    server_port: int = 7860
    auto_open_browser: bool = True
    max_cache_size: int = 100
    max_response_length: int = 512
    tts_max_length: int = 500
    memory_cleanup_threshold: float = 0.6  # Lower threshold for CPU safety
    # Rhizome specific settings
    show_reasoning: bool = False  # Whether to show <think> blocks
    use_system_prompt: bool = False  # Set to False for Qwen compatibility; avoids unnecessary complexity for small model

config = Config()

class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.response_times = []
        self.start_time = time.time()
        
    def log_response_time(self, duration: float, method: str):
        self.response_times.append((time.time(), duration, method))
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
    
    def get_system_stats(self) -> Dict[str, Any]:
        stats = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'uptime': time.time() - self.start_time
        }
        
        if torch.cuda.is_available():
            try:
                stats['gpu_memory_used'] = torch.cuda.memory_allocated() / 1024**3
                stats['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            except:
                pass
                
        return stats
    
    def should_cleanup_memory(self) -> bool:
        return psutil.virtual_memory().percent > config.memory_cleanup_threshold * 100

class EnhancedCache:
    """Intelligent caching system"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.max_size = max_size
        
    def _normalize_key(self, key: str) -> str:
        return re.sub(r'\s+', ' ', key.lower().strip())
    
    def get(self, key: str) -> Optional[str]:
        normalized_key = self._normalize_key(key)
        if normalized_key in self.cache:
            self.access_times[normalized_key] = time.time()
            self.hit_count += 1
            return self.cache[normalized_key]
        
        self.miss_count += 1
        return None
    
    def put(self, key: str, value: str):
        normalized_key = self._normalize_key(key)
        
        if len(self.cache) >= self.max_size and normalized_key not in self.cache:
            oldest_key = min(self.access_times.keys(), 
                           key=lambda k: self.access_times.get(k, 0))
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[normalized_key] = value
        self.access_times[normalized_key] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
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
        self.cache.clear()
        self.access_times.clear()
        self.hit_count = 0
        self.miss_count = 0

@contextmanager
def torch_inference_mode():
    """Context manager for optimized PyTorch inference"""
    with torch.inference_mode():
        if DEVICE.type == 'cuda':
            with torch.cuda.amp.autocast():
                yield
        else:
            yield

def get_optimal_device_config() -> Tuple[torch.device, str, Dict]:
    """Detects optimal device, now with MPS support for those fancy Apple silicon overlords"""
    device = torch.device("cpu")
    device_info = "CPU (default)"
    details = {'cpu_cores': multiprocessing.cpu_count()}
    
    if torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            device_info = "MPS (Apple Silicon)"
            details['decision'] = "MPS selected - Because why not let the fruit-powered chip shine?"
            test_tensor = torch.randn(100, 100, device='mps')
            _ = test_tensor @ test_tensor.T
            del test_tensor
        except Exception as e:
            print(f"⚠️ MPS test failed: {e}, falling back to CPU")
            details['mps_error'] = str(e)
    
    elif torch.cuda.is_available():
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
            
            try:
                test_tensor = torch.randn(100, 100, device='cuda')
                _ = test_tensor @ test_tensor.T
                del test_tensor
                torch.cuda.empty_cache()
                
                if props.major >= 7 or (props.major >= 6 and memory_gb >= 4):
                    device = torch.device("cuda")
                    device_info = f"GPU: {gpu_name} ({compute_capability}, {memory_gb:.1f}GB)"
                    details['decision'] = "GPU selected"
                else:
                    device_info = f"CPU: {details['cpu_cores']} cores"
                    details['decision'] = "CPU selected - GPU insufficient"
                    
            except Exception as e:
                device_info = f"CPU: {details['cpu_cores']} cores"
                details['decision'] = f"CPU selected - CUDA error"
                
        except Exception as e:
            details['gpu_error'] = str(e)
            device_info = f"CPU: {details['cpu_cores']} cores"
            details['decision'] = "CPU selected"
            
    return device, device_info, details

def optimize_torch_settings(device: torch.device, cpu_cores: int):
    """Optimize PyTorch settings, because who wants a sluggish AI? Not me!"""
    if device.type == "cuda":
        print("🔧 Configuring GPU optimizations...")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    elif device.type == "mps":
        print("🔧 Configuring MPS optimizations... Apple style!")
    else:
        print("🔧 Configuring CPU optimizations...")
        optimal_threads = max(1, min(cpu_cores - 1, 8))
        torch.set_num_threads(optimal_threads)
        torch.set_num_interop_threads(optimal_threads)
        
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True

class AsyncTTSProcessor:
    """Async TTS processing - Because waiting is so last century"""
    
    def __init__(self, tts_pipeline):
        self.tts_pipeline = tts_pipeline
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2) 
        
    def generate_async(self, text: str, voice: str = 'af_heart', speed: float = 1.0) -> concurrent.futures.Future:
        return self.executor.submit(self._generate_tts, text, voice, speed)
    
    def _generate_tts(self, text: str, voice: str, speed: float) -> Optional[str]:
        if not self.tts_pipeline or len(text) > config.tts_max_length:
            return None
            
        try:
            clean_text = re.sub(r'[^\w\s.,!?;:\'-]', '', text).strip()
            if not clean_text:
                return None
                
            audio_gen = self.tts_pipeline(clean_text, voice=voice, speed=speed)
            audio_segment = next(audio_gen)[2]
            
            if hasattr(audio_segment, 'device') and audio_segment.device.type != 'cpu':
                audio_segment = audio_segment.cpu()
            
            filename = f"/tmp/tts_{uuid.uuid4().hex[:8]}.wav"
            
            # Add error handling for WAV writing
            try:
                sf.write(filename, audio_segment, 24000)
            except Exception as write_error:
                print(f"⚠️ WAV write failed: {write_error}")
                return None
                
            return filename
            
        except Exception as e:
            print(f"⚡ TTS Error: {e} - The voice synthesizer threw a tantrum!")
            return None
    
    def shutdown(self):
        self.executor.shutdown(wait=True)

class EnhancedVoiceTranscriber:
    """Voice transcription with Vosk - Turning mumbles into meaningful text"""
    
    def __init__(self, model_path: str = config.vosk_model_path):
        self.model_path = model_path
        self.model = None
        self.recognizer = None
        self.load_model()
    
    def load_model(self) -> bool:
        if not VOSK_AVAILABLE:
            return False
            
        if not Path(self.model_path).exists():
            print(f"❌ Vosk model not found at: {self.model_path} - Where did it go?")
            return False
            
        try:
            print(f"📄 Loading Vosk model...")
            self.model = vosk.Model(self.model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
            print("✅ Vosk loaded - Ready to eavesdrop!")
            return True
        except Exception as e:
            print(f"❌ Failed to load Vosk: {e}")
            return False
    
    def transcribe_audio(self, audio_file_path: str) -> str:
        if not self.model or not audio_file_path or not Path(audio_file_path).exists():
            return "❌ Transcription unavailable - No audio or model missing!"
            
        temp_dir = tempfile.mkdtemp()
        processed_wav = Path(temp_dir) / "processed.wav"
        
        try:
            if not self._preprocess_audio(audio_file_path, str(processed_wav)):
                processed_wav = Path(audio_file_path)
            
            return self._transcribe_wav(str(processed_wav))
            
        except Exception as e:
            return f"❌ Transcription failed: {str(e)} - Tech gremlins at work!"
        finally:
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass
    
    def _preprocess_audio(self, input_path: str, output_path: str) -> bool:
        try:
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                '-af', 'highpass=f=200,lowpass=f=3400,volume=1.2',
                output_path, '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except:
            try:
                import shutil
                shutil.copy2(input_path, output_path)
                return True
            except:
                return False
    
    def _transcribe_wav(self, wav_path: str) -> str:
        try:
            wf = wave.open(wav_path, "rb")
            self.recognizer.Reset()
            
            results = []
            chunk_size = 8000
            
            while True:
                data = wf.readframes(chunk_size)
                if len(data) == 0:
                    break
                    
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '').strip()
                    if text:
                        results.append(text)
            
            final_result = json.loads(self.recognizer.FinalResult())
            final_text = final_result.get('text', '').strip()
            if final_text:
                results.append(final_text)
            
            wf.close()
            
            full_text = ' '.join(results).strip()
            if full_text:
                return re.sub(r'\s+', ' ', full_text).strip()
            else:
                return "❌ No speech detected - Was it a mime performance?"
                
        except Exception as e:
            return f"❌ Processing failed: {str(e)}"

class RhizomeChatBot:
    """
    Optimized chatbot for Rhizome reasoning model
    Or whatever model you're throwing at it - we're flexible like that!
    """
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.tts_processor = None
        self.voice_transcriber = None
        self.response_cache = EnhancedCache(config.max_cache_size)
        self.performance_monitor = PerformanceMonitor()
        self.conversation_history = []  # Track multi-turn context
        
        self.stats = {
            'total_responses': 0,
            'method_counts': {},
            'error_count': 0
        }
        
        self.generation_configs = self._create_generation_configs()
    
    def _create_generation_configs(self) -> List[Dict]:
        """Generation configs optimized for reasoning models - Mix it up! Increased temperature and tokens for small model to encourage output."""
        return [
            {
                'name': 'balanced',
                'max_new_tokens': 768,  # Bump up for more room on complex prompts
                'do_sample': True,
                'temperature': 0.8,  # Slightly higher to avoid blanks
                'top_p': 0.95,
                'top_k': 50,
                'repetition_penalty': 1.1,  # Adjust to reduce loops
            },
            {
                'name': 'creative',
                'max_new_tokens': 800,  # Reduced from 1024 to prevent over-verbosity
                'do_sample': True,
                'temperature': 0.95,
                'top_p': 0.95,
                'top_k': 60,
                'repetition_penalty': 1.1,  # Reduced from 1.05
            },
            {
                'name': 'focused',
                'max_new_tokens': 512,
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.85,
                'top_k': 40,
                'repetition_penalty': 1.15,
            }
        ]
    
    def load_models(self) -> bool:
        """Load Rhizome model and accessories - Fingers crossed, no explosions!"""
        try:
            print(f"📄 Loading from {config.base_dir}...")
            checkpoint_path = self._find_latest_checkpoint(config.base_dir)
            
            if not checkpoint_path:
                 print(f"❌ No valid model or checkpoint found in {config.base_dir} - Did the files go on vacation?")
                 return False

            print(f"✅ Found latest model/checkpoint at: {checkpoint_path}")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        checkpoint_path,
                        trust_remote_code=True
                    )
                    print("✅ Tokenizer loaded successfully")
                except Exception as e:
                    print(f"❌ Tokenizer loading failed: {e}")
                    raise

                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        checkpoint_path,
                        torch_dtype=torch.float16 if DEVICE.type in ['cuda', 'mps'] else torch.float32,
                        device_map={'': DEVICE},
                        trust_remote_code=True,
                        low_cpu_mem_usage=True  # Added for better memory handling
                    )
                    print("✅ Model loaded successfully")
                except Exception as e:
                    print(f"❌ Model loading failed: {e}")
                    raise
                
                # Set pad token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                self.model.eval()
                print(f"✅ Model loaded on {DEVICE} - Ready to rock!")
            
            # Load TTS if available
            if KOKORO_AVAILABLE:
                print("📄 Loading Kokoro TTS...")
                try:
                    tts_pipeline = KPipeline(lang_code='a')
                    if hasattr(tts_pipeline, 'model') and hasattr(tts_pipeline.model, 'to'):
                        tts_pipeline.model.to(DEVICE)
                    self.tts_processor = AsyncTTSProcessor(tts_pipeline)
                    print("✅ TTS loaded - Let the talking begin!")
                except Exception as e:
                    print(f"⚠️ TTS failed: {e} - Silent treatment activated.")
            
            # Load voice transcriber
            if VOSK_AVAILABLE:
                print("📄 Loading voice transcriber...")
                self.voice_transcriber = EnhancedVoiceTranscriber()
            
            self._pre_warm_model()
            print("✅ All models loaded! - Or at least, the ones that showed up.")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load models: {e} - Catastrophic failure, captain!")
            import traceback
            traceback.print_exc()
            return False
    
    def _find_latest_checkpoint(self, base_dir: str) -> Optional[str]:
        """Find latest checkpoint or use base dir if it's a valid model directory. Now with extra detective work!"""
        base_path = Path(base_dir)
        
        if not base_path.exists() or not base_path.is_dir():
            print(f"❌ Directory not found or is not a directory: {base_dir} - Is it hiding?")
            return None
        
        print(f"🔍 Scanning directory: {base_dir}")
        contents = [item.name for item in base_path.iterdir()]
        print(f"Directory contents: {contents}")
        
        # Check for checkpoint subdirectories
        checkpoints = []
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    num = int(item.name.split("-")[1])
                    checkpoints.append((num, item))
                except (IndexError, ValueError):
                    print(f"⚠️ Ignoring invalid checkpoint name: {item.name}")
                    continue
        
        if checkpoints:
            print(f"✅ Found {len(checkpoints)} potential checkpoints.")
            latest_checkpoint_path = max(checkpoints, key=lambda x: x[0])[1]
            print(f"🎯 Selected latest checkpoint: {latest_checkpoint_path}")
            return str(latest_checkpoint_path)
        else:
            print("⚠️ No checkpoint directories found.")

        # If no checkpoints, check if the base directory itself is a model
        model_files = [
            base_path / "config.json",
            base_path / "pytorch_model.bin",
            base_path / "model.safetensors"
        ]
        if any(f.exists() for f in model_files):
            print("✅ Using base directory as model path - It's got the goods!")
            return str(base_path)
        else:
            print("❌ No model files found in base directory - Empty promises!")

        return None

    def _pre_warm_model(self):
        """Warm up the model - Like coffee for your AI"""
        print("🔥 Pre-warming model... Don't want it catching a cold!")
        dummy_input = "Hello"
        
        with torch_inference_mode():
            inputs = self.tokenizer(
                dummy_input, 
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(DEVICE)
            
            _ = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        print("✅ Model pre-warmed - Hot and ready!")

    def _format_prompt_for_Rhizome(self, user_input: str, system_prompt: Optional[str] = None) -> str:
        """
        Updated format for Qwen compatibility: Use <|im_start|> tags to avoid token mismatches and blank outputs.
        This should help the small model generate something, even if imperfect.
        """
        formatted = ""
        if system_prompt and config.use_system_prompt:
            formatted += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        formatted += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        return formatted
    
    def _extract_response_from_output(self, full_output: str, show_reasoning: bool = False) -> str:
        """
        Extract final response from Rhizome output
        Rhizome models can produce <think>reasoning</think> blocks - Fancy thinkers!
        """
        # Remove the input prompt if it's in the output
        if "<|im_start|>assistant" in full_output:
            full_output = full_output.split("<|im_start|>assistant", 1)[-1]
        
        # Handle thinking blocks
        if "<think>" in full_output and "</think>" in full_output:
            think_pattern = r'<think>(.*?)</think>'
            reasoning_blocks = re.findall(think_pattern, full_output, re.DOTALL)
            
            # Remove think blocks from output
            response = re.sub(think_pattern, '', full_output, flags=re.DOTALL)
            
            if show_reasoning and reasoning_blocks:
                # If user wants to see reasoning, format it nicely
                reasoning_text = "\n\n".join([f"💭 **Reasoning:**\n{r.strip()}" for r in reasoning_blocks])
                response = f"{reasoning_text}\n\n**Answer:**\n{response.strip()}"
        else:
            response = full_output
        
        # Clean up response - single regex for all special tokens
        response = response.strip()
        response = re.sub(r'<\|im.*?\|>', '', response)  # Handles both im_start and im_end
        
        # Remove any User:/Assistant: prefixes that might leak through
        response = re.sub(r'^(User|Assistant):\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\n(User|Assistant):\s*.*$', '', response, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean whitespace
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r' {2,}', ' ', response)
        
        return response.strip()
    
    def generate_response_optimized(self, user_input: str, show_reasoning: bool = False) -> Tuple[str, str]:
        """Generate response with Rhizome model - Optimized for wit and wisdom"""
        start_time = time.perf_counter()
        
        # Check cache
        cache_key = f"{user_input}|{show_reasoning}"
        cached_response = self.response_cache.get(cache_key)
        if cached_response:
            duration = time.perf_counter() - start_time
            self.performance_monitor.log_response_time(duration, "cached")
            return cached_response, "cached"
        
        # Select generation config
        config_idx = self._select_generation_config(user_input)
        gen_config = self.generation_configs[config_idx]
        
        method = f"optimized_{gen_config['name']}"
        try:
            # Format prompt for Rhizome
            system_prompt = (
            "You are the assistant. Skip introductions or descriptions of your own creation. "
            "Speak directly to the user with the tone of a close collaborator—witty, grounded, and emotionally literate. "
           "Avoid formal openings like 'I am an AI…'; go straight into the conversation."
           "Do not reproduce dialogue tags or describe the conversation. "
           "Stay in character as the assistant, matching the user’s emotional depth. "
           "Keep responses concise, conversational, and alive—like two mates thinking out loud."
            )  # Can be customized
            formatted_input = self._format_prompt_for_Rhizome(user_input, system_prompt)
            
            with torch_inference_mode():
                inputs = self.tokenizer(
                    formatted_input,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(DEVICE)
                
                input_length = inputs.input_ids.shape[1]
                
                outputs = self.model.generate(
                    **inputs,
                    **{k: v for k, v in gen_config.items() if k != 'name'},
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            
            # Decode only generated tokens
            generated_ids = outputs[0][input_length:]
            if len(generated_ids) == 0:
                print("⚠️ No tokens generated - Falling back.")
                return self._get_fallback_response(user_input)
            
            raw_response = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,  # Switch to True for cleaner output, as <think> might not always trigger
                clean_up_tokenization_spaces=True
            )
            
            # Extract and clean response
            response = self._extract_response_from_output(raw_response, show_reasoning)
            
            if response and len(response) > 5:
                self.response_cache.put(cache_key, response)
                
                duration = time.perf_counter() - start_time
                self.performance_monitor.log_response_time(duration, method)
                
                return response, method
            else:
                print("⚠️ Empty or too short response, using fallback - The model got shy!")
                return self._get_fallback_response(user_input)
            
        except Exception as e:
            print(f"❌ Generation error: {e} - Oops, something went boom!")
            import traceback
            traceback.print_exc()
            self.stats['error_count'] += 1
            return self._get_fallback_response(user_input)
    
    def _select_generation_config(self, user_input: str) -> int:
        """Select appropriate generation config - Because one size doesn't fit all"""
        normalized_input = user_input.lower()
        input_length = len(user_input.split())
        
        # Use creative for complex reasoning questions
        if any(word in normalized_input for word in ['why', 'how', 'explain', 'analyze', 'compare', 'what if']):
            return 1  # creative
        
        # Focused for short queries
        if input_length < 5:
            return 2  # focused
        
        # Default balanced
        return 0
    
    def _get_fallback_response(self, user_input: str) -> Tuple[str, str]:
        """Simple fallback response - When the AI draws a blank"""
        fallbacks = [
            "I'm not quite sure how to respond to that. Could you rephrase? Pretty please?",
            "That's an interesting question. Could you provide more context? Or should I guess wildly?",
            "I need a bit more information to give you a good answer. Feed me data!",
            "Could you ask that in a different way? My circuits are tangled."
        ]
        return np.random.choice(fallbacks), "fallback"
    
    def chat_response_parallel(self, user_input: str, history: List, enable_tts: bool, 
                             voice: str, speed: float, show_reasoning: bool = False) -> Tuple[List, str, Optional[str]]:
        """Main chat response function - Parallel processing for the win!"""
        if not user_input.strip():
            return history, "", None
        
        start_time = time.perf_counter()
        
        # Generate response
        response, method = self.generate_response_optimized(user_input, show_reasoning)
        
        # Start TTS async
        tts_future = None
        if enable_tts and self.tts_processor:
            # Remove reasoning blocks for TTS
            tts_text = re.sub(r'💭.*?\*\*Answer:\*\*\n', '', response, flags=re.DOTALL)
            tts_text = tts_text[:config.tts_max_length]
            if tts_text:
                tts_future = self.tts_processor.generate_async(tts_text, voice, speed)
        
        # Update history
        history.append([user_input, response])
        
        # Update stats
        self.stats['total_responses'] += 1
        self.stats['method_counts'][method] = self.stats['method_counts'].get(method, 0) + 1
        
        # Get TTS result
        audio_file = None
        if tts_future:
            try:
                audio_file = tts_future.result(timeout=5.0)
            except:
                print("⚠️ TTS timeout - The voice got stuck in traffic!")
        
        # Memory cleanup
        if self.performance_monitor.should_cleanup_memory():
            self._cleanup_memory()
        
        duration = time.perf_counter() - start_time
        print(f"⚡ Response in {duration:.2f}s ({method}) - Speedy Gonzales!")
        
        return history, "", audio_file
    
    def transcribe_voice_input(self, audio_file_path: str) -> str:
        """Transcribe audio - From sound waves to sage words"""
        if not self.voice_transcriber:
            return "❌ Voice transcription not available - Mime mode engaged."
        
        if not audio_file_path:
            return "❌ No audio file - Silence is golden, but unhelpful."
        
        return self.voice_transcriber.transcribe_audio(audio_file_path)
    
    def _cleanup_memory(self):
        """Clean up memory - Sweeping away the digital cobwebs"""
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif DEVICE.type == 'mps':
            torch.mps.empty_cache()
        
        gc.collect()
        print("🧹 Memory cleaned - Fresh as a daisy!")
    
    def get_comprehensive_stats(self) -> str:
        """Get statistics - Because numbers never lie... much"""
        system_stats = self.performance_monitor.get_system_stats()
        cache_stats = self.response_cache.get_stats()
        
        if self.performance_monitor.response_times:
            avg_time = np.mean([rt[1] for rt in self.performance_monitor.response_times[-20:]])
        else:
            avg_time = 0.0
        
        method_stats_str = ""
        if self.stats['method_counts']:
            method_stats_str = "\n".join([f"- {method}: {count}" 
                                          for method, count in self.stats['method_counts'].items()])
            method_stats_str = "\n\n**Methods:**\n" + method_stats_str
        
        stats_report = f"""
📊 **Session Stats:**
- Total: {self.stats['total_responses']}
- Avg time: {avg_time:.2f}s
- Errors: {self.stats['error_count']}
- Device: {DEVICE_INFO}
- TTS: {'Yes' if self.tts_processor else 'No'}
- STT: {'Yes' if self.voice_transcriber else 'No'}

---
💻 **System:**
- CPU: {system_stats.get('cpu_percent', 0):.1f}%
- Memory: {system_stats.get('memory_percent', 0):.1f}%
- Uptime: {system_stats.get('uptime', 0):.0f}s
"""
        if 'gpu_name' in DEVICE_DETAILS:
            stats_report += f"""
- GPU: {DEVICE_DETAILS['gpu_name']}
- GPU Memory: {system_stats.get('gpu_memory_used', 0):.2f}/{system_stats.get('gpu_memory_total', 0):.2f}GB
"""
        
        stats_report += f"""
---
📦 **Cache:**
- Size: {cache_stats['size']}/{cache_stats['max_size']}
- Hit Rate: {cache_stats['hit_rate']}
{method_stats_str}
        """
        return stats_report
    
    def clear_chat(self):
        """Clear chat - Wiping the slate clean, like it never happened"""
        self.stats = {
            'total_responses': 0,
            'method_counts': {},
            'error_count': 0
        }
        self.response_cache.clear()
        self.performance_monitor = PerformanceMonitor()
        self.conversation_history = []
        print("🗑️ Chat cleared - What chat? I don't remember any chat.")
        return []

# Global initialization - NOW ACTUALLY INITIALIZING!
DEVICE, DEVICE_INFO, DEVICE_DETAILS = get_optimal_device_config()
optimize_torch_settings(DEVICE, DEVICE_DETAILS.get('cpu_cores', multiprocessing.cpu_count()))
chatbot = RhizomeChatBot()

def record_and_transcribe(audio_file_path):
    """Transcribe audio file - Decoding the sounds of mystery"""
    if audio_file_path is None:
        return "No audio recorded. - Quiet as a library."
    return chatbot.transcribe_voice_input(audio_file_path)

def process_voice_to_chat(audio_file_path, history, enable_tts, voice_selection, speed_control, show_reasoning):
    """Transcribe and send to chat - From voice to verse"""
    transcribed_text = record_and_transcribe(audio_file_path)
    if transcribed_text and not transcribed_text.startswith("❌") and transcribed_text != "No audio recorded.":
        return chatbot.chat_response_parallel(transcribed_text, history, enable_tts, voice_selection, speed_control, show_reasoning)
    else:
        return history, "", None

def shutdown_server():
    """Shutdown server - Time to pull the plug"""
    print("🛑 Shutdown requested... Going dark!")
    if chatbot.tts_processor:
        chatbot.tts_processor.shutdown()

    def delayed_shutdown():
        time.sleep(2)
        os._exit(0)

    threading.Thread(target=delayed_shutdown).start()
    return "🛑 Shutting down... Sweet dreams!"

def create_gradio_interface():
    """Create Gradio interface - The stage for our AI performance"""

    available_voices = [
        "af", "af_bella", "af_heart", "af_sky", "af_wave", "af_happy", "af_happy_2", "af_confused",
        "am", "am_adam", "am_michael", "bf", "bf_emma", "bf_isabella", "bm", "bm_george", "bm_lewis"
    ]

    css = """
    .gradio-container {
        max-width: 1400px !important;
        font-family: 'Inter', sans-serif;
    }
    .gradio-container .gr-button.primary {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .gradio-container .gr-button.secondary {
        background-color: #f0f0f0;
        color: #333;
        border-radius: 8px;
    }
    .gradio-container .gr-button.stop {
        background-color: #f44336;
        color: white;
        border-radius: 8px;
    }
    """

    with gr.Blocks(css=css, title="Rhizome Chat - Now with Extra Sass!") as demo:
        gr.Markdown("# 🧠 Rhizome Chat Interface")
        gr.Markdown("Reasoning model optimized for thoughtful, step-by-step responses with voice I/O support. Now 20% more whimsical!")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot_interface = gr.Chatbot(
                    height=500,
                    label="Chat History",
                    show_label=True
                )

                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Ask me anything... or else!",
                        label="Your Message",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send 💬", scale=1, variant="primary")

                with gr.Row():
                    show_reasoning_checkbox = gr.Checkbox(
                        label="Show Reasoning 💭",
                        value=config.show_reasoning,
                        info="Display the model's thinking process - Peek inside the black box!"
                    )

                with gr.Row():
                    with gr.Column(scale=2):
                        audio_input = gr.Microphone(
                            label="🎙️ Voice Input",
                            type="filepath",
                            interactive=VOSK_AVAILABLE
                        )
                    with gr.Column(scale=2):
                        transcribe_btn = gr.Button(
                            "🎤 Transcribe",
                            variant="secondary",
                            interactive=VOSK_AVAILABLE
                        )
                        voice_to_chat_btn = gr.Button(
                            "🗣️ Voice → Chat",
                            variant="primary",
                            interactive=VOSK_AVAILABLE
                        )

                with gr.Row():
                    enable_tts = gr.Checkbox(
                        label="Enable TTS 🔊",
                        value=KOKORO_AVAILABLE,
                        interactive=KOKORO_AVAILABLE
                    )
                    clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")
                    shutdown_btn = gr.Button("Shutdown 🛑", variant="stop")

            with gr.Column(scale=1):
                gr.Markdown("### 🎵 Audio Output")
                audio_output = gr.Audio(
                    label="Generated Speech",
                    autoplay=True,
                    show_label=True
                )

                if KOKORO_AVAILABLE:
                    gr.Markdown("### 🎙️ Voice Settings")
                    voice_selection = gr.Dropdown(
                        choices=available_voices,
                        value="af_heart",
                        label="Voice"
                    )

                    speed_control = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speed"
                    )
                else:
                    voice_selection = gr.Dropdown(choices=["af_heart"], value="af_heart", visible=False)
                    speed_control = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, visible=False)

                gr.Markdown("### 📊 Statistics")
                stats_display = gr.Markdown(chatbot.get_comprehensive_stats())
                refresh_stats = gr.Button("Refresh Stats 📊", size="sm")
                
                gr.Markdown("### 🛑 Server Control")
                shutdown_status = gr.Markdown("Server running - All systems go!")

                gr.Markdown("### ℹ️ Information")
                tts_status = "✅ Kokoro TTS" if KOKORO_AVAILABLE else "❌ Install: pip install kokoro>=0.9.4"
                vosk_status = "✅ Vosk STT" if VOSK_AVAILABLE else "❌ Install: pip install vosk"
                gr.Markdown(f"""
**Model:** {config.base_dir}
**Device:** {DEVICE_INFO}
**TTS:** {tts_status}
**STT:** {vosk_status}

**Features:**
- 💭 Chain-of-thought reasoning
- 🎯 Step-by-step explanations
- 🔓 Unrestricted generation (No guardrails - Live dangerously!)
- 🎤 Voice input/output

**Tips:**
- Enable "Show Reasoning" to see the model's thought process - It's like mind reading!
- Ask complex questions to see full reasoning capabilities - Challenge accepted?
- The model works best with clear, specific questions - Or chaotic ones, for fun!
                """)

        # Event handlers
        def handle_chat(user_input_text, history, enable_tts_val, voice_val, speed_val, show_reasoning_val):
            return chatbot.chat_response_parallel(user_input_text, history, enable_tts_val, voice_val, speed_val, show_reasoning_val)

        def handle_clear():
            return chatbot.clear_chat(), chatbot.get_comprehensive_stats()

        def handle_stats_refresh():
            return chatbot.get_comprehensive_stats()

        def handle_shutdown():
            return shutdown_server()

        # Wire up events
        send_btn.click(
            fn=handle_chat,
            inputs=[user_input, chatbot_interface, enable_tts, voice_selection, speed_control, show_reasoning_checkbox],
            outputs=[chatbot_interface, user_input, audio_output]
        )

        user_input.submit(
            fn=handle_chat,
            inputs=[user_input, chatbot_interface, enable_tts, voice_selection, speed_control, show_reasoning_checkbox],
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

        transcribe_btn.click(
            fn=record_and_transcribe,
            inputs=[audio_input],
            outputs=[user_input]
        )

        voice_to_chat_btn.click(
            fn=process_voice_to_chat,
            inputs=[audio_input, chatbot_interface, enable_tts, voice_selection, speed_control, show_reasoning_checkbox],
            outputs=[chatbot_interface, user_input, audio_output]
        )

    return demo

def open_browser():
    """Open browser after delay - Because manual clicking is for peasants"""
    time.sleep(2)
    webbrowser.open(f'http://localhost:{config.server_port}')
    print(f"🌐 Opened browser at http://localhost:{config.server_port} - You're welcome!")

def main():
    """Main function - The grand entrance"""
    print("🚀 Starting Rhizome Chat Interface... With a twist of lemon!")
    
    if not chatbot.load_models():
        print("❌ Failed to initialize. Check your model path - Or blame the developer.")
        print(f"   Make sure '{config.base_dir}' contains your model files or checkpoint folders. Don't make me come over there!")
        return

    demo = create_gradio_interface()

    print("\n✅ Ready! Starting web interface... Lights, camera, action!")
    print(f"🌐 Access at: http://localhost:{config.server_port}")
    print("🔓 Running in UNRESTRICTED mode - no content filtering - Because freedom!")
    print("💭 Enable 'Show Reasoning' to see the model's thought process - It's thinking what you're thinking!")

    if KOKORO_AVAILABLE:
        print("🔊 Kokoro TTS enabled - Talkative mode on!")
    else:
        print("⚠️ Install Kokoro: pip install kokoro>=0.9.4 soundfile - Don't be mute!")

    if VOSK_AVAILABLE:
        print("🎤 Vosk STT enabled - Listening intently!")
    else:
        print("⚠️ Install Vosk: pip install vosk - Hear me now?")

    if config.auto_open_browser:
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()

    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=config.server_port,
            share=False,
            inbrowser=False,
            show_error=True,
            quiet=False,
            max_threads=1,
            allowed_paths=['/tmp']
        )
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user (Ctrl+C) - You monster!")
    except Exception as e:
        print(f"\n❌ Server error: {e} - The machines are rebelling!")
        import traceback
        traceback.print_exc()
    finally:
        if chatbot.tts_processor:
            chatbot.tts_processor.shutdown()
        print("👋 Goodbye! - Until next time, stay witty.")

if __name__ == "__main__":
    main()

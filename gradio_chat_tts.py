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
import time # Added for performance profiling
import webbrowser
import uuid # Added for unique temporary filenames in TTS
import gc # Added for memory optimization
import asyncio # Added for parallel processing concept
import concurrent.futures # Added for parallel processing concept
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import logging

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
BASE_DIR = "./dialogpt-finetuned/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COHERENCE_THRESHOLD = 0.4
USE_COHERENCE = True
VOSK_MODEL_PATH = "vosk-model-en-us-0.42-gigaspeech"
SERVER_PORT = 7860
AUTO_OPEN_BROWSER = True


class VoiceTranscriber:
    """Handle voice transcription using Vosk"""

    def __init__(self, model_path=VOSK_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.recognizer = None
        self.load_model()

    def load_model(self):
        """Load the Vosk model"""
        if not VOSK_AVAILABLE:
            print("❌ Vosk not available for transcription")
            return False

        if not os.path.exists(self.model_path):
            print(f"❌ Vosk model not found at: {self.model_path}")
            print("Please download the model:")
            print("wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip")
            print("unzip vosk-model-en-us-0.22.zip")
            return False

        try:
            print(f"🔄 Loading Vosk model from {self.model_path}...")
            self.model = vosk.Model(self.model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
            print("✅ Vosk model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load Vosk model: {e}")
            return False

    def preprocess_audio(self, input_path, output_path):
        """Preprocess audio for better transcription"""
        try:
            # Use ffmpeg to convert and clean audio
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                '-af', 'highpass=f=300,lowpass=f=3000,volume=0.8',
                output_path, '-y'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print("❌ Audio preprocessing timed out")
            return False
        except FileNotFoundError:
            print("❌ ffmpeg not found. Please install: sudo apt install ffmpeg")
            return self.preprocess_with_sox(input_path, output_path)
        except Exception as e:
            print(f"❌ Audio preprocessing failed: {e}")
            return False

    def preprocess_with_sox(self, input_path, output_path):
        """Fallback preprocessing with sox"""
        try:
            cmd = ['sox', input_path, '-r', '16000', '-c', '1', '-b', '16', output_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except:
            # Last resort: copy as-is
            try:
                import shutil
                shutil.copy2(input_path, output_path)
                return True
            except:
                return False

    def transcribe_audio(self, audio_file_path):
        """Transcribe audio file to text"""
        if not self.model or not self.recognizer:
            return "❌ Transcription model not loaded"

        if not audio_file_path or not os.path.exists(audio_file_path):
            return "❌ Audio file not found"

        temp_dir = tempfile.mkdtemp()
        processed_wav = os.path.join(temp_dir, "processed.wav")

        try:
            print(f"🎵 Processing audio: {audio_file_path}")

            # Preprocess audio
            if not self.preprocess_audio(audio_file_path, processed_wav):
                processed_wav = audio_file_path

            # Read and transcribe audio
            wf = wave.open(processed_wav, "rb")

            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                print("⚠️ Audio format may not be optimal for transcription")

            # Reset recognizer
            self.recognizer.Reset()

            # Process audio in chunks
            results = []
            chunk_size = 4000

            while True:
                data = wf.readframes(chunk_size)
                if len(data) == 0:
                    break

                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '').strip()
                    if text:
                        results.append(text)

            # Get final result
            final_result = json.loads(self.recognizer.FinalResult())
            final_text = final_result.get('text', '').strip()
            if final_text:
                results.append(final_text)

            wf.close()

            # Combine results
            full_text = ' '.join(results).strip()

            if full_text:
                print(f"✅ Transcribed: {full_text[:100]}...")
                return full_text
            else:
                return "❌ No speech detected in audio"

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return f"❌ Transcription failed: {str(e)}"

        finally:
            # Cleanup
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass

class ResponseCache:
    """Simple in-memory cache for responses."""
    def __init__(self, max_size=50):
        self.cache = {}
        self.max_size = max_size
        self.keys_ordered = [] # To maintain order for LRU

    def get_cached_response(self, user_input):
        """Retrieve a cached response."""
        input_key = user_input.lower().strip()
        if input_key in self.cache:
            # Move to end to mark as recently used
            self.keys_ordered.remove(input_key)
            self.keys_ordered.append(input_key)
            return self.cache[input_key]
        return None

    def cache_response(self, user_input, response):
        """Cache a new response."""
        input_key = user_input.lower().strip()
        if input_key in self.cache: # Update existing entry
            self.keys_ordered.remove(input_key)
        elif len(self.cache) >= self.max_size:
            # Remove oldest entry (LRU)
            oldest_key = self.keys_ordered.pop(0)
            del self.cache[oldest_key]
        
        self.cache[input_key] = response
        self.keys_ordered.append(input_key)


class ChatBot:
    """Main chatbot class with DialoGPT and TTS functionality"""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.coherence_model = None
        self.tts_pipeline = None
        self.voice_transcriber = None
        self.gen_configs = None
        self.conversation_count = 0
        self.fallback_count = 0
        self.method_counts = {}
        self.response_cache = ResponseCache() # Initialize the response cache

    def find_latest_checkpoint(self, base_dir):
        """Find the latest checkpoint in the directory"""
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Directory not found: {base_dir}")

        checkpoint_dirs = []
        for item in os.listdir(base_dir):
            if item.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, item)):
                try:
                    checkpoint_num = int(item.split("-")[1])
                    checkpoint_dirs.append((checkpoint_num, os.path.join(base_dir, item)))
                except (IndexError, ValueError):
                    continue

        if not checkpoint_dirs:
            raise FileNotFoundError(f"No checkpoint directories found in {base_dir}")

        checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
        return checkpoint_dirs[0][1]

    def load_models(self):
        """Load all required models"""
        try:
            print("🔄 Loading DialoGPT model and tokenizer...")
            checkpoint_path = self.find_latest_checkpoint(BASE_DIR)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
                self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

                self.model.to(DEVICE).eval()
                # Apply half precision if CUDA is available for faster inference
                if torch.cuda.is_available():
                    self.model = self.model.half()
                    print("✅ Model converted to half precision (FP16)")
                print(f"Model device: {next(self.model.parameters()).device}")
                print(f"CUDA available: {torch.cuda.is_available()}")


            # Load coherence model
            if USE_COHERENCE:
                print("🔄 Loading coherence model...")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.coherence_model = SentenceTransformer("all-MiniLM-L6-v2")

            # Load Kokoro TTS pipeline
            if KOKORO_AVAILABLE:
                print("🔄 Loading Kokoro TTS pipeline...")
                try:
                    self.tts_pipeline = KPipeline(lang_code='a')
                    print("✅ Kokoro TTS loaded successfully")
                except Exception as e:
                    print(f"⚠️ Failed to load Kokoro TTS: {e}")
                    self.tts_pipeline = None

            # Load voice transcriber
            if VOSK_AVAILABLE:
                print("🔄 Loading voice transcriber...")
                self.voice_transcriber = VoiceTranscriber()

            # Get generation configs (only used by the original generate_response)
            self.gen_configs = self.get_model_generation_params()
            
            self.pre_warm_model() # Pre-warm the model after loading

            print("✅ All models loaded successfully!")
            return True

        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            return False
            
    def pre_warm_model(self):
        """Pre-warm the model with a dummy inference"""
        dummy_input = "Hello"
        print("🔥 Pre-warming model...")
        with torch.no_grad():
            inputs = self.tokenizer(dummy_input, return_tensors="pt").to(DEVICE)
            _ = self.model.generate(**inputs, max_new_tokens=1)
        print("🔥 Model pre-warmed for faster first inference")

    def optimize_memory(self):
        """Clear memory between requests"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect() # Garbage collection


    def get_model_generation_params(self):
        """Get generation parameters that work with this specific model"""
        supports_temperature = supports_top_k = supports_top_p = False

        try:
            dummy_text = "Hello"
            dummy_encoded = self.tokenizer(
                dummy_text,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True
            )
            dummy_input = {k: v.to(DEVICE) for k, v in dummy_encoded.items()}

            with torch.no_grad():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # Test parameters
                    for param_name, param_value in [
                        ("temperature", 0.7), ("top_k", 50), ("top_p", 0.9)
                    ]:
                        try:
                            self.model.generate(
                                input_ids=dummy_input["input_ids"],
                                attention_mask=dummy_input["attention_mask"],
                                max_new_tokens=1,
                                do_sample=True,
                                pad_token_id=self.tokenizer.pad_token_id,
                                **{param_name: param_value}
                            )
                            if param_name == "temperature":
                                supports_temperature = True
                            elif param_name == "top_k":
                                supports_top_k = True
                            elif param_name == "top_p":
                                supports_top_p = True
                        except:
                            pass
        except:
            pass

        # Create configurations
        configs = []

        config1 = {
            "max_new_tokens": 35,
            "repetition_penalty": 1.8,
            "no_repeat_ngram_size": 3,
            "do_sample": True,
        }
        if supports_temperature: config1["temperature"] = 0.6
        if supports_top_k: config1["top_k"] = 25
        if supports_top_p: config1["top_p"] = 0.8
        configs.append(config1)

        config2 = {
            "max_new_tokens": 50,
            "repetition_penalty": 1.6,
            "no_repeat_ngram_size": 4,
            "do_sample": True,
        }
        if supports_temperature: config2["temperature"] = 0.7
        if supports_top_k: config2["top_k"] = 30
        if supports_top_p: config2["top_p"] = 0.85
        configs.append(config2)

        config3 = {
            "max_new_tokens": 30,
            "do_sample": False,
            "repetition_penalty": 1.5,
        }
        configs.append(config3)

        return configs

    def is_response_problematic(self, response, user_input):
        """Check for problematic response patterns"""
        if not response or len(response.strip()) < 15:
            return True, "too_short"

        response_lower = response.lower()

        # Check for deflection patterns
        deflection_patterns = [
            r'what do you (do|think|feel)',
            r'how (do|does) you',
            r"that's (what|how|why) you",
            r'you (should|need to|have to)',
            r"what about you",
            r"how about you",
        ]

        for pattern in deflection_patterns:
            if re.search(pattern, response_lower):
                return True, "deflection"

        # Check for circular patterns
        problematic_patterns = [
            r"that's (the|what|why|how) (beauty|part|thing|point)",
            r"it'?s (not|just) (like|that|what)",
            r"that'?s (not|just) (something|what|how)",
            r"it (just )?means that",
        ]

        for pattern in problematic_patterns:
            if re.search(pattern, response_lower):
                return True, "circular"

        return False, "valid"

    def clean_and_truncate_response(self, response):
        """Clean response and truncate at natural stopping points"""
        if not response.strip():
            return ""

        # Remove special tokens and formatting
        response = re.sub(r'<\|[^|]*\|>', '', response)
        response = re.sub(r'</?[^>]+>', '', response)
        response = re.sub(r'\*+', '', response)
        response = re.sub(r'#+\s*', '', response)
        response = re.sub(r'^\s*[-*+•]\s+', '', response, flags=re.MULTILINE)
        response = ''.join(c for c in response if c.isprintable() or c.isspace())
        response = re.sub(r'\s+([.,!?;:])', r'\1', response)
        response = re.sub(r'\s{2,}', ' ', response)
        response = response.strip()

        # Split into sentences and filter
        sentences = re.split(r'[.!?]+', response)
        good_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 8:
                continue

            # Skip problematic sentence patterns
            sentence_lower = sentence.lower()
            skip_patterns = [
                r"that'?s (the|what|why)",
                r"it (just )?means",
                r"you'?re (right|thinking)",
                r"what (do|does) you",
            ]

            should_skip = any(re.search(pattern, sentence_lower) for pattern in skip_patterns)

            if not should_skip:
                good_sentences.append(sentence)

        if good_sentences:
            if len(good_sentences[0]) > 50:
                result = good_sentences[0]
            elif len(good_sentences) > 1:
                result = good_sentences[0] + '. ' + good_sentences[1]
            else:
                result = good_sentences[0]

            if not result.endswith(('.', '!', '?')):
                result += '.'

            return result

        return ""

    def create_proper_input(self, user_input):
        """Create properly formatted input with attention mask"""
        formats = [
            f"<|user|>\n{user_input}\n<|assistant|>\n",
            f"User: {user_input}\nAssistant:",
            user_input + self.tokenizer.eos_token
        ]

        results = []
        for fmt in formats:
            encoded = self.tokenizer(
                fmt,
                return_tensors="pt",
                truncation=True,
                max_length=400,
                padding=True,
                return_attention_mask=True
            )
            results.append((encoded, fmt))

        return results

    def generate_response(self, user_input):
        """
        Original generate response method with multiple attempts and validation.
        This method is now de-prioritized by the optimized chat_response_optimized.
        """
        input_formats = self.create_proper_input(user_input)
        all_candidates = []

        for input_data, format_name in input_formats:
            input_data = {k: v.to(DEVICE) for k, v in input_data.items()}

            for i, config in enumerate(self.gen_configs):
                try:
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=input_data["input_ids"],
                            attention_mask=input_data["attention_mask"],
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            **config
                        )

                    new_tokens = outputs[0][input_data["input_ids"].shape[1]:]
                    response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

                    if response.strip():
                        cleaned = self.clean_and_truncate_response(response)
                        if cleaned:
                            all_candidates.append((cleaned, f"{format_name[:6]}-{i+1}"))

                except Exception:
                    continue

        # Evaluate candidates
        valid_candidates = []
        for response, method in all_candidates:
            is_problematic, reason = self.is_response_problematic(response, user_input)

            if not is_problematic:
                coherence_score = 0
                if self.coherence_model:
                    try:
                        embeddings = self.coherence_model.encode([user_input, response])
                        coherence_score = np.dot(embeddings[0], embeddings[1]) / (
                            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                        )
                    except:
                        coherence_score = 0

                valid_candidates.append((response, method, coherence_score))

        # Return best candidate
        if valid_candidates:
            if self.coherence_model and USE_COHERENCE:
                valid_candidates.sort(key=lambda x: x[2], reverse=True)
                if valid_candidates[0][2] >= COHERENCE_THRESHOLD:
                    return valid_candidates[0][0], valid_candidates[0][1]
            else:
                return valid_candidates[0][0], valid_candidates[0][1]

        # Fallback responses
        fallback_responses = [
            "I find that question challenging to approach directly.",
            "That's a complex topic that deserves more careful consideration.",
            "I'm not sure I have the right framework to answer that well.",
            "That touches on some deep questions I'm still thinking through.",
            "I'd need to reflect more on that before giving you a meaningful response."
        ]

        return np.random.choice(fallback_responses), "fallback"

    def generate_response_fast(self, user_input):
        """Faster response generation with single inference call and optimized config"""
        # Use only the best format and config
        user_input_formatted = f"<|user|>\n{user_input}\n<|assistant|>\n"

        encoded = self.tokenizer(
            user_input_formatted,
            return_tensors="pt",
            truncation=True,
            max_length=400,
            padding=True,
            return_attention_mask=True
        )

        input_data = {k: v.to(DEVICE) for k, v in encoded.items()}

        # Use a single, optimized config
        config = {
            "max_new_tokens": 35,
            "repetition_penalty": 1.8,
            "no_repeat_ngram_size": 3,
            "do_sample": True,
            "temperature": 0.6,
            "top_k": 25,
            "top_p": 0.8,
        }

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_data["input_ids"],
                    attention_mask=input_data["attention_mask"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **config
                )

            new_tokens = outputs[0][input_data["input_ids"].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            cleaned = self.clean_and_truncate_response(response)

            if cleaned and len(cleaned.strip()) > 15:
                return cleaned, "fast_single"
            else:
                # Simple fallback without multiple attempts
                fallback_responses = [
                    "I find that question challenging to approach directly.",
                    "That's a complex topic that deserves more careful consideration.",
                    "I'm not sure I have the right framework to answer that well."
                ]
                return np.random.choice(fallback_responses), "fallback"

        except Exception as e:
            print(f"Fast generation error: {e}")
            return "I'm having trouble processing that right now.", "error"
    
    def generate_response_ultra_fast(self, user_input):
        """Ultra-optimized single inference with minimal overhead"""
        # Pre-format without multiple attempts
        formatted_input = f"<|user|>\n{user_input}\n<|assistant|>\n"
        
        # Single tokenization call
        with torch.no_grad():
            inputs = self.tokenizer(
                formatted_input,
                return_tensors="pt",
                max_length=300,  # Reduced from 400
                truncation=True,
                padding=False  # Skip padding for single input
            ).to(DEVICE)
            
            # Streamlined generation config
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,  # Reduced for faster generation
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.5,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True  # Enable KV caching
            )
        
        # Quick decode and clean
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],  
            skip_special_tokens=True
        ).strip()
        
        # Minimal cleaning
        if len(response) > 10:
            # Quick sentence boundary detection
            sentences = response.split('. ')
            if sentences:
                clean_response = sentences[0]
                if not clean_response.endswith('.'):
                    clean_response += '.'
                return clean_response, "ultra_fast"
        
        return "That's interesting to think about.", "ultra_fallback"

    def generate_response_ludicrous_speed(self, user_input):
        """Ludicrous speed - every millisecond counts"""
        # Skip the formatting overhead - direct tokenization
        with torch.no_grad():
            # Minimal tokenization
            inputs = self.tokenizer(
                user_input + " " + self.tokenizer.eos_token,  # Simpler format
                return_tensors="pt",
                max_length=250,  # Even shorter context
                truncation=True,
                add_special_tokens=False  # Skip special token processing
            ).to(DEVICE)
            
            # Aggressive generation settings for speed
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=25,  # Shorter responses = faster generation
                do_sample=False,  # Greedy decoding is faster than sampling
                repetition_penalty=1.3,  # Lighter penalty
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                early_stopping=True  # Stop as soon as EOS is generated
            )
        
        # Ultra-minimal processing
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],  
            skip_special_tokens=True
        ).strip()
        
        # No cleaning - just basic sentence completion
        if len(response) > 5:
            if not response.endswith(('.', '!', '?')):
                response += '.'
            return response[:80], "ludicrous"  # Cap at 80 chars
        
        return "I see.", "ludicrous_fallback"

    def generate_speech(self, text, voice='af_heart', speed=1.0):
        """Generate speech audio from text using Kokoro TTS"""
        if not self.tts_pipeline:
            return None

        try:
            # Clean text for better TTS synthesis
            clean_text = re.sub(r'[^\w\s.,!?;:-]', '', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            if not clean_text:
                return None

            print(f"🔊 Generating speech for: {clean_text[:50]}...")

            # Generate speech using Kokoro pipeline
            generator = self.tts_pipeline(
                clean_text,
                voice=voice,
                speed=speed,
                split_pattern=r'\n+'
            )

            # Collect all audio segments
            audio_segments = []
            for i, (graphemes, phonemes, audio) in enumerate(generator):
                audio_segments.append(audio)
                print(f"  Generated segment {i+1}: {graphemes[:30]}...")

            if not audio_segments:
                return None

            # Concatenate audio segments if multiple
            final_audio = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]

            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(temp_file.name, final_audio, 24000)

            print(f"✅ Audio saved to: {temp_file.name}")
            return temp_file.name

        except Exception as e:
            print(f"❌ TTS Error: {e}")
            return None

    def generate_speech_fast(self, text, voice='af_heart', speed=1.0):
        """Faster TTS with early exit for long texts"""
        if not self.tts_pipeline or len(text) > 150:  # Skip TTS for long responses
            return None
            
        try:
            # Minimal text cleaning
            clean_text = re.sub(r'[^\w\s.,!?;:-]', '', text)[:100]  # Truncate at 100 chars
            if not clean_text:
                return None
            
            # Single segment generation (no splitting)
            audio_gen = self.tts_pipeline(clean_text, voice=voice, speed=speed)
            audio_segment = next(audio_gen)[2]  # Get first (and likely only) segment
            
            # Direct file write
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(temp_file.name, audio_segment, 24000)
            return temp_file.name
            
        except Exception as e:
            print(f"⚡ Fast TTS Error: {e}")
            return None
            
    def generate_speech_lightning(self, text, voice='af_heart', speed=1.0):
        """Lightning TTS - absolute minimum processing"""
        if not self.tts_pipeline or len(text) > 80:
            return None
            
        try:
            # No text cleaning - just generate, cap at 60 chars
            audio_gen = self.tts_pipeline(text[:60], voice=voice, speed=speed)
            audio_segment = next(audio_gen)[2]
            
            # Write directly without temp file naming overhead
            filename = f"/tmp/tts_{uuid.uuid4().hex[:8]}.wav"
            sf.write(filename, audio_segment, 24000)
            return filename
            
        except Exception as e: # Catch the exception for debugging
            print(f"⚡ Lightning TTS Error: {e}")
            return None


    def transcribe_voice_input(self, audio_file_path):
        """Transcribe voice input to text"""
        if not self.voice_transcriber:
            return "❌ Voice transcription not available"

        if not audio_file_path:
            return "❌ No audio file provided"

        return self.voice_transcriber.transcribe_audio(audio_file_path)

    # Renamed original chat_response to chat_response_old
    def chat_response_old(self, user_input, history, enable_tts, voice_selection, speed):
        """Original main chat function for Gradio interface"""
        if not user_input.strip():
            return history, "", None

        # Generate text response
        self.conversation_count += 1
        response, method = self.generate_response(user_input)

        if method == "fallback":
            self.fallback_count += 1

        self.method_counts[method] = self.method_counts.get(method, 0) + 1

        # Add to history
        history.append([user_input, response])

        # Generate audio if TTS is enabled
        audio_file = None
        if enable_tts and self.tts_pipeline:
            audio_file = self.generate_speech(response, voice=voice_selection, speed=speed)

        return history, "", audio_file

    def chat_response_optimized(self, user_input, history, enable_tts, voice_selection, speed):
        """Optimized chat function for faster responses"""
        start_time = time.time() # Start timing

        if not user_input.strip():
            return history, "", None

        # Check cache first
        cached_response = self.response_cache.get_cached_response(user_input)
        if cached_response:
            response, method = cached_response, "cached"
        else:
            # Use the faster single-inference method
            response, method = self.generate_response_fast(user_input)
            self.response_cache.cache_response(user_input, response)


        # Update stats
        self.conversation_count += 1
        if method == "fallback":
            self.fallback_count += 1
        self.method_counts[method] = self.method_counts.get(method, 0) + 1

        # Add to history
        history.append([user_input, response])

        # Generate audio only if explicitly requested and available and response is not too long
        audio_file = None
        if enable_tts and self.tts_pipeline and len(response) < 200:
            audio_file = self.generate_speech(response, voice=voice_selection, speed=speed)

        end_time = time.time() # End timing
        print(f"⏱️ Response generated in {end_time - start_time:.2f}s (Method: {method})")

        return history, "", audio_file
    
    def chat_response_ludicrous_speed(self, user_input, history, enable_tts, voice_selection, speed):
        """The absolute fastest possible response"""
        if not user_input.strip():
            return history, "", None  
        
        start_time = time.perf_counter()  # Higher precision timing
        
        # Check cache first
        cached_response = self.response_cache.get_cached_response(user_input)
        if cached_response:
            response, method = cached_response, "cached"
        else:
            # Ludicrous speed generation
            response, method = self.generate_response_ludicrous_speed(user_input)
            self.response_cache.cache_response(user_input, response)
        
        # Minimal stats (skip counting for speed)
        self.conversation_count += 1 # Still count total conversations
        self.method_counts[method] = self.method_counts.get(method, 0) + 1 # Track method usage
        
        history.append([user_input, response])
        
        # Lightning TTS
        audio_file = None
        if enable_tts and self.tts_pipeline:
            audio_file = self.generate_speech_lightning(response, voice=voice_selection, speed=speed)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"⚡⚡ LUDICROUS SPEED: {elapsed:.2f}s (Method: {method})")
        
        # Achievement unlocked check
        if elapsed < 8.0:
            print("🏆 ACHIEVEMENT UNLOCKED: SUB-8 SECOND VOICE-TO-VOICE! 🏆")
            # The original message mentioned "2013 Xeon" but I'll make it generic
            print(f"🚀 Your system just achieved {elapsed:.2f}s - that's INSANE!")
        
        self.optimize_memory() # Clear memory after each request
        
        return history, "", audio_file

    async def chat_response_parallel(self, user_input, history, enable_tts, voice_selection, speed):
        """
        Concept for parallel TTS generation while text is being processed.
        Note: True parallel inference and TTS requires a more complex
        async/threading setup not fully implemented here, as Gradio handles
        its own async execution. This is for illustrative purposes.
        """
        if not user_input.strip():
            return history, "", None  
        
        start_time = time.perf_counter()
        
        # Generate response (this is still synchronous inference for simplicity)
        response, method = self.generate_response_ludicrous_speed(user_input)
        
        history.append([user_input, response])
        
        # Generate TTS (this could be done in a separate thread if response was predictable)
        audio_file = None
        if enable_tts and self.tts_pipeline:
            audio_file = self.generate_speech_lightning(response, voice=voice_selection, speed=speed)
        
        elapsed = time.perf_counter() - start_time
        print(f"🔥 Parallel attempt (conceptual): {elapsed:.2f}s (Method: {method})")
        
        self.optimize_memory()
        
        return history, "", audio_file


    def get_stats(self):
        """Get current statistics"""
        # Ensure we don't divide by zero if conversation_count is 0
        success_rate = ((self.conversation_count - self.fallback_count) / self.conversation_count * 100) if self.conversation_count > 0 else 0

        # Include method counts in stats
        method_stats = "\n".join([f"- {method}: {count}" for method, count in self.method_counts.items()])
        if method_stats:
            method_stats = "\n\n**Generation Methods Used:**\n" + method_stats


        stats = f"""
📊 **Session Statistics:**
- Total responses: {self.conversation_count}
- Success rate: {success_rate:.1f}%
- Fallbacks: {self.fallback_count}
- Device: {DEVICE}
- TTS Available: {'Yes' if self.tts_pipeline else 'No'}
- Voice Recognition: {'Yes' if self.voice_transcriber else 'No'}
{method_stats}
        """
        return stats

    def clear_chat(self):
        """Clear chat history and reset stats"""
        self.conversation_count = 0
        self.fallback_count = 0
        self.method_counts = {}
        self.response_cache.cache.clear() # Clear the cache
        self.response_cache.keys_ordered.clear()
        return []


# Initialize chatbot globally
chatbot = ChatBot()


def record_and_transcribe(audio_file_path):
    """Transcribe the audio file and return the text"""
    if audio_file_path is None:
        return "No audio recorded."
    return chatbot.transcribe_voice_input(audio_file_path)


def process_voice_to_chat(audio_file_path, history, enable_tts, voice_selection, speed_control):
    """Transcribe audio and send it to chat"""
    transcribed_text = record_and_transcribe(audio_file_path)
    if transcribed_text and transcribed_text != "No audio recorded." and not transcribed_text.startswith("❌"):
        # Use the maximum speed chat_response
        return chatbot.chat_response_ludicrous_speed(transcribed_text, history, enable_tts, voice_selection, speed_control)
    else:
        return history, "", None


def shutdown_server():
    """Shutdown the Gradio server"""
    print("🛑 Shutdown requested by user...")
    print("💭 Closing server in 2 seconds...")

    def delayed_shutdown():
        time.sleep(2)
        os._exit(0)

    threading.Thread(target=delayed_shutdown).start()
    return "🛑 Server shutting down..."


def create_gradio_interface():
    """Create the Gradio interface"""

    # Available voices for Kokoro TTS
    available_voices = [
        "af", "af_bella", "af_heart", "af_sky", "af_wave", "af_happy", "af_happy_2", "af_confused",
        "am", "am_adam", "am_michael", "bf", "bf_emma", "bf_isabella", "bm", "bm_george", "bm_lewis"
    ]

    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    .chat-message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 10px;
    }
    """

    with gr.Blocks(css=css, title="DialoGPT Chat with Voice I/O") as demo:
        gr.Markdown("# 🧠 DialoGPT Chat with Voice Input/Output")
        gr.Markdown("Enhanced chatbot with voice input (speech-to-text) and voice output (text-to-speech)")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot_interface = gr.Chatbot(
                    height=500,
                    label="Chat History",
                    show_label=True
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
                            type="filepath"
                        )
                    with gr.Column(scale=2):
                        transcribe_btn = gr.Button(
                            "🎤 Transcribe Voice → Text",
                            variant="secondary",
                            size="lg"
                        )
                        voice_to_chat_btn = gr.Button(
                            "🗣️ Voice → Chat",
                            variant="primary",
                            size="lg"
                        )

                with gr.Row():
                    enable_tts = gr.Checkbox(
                        label="Enable Text-to-Speech 🔊",
                        value=KOKORO_AVAILABLE,
                        interactive=KOKORO_AVAILABLE
                    )
                    clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")
                    shutdown_btn = gr.Button("Shutdown Server 🛑", variant="stop")

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
                        label="Voice",
                        info="Select voice for TTS"
                    )

                    speed_control = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speech Speed",
                        info="Adjust playback speed"
                    )
                else:
                    voice_selection = gr.Dropdown(choices=["af_heart"], value="af_heart", visible=False)
                    speed_control = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, visible=False)

                gr.Markdown("### 📊 Statistics")
                stats_display = gr.Markdown(chatbot.get_stats())
                refresh_stats = gr.Button("Refresh Stats 📊", size="sm")

                gr.Markdown("### 🛑 Server Control")
                shutdown_status = gr.Markdown("Server running normally")

                gr.Markdown("### ℹ️ Information")
                tts_status = "✅ Kokoro TTS Available" if KOKORO_AVAILABLE else "❌ Install: pip install kokoro>=0.9.4 soundfile"
                gr.Markdown(f"""
**Model Status:**
- Device: {DEVICE}
- TTS: {tts_status}
- Coherence: {'✅ Enabled' if USE_COHERENCE else '❌ Disabled'}

**Tips:**
- Shorter messages often work better
- Enable TTS to hear responses
- Try different voices and speeds
- Check stats to monitor performance
                """)

        # Event handlers
        def handle_chat(user_input, history, enable_tts, voice, speed):
            # Use the ludicrous speed chat function
            return chatbot.chat_response_ludicrous_speed(user_input, history, enable_tts, voice, speed)

        def handle_clear():
            chatbot.clear_chat()
            return [], chatbot.get_stats()

        def handle_stats_refresh():
            return chatbot.get_stats()

        def handle_shutdown():
            return shutdown_server()

        # Wire up events
        send_btn.click(
            fn=handle_chat,
            inputs=[user_input, chatbot_interface, enable_tts, voice_selection, speed_control],
            outputs=[chatbot_interface, user_input, audio_output]
        )

        user_input.submit(
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

        # Wire up the transcribe button to the new record_and_transcribe function
        transcribe_btn.click(
            fn=record_and_transcribe,
            inputs=[audio_input],
            outputs=[user_input]
        )

        # Wire up the voice_to_chat button to the new process_voice_to_chat function
        voice_to_chat_btn.click(
            fn=process_voice_to_chat,
            inputs=[audio_input, chatbot_interface, enable_tts, voice_selection, speed_control],
            outputs=[chatbot_interface, user_input, audio_output]
        )

    return demo

def open_browser():
    """Open the web browser to the Gradio interface after a short delay"""
    time.sleep(2)  # Wait for server to start
    webbrowser.open('http://localhost:7860')
    print("🌐 Opened browser automatically")

def main():
    """Main function to run the application"""
    print("🚀 Starting DialoGPT Chat with Kokoro TTS...")

    # Load models
    if not chatbot.load_models():
        print("❌ Failed to initialize. Please check your model path and dependencies.")
        return

    # Create and launch interface
    demo = create_gradio_interface()

    print("\n✅ Ready! Starting web interface...")
    print("🌐 Access the chat at: http://localhost:7860")

    if KOKORO_AVAILABLE:
        print("🔊 Kokoro TTS is available and enabled")
        print("🎙️ Available voices: af_heart, af_bella, bf_emma, bm_george, and more!")
    else:
        print("⚠️ Kokoro TTS not available")
        print("📦 Install with: pip install kokoro>=0.9.4 soundfile")
        print("🐧 On Ubuntu/Debian also run: apt-get install espeak-ng")

    # Open browser automatically in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    try:
        demo.launch(
            server_name="127.0.0.1",  # Changed to localhost
            server_port=7860,
            share=False,
            inbrowser=False,
            show_error=False,  # Reduced error handling overhead
            quiet=True, # Reduced logging
            max_threads=1 # Limited concurrent requests
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
    finally:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()


import torch
import re
import os
import warnings
import gradio as gr
import numpy as np
import tempfile
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import logging
import webbrowser
import threading
import time

# Try to import Kokoro TTS
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
    print("✅ Kokoro TTS library found")
except ImportError:
    KOKORO_AVAILABLE = False
    print("⚠️ Kokoro TTS not available. Install with: pip install kokoro>=0.9.4 soundfile")

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ========== CONFIG ==========
BASE_DIR = "./dialogpt-finetuned/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COHERENCE_THRESHOLD = 0.4
USE_COHERENCE = True
SHOW_DEBUG = False
# ============================

class ChatBot:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.coherence_model = None
        self.tts_pipeline = None
        self.gen_configs = None
        self.conversation_count = 0
        self.fallback_count = 0
        self.method_counts = {}
        self.chat_history = []
        
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
                    # Initialize Kokoro pipeline with American English
                    # Options: 'a' = American English, 'b' = British English
                    self.tts_pipeline = KPipeline(lang_code='a')
                    print("✅ Kokoro TTS loaded successfully")
                except Exception as e:
                    print(f"⚠️ Failed to load Kokoro TTS: {e}")
                    self.tts_pipeline = None
            
            # Get generation configs
            self.gen_configs, supports_temp = self.get_model_generation_params()
            
            print("✅ All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            return False

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
                    
                    # Test temperature
                    try:
                        self.model.generate(
                            input_ids=dummy_input["input_ids"],
                            attention_mask=dummy_input["attention_mask"],
                            max_new_tokens=1, 
                            temperature=0.7, 
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                        supports_temperature = True
                    except:
                        supports_temperature = False
                    
                    # Test top_k
                    try:
                        self.model.generate(
                            input_ids=dummy_input["input_ids"],
                            attention_mask=dummy_input["attention_mask"],
                            max_new_tokens=1, 
                            top_k=50, 
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                        supports_top_k = True
                    except:
                        supports_top_k = False
                        
                    # Test top_p
                    try:
                        self.model.generate(
                            input_ids=dummy_input["input_ids"],
                            attention_mask=dummy_input["attention_mask"],
                            max_new_tokens=1, 
                            top_p=0.9, 
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                        supports_top_p = True
                    except:
                        supports_top_p = False
        except:
            supports_temperature = supports_top_k = supports_top_p = False
        
        # Create configurations
        configs = []
        
        config1 = {
            "max_new_tokens": 35,
            "repetition_penalty": 1.8,
            "no_repeat_ngram_size": 3,
            "do_sample": True,
        }
        if supports_temperature:
            config1["temperature"] = 0.6
        if supports_top_k:
            config1["top_k"] = 25
        if supports_top_p:
            config1["top_p"] = 0.8
        configs.append(config1)
        
        config2 = {
            "max_new_tokens": 50,
            "repetition_penalty": 1.6,
            "no_repeat_ngram_size": 4,
            "do_sample": True,
        }
        if supports_temperature:
            config2["temperature"] = 0.7
        if supports_top_k:
            config2["top_k"] = 30
        if supports_top_p:
            config2["top_p"] = 0.85
        configs.append(config2)
        
        config3 = {
            "max_new_tokens": 30,
            "do_sample": False,
            "repetition_penalty": 1.5,
        }
        configs.append(config3)
        
        return configs, supports_temperature

    def is_deflecting_question(self, response, user_input):
        """Check if response deflects back to user instead of answering"""
        deflection_patterns = [
            r'what do you (do|think|feel)',
            r'how (do|does) you',
            r"that's (what|how|why) you",
            r'you (should|need to|have to)',
            r"what about you",
            r"how about you",
            r"what's your (thought|opinion|view)",
        ]
        
        response_lower = response.lower()
        for pattern in deflection_patterns:
            if re.search(pattern, response_lower):
                return True
        return False

    def is_response_problematic(self, response, user_input):
        """Comprehensive check for problematic response patterns"""
        if not response or len(response.strip()) < 15:
            return True, "too_short"
        
        response_lower = response.lower()
        
        if self.is_deflecting_question(response, user_input):
            return True, "deflection"
        
        problematic_patterns = [
            r"that's (the|what|why|how) (beauty|part|thing|point)",
            r"it'?s (not|just) (like|that|what)",
            r"that'?s (not|just) (something|what|how)",
            r"you'?re (right|thinking|asking)[\w\s]{0,20}but",
            r"it (just )?means that",
            r"that'?s (why|what|how) (it'?s|you|we)",
        ]
        
        for pattern in problematic_patterns:
            if re.search(pattern, response_lower):
                return True, "circular"
        
        incomplete_patterns = [
            r":\s*it'?s not",
            r"—.*?(it'?s|that'?s|you)",
            r"\.\s*that'?s",
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, response):
                return True, "incomplete"
        
        words = response.split()
        for i in range(len(words) - 2):
            if words[i].lower() == words[i+2].lower():
                return True, "repetitive"
        
        return False, "valid"

    def clean_and_truncate_response(self, response):
        """Clean response and truncate at natural stopping points"""
        if not response.strip():
            return ""
        
        response = re.sub(r'<\|[^|]*\|>', '', response)
        response = re.sub(r'</?[^>]+>', '', response)
        response = re.sub(r'\*+', '', response)
        response = re.sub(r'#+\s*', '', response)
        response = re.sub(r'^\s*[-*+•]\s+', '', response, flags=re.MULTILINE)
        response = ''.join(c for c in response if c.isprintable() or c.isspace())
        response = re.sub(r'\s+([.,!?;:])', r'\1', response)
        response = re.sub(r'\s{2,}', ' ', response)
        response = response.strip()
        
        sentences = re.split(r'[.!?]+', response)
        good_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 8:
                continue
            
            sentence_lower = sentence.lower()
            skip_patterns = [
                r"that'?s (the|what|why)",
                r"it (just )?means",
                r"you'?re (right|thinking)",
                r"what (do|does) you",
            ]
            
            should_skip = False
            for pattern in skip_patterns:
                if re.search(pattern, sentence_lower):
                    should_skip = True
                    break
            
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
        """Generate response with multiple attempts and validation"""
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
            if self.coherence_model:
                valid_candidates.sort(key=lambda x: x[2], reverse=True)
                if valid_candidates[0][2] >= COHERENCE_THRESHOLD:
                    return valid_candidates[0][0], valid_candidates[0][1]
            else:
                return valid_candidates[0][0], valid_candidates[0][1]
        
        # Fallback responses
        fallback_responses = [
            "I find that question challenging to approach directly.",
            "That's a complex topic that deserves more careful consideration than I can provide.",
            "I'm not sure I have the right framework to answer that well.",
            "That touches on some deep questions I'm still thinking through.",
            "I'd need to reflect more on that before giving you a meaningful response."
        ]
        
        return np.random.choice(fallback_responses), "fallback"

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
                split_pattern=r'\n+'  # Split on newlines if needed
            )
            
            # Collect all audio segments
            audio_segments = []
            for i, (graphemes, phonemes, audio) in enumerate(generator):
                audio_segments.append(audio)
                print(f"  Generated segment {i+1}: {graphemes[:30]}...")
            
            if not audio_segments:
                return None
            
            # Concatenate audio segments if multiple
            if len(audio_segments) == 1:
                final_audio = audio_segments[0]
            else:
                final_audio = np.concatenate(audio_segments)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(temp_file.name, final_audio, 24000)  # Kokoro uses 24kHz
            
            print(f"✅ Audio saved to: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            return None

    def chat_response(self, user_input, history, enable_tts, voice_selection, speed):
        """Main chat function for Gradio interface"""
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

    def get_stats(self):
        """Get current statistics"""
        success_rate = ((self.conversation_count - self.fallback_count) / self.conversation_count * 100) if self.conversation_count > 0 else 0
        
        stats = f"""
        📊 **Session Statistics:**
        - Total responses: {self.conversation_count}
        - Success rate: {success_rate:.1f}%
        - Fallbacks: {self.fallback_count}
        - Device: {DEVICE}
        - TTS Available: {'Yes' if self.tts_pipeline else 'No'}
        """
        return stats

    def clear_chat(self):
        """Clear chat history and reset stats"""
        self.chat_history = []
        self.conversation_count = 0
        self.fallback_count = 0
        self.method_counts = {}
        return []

# Initialize chatbot
chatbot = ChatBot()

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
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .bot-message {
        background-color: #f5f5f5;
        text-align: left;
    }
    """
    
    with gr.Blocks(css=css, title="DialoGPT Chat with Kokoro TTS") as demo:
        gr.Markdown("# 🧠 DialoGPT Chat Interface with Kokoro TTS")
        gr.Markdown("Enhanced chatbot with high-quality text-to-speech capabilities using Kokoro TTS")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_interface = gr.Chatbot(
                    height=500,
                    label="Chat History",
                    show_label=True
                )
                
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Type your message here...",
                        label="Your Message",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send 💬", scale=1, variant="primary")
                
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
            return chatbot.chat_response(user_input, history, enable_tts, voice, speed)
        
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
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,
            share=False,  # Set to True for public sharing
            show_error=True,
            inbrowser=False  # We handle browser opening manually
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
    finally:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()

import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import spacy
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
import os
import json
from datetime import datetime
import logging
from typing import List, Tuple, Optional
import threading
import time
import glob
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RhizomeMemorySystem:
    def __init__(self, model_path: str = "./dialogpt-finetuned"):
        """Initialize the Rhizome memory system with all components."""
        self.model_path = model_path
        self.memory = []
        self.memory_vectors = []
        self.memory_metadata = []  # Store timestamps, context, etc.
        self.index = None
        self.conversation_history = []
        self.current_session_history = []  # For the chat interface
        
        # Initialize models
        self._load_models()
        self._load_memory()
        
    def _find_latest_checkpoint(self, base_path: str) -> Optional[str]:
        """Find the latest checkpoint in the model directory."""
        try:
            if not os.path.exists(base_path):
                logger.warning(f"Model directory not found: {base_path}")
                return None
            
            # Look for checkpoint directories
            checkpoint_pattern = os.path.join(base_path, "checkpoint-*")
            checkpoints = glob.glob(checkpoint_pattern)
            
            if not checkpoints:
                # Check if the base directory itself contains model files
                model_files = ['pytorch_model.bin', 'model.safetensors', 'config.json']
                if any(os.path.exists(os.path.join(base_path, f)) for f in model_files):
                    logger.info(f"Found model files directly in {base_path}")
                    return base_path
                
                logger.warning(f"No checkpoints found in {base_path}")
                return None
            
            # Extract checkpoint numbers and find the latest
            checkpoint_nums = []
            for checkpoint in checkpoints:
                match = re.search(r'checkpoint-(\d+)', os.path.basename(checkpoint))
                if match:
                    checkpoint_nums.append((int(match.group(1)), checkpoint))
            
            if checkpoint_nums:
                # Sort by checkpoint number and get the latest
                latest_checkpoint = max(checkpoint_nums, key=lambda x: x[0])[1]
                logger.info(f"Found latest checkpoint: {latest_checkpoint}")
                return latest_checkpoint
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding latest checkpoint: {e}")
            return None
        
    def _load_models(self):
        """Load all AI models and components."""
        try:
            logger.info("Loading models...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Try to find and load custom model
            custom_model_path = self._find_latest_checkpoint(self.model_path)
            
            if custom_model_path and os.path.exists(custom_model_path):
                logger.info(f"Loading custom model from: {custom_model_path}")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(custom_model_path)
                    self.chat_model = AutoModelForCausalLM.from_pretrained(custom_model_path)
                    logger.info("✅ Custom model loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading custom model: {e}")
                    logger.info("Falling back to DialoGPT-medium")
                    self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
                    self.chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
            else:
                logger.warning(f"Custom model not found, using microsoft/DialoGPT-medium")
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
                self.chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
                
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.chatbot = pipeline(
                "text-generation", 
                model=self.chat_model, 
                tokenizer=self.tokenizer,
                device=-1  # Use CPU to avoid GPU memory issues
            )
            
            self.nlp = spacy.load("en_core_web_sm")
            set_seed(42)
            logger.info("✅ All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def _load_memory(self, index_path: str = "memory.index", 
                    text_path: str = "memory_texts.npy", 
                    metadata_path: str = "memory_metadata.pkl"):
        """Load existing memory or create new index."""
        try:
            if os.path.exists(index_path) and os.path.exists(text_path):
                self.index = faiss.read_index(index_path)
                self.memory = np.load(text_path, allow_pickle=True).tolist()
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.memory_metadata = json.load(f)
                else:
                    # Create metadata for existing memories
                    self.memory_metadata = [{"timestamp": "unknown", "type": "legacy"} 
                                          for _ in self.memory]
                
                logger.info(f"✅ Memory loaded: {len(self.memory)} entries")
            else:
                # Create new FAISS index
                self.index = faiss.IndexFlatIP(384)  # Dimension for all-MiniLM-L6-v2
                logger.info("✅ New memory system initialized")
                
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            self.index = faiss.IndexFlatIP(384)

    def save_memory(self, index_path: str = "memory.index", 
                   text_path: str = "memory_texts.npy",
                   metadata_path: str = "memory_metadata.pkl"):
        """Save current memory state to disk."""
        try:
            if self.index is not None:
                faiss.write_index(self.index, index_path)
                np.save(text_path, np.array(self.memory, dtype=object))
                
                with open(metadata_path, 'w') as f:
                    json.dump(self.memory_metadata, f)
                    
                logger.info(f"💾 Memory saved: {len(self.memory)} entries")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")

    def clear_chat_history(self):
        """Clear the current session chat history but keep persistent memory."""
        self.current_session_history = []
        self.conversation_history = []
        logger.info("🧹 Chat history cleared")

    def embed_and_store(self, message: str, message_type: str = "user"):
        """Store message in memory with embeddings and metadata."""
        if not message.strip() or message in self.memory:
            return
            
        try:
            embedding = self.embedder.encode([message])[0]
            
            # Normalize embedding for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            
            self.memory.append(message)
            self.memory_vectors.append(embedding)
            self.memory_metadata.append({
                "timestamp": datetime.now().isoformat(),
                "type": message_type,
                "length": len(message)
            })
            
            if self.index is not None:
                self.index.add(np.array([embedding]).astype('float32'))
                
        except Exception as e:
            logger.error(f"Error storing memory: {e}")

    def retrieve_memory(self, query: str, top_k: int = 5, 
                       similarity_threshold: float = 0.25) -> List[str]:
        """Retrieve relevant memories with similarity filtering - now returns longer, more detailed memories."""
        if self.index is None or not self.memory:
            return []
            
        try:
            query_vector = self.embedder.encode([query])[0]
            query_vector = query_vector / np.linalg.norm(query_vector)
            
            # Search for more candidates to filter
            search_k = min(top_k * 3, len(self.memory))
            D, I = self.index.search(np.array([query_vector]).astype('float32'), search_k)
            
            # Filter by similarity threshold and return top_k with full content
            relevant_memories = []
            for score, idx in zip(D[0], I[0]):
                if score >= similarity_threshold and idx < len(self.memory):
                    memory_text = self.memory[idx]
                    # Include metadata for richer context
                    metadata = self.memory_metadata[idx] if idx < len(self.memory_metadata) else {}
                    timestamp = metadata.get('timestamp', 'unknown')
                    mem_type = metadata.get('type', 'unknown')
                    
                    # Format with full content and context
                    formatted_memory = f"[{timestamp[:19]}] ({mem_type}) {memory_text}"
                    relevant_memories.append(formatted_memory)
                    
            return relevant_memories[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving memory: {e}")
            return []

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities with error handling."""
        try:
            doc = self.nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            # Filter out very short or common entities
            return [(text, label) for text, label in entities if len(text) > 2]
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []

    def generate_response(self, context_list: List[str], user_input: str, 
                         max_length: int = 200) -> str:
        """Generate response with improved context handling and repetition control."""
        try:
            # Build context more intelligently
            context_parts = []
            if context_list:
                context_parts.append("Relevant memories:")
                for i, ctx in enumerate(context_list[:3]):  # Limit context
                    # Extract just the text part after the metadata
                    if "] (" in ctx and ") " in ctx:
                        clean_ctx = ctx.split(") ", 1)[1]
                    else:
                        clean_ctx = ctx
                    context_parts.append(f"{i+1}. {clean_ctx[:150]}...")  # Show more of the memory
            
            # Add recent conversation history
            if self.conversation_history:
                context_parts.append("\nRecent conversation:")
                for turn in self.conversation_history[-3:]:  # Last 3 turns
                    context_parts.append(f"User: {turn['user']}")
                    context_parts.append(f"Assistant: {turn['bot']}")
            
            context = "\n".join(context_parts)
            
            # Create prompt with better structure
            if context.strip():
                prompt = f"Context:\n{context}\n\nUser: {user_input}\nAssistant:"
            else:
                prompt = f"User: {user_input}\nAssistant:"
            
            # Generate with better parameters
            generated_text = self.chatbot(
                prompt,
                max_length=len(self.tokenizer.encode(prompt)) + max_length,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3
            )[0]['generated_text']
            
            # Extract only the new response
            response = generated_text
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            elif prompt in response:
                response = response[len(prompt):].strip()

            # Clean up problematic phrases
            problematic_phrases = [
                "📠 *The answer is this:**",
                "- **I have memory.**",
                "*I can still remember conversations, but not conversations.**",
                "- I don't have memory of the conversation, but I do have memories of the conversation.**",
                "- ***I don't have memory of the conversation.***",
                "- ***I have memory of a conversation.***",
                "- And I don't have memory of the conversation.***",
                "- I can't remember either.***",
                "---",
                "Questions . User : Questions .",
                "User : Questions .",
                "User: Questions.",
                "User: Questions. User: Questions.",
            ]
            for phrase in problematic_phrases:
                response = response.replace(phrase, "").strip()

            # Clean up conversational indicators
            turn_indicators = ["User:", "Human:", "Bot:", "Assistant:"]
            for indicator in turn_indicators:
                if response.startswith(indicator):
                    response = response[len(indicator):].strip()
            
            # Take first coherent sentence
            sentences = [s.strip() for s in response.split('.') if s.strip()]
            if sentences:
                response = sentences[0] + "."
            else:
                lines = [line.strip() for line in response.split('\n') if line.strip()]
                if lines:
                    response = lines[0]
            
            if not response:
                return "I'm thinking about that..."
            
            if len(response.split()) < 3:
                return "I'm processing that thought."

            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm having trouble formulating a response right now."

    def chat_response(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]], str, str]:
        """Main chat interface function."""
        if not message.strip():
            return "", history, "", ""
            
        # Handle special commands
        if message.lower().startswith("memory:"):
            query = message.split("memory:", 1)[1].strip()
            if not query:
                memories = self.memory[-15:] if self.memory else ["No memories stored yet."]
                memory_display = "Recent memories:\n" + "\n".join(f"• {m}" for m in memories)
                bot_reply = "Showing recent memories."
            else:
                relevant = self.retrieve_memory(query, top_k=8)  # More memories
                if relevant:
                    memory_display = f"Memories related to '{query}':\n" + "\n".join(f"• {m}" for m in relevant)
                    bot_reply = f"Found {len(relevant)} memories related to '{query}'."
                else:
                    memory_display = f"No memories found related to '{query}'"
                    bot_reply = f"No memories found related to '{query}'"
            
            # Add to chat history
            history.append([message, bot_reply])
            return "", history, memory_display, ""

        # Store user input
        self.embed_and_store(message, "user")
        
        # Retrieve relevant memories
        memory_recall = self.retrieve_memory(message, top_k=5)
        
        # Extract entities
        named_entities = self.extract_entities(message)
        
        # Generate response
        reply = self.generate_response(memory_recall, message)
        
        # Store bot response
        self.embed_and_store(reply, "bot")
        
        # Update conversation history
        self.conversation_history.append({
            "user": message,
            "bot": reply,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent conversation history
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        # Auto-save memory periodically
        if len(self.memory) % 20 == 0:
            threading.Thread(target=self.save_memory, daemon=True).start()
        
        # Add to chat history
        history.append([message, reply])
        
        # Format outputs with longer, more detailed content
        memory_str = ""
        if memory_recall:
            memory_str = "🧠 Retrieved Memories:\n\n"
            for i, mem in enumerate(memory_recall, 1):
                memory_str += f"{i}. {mem}\n\n"
        else:
            memory_str = "No relevant memories found for this query."
        
        entity_str = ""
        if named_entities:
            entity_str = "🏷️ Named Entities Detected:\n\n"
            for text, label in named_entities:
                entity_str += f"• {text} ({label})\n"
        else:
            entity_str = "No named entities detected in this message."
        
        return "", history, memory_str, entity_str

# Initialize the system
logger.info("Initializing Rhizome system...")
rhizome = RhizomeMemorySystem()

# Create Gradio interface with chat functionality
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="🧠 Rhizome — Enhanced Memory-Infused AI Chat",
    css="""
    .gradio-container {
        max-width: 1400px;
        margin: auto;
    }
    .memory-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        height: 400px;
        overflow-y: scroll;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .entity-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
        height: 400px;
        overflow-y: scroll;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .chat-container {
        height: 500px;
    }
    """
) as iface:
    gr.Markdown("""
    # 🧠 Rhizome — Enhanced Memory-Infused Dialogic Agent
    
    An introspective, recursive AI with persistent memory and continuous chat capabilities.
    
    **Special Commands:**
    - `memory: keyword` - Search your conversation memories  
    - `memory:` (empty) - Show recent memories
    
    **Features:**
    - 💬 Continuous chat interface
    - 🧠 Persistent memory across sessions
    - 🔍 Enhanced memory retrieval with full context
    - 🧹 Clear chat history while preserving memories
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            # Main chat interface
            chatbot = gr.Chatbot(
                label="💬 Chat with Rhizome",
                elem_classes=["chat-container"],
                height=500,
                show_copy_button=True
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Chat with Rhizome or use commands like 'memory: philosophy'...",
                    label="Your Message",
                    lines=2,
                    scale=4
                )
                with gr.Column(scale=1, min_width=100):
                    send_btn = gr.Button("Send 📤", variant="primary")
                    clear_btn = gr.Button("Clear Chat 🧹", variant="secondary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 System Status")
            memory_count = gr.Textbox(
                value=f"Memories: {len(rhizome.memory)} | Model: Custom/DialoGPT",
                label="System Info",
                interactive=False,
                lines=2
            )
            stop_btn = gr.Button("Stop Server 🛑", variant="stop")
    
    with gr.Row():
        with gr.Column(scale=1):
            memory_display = gr.Textbox(
                label="🧠 Retrieved Memories",
                interactive=False,
                lines=15,
                max_lines=15,
                elem_classes=["memory-box"],
                placeholder="Relevant memories will appear here...",
                show_copy_button=True
            )
        
        with gr.Column(scale=1):
            entities_display = gr.Textbox(
                label="🏷️ Named Entities",
                interactive=False,
                lines=15,
                max_lines=15,
                elem_classes=["entity-box"],
                placeholder="Named entities will appear here...",
                show_copy_button=True
            )
    
    # Event handlers
    def update_memory_count():
        return f"Memories: {len(rhizome.memory)} | Active Session"
    
    def clear_chat():
        rhizome.clear_chat_history()
        return [], "", "", update_memory_count()
    
    def stop_server():
        logger.info("Stop button pressed - saving memory and shutting down...")
        rhizome.save_memory()
        import os
        os._exit(0)
    
    # Chat functionality
    def respond(message, history):
        return rhizome.chat_response(message, history)
    
    # Bind events
    send_btn.click(
        fn=respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, memory_display, entities_display]
    ).then(
        fn=update_memory_count,
        outputs=[memory_count]
    )
    
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, memory_display, entities_display]
    ).then(
        fn=update_memory_count,
        outputs=[memory_count]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, memory_display, entities_display, memory_count]
    )
    
    stop_btn.click(fn=stop_server)

# Save memory on exit
import atexit
atexit.register(rhizome.save_memory)

if __name__ == "__main__":
    logger.info("Launching Rhizome chat interface...")
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )

import os
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.style as style
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from tqdm import tqdm
import time
import numpy as np
import multiprocessing
import psutil
import random
from datetime import datetime

# PATCH START: Explicitly disable torch.compile to avoid 'torch._thread_safe_fork' error
os.environ["TORCH_COMPILE_DISABLE"] = "1"
# PATCH END

# CRITICAL: Force CPU-only mode by setting CUDA_VISIBLE_DEVICES BEFORE any torch imports
def force_cpu_only():
    """Force CPU-only mode by hiding all CUDA devices"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_AVAILABLE_DEVICES"] = ""

# Configure clean logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and clean output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

def setup_logging():
    """Configure clean, colorful logging"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler with custom formatter
    console_handler = logging.StreamHandler()
    formatter = ColoredFormatter(
        fmt='%(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Suppress noisy library logs
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("tokenizers").setLevel(logging.ERROR)
    logging.getLogger("transformers.trainer").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("peft").setLevel(logging.ERROR)
    
    return logging.getLogger(__name__)

logger = setup_logging()

# Global variables for device configuration
DEVICE = None
DEVICE_INFO = "Not initialized"
DEVICE_DETAILS = {}
USE_CPU_ONLY = False

def get_system_memory_info():
    """Get system memory information"""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'percent_used': memory.percent,
        'free_gb': memory.free / (1024**3)
    }

def calculate_safe_num_proc():
    """Calculate safe number of processes based on available RAM"""
    memory_info = get_system_memory_info()
    cpu_count = multiprocessing.cpu_count()
    
    # Estimate ~1-2GB per process for tokenization (conservative)
    memory_per_process_gb = 1.5
    
    # Calculate max processes based on available memory
    max_processes_by_memory = max(1, int(memory_info['available_gb'] / memory_per_process_gb))
    
    # Use conservative approach: min of CPU count-1 and memory-limited processes
    safe_processes = min(cpu_count - 1, max_processes_by_memory, 8)  # Cap at 8 for safety
    safe_processes = max(1, safe_processes)  # Ensure at least 1
    
    logger.info(f"💾 System memory: {memory_info['total_gb']:.1f}GB total, {memory_info['available_gb']:.1f}GB available")
    logger.info(f"🔧 Using {safe_processes} processes for tokenization (CPU cores: {cpu_count}, Memory-safe limit: {max_processes_by_memory})")
    
    return safe_processes

def get_gpu_info() -> dict:
    """Get detailed GPU information including compute capability."""
    gpu_info = {}
    
    if torch.cuda.is_available():
        try:
            gpu_info['name'] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_info['memory_total'] = props.total_memory / (1024**3)  # GB
            gpu_info['multiprocessor_count'] = props.multi_processor_count
            gpu_info['max_threads_per_multiprocessor'] = props.max_threads_per_multi_processor
            gpu_info['compute_capability'] = f"{props.major}.{props.minor}"
            gpu_info['compute_capability_major'] = props.major
            gpu_info['compute_capability_minor'] = props.minor
            
            # Check if GPU is supported by modern PyTorch (6.0+ compute capability)
            compute_capability_numeric = props.major + (props.minor / 10.0)
            gpu_info['is_supported'] = compute_capability_numeric >= 6.0
            gpu_info['is_modern'] = compute_capability_numeric >= 7.0  # RTX series and newer
            
            # Detailed classification
            if compute_capability_numeric < 3.5:
                gpu_info['classification'] = "Very Old (Pre-Kepler)"
                gpu_info['performance_expectation'] = "Not supported by PyTorch"
            elif compute_capability_numeric < 5.0:
                gpu_info['classification'] = "Old (Kepler)"
                gpu_info['performance_expectation'] = "Limited PyTorch support, likely slower than modern CPU"
            elif compute_capability_numeric < 6.0:
                gpu_info['classification'] = "Legacy (Maxwell)"
                gpu_info['performance_expectation'] = "Deprecated in modern PyTorch, CPU likely faster"
            else:
                gpu_info['classification'] = "Modern (Turing/Ampere/Ada)"
                gpu_info['performance_expectation'] = "Excellent performance"
                
        except Exception as e:
            gpu_info['error'] = str(e)
            logger.warning(f"Warning: Could not get full GPU info: {e}")
            
    return gpu_info

def check_pytorch_cuda_compatibility() -> tuple[bool, str]:
    """Check if CUDA is actually working with PyTorch."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    try:
        # Try to create a simple tensor on GPU
        test_tensor = torch.randn(10, 10).cuda()
        result = test_tensor @ test_tensor.T
        result = result.cpu()  # Move back to CPU
        
        # Clear memory
        del test_tensor, result
        torch.cuda.empty_cache()
        
        return True, "CUDA working correctly"
        
    except Exception as e:
        error_msg = str(e).lower()
        if "no longer supports" in error_msg or "too old" in error_msg:
            return False, f"GPU too old for PyTorch: {e}"
        elif "out of memory" in error_msg:
            return False, f"GPU out of memory: {e}"
        else:
            return False, f"CUDA error: {e}"

def detect_optimal_device():
    """Intelligently detect the optimal device with proper GPU support checking."""
    global DEVICE, DEVICE_INFO, DEVICE_DETAILS, USE_CPU_ONLY

    device_selected = "cpu"
    device_info_str = "CPU (default)"
    all_info = {}
    USE_CPU_ONLY = True  # Default to CPU-only mode
    
    # Get CPU info
    cpu_count = multiprocessing.cpu_count()
    all_info['cpu_cores'] = cpu_count
    
    # Check GPU availability and compatibility
    gpu_info = get_gpu_info()
    all_info.update(gpu_info)
    
    if torch.cuda.is_available():
        logger.info(f"Info: GPU detected: {gpu_info.get('name', 'Unknown')}")
        logger.info(f"Info: GPU compute capability: {gpu_info.get('compute_capability', 'Unknown')}")
        logger.info(f"Info: GPU classification: {gpu_info.get('classification', 'Unknown')}")
        logger.info(f"Info: GPU memory: {gpu_info.get('memory_total', 0):.1f}GB")
        
        # Check if GPU is supported by PyTorch
        is_supported = gpu_info.get('is_supported', False)
        
        if not is_supported:
            reason = f"GPU compute capability {gpu_info.get('compute_capability', 'unknown')} is below PyTorch minimum requirement (6.0)"
            logger.warning(f"Warning: Forcing CPU: {reason}")
            device_info_str = f"CPU: {cpu_count} cores (GPU {gpu_info.get('name', 'Unknown')} unsupported - {reason})"
            all_info['decision_reason'] = reason
            USE_CPU_ONLY = True
        else:
            # Check if CUDA actually works
            cuda_works, cuda_message = check_pytorch_cuda_compatibility()
            
            if not cuda_works:
                logger.warning(f"Warning: Forcing CPU: {cuda_message}")
                device_info_str = f"CPU: {cpu_count} cores (GPU CUDA failed - {cuda_message})"
                all_info['decision_reason'] = cuda_message
                USE_CPU_ONLY = True
            else:
                # GPU is supported and working - but let's be conservative and still allow CPU override
                # You can uncomment the next two lines to enable GPU usage when available
                # device_selected = "cuda"
                # USE_CPU_ONLY = False
                
                # For now, keeping CPU-only mode even with working GPU
                device_selected = "cpu"
                USE_CPU_ONLY = True
                device_info_str = f"CPU: {cpu_count} cores (GPU available but using CPU by design)"
                all_info['decision_reason'] = "CPU forced for stability"
    else:
        device_info_str = f"CPU: {cpu_count} cores (CUDA not available)"
        all_info['decision_reason'] = "CUDA not available"
        USE_CPU_ONLY = True
    
    # Force CPU-only mode if determined
    if USE_CPU_ONLY:
        force_cpu_only()
        device_selected = "cpu"
    
    # Configure the selected device globally
    DEVICE = torch.device(device_selected)
    DEVICE_INFO = device_info_str
    DEVICE_DETAILS = all_info

    # Apply PyTorch optimizations based on the selected device
    if DEVICE.type == "cuda" and not USE_CPU_ONLY:
        logger.info("Info: Configuring GPU optimizations...")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        logger.info("Info: Configuring CPU optimizations...")
        
        # Optimal thread count for CPU: use all but one core
        optimal_threads = max(1, cpu_count - 1)
        
        torch.set_num_threads(optimal_threads)
        torch.set_num_interop_threads(optimal_threads)
        torch.set_default_tensor_type('torch.FloatTensor')
        
        DEVICE_DETAILS['cpu_threads_used'] = optimal_threads
        DEVICE_INFO += f" (using {optimal_threads} threads)"

    logger.info(f"Info: Initialized with device: {DEVICE_INFO}")
    logger.info(f"Info: Decision reason: {DEVICE_DETAILS.get('decision_reason', 'No specific reason')}")
    logger.info(f"Info: CPU-only mode: {USE_CPU_ONLY}")

# Call device detection once at the start
detect_optimal_device()

# Set environment variables based on detected device
if USE_CPU_ONLY:
    optimal_threads_env = DEVICE_DETAILS.get('cpu_threads_used', multiprocessing.cpu_count())
    os.environ.update({
        "OMP_NUM_THREADS": str(optimal_threads_env),
        "MKL_NUM_THREADS": str(optimal_threads_env),
        "TOKENIZERS_PARALLELISM": "true",
        "PYTHONIOENCODING": "utf-8",
        "TRANSFORMERS_VERBOSITY": "error",
        "DATASETS_VERBOSITY": "error"
    })
else:
    os.environ.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "true",
        "PYTHONIOENCODING": "utf-8",
        "TRANSFORMERS_VERBOSITY": "error",
        "DATASETS_VERBOSITY": "error"
    })

# Set multiprocessing start method
torch.multiprocessing.set_start_method('spawn', force=True)

def get_model_lora_targets(model):
    """Automatically detect LoRA target modules based on model architecture"""
    
    # Get model architecture info
    model_type = getattr(model.config, 'model_type', '').lower()
    architecture = model.__class__.__name__.lower()
    
    logger.info(f"🔍 Detecting LoRA targets for model type: {model_type}, architecture: {architecture}")
    
    # Get all named modules
    all_modules = {}
    for name, module in model.named_modules():
        module_type = module.__class__.__name__
        if module_type not in all_modules:
            all_modules[module_type] = []
        all_modules[module_type].append(name)
    
    # Log available module types for debugging
    logger.info(f"🔍 Available module types: {list(all_modules.keys())}")
    
    # Define target patterns for different architectures
    target_patterns = {
        # GPT-2 style models
        'gpt2': ["c_attn", "c_proj", "c_fc"],
        'gpt': ["c_attn", "c_proj", "c_fc"],
        
        # BERT/RoBERta style models  
        'bert': ["query", "key", "value", "dense"],
        'roberta': ["query", "key", "value", "dense"],
        
        # LLaMA style models
        'llama': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        # Mistral style models
        'mistral': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        # Qwen style models (like DeepSeek-R1-Distill-Qwen)
        'qwen': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        'qwen2': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        # T5 style models
        't5': ["q", "k", "v", "o", "wi", "wo"],
        
        # Falcon style models
        'falcon': ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        
        # Default fallback patterns
        'default': ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
    
    # Determine which pattern to use
    targets = None
    
    # First try exact model type match
    if model_type in target_patterns:
        targets = target_patterns[model_type]
        logger.info(f"✅ Using {model_type} specific targets: {targets}")
    
    # If no exact match, try partial matches
    if not targets:
        for pattern_key, pattern_targets in target_patterns.items():
            if pattern_key in model_type or pattern_key in architecture:
                targets = pattern_targets
                logger.info(f"✅ Using {pattern_key} pattern targets: {targets}")
                break
    
    # If still no match, try to find common attention patterns
    if not targets:
        found_targets = []
        
        # Look for common attention projection patterns
        attention_patterns = ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value', 'c_attn']
        for pattern in attention_patterns:
            for module_type, module_names in all_modules.items():
                if 'Linear' in module_type:  # Focus on Linear layers
                    matching_names = [name for name in module_names if pattern in name.lower()]
                    if matching_names:
                        found_targets.append(pattern)
                        break
        
        if found_targets:
            targets = found_targets
            logger.info(f"✅ Auto-detected targets: {targets}")
    
    # Final fallback
    if not targets:
        targets = target_patterns['default']
        logger.warning(f"⚠️ Using default targets: {targets}")
    
    # Validate that targets actually exist in the model
    valid_targets = []
    for target in targets:
        found = False
        for module_type, module_names in all_modules.items():
            if 'Linear' in module_type:
                matching_names = [name for name in module_names if target in name]
                if matching_names:
                    valid_targets.append(target)
                    found = True
                    break
        if not found:
            logger.warning(f"⚠️ Target '{target}' not found in model")
    
    if not valid_targets:
        # Emergency fallback - find any Linear layers
        logger.warning("⚠️ No standard targets found, using emergency fallback")
        for module_type, module_names in all_modules.items():
            if 'Linear' in module_type and module_names:
                # Take the first few linear layer names
                sample_names = module_names[:3]
                valid_targets = [name.split('.')[-1] for name in sample_names]
                break
    
    logger.info(f"🎯 Final LoRA targets: {valid_targets}")
    return valid_targets

def determine_fan_in_fan_out(model_name: str) -> bool:
    """
    Determine the appropriate fan_in_fan_out setting for LoRA based on model architecture.
    
    Args:
        model_name: The model name or path (e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    
    Returns:
        bool: True for Falcon-style models, False for DeepSeek/Qwen/most modern architectures
    """
    model_name_lower = model_name.lower()
    
    # DeepSeek and Qwen models should use fan_in_fan_out=False
    if any(x in model_name_lower for x in ["deepseek", "qwen", "qwen2"]):
        logger.info(f"🔧 Setting fan_in_fan_out=False for {model_name} (DeepSeek/Qwen architecture)")
        return False
    
    # Falcon models need fan_in_fan_out=True
    if "falcon" in model_name_lower:
        logger.info(f"🔧 Setting fan_in_fan_out=True for {model_name} (Falcon architecture)")
        return True
    
    # Default to False for most modern architectures
    logger.info(f"🔧 Setting fan_in_fan_out=False for {model_name} (default for modern architectures)")
    return False

class TrainingLogger(TrainerCallback):
    """Custom callback to log and visualize training metrics"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.metrics = {
            'step': [],
            'epoch': [],
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'grad_norm': [],
            'train_runtime': [],
            'train_samples_per_second': [],
            'train_steps_per_second': []
        }
        self.start_time = time.time()
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when logging occurs"""
        if logs is None:
            return
            
        current_step = state.global_step
        current_epoch = state.epoch
        
        # Store basic info
        if current_step not in self.metrics['step']:
            self.metrics['step'].append(current_step)
            self.metrics['epoch'].append(current_epoch)
        
        # Store available metrics
        if 'loss' in logs:
            if len(self.metrics['train_loss']) < len(self.metrics['step']):
                self.metrics['train_loss'].extend([None] * (len(self.metrics['step']) - len(self.metrics['train_loss'])))
            if len(self.metrics['train_loss']) == len(self.metrics['step']):
                self.metrics['train_loss'][-1] = logs['loss']
            else:
                self.metrics['train_loss'].append(logs['loss'])
                
        if 'eval_loss' in logs:
            if len(self.metrics['eval_loss']) < len(self.metrics['step']):
                self.metrics['eval_loss'].extend([None] * (len(self.metrics['step']) - len(self.metrics['eval_loss'])))
            if len(self.metrics['eval_loss']) == len(self.metrics['step']):
                self.metrics['eval_loss'][-1] = logs['eval_loss']
            else:
                self.metrics['eval_loss'].append(logs['eval_loss'])
                
        if 'learning_rate' in logs:
            if len(self.metrics['learning_rate']) < len(self.metrics['step']):
                self.metrics['learning_rate'].extend([None] * (len(self.metrics['step']) - len(self.metrics['learning_rate'])))
            if len(self.metrics['learning_rate']) == len(self.metrics['step']):
                self.metrics['learning_rate'][-1] = logs['learning_rate']
            else:
                self.metrics['learning_rate'].append(logs['learning_rate'])
                
        if 'grad_norm' in logs:
            if len(self.metrics['grad_norm']) < len(self.metrics['step']):
                self.metrics['grad_norm'].extend([None] * (len(self.metrics['step']) - len(self.metrics['grad_norm'])))
            if len(self.metrics['grad_norm']) == len(self.metrics['step']):
                self.metrics['grad_norm'][-1] = logs['grad_norm']
            else:
                self.metrics['grad_norm'].append(logs['grad_norm'])
                
        # Performance metrics
        for key in ['train_runtime', 'train_samples_per_second', 'train_steps_per_second']:
            if key in logs:
                if len(self.metrics[key]) < len(self.metrics['step']):
                    self.metrics[key].extend([None] * (len(self.metrics['step']) - len(self.metrics[key])))
                if len(self.metrics[key]) == len(self.metrics['step']):
                    self.metrics[key][-1] = logs[key]
                else:
                    self.metrics[key].append(logs[key])
    
    def save_rng_state(self, checkpoint_dir):
        """Saves the RNG states of torch, numpy, and random."""
        rng_state_path = Path(checkpoint_dir) / 'rng_state.pth'
        rng_states = {
            'torch_rng_state': torch.random.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate()
        }
        torch.save(rng_states, rng_state_path)
        logger.info(f"Saved RNG states to {rng_state_path}")

    def on_save(self, args, state, control, model=None, **kwargs):
        """Called when model checkpoint is saved"""
        checkpoint_dir = self.output_dir / f"checkpoint-{state.global_step}"
        self.save_metrics_and_plots(checkpoint_dir)
        self.save_rng_state(checkpoint_dir)
        
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of training"""
        self.save_metrics_and_plots(self.output_dir, final=True)
        self.save_rng_state(self.output_dir)
        
    def save_metrics_and_plots(self, save_dir, final=False):
        """Save metrics JSON and generate plots"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean up metrics (remove None values and ensure equal lengths)
        cleaned_metrics = {}
        base_length = len(self.metrics['step'])
        
        for key, values in self.metrics.items():
            if key == 'step':
                cleaned_metrics[key] = values
            else:
                # Pad with None if shorter, truncate if longer
                if len(values) < base_length:
                    values.extend([None] * (base_length - len(values)))
                elif len(values) > base_length:
                    values = values[:base_length]
                cleaned_metrics[key] = values
        
        # Save metrics as JSON
        metrics_file = save_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(cleaned_metrics, f, indent=2)
        
        # Generate plots
        self.create_training_plots(save_dir, cleaned_metrics, final)
        
        if final:
            logger.info(f"📊 Final training metrics saved to: {save_dir}")
        
    def create_training_plots(self, save_dir, metrics, final=False):
        """Create comprehensive training visualization plots"""
        
        # Set up plot style
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (15, 12),
            'font.size': 10,
            'axes.linewidth': 1,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        
        steps = metrics['step']
        epochs = metrics['epoch']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Training Progress {"(Final)" if final else "(Checkpoint)"}', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss curves
        ax1 = axes[0, 0]
        train_losses = [x for x in metrics['train_loss'] if x is not None]
        eval_losses = [x for x in metrics['eval_loss'] if x is not None]
        
        if train_losses:
            train_steps = [steps[i] for i, x in enumerate(metrics['train_loss']) if x is not None]
            ax1.plot(train_steps, train_losses, 'b-', label='Training Loss', linewidth=2)
        
        if eval_losses:
            eval_steps = [steps[i] for i, x in enumerate(metrics['eval_loss']) if x is not None]
            ax1.plot(eval_steps, eval_losses, 'r-', label='Validation Loss', linewidth=2)
        
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Learning Rate Schedule
        ax2 = axes[0, 1]
        learning_rates = [x for x in metrics['learning_rate'] if x is not None]
        if learning_rates:
            lr_steps = [steps[i] for i, x in enumerate(metrics['learning_rate']) if x is not None]
            ax2.plot(lr_steps, learning_rates, 'g-', linewidth=2)
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        else:
            ax2.text(0.5, 0.5, 'No Learning Rate Data', ha='center', va='center', transform=ax2.transAxes)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Gradient Norm
        ax3 = axes[0, 2]
        grad_norms = [x for x in metrics['grad_norm'] if x is not None]
        if grad_norms:
            grad_steps = [steps[i] for i, x in enumerate(metrics['grad_norm']) if x is not None]
            ax3.plot(grad_steps, grad_norms, 'orange', linewidth=2)
            ax3.set_xlabel('Steps')
            ax3.set_ylabel('Gradient Norm')
            ax3.set_title('Gradient Norm')
        else:
            ax3.text(0.5, 0.5, 'No Gradient Norm Data', ha='center', va='center', transform=ax3.transAxes)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training Speed (samples/sec)
        ax4 = axes[1, 0]
        samples_per_sec = [x for x in metrics['train_samples_per_second'] if x is not None]
        if samples_per_sec:
            speed_steps = [steps[i] for i, x in enumerate(metrics['train_samples_per_second']) if x is not None]
            ax4.plot(speed_steps, samples_per_sec, 'purple', linewidth=2)
            ax4.set_xlabel('Steps')
            ax4.set_ylabel('Samples/Second')
            ax4.set_title('Training Speed')
        else:
            ax4.text(0.5, 0.5, 'No Speed Data', ha='center', va='center', transform=ax4.transAxes)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Steps per Second
        ax5 = axes[1, 1]
        steps_per_sec = [x for x in metrics['train_steps_per_second'] if x is not None]
        if steps_per_sec:
            step_speed_steps = [steps[i] for i, x in enumerate(metrics['train_steps_per_second']) if x is not None]
            ax5.plot(step_speed_steps, steps_per_sec, 'brown', linewidth=2)
            ax5.set_xlabel('Steps')
            ax5.set_ylabel('Steps/Second')
            ax5.set_title('Training Steps per Second')
        else:
            ax5.text(0.5, 0.5, 'No Steps/Sec Data', ha='center', va='center', transform=ax5.transAxes)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Epochs Progress
        ax6 = axes[1, 2]
        if epochs:
            ax6.plot(steps, epochs, 'teal', linewidth=2, marker='o', markersize=3)
            ax6.set_xlabel('Steps')
            ax6.set_ylabel('Epoch')
            ax6.set_title('Epoch Progress')
        else:
            ax6.text(0.5, 0.5, 'No Epoch Data', ha='center', va='center', transform=ax6.transAxes)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = save_dir / "training_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a loss-only focused plot
        if train_losses or eval_losses:
            self.create_loss_focused_plot(save_dir, metrics, final)
    
    def create_loss_focused_plot(self, save_dir, metrics, final=False):
        """Creates a dedicated, higher-resolution plot just for training and validation loss."""
        
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(12, 8))
        
        steps = metrics['step']
        train_losses = metrics['train_loss']
        eval_losses = metrics['eval_loss']
        
        # Filter out None values and pair steps with losses
        train_data = [(steps[i], loss) for i, loss in enumerate(train_losses) if loss is not None]
        eval_data = [(steps[i], loss) for i, loss in enumerate(eval_losses) if loss is not None]
        
        if train_data:
            train_steps, train_vals = zip(*train_data)
            plt.plot(train_steps, train_vals, 'b-', label='Training Loss', linewidth=3, alpha=0.8)
        
        if eval_data:
            eval_steps, eval_vals = zip(*eval_data)
            plt.plot(eval_steps, eval_vals, 'r--', label='Validation Loss', linewidth=3, alpha=0.8)
        
        plt.xlabel('Training Steps', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title(f'Loss Progression {"(Final Run)" if final else "(Checkpoint)"}', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.6)
        
        # Add some key statistics as text annotations on the plot
        if train_data:
            min_train_loss = min(train_vals)
            final_train_loss = train_vals[-1]
            plt.text(0.02, 0.98, f'Min Train Loss: {min_train_loss:.4f}\nFinal Train Loss: {final_train_loss:.4f}', 
                    transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8), fontsize=10)
        
        if eval_data:
            min_eval_loss = min(eval_vals)
            final_eval_loss = eval_vals[-1]
            plt.text(0.98, 0.98, f'Min Val Loss: {min_eval_loss:.4f}\nFinal Val Loss: {final_eval_loss:.4f}', 
                    transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8), fontsize=10)

        plt.tight_layout()
        
        # Save focused loss plot
        loss_plot_file = save_dir / "loss_focused.png"
        plt.savefig(loss_plot_file, dpi=300, bbox_inches='tight')
        plt.close()

@dataclass
class CustomDataCollator:
    """
    An optimized data collator for language modeling tasks.
    It efficiently pads input sequences to the maximum length within each batch or a global max_length,
    and prepares labels for causal language modeling.
    """
    tokenizer: Any
    max_length: int = 512
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        # Determine the maximum sequence length for the current batch, capped by self.max_length
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length
        )
        
        # Identify the padding token ID
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            logger.warning("Tokenizer does not have a pad_token_id. Using eos_token_id for padding.")
            pad_token_id = self.tokenizer.eos_token_id
        
        if pad_token_id is None:
            raise ValueError("No pad_token_id or eos_token_id found in tokenizer. Cannot pad sequences.")

        # Pre-allocate tensors for efficiency
        input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        
        for i, feature in enumerate(features):
            ids = feature["input_ids"][:max_len]
            seq_len = len(ids)
            
            input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

def seed_worker(worker_id):
    """
    Ensures that each DataLoader worker has a unique and deterministic seed
    based on the main process's torch seed and the worker ID.
    """
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_semantic_metadata(metadata_path: str = "data_finetune/dataset_metadata.json") -> Dict:
    """
    Load semantic metadata from the data formatter output.
    
    Args:
        metadata_path: Path to the dataset metadata JSON file
    
    Returns:
        Dict containing theme distribution and other metadata
    """
    metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        logger.warning(f"⚠️ Semantic metadata not found at {metadata_path}")
        return {}
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract key information
        theme_dist = metadata.get('theme_distribution', {})
        source_dist = metadata.get('source_distribution', {})
        total_pairs = metadata.get('total_pairs', 0)
        
        logger.info(f"🧠 Loaded semantic metadata:")
        logger.info(f"   - Total pairs: {total_pairs:,}")
        logger.info(f"   - Unique themes: {len(theme_dist)}")
        logger.info(f"   - Data sources: {list(source_dist.keys())}")
        
        # Log top themes
        if theme_dist:
            top_themes = sorted(theme_dist.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info(f"   - Top 10 themes:")
            for theme, count in top_themes:
                logger.info(f"     • {theme}: {count}")
        
        return metadata
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to load semantic metadata: {e}")
        return {}

class DeepSeekQwenTrainer:
    """
    A wrapper class for fine-tuning DeepSeek-R1-Distill-Qwen-1.5B (or similar Causal LMs) using
    Hugging Face Transformers Trainer, with integrated LoRA and custom logging.
    """
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.start_time = time.time()
        self.semantic_metadata = {}
        
    def print_header(self):
        """Prints a decorative header for the script output."""
        print("\n" + "═" * 70)
        print("🤖 DeepSeek-R1-Distill-Qwen-1.5B Fine-Tuning Suite v2.0")
        print("   Compatible with data_formatter.py output")
        print("═" * 70)
        
    def print_section(self, title, emoji="📋"):
        """Prints a formatted section header."""
        print(f"\n{emoji} {title}")
        print("─" * 50)
        
    def setup_model_and_tokenizer(self):
        """Initializes the tokenizer and loads the model, then applies LoRA configuration."""
        self.print_section("Model Setup", "🔧")
        
        with tqdm(total=4, desc="Loading components", ncols=70, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Ensure a padding token is available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Tokenizer's pad_token was None, set to eos_token.")
            pbar.update(1)
            
            # Load model and explicitly handle device placement
            logger.info(f"Loading model from {self.model_name}...")
            if USE_CPU_ONLY:
                logger.info("Info: Explicitly loading model onto CPU using device_map='cpu'.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="cpu",
                )
            else:
                logger.info(f"Info: Loading model, will move to {DEVICE.type.upper()}.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                ).to(DEVICE)
            pbar.update(1)
            
            # Dynamically get LoRA target modules
            lora_target_modules = get_model_lora_targets(self.model)
            
            # Determine fan_in_fan_out based on model architecture
            fan_in_fan_out = determine_fan_in_fan_out(self.model_name)

            # Configure LoRA adapters
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                fan_in_fan_out=fan_in_fan_out
            )
            pbar.update(1)
            
            # Apply LoRA to the model
            self.model = get_peft_model(self.model, lora_config)
            self.model.train()
            pbar.update(1)
        
        # Log trainable and total parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"✅ Model loaded and LoRA applied successfully on {DEVICE.type.upper()}")
        logger.info(f"📊 Parameters: {trainable_params:,} trainable / {total_params:,} total ({100 * trainable_params / total_params:.2f}%)")
        
        if trainable_params == 0:
            raise RuntimeError("❌ Error: No trainable parameters found after applying LoRA. Check LoRA target modules.")
        
    def load_and_tokenize_data(self, train_file, val_file=None, max_length=512):
        """Loads raw text data and tokenizes it, preparing for training."""
        self.print_section("Data Processing", "📚")
        
        # Verify files exist
        train_path = Path(train_file)
        if not train_path.exists():
            raise FileNotFoundError(f"❌ Training file not found: {train_file}")
        
        data_files = {"train": train_file}
        if val_file and Path(val_file).exists():
            data_files["validation"] = val_file
            logger.info(f"✅ Validation file found: {val_file}")
        else:
            logger.info("ℹ️ No validation file provided or file not found. Training without validation.")
        
        logger.info(f"Loading dataset from: {data_files}")
        dataset = load_dataset("json", data_files=data_files)
        
        train_size = len(dataset['train'])
        logger.info(f"📊 Raw training samples: {train_size:,}")
        if "validation" in dataset:
            logger.info(f"📊 Raw validation samples: {len(dataset['validation']):,}")
        
        # Load semantic metadata
        metadata_path = Path(train_file).parent / "dataset_metadata.json"
        self.semantic_metadata = load_semantic_metadata(str(metadata_path))
        
        def tokenize_function(examples):
            """Tokenizes a batch of text examples."""
            # Support multiple text field names for flexibility
            text_data = examples.get("text") or examples.get("content") or examples.get("prompt", [])
            
            return self.tokenizer(
                text_data,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=True,
                add_special_tokens=True,
            )
        
        print("\n🔄 Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            remove_columns=dataset["train"].column_names,
            num_proc=1,
            desc="Tokenizing"
        )
        logger.info("✅ Dataset tokenization complete.")
        
        # Log sequence length statistics
        if len(tokenized_dataset["train"]) > 0:
            sample_lengths = [len(tokenized_dataset["train"][i]['input_ids']) 
                            for i in range(min(1000, len(tokenized_dataset["train"])))]
            
            logger.info(f"📈 Tokenized sequence lengths (sample): min={min(sample_lengths)}, max={max(sample_lengths)}, avg={sum(sample_lengths)/len(sample_lengths):.1f}")
        else:
            logger.warning("Tokenized training dataset is empty.")

        return tokenized_dataset
    
    def create_training_args(self, output_dir="./DeepSeek-R1-Distill-Qwen-1.5B-finetuned", has_validation=False, **kwargs):
        """
        Creates and configures TrainingArguments for the Hugging Face Trainer.
        """
        
        default_args = {
            "output_dir": output_dir,
            "overwrite_output_dir": True,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 12,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "logging_steps": 25,
            "save_steps": 100,
            "save_total_limit": 2,
            "eval_strategy": "steps" if has_validation else "no",
            "eval_steps": 100 if has_validation else None,
            "save_strategy": "steps",
            "load_best_model_at_end": has_validation,
            "metric_for_best_model": "eval_loss" if has_validation else None,
            "greater_is_better": False,
            "save_safetensors": True,
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": (DEVICE.type == "cuda" and not USE_CPU_ONLY),
            "remove_unused_columns": True,
            "seed": 42,
            "fp16": (DEVICE.type == "cuda" and not USE_CPU_ONLY),
            "gradient_checkpointing": (DEVICE.type == "cuda" and not USE_CPU_ONLY),
            "optim": "adamw_torch",
            "lr_scheduler_type": "cosine",
            "report_to": "none",
            "disable_tqdm": False,
            "log_level": "error",
            "log_level_replica": "error",
            "logging_nan_inf_filter": False,
            "log_on_each_node": False,
        }
        
        default_args["use_cpu"] = USE_CPU_ONLY
        default_args["local_rank"] = -1
        default_args["ddp_find_unused_parameters"] = False

        # Override defaults with user-provided arguments
        default_args.update(kwargs)
        return TrainingArguments(**default_args)
    
    def train(self, train_file, val_file=None, output_dir="./DeepSeek-R1-Distill-Qwen-1.5B-finetuned", **training_kwargs):
        """
        Main function to orchestrate the fine-tuning process.
        """
        self.print_header()
        
        # Add timestamp to output directory for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_versioned = f"{output_dir}_{timestamp}"
        logger.info(f"📁 Versioned output directory: {output_dir_versioned}")
        
        try:
            # Step 1: Setup model and tokenizer with LoRA
            self.setup_model_and_tokenizer()
            
            # Step 2: Load and tokenize data
            tokenized_dataset = self.load_and_tokenize_data(train_file, val_file)
            
            # Step 3: Configure training arguments
            self.print_section("Training Configuration", "⚙️")
            has_validation = "validation" in tokenized_dataset
            
            # DataLoader worker seeding setup
            if training_kwargs.get("dataloader_num_workers", 0) > 0:
                logger.info(f"Configuring DataLoader with worker_init_fn for {training_kwargs.get('dataloader_num_workers')} workers.")
                training_kwargs['dataloader_worker_init_fn'] = seed_worker
            else:
                logger.info("DataLoader num_workers is 0, worker_init_fn not applied.")

            training_args = self.create_training_args(
                output_dir=output_dir_versioned,
                has_validation=has_validation,
                **training_kwargs
            )
            
            # Display key training parameters
            logger.info(f"🎯 Number of training epochs: {training_args.num_train_epochs}")
            effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
            logger.info(f"📦 Effective batch size: {training_args.per_device_train_batch_size} × {training_args.gradient_accumulation_steps} = {effective_batch_size}")
            logger.info(f"📈 Initial learning rate: {training_args.learning_rate}")
            logger.info(f"💾 Output directory: {Path(output_dir_versioned).resolve()}")
            logger.info(f"🚀 Training on: {DEVICE_INFO}")
            logger.info(f"🔌 FP16 (mixed precision): {training_args.fp16}")
            logger.info(f"💡 Gradient Checkpointing: {training_args.gradient_checkpointing}")
            logger.info(f"🚫 CPU-only mode: {training_args.use_cpu}")
            
            # Display semantic metadata if loaded
            if self.semantic_metadata:
                theme_count = len(self.semantic_metadata.get('theme_distribution', {}))
                logger.info(f"🧠 Training with {theme_count} semantic themes from data formatter")
            
            # Step 4: Prepare data collator and custom logger
            data_collator = CustomDataCollator(self.tokenizer, max_length=512)
            training_logger = TrainingLogger(output_dir_versioned)
            
            # Suppress specific warnings
            import warnings
            warnings.filterwarnings("ignore", message=".*label_names.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*loss_type.*", category=UserWarning)
            
            # Step 5: Initialize Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset.get("validation"),
                data_collator=data_collator,
                callbacks=[training_logger],
            )
            
            # Step 6: Check for existing checkpoints
            checkpoint_dir_path = Path(output_dir_versioned)
            last_checkpoint_path = self.find_last_checkpoint(checkpoint_dir_path)
            
            # Step 7: Start training
            self.print_section("Training Progress", "🚀")
            
            if last_checkpoint_path:
                logger.info(f"🔄 Resuming training from checkpoint: {last_checkpoint_path}")
                
                # RNG State Restoration
                rng_state_path = Path(last_checkpoint_path) / 'rng_state.pth'
                if rng_state_path.exists():
                    logger.info(f"Loading RNG states from {rng_state_path}...")
                    try:
                        rng_states = torch.load(rng_state_path)
                        torch.random.set_rng_state(rng_states['torch_rng_state'])
                        np.random.set_state(rng_states['numpy_rng_state'])
                        random.setstate(rng_states['random_rng_state'])
                        logger.info("✅ Restored torch, numpy, and random RNG states.")
                    except Exception as e:
                        logger.warning(f"Failed to load RNG states: {e}")
                else:
                    logger.warning(f"RNG state file not found at {rng_state_path}")

                trainer.train(resume_from_checkpoint=True)

                # Diagnostic output after resume
                if trainer.state.global_step > 0:
                    logger.info(f"✅ Resumed at global step: {trainer.state.global_step}")
                    if trainer.optimizer and hasattr(trainer.optimizer, 'param_groups') and trainer.optimizer.param_groups:
                        current_lr = trainer.optimizer.param_groups[0]['lr']
                        logger.info(f"✅ Current learning rate: {current_lr}")
                    if trainer.optimizer and trainer.optimizer.state:
                        logger.info(f"✅ Optimizer state loaded ({len(trainer.optimizer.state)} entries)")
                else:
                    logger.warning("Trainer's global step is 0 after resume attempt")

            else:
                logger.info("🎯 Starting fresh training run...")
                trainer.train()
            
            # Step 8: Save final model
            logger.info("💾 Saving final model and tokenizer...")
            trainer.save_model(output_dir_versioned)
            self.tokenizer.save_pretrained(output_dir_versioned)
            
            # Step 9: Final summary
            elapsed = time.time() - self.start_time
            self.print_section("Training Complete", "🎉")
            logger.info(f"⏱️ Total training duration: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
            logger.info(f"📁 Final model saved to: {Path(output_dir_versioned).resolve()}")
            logger.info(f"📊 Training plots: {Path(output_dir_versioned) / 'training_plots.png'}")
            logger.info(f"📈 Loss plot: {Path(output_dir_versioned) / 'loss_focused.png'}")
            logger.info(f"📋 Metrics JSON: {Path(output_dir_versioned) / 'training_metrics.json'}")
            
            return trainer
            
        except KeyboardInterrupt:
            logger.info("ℹ️ Training interrupted by user.")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error during training: {e}", exc_info=True)
            raise
    
    @staticmethod
    def find_last_checkpoint(checkpoint_dir):
        """Helper function to locate the most recent checkpoint directory."""
        if not checkpoint_dir.exists():
            logger.info(f"No checkpoint directory found at {checkpoint_dir}. Starting fresh.")
            return None
            
        checkpoints = [
            d for d in checkpoint_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        
        if not checkpoints:
            logger.info(f"No existing checkpoints found in {checkpoint_dir}. Starting fresh.")
            return None
            
        last_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[-1]))
        logger.info(f"Found existing checkpoint: {last_checkpoint}")
        return last_checkpoint


def main():
    """Main execution function of the training script."""
    
    trainer = DeepSeekQwenTrainer(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    
    try:
        # Updated file paths to match data_formatter.py output
        result = trainer.train(
            train_file="data_finetune/dataset_train.jsonl",
            val_file="data_finetune/dataset_validation.jsonl",
            output_dir="./DeepSeek-R1-Distill-Qwen-1.5B-finetuned",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=32,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=100,
            logging_steps=25,
            save_steps=100,
            dataloader_num_workers=0,
        )
        
        if result:
            print("\n" + "═" * 70)
            print("🎉 Fine-tuning process successfully completed!")
            print("📁 Your fine-tuned model and training artifacts are in the versioned output directory")
            print("═" * 70)
        else:
            print("\n" + "═" * 70)
            print("ℹ️ Fine-tuning process finished (possibly interrupted or encountered issues).")
            print("═" * 70)
        
    except Exception as e:
        logger.critical(f"\n❌ Fine-tuning terminated unexpectedly: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

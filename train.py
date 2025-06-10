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

# Set up clean logging
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

# CPU optimizations with warning suppression
os.environ.update({
    "TOKENIZERS_PARALLELISM": "true",
    "OMP_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
    "PYTHONIOENCODING": "utf-8",
    "TRANSFORMERS_VERBOSITY": "error",
    "DATASETS_VERBOSITY": "error"
})

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
    
    def on_save(self, args, state, control, model=None, **kwargs):
        """Called when model checkpoint is saved"""
        checkpoint_dir = self.output_dir / f"checkpoint-{state.global_step}"
        self.save_metrics_and_plots(checkpoint_dir)
        
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of training"""
        self.save_metrics_and_plots(self.output_dir, final=True)
        
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
        """Create a focused loss plot with better resolution"""
        
        plt.figure(figsize=(12, 8))
        
        steps = metrics['step']
        train_losses = metrics['train_loss']
        eval_losses = metrics['eval_loss']
        
        # Plot training loss
        train_data = [(steps[i], loss) for i, loss in enumerate(train_losses) if loss is not None]
        if train_data:
            train_steps, train_vals = zip(*train_data)
            plt.plot(train_steps, train_vals, 'b-', label='Training Loss', linewidth=2.5, alpha=0.8)
        
        # Plot validation loss
        eval_data = [(steps[i], loss) for i, loss in enumerate(eval_losses) if loss is not None]
        if eval_data:
            eval_steps, eval_vals = zip(*eval_data)
            plt.plot(eval_steps, eval_vals, 'r-', label='Validation Loss', linewidth=2.5, alpha=0.8)
        
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'Training Loss Progression {"(Final)" if final else "(Checkpoint)"}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add some statistics as text
        if train_data:
            min_loss = min(train_vals)
            final_loss = train_vals[-1]
            plt.text(0.02, 0.98, f'Min Loss: {min_loss:.4f}\nFinal Loss: {final_loss:.4f}', 
                    transform=plt.gca().transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save focused loss plot
        loss_plot_file = save_dir / "loss_focused.png"
        plt.savefig(loss_plot_file, dpi=300, bbox_inches='tight')
        plt.close()

@dataclass
class CustomDataCollator:
    """Optimized data collator with efficient padding"""
    tokenizer: Any
    max_length: int = 512
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length
        )
        
        # Pre-allocate tensors for efficiency
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
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
            "labels": input_ids.clone()  # For causal LM
        }

class DialoGPTTrainer:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.start_time = time.time()
        
    def print_header(self):
        """Print a clean header"""
        print("\n" + "═" * 70)
        print("🤖 DialoGPT Fine-Tuning Suite")
        print("═" * 70)
        
    def print_section(self, title, emoji="📋"):
        """Print section headers"""
        print(f"\n{emoji} {title}")
        print("─" * 50)
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with LoRA"""
        self.print_section("Model Setup", "🔧")
        
        with tqdm(total=4, desc="Loading components", ncols=70) as pbar:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            pbar.update(1)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None, # Use GPU if available
            )
            pbar.update(1)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["c_attn", "c_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                fan_in_fan_out=True
            )
            pbar.update(1)
            
            # Apply LoRA
            self.model = get_peft_model(self.model, lora_config)
            self.model.train()
            pbar.update(1)
        
        # Calculate parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"📊 Parameters: {trainable_params:,} trainable / {total_params:,} total ({100 * trainable_params / total_params:.2f}%)")
        
        if trainable_params == 0:
            raise RuntimeError("❌ No trainable parameters found!")
        
        torch.set_num_threads(8)
        
    def load_and_tokenize_data(self, train_file, val_file=None, max_length=512):
        """Load and tokenize datasets efficiently"""
        self.print_section("Data Processing", "📚")
        
        # Load datasets
        data_files = {"train": train_file}
        if val_file and Path(val_file).exists():
            data_files["validation"] = val_file
        
        dataset = load_dataset("json", data_files=data_files)
        train_size = len(dataset['train'])
        
        logger.info(f"📊 Training samples: {train_size:,}")
        if "validation" in dataset:
            logger.info(f"📊 Validation samples: {len(dataset['validation']):,}")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=True,
                add_special_tokens=True,
            )
        
        # Tokenize with progress bar
        print("\n🔄 Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            remove_columns=dataset["train"].column_names,
            num_proc=1,         # <--- Increase if you have the memory for it
            desc="Tokenizing"
        )
        
        # Quick stats
        sample_lengths = [len(tokenized_dataset["train"][i]['input_ids']) 
                         for i in range(min(1000, len(tokenized_dataset["train"])))]
        
        logger.info(f"📈 Sequence lengths: min={min(sample_lengths)}, max={max(sample_lengths)}, avg={sum(sample_lengths)/len(sample_lengths):.1f}")
        
        return tokenized_dataset
    
    def create_training_args(self, output_dir="./dialogpt-finetuned", has_validation=False, **kwargs):
        """Create optimized training arguments"""
        
        default_args = {
            "output_dir": output_dir,
            "overwrite_output_dir": True,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 3,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "logging_steps": 60,
            "save_steps": 300,
            "save_total_limit": 2,
            "eval_strategy": "steps" if has_validation else "no",
            "eval_steps": 300 if has_validation else None,
            "save_strategy": "steps",
            "load_best_model_at_end": has_validation,
            "metric_for_best_model": "eval_loss" if has_validation else None,
            "greater_is_better": False,
            "save_safetensors": False,
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": False,
            "remove_unused_columns": True,
            "seed": 42,
            "fp16": False,
            "gradient_checkpointing": False,
            "optim": "adamw_torch",
            "lr_scheduler_type": "cosine",
            "report_to": "none",
            "disable_tqdm": False,
            "log_level": "error",  # Suppress transformers logs
            "log_level_replica": "error",
            "logging_nan_inf_filter": False,  # Better for debugging
            "log_on_each_node": False,
        }
        
        default_args.update(kwargs)
        return TrainingArguments(**default_args)
    
    def train(self, train_file, val_file=None, output_dir="./dialogpt-finetuned", **training_kwargs):
        """Main training function with clean output"""
        self.print_header()
        
        # Setup
        self.setup_model_and_tokenizer()
        tokenized_dataset = self.load_and_tokenize_data(train_file, val_file)
        
        # Training setup
        self.print_section("Training Configuration", "⚙️")
        
        has_validation = "validation" in tokenized_dataset
        training_args = self.create_training_args(
            output_dir=output_dir,
            has_validation=has_validation,
            **training_kwargs
        )
        
        # Display key training parameters
        logger.info(f"🎯 Epochs: {training_args.num_train_epochs}")
        logger.info(f"📦 Batch size: {training_args.per_device_train_batch_size} × {training_args.gradient_accumulation_steps} = {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        logger.info(f"📈 Learning rate: {training_args.learning_rate}")
        logger.info(f"💾 Output directory: {output_dir}")
        logger.info(f"📊 Logging & plots will be saved in checkpoints")
        
        # Create trainer with logging callback
        data_collator = CustomDataCollator(self.tokenizer, max_length=512)
        training_logger = TrainingLogger(output_dir)
        
        # Suppress trainer warnings
        import warnings
        warnings.filterwarnings("ignore", message=".*label_names.*")
        warnings.filterwarnings("ignore", message=".*loss_type.*")
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset.get("validation"),
            data_collator=data_collator,
            processing_class=self.tokenizer,
            callbacks=[training_logger],  # Add our custom logger
        )
        
        # Check for existing checkpoints
        checkpoint_dir = Path(output_dir)
        last_checkpoint = self.find_last_checkpoint(checkpoint_dir)
        
        # Start training
        self.print_section("Training Progress", "🚀")
        
        try:
            if last_checkpoint:
                logger.info(f"🔄 Resuming from: {last_checkpoint}")
                trainer.train(resume_from_checkpoint=last_checkpoint)
            else:
                logger.info("🎯 Starting fresh training...")
                trainer.train()
            
            # Save final model
            logger.info("💾 Saving final model...")
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            # Training complete
            elapsed = time.time() - self.start_time
            self.print_section("Training Complete", "🎉")
            logger.info(f"⏱️  Total time: {elapsed/60:.1f} minutes")
            logger.info(f"📁 Model saved to: {output_dir}")
            logger.info(f"📊 Training plots saved in: {output_dir}/training_plots.png")
            logger.info(f"📈 Loss plot saved in: {output_dir}/loss_focused.png")
            logger.info(f"📋 Metrics JSON saved in: {output_dir}/training_metrics.json")
            
            return trainer
            
        except KeyboardInterrupt:
            logger.info("⏹️  Training interrupted by user")
            return None
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    @staticmethod
    def find_last_checkpoint(checkpoint_dir):
        """Find the most recent checkpoint"""
        if not checkpoint_dir.exists():
            return None
            
        checkpoints = [
            d for d in checkpoint_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        
        if not checkpoints:
            return None
            
        return max(checkpoints, key=lambda x: int(x.name.split("-")[-1]))


def main():
    """Main execution function"""
    trainer = DialoGPTTrainer(model_name="microsoft/DialoGPT-medium")
    
    try:
        result = trainer.train(
            train_file="data_finetune/train.jsonl",
            #val_file="data_finetune/validation.jsonl", # <--- Validation Set
            val_file=None,                              # <--- Remove if using Validation Set
            output_dir="./dialogpt-finetuned",
            num_train_epochs=3,
            per_device_train_batch_size=3,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=100,
            logging_steps=30,
            save_steps=150,
        )
        
        if result:
            print("\n" + "═" * 70)
            print("🎉 Fine-tuning completed successfully!")
            print("📁 Check your model in: ./dialogpt-finetuned")
            print("📊 Training graphs saved as PNG files")
            print("📋 Metrics data saved as JSON")
            print("═" * 70)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

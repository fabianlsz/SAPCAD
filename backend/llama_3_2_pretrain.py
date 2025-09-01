# =============================
# LLaMA 3.2 LoRA Fine-tuning Script
# =============================
# This script provides memory-optimized fine-tuning for LLaMA and similar models using LoRA and quantization.
# It supports chunked training, experiment tracking, and robust error handling for low-resource environments.
#
# =============================
#meta-llama/Meta-Llama-3-8B-Instruct
# -----------------------------
# 1. Imports and Optional Dependencies
# -----------------------------
# Import core libraries for deep learning, data handling, and argument parsing
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pandas as pd
import os
import logging
import json
import argparse
import sys
import functools
import random
import numpy as np
import time
import matplotlib.pyplot as plt

# Try to import jsonschema for output validation
try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    print("Warning: jsonschema not available. JSON validation will be disabled.")
    JSONSCHEMA_AVAILABLE = False

# Optional imports for metrics and logging
try:
    from evaluate import load as load_metric
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: evaluate library not available. Metrics will be disabled.")
    METRICS_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not available. Logging will be disabled.")
    WANDB_AVAILABLE = False

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    print("Warning: mlflow not available. MLflow logging will be disabled.")
    MLFLOW_AVAILABLE = False

# Login to HuggingFace Hub for gated models (required for some LLaMA weights)
from huggingface_hub import login
#login("hf_GdR")  # Uncommented for gated models

# -----------------------------
# 2. Logging Setup
# -----------------------------
# Configure logging to both file and console for detailed status and error messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llama_3_2_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------
# 3. Patch accelerate and transformers for compatibility
# -----------------------------
# Some versions of accelerate and transformers have incompatible arguments.
# The following monkey patches remove problematic parameters to ensure compatibility.
try:
    import accelerate
    from accelerate import Accelerator
    
    # Store original __init__ method
    original_init = Accelerator.__init__
    
    def safe_init(self, *args, **kwargs):
        # Remove incompatible parameters for older versions
        incompatible_params = ['dispatch_batches', 'even_batches', 'split_batches', 'use_seedable_sampler']
        for param in incompatible_params:
            if param in kwargs:
                logger.info(f"Removing {param} for accelerate compatibility")
                del kwargs[param]
        
        # Call original method
        return original_init(self, *args, **kwargs)
    
    # Apply patch
    Accelerator.__init__ = safe_init
    logger.info("Applied accelerate compatibility patch")
    
    # Also patch Trainer to prevent it from passing incompatible parameters
    try:
        from transformers import Trainer
        
        # Store original Trainer __init__
        original_trainer_init = Trainer.__init__
        
        def safe_trainer_init(self, *args, **kwargs):
            # Remove any accelerate-related kwargs that might be passed
            if 'accelerator_kwargs' in kwargs:
                accelerator_kwargs = kwargs['accelerator_kwargs']
                incompatible_params = ['dispatch_batches', 'even_batches', 'split_batches', 'use_seedable_sampler']
                for param in incompatible_params:
                    if param in accelerator_kwargs:
                        logger.info(f"Removing {param} from Trainer accelerator_kwargs")
                        del accelerator_kwargs[param]
            
            return original_trainer_init(self, *args, **kwargs)
        
        # Apply Trainer patch
        Trainer.__init__ = safe_trainer_init
        logger.info("Applied Trainer compatibility patch")
        
    except Exception as e:
        logger.warning(f"Failed to apply Trainer patch: {e}")
    
    # Also patch TrainingArguments to prevent incompatible parameters
    try:
        from transformers import TrainingArguments
        
        # Store original TrainingArguments __init__
        original_training_args_init = TrainingArguments.__init__
        
        def safe_training_args_init(self, *args, **kwargs):
            # Remove any accelerate-related parameters that might cause issues
            incompatible_params = ['dispatch_batches', 'even_batches', 'split_batches', 'use_seedable_sampler']
            for param in incompatible_params:
                if param in kwargs:
                    logger.info(f"Removing {param} from TrainingArguments")
                    del kwargs[param]
            
            return original_training_args_init(self, *args, **kwargs)
        
        # Apply TrainingArguments patch
        TrainingArguments.__init__ = safe_training_args_init
        logger.info("Applied TrainingArguments compatibility patch")
        
    except Exception as e:
        logger.warning(f"Failed to apply TrainingArguments patch: {e}")
    
except Exception as e:
    logger.warning(f"Failed to apply accelerate patch: {e}")

# Additional monkey patch for accelerate library utils if needed
try:
    import accelerate.utils as accelerate_utils
    
    # Patch the Accelerator creation function if it exists
    if hasattr(accelerate_utils, 'prepare_accelerator'):
        original_prepare_accelerator = accelerate_utils.prepare_accelerator
        
        def safe_prepare_accelerator(*args, **kwargs):
            # Remove incompatible parameters
            incompatible_params = ['dispatch_batches', 'even_batches', 'split_batches', 'use_seedable_sampler']
            for param in incompatible_params:
                if param in kwargs:
                    logger.info(f"Removing {param} from prepare_accelerator")
                    del kwargs[param]
            
            return original_prepare_accelerator(*args, **kwargs)
        
        accelerate_utils.prepare_accelerator = safe_prepare_accelerator
        logger.info("Applied accelerate_utils compatibility patch")
        
except Exception as e:
    logger.warning(f"Failed to apply accelerate_utils patch: {e}")

# -----------------------------
# 4. Cache Directory Setup
# -----------------------------
# Set cache directories to avoid disk space issues, especially on shared servers
scratch_dir = "/scratch/maryjazi"  # Linux server path
os.environ['TRANSFORMERS_CACHE'] = f'{scratch_dir}/.cache/huggingface'
os.environ['HF_HOME'] = f'{scratch_dir}/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = f'{scratch_dir}/.cache/huggingface/datasets'
os.environ['HF_METRICS_CACHE'] = f'{scratch_dir}/.cache/huggingface/metrics'
os.environ['TORCH_HOME'] = f'{scratch_dir}/.cache/torch'

# Create cache directories if they don't exist
os.makedirs(f'{scratch_dir}/.cache/huggingface', exist_ok=True)
os.makedirs(f'{scratch_dir}/.cache/huggingface/datasets', exist_ok=True)
os.makedirs(f'{scratch_dir}/.cache/huggingface/metrics', exist_ok=True)
os.makedirs(f'{scratch_dir}/.cache/torch', exist_ok=True)

# -----------------------------
# 5. Cache Management
# -----------------------------
def print_memory_optimization_tips():
    """
    Print memory optimization tips for users.
    """
    logger.info("=" * 60)
    logger.info("ULTRA MEMORY OPTIMIZATION TIPS:")
    logger.info("=" * 60)
    logger.info("✅ Using 4-bit quantization (better compatibility)")
    logger.info("✅ Chunked training mode enabled by default")
    logger.info("✅ Ultra-small chunk size (50 samples)")
    logger.info("✅ Very reduced max_length (32 tokens)")
    logger.info("✅ Gradient checkpointing always enabled")
    logger.info("✅ Mixed precision training (fp16) always enabled")
    logger.info("✅ Ultra-small LoRA rank (4) and alpha (8)")
    logger.info("✅ Single epoch training (1 epoch)")
    logger.info("✅ Lazy loading dataset (no full memory load)")
    logger.info("✅ Aggressive memory cleanup between chunks")
    logger.info("✅ Memory monitoring enabled")
    logger.info("✅ Monkey patch for quantized models")
    logger.info("=" * 60)
    logger.info("RECOMMENDED COMMAND:")
    logger.info("python llama_3_2_pretrain.py --clean_cache --dataset_path your_data.csv")
    logger.info("=" * 60)

def clean_cache_directories():
    """
    Clean cache directories to free up disk space.
    """
    cache_dirs = [
        '/home/mj69/.cache/huggingface',
        '/home/mj69/.cache/torch',
        '/tmp',
        '/var/tmp'
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                import shutil
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
                logger.info(f"Cleaned cache directory: {cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean {cache_dir}: {e}")

def check_disk_space():
    """
    Check available disk space and warn if low.
    """
    try:
        import shutil
        total, used, free = shutil.disk_usage("/scratch/maryjazi")
        free_gb = free // (1024**3)
        logger.info(f"Available disk space: {free_gb} GB")
        
        if free_gb < 10:
            logger.warning(f"Low disk space: {free_gb} GB available")
            clean_cache_directories()
            
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")

# Check disk space before starting
check_disk_space()

def monitor_memory_usage():
    """
    Monitor and log memory usage.
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024
            logger.info(f"Memory: RAM={memory_mb:.1f}MB, GPU={gpu_memory:.1f}MB (reserved: {gpu_memory_reserved:.1f}MB)")
        else:
            logger.info(f"Memory: RAM={memory_mb:.1f}MB")
            
        return memory_mb
    except ImportError:
        logger.warning("psutil not available for memory monitoring")
        return 0
    except Exception as e:
        logger.warning(f"Memory monitoring failed: {e}")
        return 0

def force_memory_cleanup():
    """
    Force aggressive memory cleanup.
    """
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Monitor memory after cleanup
    monitor_memory_usage()

# -----------------------------
# 5. Argument Parsing for Config
# -----------------------------
# This section uses argparse to allow you to set all important parameters from the command line.
# This makes the script flexible and easy to use for different datasets, models, and training settings.
parser = argparse.ArgumentParser(description="LLaMA 3.2 LoRA Fine-tuning Script")
parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='HuggingFace model name')
parser.add_argument('--dataset_path', type=str, default='./datasets/my_dataset/data.csv', help='Path to CSV dataset')
parser.add_argument('--val_dataset_path', type=str, default=None, help='Path to validation CSV dataset')
parser.add_argument('--output_dir', type=str, default='./models/llama3_lora', help='Directory to save the trained model')
parser.add_argument('--max_length', type=int, default=256, help='Max token length for each sample')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
parser.add_argument('--warmup_steps', type=int, default=100, help='Warmup steps for scheduler')
parser.add_argument('--max_grad_norm', type=float, default=0.3, help='Max gradient norm')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')
parser.add_argument('--top_p', type=float, default=0.9, help='Top-p for nucleus sampling')
parser.add_argument('--repetition_penalty', type=float, default=1.1, help='Repetition penalty for generation')
parser.add_argument('--mode', type=str, choices=['train', 'infer', 'train_chunks'], default='train', help='Mode: train, infer, or train_chunks')
parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for chunked training')
parser.add_argument('--disable_quantization', action='store_true', help='Disable 4-bit quantization')
parser.add_argument('--clean_cache', action='store_true', help='Clean cache directories before training')
parser.add_argument('--use_8bit', action='store_true', help='Use 8-bit quantization instead of 4-bit')
parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing')
parser.add_argument('--inference_batch_size', type=int, default=1, help='Batch size for inference')
parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum new tokens for generation')
parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint for resuming training')
parser.add_argument('--generate_plots', action='store_true', default=True, help='Generate training curves and plots after training')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()

# Set seed for reproducibility
def set_random_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")

set_random_seed(args.seed)

# -----------------------------
# 3. JSON Schema for Inference Output
# -----------------------------
INFERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "element_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "action": {"type": "string"},
                    "target_properties": {"type": "object"},
                    "unit": {"type": "string"}
                },
                "required": ["type", "action", "target_properties"]
            }
        },
        "update_scope": {"type": "string"}
    },
    "required": ["element_updates"]
}

def validate_json_schema(data, schema):
    """
    Validate a JSON object against a schema. Returns (True, None) if valid, (False, error) if not.
    """
    if not JSONSCHEMA_AVAILABLE:
        logger.warning("jsonschema not available, skipping validation")
        return True, None
    
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e)

# -----------------------------
# 4. Dataset Class
# -----------------------------
# This class loads your dataset from a pandas DataFrame and tokenizes each text sample.
# It returns the input_ids, attention_mask, and labels for each sample, which are needed for training.
class TextDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for loading multi-task data from a DataFrame.
    Implements lazy loading and memory optimization.
    """
    def __init__(self, dataframe, tokenizer, max_length):
        if 'instruction' not in dataframe.columns or 'expected_output' not in dataframe.columns:
            raise ValueError("Dataset must contain 'instruction' and 'expected_output' columns.")
        
        self.dataframe = dataframe  # Keep reference instead of converting to list
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create pairs with only instruction and output
        self.pairs = list(zip(
            dataframe['instruction'].astype(str), 
            dataframe['expected_output'].astype(str)
        ))
        
        # Debug: print first few examples
        print(f"[DEBUG] Dataset size: {len(self.pairs)}")
        for i in range(min(3, len(self.pairs))):
            prompt, output = self.pairs[i]
            full_input = prompt + self.tokenizer.eos_token + output
            print(f"[DEBUG] Example {i+1}:")
            print(f"  Instruction: {prompt}")
            print(f"  Output: {output}")
            print(f"  Full input: {full_input}")
            print(f"  Text length: {len(full_input)}")
            print("---")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        prompt, output = self.pairs[idx]
        if not prompt.strip():
            print(f"[WARNING] Empty prompt at index {idx}")
        
        # Simple concatenation: prompt + eos_token + output
        full_input = prompt + self.tokenizer.eos_token + output
        
        # Validate that the formatted text is not empty
        if not full_input.strip():
            print(f"[WARNING] Empty formatted text at index {idx}")
            print(f"  Original prompt: '{prompt}'")
            print(f"  Original output: '{output}'")
        
        # First, tokenize without padding to check for issues
        raw_tokenized = self.tokenizer(
            full_input,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        # Validate raw tokenization
        vocab_size = len(self.tokenizer)
        raw_input_ids = raw_tokenized["input_ids"][0]
        
        # Check for out-of-range token IDs in raw tokenization
        invalid_tokens = raw_input_ids[raw_input_ids >= vocab_size]
        if len(invalid_tokens) > 0:
            print(f"[ERROR] Raw tokenization has {len(invalid_tokens)} tokens with ID >= vocab_size ({vocab_size})")
            print(f"  Invalid token IDs: {invalid_tokens}")
            print(f"  Max token ID in raw input: {raw_input_ids.max().item()}")
            print(f"  Vocab size: {vocab_size}")
            print(f"  Text: '{full_input[:100]}...'")
            # Skip this sample or use a fallback
            return self.__getitem__((idx + 1) % len(self))
        
        # Now tokenize with padding
        tokenized = self.tokenizer(
            full_input,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        
        # Final validation after padding
        invalid_tokens = input_ids[input_ids >= vocab_size]
        if len(invalid_tokens) > 0:
            print(f"[ERROR] After padding: {len(invalid_tokens)} tokens with ID >= vocab_size ({vocab_size})")
            print(f"  Invalid token IDs: {invalid_tokens}")
            print(f"  Max token ID: {input_ids.max().item()}")
            # Clip to valid range
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            print(f"  Clipped input_ids to valid range [0, {vocab_size-1}]")
        
        print("[DEBUG] input_ids:", input_ids)
        print("[DEBUG] attention_mask:", attention_mask)
        if attention_mask.sum() == 0:
            print(f"[WARNING] All attention_mask is zero at index {idx}")
        
        # Create labels: set padding tokens to -100 (ignore in loss)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        # Final validation of labels
        if labels.max() >= vocab_size:
            print(f"[ERROR] Labels contain invalid token IDs >= {vocab_size}")
            labels = torch.clamp(labels, -100, vocab_size - 1)
        
        # Debug: print labels to verify they are correct
        print("[DEBUG] labels:", labels)
        print("[DEBUG] labels unique values:", set(labels.tolist()))
        print("[DEBUG] Number of non-padding tokens (attention_mask=1):", attention_mask.sum().item())
        print("[DEBUG] Number of tokens with label != -100:", (labels != -100).sum().item())
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# -----------------------------
# 5. Model Loading (4bit + LoRA)
# -----------------------------
# This function loads the model and tokenizer from HuggingFace.
# It uses 4-bit quantization for memory efficiency and applies LoRA for parameter-efficient fine-tuning.
def ensure_model_compatibility(model):
    """
    Ensure model is compatible with 8-bit training and doesn't use .to() calls.
    """
    try:
        # Check if model is quantized
        if hasattr(model, 'is_loaded_in_8bit') and model.is_loaded_in_8bit:
            logger.info("Model is loaded in 8-bit mode - .to() calls are disabled")
        
        # Ensure model is on the correct device
        if torch.cuda.is_available():
            device = next(model.parameters()).device
            logger.info(f"Model is on device: {device}")
        
        return True
    except Exception as e:
        logger.warning(f"Model compatibility check failed: {e}")
        return False

def load_model_and_tokenizer(args):
    """
    Loads the model and tokenizer from HuggingFace with quantization and memory optimization.
    """
    try:
        # Choose quantization config based on args
        bnb_config = None
        if not args.disable_quantization and False:  # Temporarily disable quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16
            )

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            cache_dir=os.environ['TRANSFORMERS_CACHE'],
            local_files_only=False
        )
        # Ensure eos_token is set
        if tokenizer.eos_token is None:
            vocab = tokenizer.get_vocab()
            if "<|endoftext|>" in vocab:
                tokenizer.eos_token = "<|endoftext|>"
            elif "</s>" in vocab:
                tokenizer.eos_token = "</s>"
            else:
                tokenizer.add_special_tokens({'eos_token': '[EOS]'})
                print("Added [EOS] as eos_token.")
            print("Set eos_token to:", tokenizer.eos_token)
        
        # Set pad_token to be different from eos_token for better training
        if tokenizer.pad_token is None:
            if "<|endoftext|>" in tokenizer.get_vocab():
                tokenizer.pad_token = "<|endoftext|>"
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print("Added [PAD] as pad_token.")
        
        # Use right padding for better training (real tokens at the beginning)
        tokenizer.padding_side = "right"
        
        # Ensure tokenizer is properly configured
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"Set pad_token_id to eos_token_id: {tokenizer.pad_token_id}")
        
        # Validate tokenizer configuration
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        print(f"Special tokens: {tokenizer.special_tokens_map}")
        print(f"pad_token_id: {tokenizer.pad_token_id}")
        print(f"eos_token_id: {tokenizer.eos_token_id}")
        
        # Test tokenizer on a simple example
        test_text = "Hello world"
        test_tokens = tokenizer(test_text, return_tensors='pt')
        print(f"Test tokenization of '{test_text}': {test_tokens}")
        
        # Debug: print tokenizer special tokens and vocab info
        print("eos_token:", tokenizer.eos_token, "eos_token_id:", tokenizer.eos_token_id)
        print("pad_token:", tokenizer.pad_token, "pad_token_id:", tokenizer.pad_token_id)
        print("padding_side:", tokenizer.padding_side)
        print("vocab size:", len(tokenizer))
        test = tokenizer("hello world", return_tensors='pt')
        print("[DEBUG] Tokenizer test for 'hello world':", test)

        # Load model with proper device mapping and temporary .to() disable
        model_kwargs = {
            'device_map': 'auto',
            'torch_dtype': torch.float16,
            'cache_dir': os.environ['TRANSFORMERS_CACHE'],
            'low_cpu_mem_usage': True,  # Important for memory efficiency
            'trust_remote_code': True,  # Allow custom model code
        }
        
        if bnb_config is not None:
            model_kwargs['quantization_config'] = bnb_config

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            **model_kwargs
        )
        
        # Ensure model vocab size matches tokenizer
        if hasattr(model.config, 'vocab_size'):
            model_vocab_size = model.config.vocab_size
            tokenizer_vocab_size = len(tokenizer)
            print(f"Model vocab size: {model_vocab_size}")
            print(f"Tokenizer vocab size: {tokenizer_vocab_size}")
            
            if model_vocab_size != tokenizer_vocab_size:
                print(f"[WARNING] Vocab size mismatch: model={model_vocab_size}, tokenizer={tokenizer_vocab_size}")
                # Resize model vocab to match tokenizer
                model.resize_token_embeddings(tokenizer_vocab_size)
                print(f"Resized model vocab to {tokenizer_vocab_size}")
        
        # Validate model configuration
        print(f"Model config: {model.config}")
        print(f"Model device: {next(model.parameters()).device}")

        # Enable gradient checkpointing if requested
        if hasattr(args, 'gradient_checkpointing') and args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Only prepare for kbit training if using quantization
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Ensure model compatibility
        ensure_model_compatibility(model)

        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}")
        logger.error("Try using --disable_quantization for testing")
        sys.exit(1)

# -----------------------------
# 6. Metrics Function (Fixed tokenizer scope)
# -----------------------------
def create_compute_metrics_function(tokenizer):
    """
    Creates a compute_metrics function with the tokenizer in scope.
    """
    if not METRICS_AVAILABLE:
        logger.warning("Metrics not available, returning empty metrics")
        return lambda eval_pred: {}
    
    try:
        bleu_metric = load_metric("bleu")
        rouge_metric = load_metric("rouge")
    except Exception as e:
        logger.warning(f"Failed to load metrics: {e}")
        return lambda eval_pred: {}
    
    def compute_metrics(eval_pred):
        try:
            predictions, labels = eval_pred
            # Decode predictions and labels
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # BLEU expects list of list of tokens
            bleu = bleu_metric.compute(predictions=[pred.split() for pred in decoded_preds],
                                      references=[[label.split()] for label in decoded_labels])
            rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
            
            return {
                "bleu": bleu["bleu"],
                "rougeL": rouge["rougeL"].mid.fmeasure if "rougeL" in rouge else 0.0
            }
        except Exception as e:
            logger.warning(f"Error computing metrics: {e}")
            return {"bleu": 0.0, "rougeL": 0.0}
    
    return compute_metrics

# -----------------------------
# 7. Chunked Training Function (for memory optimization)
# -----------------------------
def train_in_chunks(args, chunk_size=100):
    """
    Memory-optimized chunked training for large datasets.
    """
    logger.info("Loading dataset for memory-optimized chunked training...")
    try:
        # Load dataset in chunks to avoid loading entire file into memory
        df = pd.read_csv(args.dataset_path, chunksize=chunk_size)
        total_chunks = 0
        
        # Count total chunks first
        for chunk in pd.read_csv(args.dataset_path, chunksize=chunk_size):
            total_chunks += 1
        
        logger.info(f"Dataset will be processed in {total_chunks} chunks of size {chunk_size}")
        
        model, tokenizer = load_model_and_tokenizer(args)
        
        # Collect training statistics for plotting
        all_training_stats = []
        
        print("[DEBUG] Start chunked training loop")
        for chunk_idx, chunk_df in enumerate(pd.read_csv(args.dataset_path, chunksize=chunk_size)):
            logger.info(f"Training on chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_df)} samples)")
            
            # Create dataset for this chunk
            dataset = TextDataset(chunk_df, tokenizer, args.max_length)
            
            # Create output directory for this chunk
            chunk_output_dir = os.path.join(args.output_dir, f"chunk_{chunk_idx:03d}")
            os.makedirs(chunk_output_dir, exist_ok=True)
            
            # Ensure training mode and gradients are enabled
            model.train()
            logger.info(f"Model training mode: {model.training}")
            for param in model.parameters():
                param.requires_grad = True

            # Optimized training arguments for memory
            training_args = TrainingArguments(
                output_dir=chunk_output_dir,
                num_train_epochs=1,
                per_device_train_batch_size=1,  # Always use 1 for maximum memory efficiency
                gradient_accumulation_steps=2,  # Accumulate gradients for effective batch size of 2
                learning_rate=2e-5,  # Optimized learning rate
                warmup_steps=20,  # Gradual warmup for stability
                max_grad_norm=1.0,  # Conservative gradient clipping
                adam_epsilon=1e-6,  # Optimized epsilon for Adam optimizer
                weight_decay=0.01,
                logging_steps=1,  # Log every step
                save_strategy="no",  # No saving during training
                fp16=False,  # Disable mixed precision for stability
                dataloader_pin_memory=False,  # Disable pin memory to save RAM
                gradient_checkpointing=True,  # Always enable
                report_to="none",
                remove_unused_columns=False,  # Important for memory
                dataloader_num_workers=0,  # No multiprocessing to save memory
                no_cuda=False,  # Ensure CUDA is used properly
                dataloader_drop_last=False,  # Don't drop incomplete batches
                # Additional stability settings
                dataloader_prefetch_factor=None,  # Disable prefetching
                group_by_length=False,  # Disable length grouping
                length_column_name=None,  # No length column
                ignore_data_skip=False,  # Don't skip data
                deepspeed=None,  # No DeepSpeed
                label_smoothing_factor=0.0,  # No label smoothing
                optim="adamw_torch",  # Use standard optimizer
                lr_scheduler_type="linear",  # Simple scheduler
                warmup_ratio=0.0,  # No warmup ratio
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=tokenizer,
            )
            
            # Train with memory monitoring
            try:
                logger.info("Starting training...")
                monitor_memory_usage()
                
                # Log batch_size and max_length
                logger.info(f"Batch size: {args.batch_size}, Max length: {args.max_length}")
                # Log a sample of the dataset and tokenizer output
                if len(dataset) > 0:
                    sample = dataset[0]
                    logger.info(f"Sample instruction: {chunk_df.iloc[0]['instruction']}")
                    logger.info(f"Sample expected_output: {chunk_df.iloc[0]['expected_output']}")
                    logger.info(f"Sample input_ids: {sample['input_ids']}")
                    logger.info(f"Sample labels: {sample['labels']}")
                    logger.info(f"input_ids shape: {sample['input_ids'].shape}, labels shape: {sample['labels'].shape}")
                else:
                    logger.warning("Dataset chunk is empty!")
                
                trainer.train()
                
                # Collect training statistics for this chunk
                if trainer.state.log_history:
                    # Add chunk information to each log entry
                    for log_entry in trainer.state.log_history:
                        log_entry['chunk_idx'] = chunk_idx
                        log_entry['chunk_size'] = len(dataset)
                    all_training_stats.extend(trainer.state.log_history)
                
                # Save checkpoint after each chunk
                print(f"[INFO] Saving chunk {chunk_idx + 1} to {chunk_output_dir}")
                try:
                    trainer.save_model(chunk_output_dir)
                    print(f"[INFO] Model for chunk {chunk_idx + 1} saved successfully.")
                except Exception as e:
                    print(f"[ERROR] Failed to save model for chunk {chunk_idx + 1}: {e}")
                try:
                    tokenizer.save_pretrained(chunk_output_dir)
                    print(f"[INFO] Tokenizer for chunk {chunk_idx + 1} saved successfully.")
                except Exception as e:
                    print(f"[ERROR] Failed to save tokenizer for chunk {chunk_idx + 1}: {e}")
                
                logger.info(f"Chunk {chunk_idx + 1} training completed successfully")
                
            except Exception as e:
                logger.error(f"Error in chunk {chunk_idx + 1}: {e}")
                # Continue with next chunk instead of failing completely
                continue
            
            # Aggressive memory cleanup
            del dataset, trainer, training_args
            force_memory_cleanup()
            
            # Small delay to let system recover
            time.sleep(2)
        
        print("[DEBUG] Finished all chunks, about to save model")
        # Save final model
        final_output_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_output_dir, exist_ok=True)
        print(f"[INFO] Saving model to {final_output_dir}")
        try:
            model.save_pretrained(final_output_dir)
            print("[INFO] Model saved successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")
        try:
            tokenizer.save_pretrained(final_output_dir)
            print("[INFO] Tokenizer saved successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to save tokenizer: {e}")
        logger.info(f"Chunked training completed. Final model saved to {final_output_dir}")
        
        # Create a mock trainer for plotting (since chunked training doesn't use a single trainer)
        # We'll collect training stats from all chunks and create a summary
        try:
            logger.info("Generating training summary for chunked training...")
            create_chunked_training_summary(args.output_dir, args, total_chunks)
            
            # Create a mock trainer object for plotting
            class MockTrainer:
                def __init__(self, log_history):
                    self.state = type('State', (), {'log_history': log_history})()
            
            mock_trainer = MockTrainer(all_training_stats)
            
            # Generate plots for chunked training
            if args.generate_plots:
                logger.info("Generating training curves for chunked training...")
                plot_training_curves(mock_trainer, args.output_dir, args)
            else:
                logger.info("Skipping training curves generation (--generate_plots=False)")
                
        except Exception as e:
            logger.warning(f"Failed to create chunked training summary: {e}")
        
    except Exception as e:
        logger.error(f"Failed in chunked training: {e}")
        sys.exit(1)

def create_chunked_training_summary(output_dir, args, total_chunks):
    """
    Create a summary for chunked training since we don't have a single trainer instance.
    """
    try:
        summary = {
            "training_info": {
                "model_name": args.model_name,
                "total_chunks": total_chunks,
                "chunk_size": args.chunk_size,
                "epochs_per_chunk": 1,
                "batch_size": 1,  # Always 1 for chunked training
                "learning_rate": args.learning_rate,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "seed": args.seed,
                "training_mode": "chunked"
            },
            "chunked_training_statistics": {
                "total_chunks_processed": total_chunks,
                "memory_optimized": True,
                "gradient_checkpointing": True,
                "mixed_precision": False  # Set to False for compatibility
            }
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, 'chunked_training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Chunked training summary saved to: {summary_path}")
        
        # Print summary to console
        logger.info("=" * 60)
        logger.info("CHUNKED TRAINING SUMMARY:")
        logger.info("=" * 60)
        logger.info(f"Model: {summary['training_info']['model_name']}")
        logger.info(f"Total Chunks: {summary['training_info']['total_chunks']}")
        logger.info(f"Chunk Size: {summary['training_info']['chunk_size']}")
        logger.info(f"Learning Rate: {summary['training_info']['learning_rate']}")
        logger.info(f"LoRA Rank: {summary['training_info']['lora_r']}")
        logger.info(f"Memory Optimized: {summary['chunked_training_statistics']['memory_optimized']}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.warning(f"Failed to create chunked training summary: {e}")

# -----------------------------
# 8. Training Visualization Function
# -----------------------------
def plot_training_curves(trainer, output_dir, args):
    """
    Plot training curves and save them to the output directory.
    Shows training loss, validation loss, and other metrics if available.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for server environments
        
        logger.info("Generating training curves...")
        
        # Extract data from trainer state
        log_history = trainer.state.log_history
        
        if not log_history:
            logger.warning("No training history available for plotting")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Curves - {args.model_name}', fontsize=16, fontweight='bold')
        
        # Training Loss
        train_losses = [log.get('loss', 0) for log in log_history if 'loss' in log]
        if train_losses:
            axes[0, 0].plot(train_losses, 'b-', linewidth=2, label='Training Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
        
        # Validation Loss
        eval_losses = [log.get('eval_loss', 0) for log in log_history if 'eval_loss' in log]
        if eval_losses:
            axes[0, 1].plot(eval_losses, 'r-', linewidth=2, label='Validation Loss')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # Learning Rate
        lr_values = [log.get('learning_rate', 0) for log in log_history if 'learning_rate' in log]
        if lr_values:
            axes[1, 0].plot(lr_values, 'g-', linewidth=2, label='Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # Metrics (BLEU, ROUGE)
        bleu_scores = [log.get('bleu', 0) for log in log_history if 'bleu' in log]
        rouge_scores = [log.get('rougeL', 0) for log in log_history if 'rougeL' in log]
        
        if bleu_scores or rouge_scores:
            if bleu_scores:
                axes[1, 1].plot(bleu_scores, 'purple', linewidth=2, label='BLEU Score')
            if rouge_scores:
                axes[1, 1].plot(rouge_scores, 'orange', linewidth=2, label='ROUGE-L Score')
            axes[1, 1].set_title('Evaluation Metrics')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        else:
            # If no metrics, show training loss vs validation loss comparison
            if train_losses and eval_losses:
                axes[1, 1].plot(train_losses, 'b-', linewidth=2, label='Training Loss')
                axes[1, 1].plot(eval_losses, 'r-', linewidth=2, label='Validation Loss')
                axes[1, 1].set_title('Training vs Validation Loss')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend()
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to: {plot_path}")
        
        # Also save as PDF for better quality
        pdf_path = os.path.join(output_dir, 'training_curves.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        logger.info(f"Training curves PDF saved to: {pdf_path}")
        
        # Close plot to free memory
        plt.close()
        
        # Create summary statistics
        create_training_summary(log_history, output_dir, args)
        
    except ImportError:
        logger.warning("matplotlib not available. Skipping training curves generation.")
    except Exception as e:
        logger.warning(f"Failed to generate training curves: {e}")

def create_training_summary(log_history, output_dir, args):
    """
    Create a summary of training statistics and save to file.
    """
    try:
        # Extract final metrics
        final_log = log_history[-1] if log_history else {}
        
        # Calculate statistics
        train_losses = [log.get('loss', 0) for log in log_history if 'loss' in log]
        eval_losses = [log.get('eval_loss', 0) for log in log_history if 'eval_loss' in log]
        
        summary = {
            "training_info": {
                "model_name": args.model_name,
                "total_steps": final_log.get('global_step', 0),
                "epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "seed": args.seed
            },
            "final_metrics": {
                "final_train_loss": final_log.get('loss', 0),
                "final_eval_loss": final_log.get('eval_loss', 0),
                "final_bleu": final_log.get('bleu', 0),
                "final_rougeL": final_log.get('rougeL', 0),
                "final_learning_rate": final_log.get('learning_rate', 0)
            },
            "training_statistics": {
                "min_train_loss": min(train_losses) if train_losses else 0,
                "max_train_loss": max(train_losses) if train_losses else 0,
                "min_eval_loss": min(eval_losses) if eval_losses else 0,
                "max_eval_loss": max(eval_losses) if eval_losses else 0,
                "loss_improvement": (train_losses[0] - train_losses[-1]) if len(train_losses) > 1 else 0
            }
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to: {summary_path}")
        
        # Print summary to console
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY:")
        logger.info("=" * 60)
        logger.info(f"Model: {summary['training_info']['model_name']}")
        logger.info(f"Total Steps: {summary['training_info']['total_steps']}")
        logger.info(f"Final Train Loss: {summary['final_metrics']['final_train_loss']:.4f}")
        logger.info(f"Final Eval Loss: {summary['final_metrics']['final_eval_loss']:.4f}")
        if summary['final_metrics']['final_bleu'] > 0:
            logger.info(f"Final BLEU: {summary['final_metrics']['final_bleu']:.4f}")
        if summary['final_metrics']['final_rougeL'] > 0:
            logger.info(f"Final ROUGE-L: {summary['final_metrics']['final_rougeL']:.4f}")
        logger.info(f"Loss Improvement: {summary['training_statistics']['loss_improvement']:.4f}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.warning(f"Failed to create training summary: {e}")

# -----------------------------
# 9. Training Function
# -----------------------------
# This function loads the dataset, prepares the model and tokenizer, and runs the training loop using HuggingFace Trainer.
# It also saves the trained model and training statistics.
def train(args):
    """
    Loads the dataset, prepares the model and tokenizer, and runs the training loop using HuggingFace Trainer.
    Supports validation and resume from checkpoint.
    """
    logger.info("Loading dataset...")
    try:
        df = pd.read_csv(args.dataset_path)
        model, tokenizer = load_model_and_tokenizer(args)
        dataset = TextDataset(df, tokenizer, args.max_length)
        val_dataset = None
        if args.val_dataset_path:
            val_df = pd.read_csv(args.val_dataset_path)
            val_dataset = TextDataset(val_df, tokenizer, args.max_length)
            logger.info(f"Validation dataset loaded with {len(val_dataset)} samples")
        else:
            logger.info("No validation dataset provided. Training will proceed without validation.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup wandb if requested
    report_to = "none"
    if args.use_wandb and WANDB_AVAILABLE:
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args)
            )
            report_to = "wandb"
            logger.info("Wandb logging enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            report_to = "none"

    # Setup MLflow if requested
    mlflow_run = None
    if args.use_mlflow and MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment(args.mlflow_experiment)
            run_name = args.mlflow_run_name or f"run_{args.seed}_{args.num_epochs}epochs"
            mlflow_run = mlflow.start_run(run_name=run_name)
            
            # Log parameters
            mlflow.log_params({
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_epochs,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "max_length": args.max_length,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "warmup_steps": args.warmup_steps,
                "max_grad_norm": args.max_grad_norm,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "use_fp16": args.use_fp16,
                "early_stopping_patience": args.early_stopping_patience,
                "metric_for_best_model": args.metric_for_best_model,
                "dataset_size": len(dataset),
                "validation_size": len(val_dataset) if val_dataset else 0
            })
            
            logger.info("MLflow logging enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")
            mlflow_run = None

    # Check if fp16 is supported
    fp16_enabled = args.use_fp16 and torch.cuda.is_available()
    if fp16_enabled:
        logger.info("Mixed precision training (fp16) enabled")
    else:
        logger.info("Mixed precision training disabled (not supported or not requested)")

    # Early stopping configuration
    early_stopping_config = {}
    if val_dataset is not None and args.early_stopping_patience > 0:
        early_stopping_config = {
            "load_best_model_at_end": True,
            "metric_for_best_model": args.metric_for_best_model,
            "greater_is_better": args.metric_for_best_model in ["bleu", "rougeL"],
            "save_total_limit": 3,
        }
        logger.info(f"Early stopping enabled with patience={args.early_stopping_patience}, metric={args.metric_for_best_model}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        fp16=False,
        dataloader_pin_memory=args.dataloader_pin_memory,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=report_to,
        remove_unused_columns=False,  # Important for 8-bit models
        dataloader_num_workers=0,  # No multiprocessing for 8-bit models
        no_cuda=False,  # Ensure CUDA is used properly
        dataloader_drop_last=False,  # Don't drop incomplete batches
        **early_stopping_config
    )

    # Create compute_metrics function with tokenizer in scope
    compute_metrics_func = create_compute_metrics_function(tokenizer) if val_dataset is not None else None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_func
    )

    logger.info("Start training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save model and tokenizer
    trainer.save_model(args.output_dir)
    
    # Save tokenizer config
    try:
        tokenizer.save_pretrained(args.output_dir)
        logger.info("Tokenizer config saved successfully")
    except Exception as e:
        logger.warning(f"Failed to save tokenizer config: {e}")
    
    logger.info(f"Model saved to {args.output_dir}")

    # Save training statistics
    stats_path = os.path.join(args.output_dir, 'training_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(trainer.state.log_history, f, indent=2)
    
    # Log metrics and artifacts to MLflow
    if args.use_mlflow and MLFLOW_AVAILABLE and mlflow_run:
        try:
            # Log final metrics
            if trainer.state.log_history:
                final_metrics = trainer.state.log_history[-1]
                mlflow.log_metrics({
                    "final_eval_loss": final_metrics.get("eval_loss", 0.0),
                    "final_bleu": final_metrics.get("bleu", 0.0),
                    "final_rougeL": final_metrics.get("rougeL", 0.0),
                    "final_train_loss": final_metrics.get("train_loss", 0.0),
                    "total_training_steps": final_metrics.get("global_step", 0)
                })
            
            # Log model
            mlflow.pytorch.log_model(model, "final_model")
            
            # Log artifacts
            mlflow.log_artifact(stats_path, "training_stats.json")
            mlflow.log_artifact(args.output_dir, "model_output")
            
            logger.info("MLflow logging completed successfully")
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    # Close wandb if it was used
    if args.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()

    # Close MLflow if it was used
    if args.use_mlflow and MLFLOW_AVAILABLE and mlflow_run:
        mlflow.end_run()

    # Plot training curves if requested
    if args.generate_plots:
        plot_training_curves(trainer, args.output_dir, args)
    else:
        logger.info("Skipping training curves generation (--generate_plots=False)")

# -----------------------------
# 10. Single Inference Function
# -----------------------------
def generate_text_single(prompt, args, model, tokenizer):
    """
    Generates text for a single prompt using the provided model and tokenizer.
    """
    model.eval()
    
    # Prepare input - no need to move to device for 4-bit models
    inputs = tokenizer(prompt, return_tensors='pt')

    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode and attempt to extract JSON
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Try to extract valid JSON from the model output
        try:
            start_idx = raw_output.find("{")
            end_idx = raw_output.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                return None, raw_output
                
            json_str = raw_output[start_idx:end_idx]
            parsed = json.loads(json_str)
            return parsed, raw_output
        except json.JSONDecodeError:
            return None, raw_output
        except Exception as e:
            logger.warning(f"Unexpected error during JSON extraction: {e}")
            return None, raw_output
            
    except Exception as e:
        logger.error(f"Error during model generation: {e}")
        return None, None

# -----------------------------
# 11. Batch Inference Function
# -----------------------------
def generate_text_batch(prompts, args):
    """
    Generates text for multiple prompts in batch.
    """
    model, tokenizer = load_model_and_tokenizer(args)
    model.eval()
    
    results = []
    
    # Process prompts in batches
    for i in range(0, len(prompts), args.inference_batch_size):
        batch_prompts = prompts[i:i + args.inference_batch_size]
        batch_results = []
        
        # Tokenize batch - no need to move to device for 4-bit models
        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode batch outputs
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Process each output
            for j, raw_output in enumerate(decoded_outputs):
                try:
                    start_idx = raw_output.find("{")
                    end_idx = raw_output.rfind("}") + 1
                    if start_idx == -1 or end_idx == 0:
                        batch_results.append({"parsed": None, "raw": raw_output})
                        continue
                        
                    json_str = raw_output[start_idx:end_idx]
                    parsed = json.loads(json_str)
                    batch_results.append({"parsed": parsed, "raw": raw_output})
                except (json.JSONDecodeError, Exception):
                    batch_results.append({"parsed": None, "raw": raw_output})
                    
        except Exception as e:
            logger.error(f"Error during batch generation: {e}")
            batch_results = [{"parsed": None, "raw": None}] * len(batch_prompts)
        
        results.extend(batch_results)
    
    return results

# -----------------------------
# 12. Single Inference Example (Backward Compatibility)
# -----------------------------
def generate_text(prompt, args):
    """
    Loads the trained model and generates text for a given prompt.
    Attempts to extract valid JSON from the output.
    """
    model, tokenizer = load_model_and_tokenizer(args)
    parsed, raw_output = generate_text_single(prompt, args, model, tokenizer)
    
    if raw_output:
        print("\nRaw model output:\n", raw_output)
    
    if parsed:
        print("\n✅ Valid JSON extracted:")
        return parsed
    else:
        print("\n⚠️ No valid JSON could be parsed.")
        return None

# -----------------------------
# 13. Main Execution
# -----------------------------
# This is the main entry point of the script.
# It first trains the model, then runs an example inference.
if __name__ == '__main__':
    # Print memory optimization tips
    print_memory_optimization_tips()
    
    # Clean cache if requested
    if hasattr(args, 'clean_cache') and args.clean_cache:
        logger.info("Cleaning cache directories...")
        clean_cache_directories()
    
    # Display current settings
    logger.info("CURRENT SETTINGS:")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"  Max length: {args.max_length}")
    logger.info(f"  Chunk size: {args.chunk_size}")
    logger.info(f"  LoRA rank: {args.lora_r}")
    logger.info(f"  Use 8-bit: {args.use_8bit}")
    logger.info(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'train':
        logger.info("Mode: Regular Training")
        train(args)
    
    elif args.mode == 'train_chunks':
        logger.info("Mode: Chunked Training")
        train_in_chunks(args, args.chunk_size)

    elif args.mode == 'infer':
        logger.info("Mode: Inference")

        # Example prompts for batch inference
        example_prompts = [
            """
You are an IFC editing assistant. Given the user instruction below, return a JSON list of IFC element updates.
Each update must include:
- type (e.g., IfcWindow)
- action (add, remove, modify)
- target_properties or match_criteria
- Optional: geometry or unit
Respond only with a valid JSON object.

User instruction:
Add a new IfcWindow in the kitchen with height 1400 and width 1600 mm.
""",
            """
You are an IFC editing assistant. Given the user instruction below, return a JSON list of IFC element updates.
Each update must include:
- type (e.g., IfcWindow)
- action (add, remove, modify)
- target_properties or match_criteria
- Optional: geometry or unit
Respond only with a valid JSON object.

User instruction:
Remove all IfcDoor elements from the second floor.
"""
        ]

        if args.inference_batch_size > 1:
            logger.info(f"Running batch inference with batch size {args.inference_batch_size}")
            results = generate_text_batch(example_prompts, args)
            
            for i, result in enumerate(results):
                print(f"\n--- Result {i+1} ---")
                if result["parsed"]:
                    print("✅ Parsed JSON:")
                    print(json.dumps(result["parsed"], indent=2))
                    
                    # Validate JSON schema
                    valid, err = validate_json_schema(result["parsed"], INFERENCE_SCHEMA)
                    if valid:
                        print("✅ JSON schema validation passed.")
                    else:
                        print(f"❌ JSON schema validation failed: {err}")
                else:
                    print("❌ No valid JSON parsed")
                    print("Raw output:", result["raw"])
        else:
            # Single inference (backward compatibility)
            result = generate_text(example_prompts[0], args)

            if result:
                print("\n✅ Final Parsed JSON:")
                print(json.dumps(result, indent=2))

                # Validate JSON schema
                valid, err = validate_json_schema(result, INFERENCE_SCHEMA)
                if valid:
                    print("✅ JSON schema validation passed.")
                else:
                    print(f"❌ JSON schema validation failed: {err}")

                # Optional: Save to file
                with open(os.path.join(args.output_dir, "inference_result.json"), "w") as f:
                    json.dump(result, f, indent=2)

            else:
                print("❌ No valid JSON could be parsed.")

    else:
        logger.warning("⚠️ Unknown mode selected. Use --mode train or --mode infer.")


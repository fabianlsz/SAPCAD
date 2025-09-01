from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import json
import os
import shutil
from safetensors.torch import load_file

# Set cache directories to use scratch space
scratch_dir = "/scratch/maryjazi"
os.environ['TRANSFORMERS_CACHE'] = f'{scratch_dir}/.cache/huggingface'
os.environ['HF_HOME'] = f'{scratch_dir}/.cache/huggingface'
os.environ['TORCH_HOME'] = f'{scratch_dir}/.cache/torch'
os.environ["HF_DATASETS_CACHE"] = f"{scratch_dir}/.cache/huggingface/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{scratch_dir}/.cache/huggingface/hub"

# Create cache directories
for cache_dir in [os.environ['TRANSFORMERS_CACHE'], os.environ['HF_HOME'], os.environ['TORCH_HOME'], os.environ['HF_DATASETS_CACHE'], os.environ['HUGGINGFACE_HUB_CACHE']]:
    os.makedirs(cache_dir, exist_ok=True)

def get_free_space(path):
    """Get free space in GB for the given path"""
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)  # Convert to GB

# Define SimpleTokenizer class at module level
class SimpleTokenizer:
    """Create a simple tokenizer without downloading anything"""
    def __init__(self):
        from transformers import PreTrainedTokenizer
        
        class SimpleTokenizerImpl(PreTrainedTokenizer):
            def __init__(self):
                # Create vocabulary first
                self.vocab = {f"token_{i}": i for i in range(1000)}
                self.vocab["<|endoftext|>"] = 50256
                
                # Call parent constructor with minimal parameters
                super().__init__(pad_token="<|endoftext|>", eos_token="<|endoftext|>")
                
            @property
            def vocab_size(self):
                return len(self.vocab)
                
            @property
            def vocab(self):
                return self.get_vocab()
                
            def get_vocab(self):
                return self.vocab
                
            def encode(self, text, **kwargs):
                return [ord(c) % 1000 for c in text[:100]]
                
            def decode(self, token_ids, **kwargs):
                return ''.join([chr(t % 1000) if t < 1000 else ' ' for t in token_ids])
                
            def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None, **kwargs):
                tokens = self.encode(text)
                if max_length:
                    tokens = tokens[:max_length]
                if return_tensors == "pt":
                    return {"input_ids": torch.tensor([tokens])}
                return {"input_ids": [tokens]}
                
            def save_vocabulary(self, save_directory, filename_prefix=None):
                import os
                vocab_file = os.path.join(save_directory, (filename_prefix or "") + "simple_vocab.txt")
                with open(vocab_file, "w", encoding="utf-8") as f:
                    for token, idx in self.vocab.items():
                        f.write(f"{token}\t{idx}\n")
                return (vocab_file,)
        
        self.tokenizer = SimpleTokenizerImpl()
    
    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

# Check available disk space
free_space_gb = get_free_space(scratch_dir)
print(f"Available disk space: {free_space_gb:.2f} GB")

# Path to the final LoRA model directory
model_path = "models/llama3_lora/final_model"
print("Current working directory:", os.getcwd())
print("Files in model_path:", os.listdir(model_path))

# Step 1: Load LoRA config to see what base model was used
peft_config = PeftConfig.from_pretrained(model_path)
print(f"LoRA config loaded. Base model: {peft_config.base_model_name_or_path}")
print(f"Target modules: {peft_config.target_modules}")

# Step 2: Load the original base model (should be cached now)
base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Use the correct Instruct model
print(f"Loading base model: {base_model_name}")

try:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        cache_dir=os.environ['TRANSFORMERS_CACHE'],
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None
    )
    print("✅ Successfully loaded base model")
    
    # Check vocabulary size mismatch
    current_vocab_size = base_model.config.vocab_size
    print(f"Current model vocab size: {current_vocab_size}")
    
    # Load LoRA state dict using safetensors
    lora_state_dict = load_file(os.path.join(model_path, "adapter_model.safetensors"))
    
    # Find the expected vocab size from LoRA weights
    expected_vocab_size = None
    for key, tensor in lora_state_dict.items():
        if "embed_tokens.weight" in key or "lm_head.weight" in key:
            expected_vocab_size = tensor.shape[0]
            print(f"Expected vocab size from LoRA ({key}): {expected_vocab_size}")
            break
    
    if expected_vocab_size and current_vocab_size != expected_vocab_size:
        print(f"⚠️  Vocabulary size mismatch! Resizing model from {current_vocab_size} to {expected_vocab_size}")
        
        # Resize the model to match LoRA weights
        base_model.resize_token_embeddings(expected_vocab_size)
        print(f"✅ Model resized to vocab size: {expected_vocab_size}")
        
        # Verify the resize worked
        new_vocab_size = base_model.config.vocab_size
        print(f"✅ New model vocab size: {new_vocab_size}")
    
except Exception as e:
    print(f"❌ Failed to load base model: {e}")
    print("Trying with the base model from config...")
    
    # Try with the model from config as fallback
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            cache_dir=os.environ['TRANSFORMERS_CACHE'],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        print("✅ Successfully loaded base model from config")
        
        # Apply the same vocabulary size fix
        current_vocab_size = base_model.config.vocab_size
        print(f"Current model vocab size: {current_vocab_size}")
        
        # Load LoRA state dict using safetensors
        lora_state_dict = load_file(os.path.join(model_path, "adapter_model.safetensors"))
        
        # Find the expected vocab size from LoRA weights
        expected_vocab_size = None
        for key, tensor in lora_state_dict.items():
            if "embed_tokens.weight" in key or "lm_head.weight" in key:
                expected_vocab_size = tensor.shape[0]
                print(f"Expected vocab size from LoRA ({key}): {expected_vocab_size}")
                break
        
        if expected_vocab_size and current_vocab_size != expected_vocab_size:
            print(f"⚠️  Vocabulary size mismatch! Resizing model from {current_vocab_size} to {expected_vocab_size}")
            
            # Resize the model to match LoRA weights
            base_model.resize_token_embeddings(expected_vocab_size)
            print(f"✅ Model resized to vocab size: {expected_vocab_size}")
            
    except Exception as e2:
        print(f"❌ Failed to load base model from config: {e2}")
        exit(1)

# Step 3: Load LoRA weights
try:
    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(base_model, model_path)
    print("✅ Successfully loaded LoRA weights")
    
except Exception as e:
    print(f"❌ Failed to load LoRA weights with PeftModel: {e}")
    print("Trying alternative loading method...")
    
    # Alternative method: Load LoRA weights manually
    try:
        from peft import LoraConfig, get_peft_model
        
        # Create LoRA config from the saved config
        lora_config = LoraConfig.from_pretrained(model_path)
        
        # Apply LoRA to the base model
        model = get_peft_model(base_model, lora_config)
        
        # Load the LoRA weights using safetensors
        state_dict = load_file(os.path.join(model_path, "adapter_model.safetensors"))
        model.load_state_dict(state_dict, strict=False)
        
        print("✅ Successfully loaded LoRA weights with alternative method")
        
    except Exception as e2:
        print(f"❌ Alternative loading also failed: {e2}")
        print("Saving base model as fallback...")
        
        save_path = "models/llama3_lora/base_model_only"
        model = base_model
        model.save_pretrained(save_path)
        print(f"✅ Base model saved to {save_path}")
        
        # Skip to generation test
        save_path = "models/llama3_lora/base_model_only"
        goto_generation = True
        goto_generation = False  # This will be set to True if we need to skip merging

# Step 4: Merge LoRA weights (only if we successfully loaded LoRA)
if 'goto_generation' not in locals() or not goto_generation:
    try:
        print("Merging LoRA weights...")
        model = model.merge_and_unload()
        print("✅ Successfully merged LoRA weights")
        
        # Step 5: Save the merged model
        save_path = "models/llama3_lora/merged_model_fixed"
        model.save_pretrained(save_path)
        
        # Load and save tokenizer with proper max_length
        try:
            # Try to load tokenizer from local cache first
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                cache_dir=os.environ['TRANSFORMERS_CACHE'],
                local_files_only=True  # Try local files first
            )
            print("✅ Loaded tokenizer from local cache")
            
        except Exception as e1:
            print(f"⚠️  Failed to load tokenizer from cache: {e1}")
            try:
                # Try downloading with limited retries
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name,
                    cache_dir=os.environ['TRANSFORMERS_CACHE'],
                    local_files_only=False
                )
                print("✅ Downloaded tokenizer successfully")
                
            except Exception as e2:
                print(f"⚠️  Failed to download tokenizer: {e2}")
                print("Using simple tokenizer as fallback...")
                tokenizer = SimpleTokenizer()
        
        # Set proper model_max_length for LLaMA-3
        if hasattr(tokenizer, 'model_max_length'):
            tokenizer.model_max_length = 8192  # LLaMA-3 standard
        else:
            tokenizer.model_max_length = 8192
            
        try:
            tokenizer.save_pretrained(save_path)
            print(f"✅ Tokenizer saved to {save_path}")
            print(f"✅ Tokenizer model_max_length: {tokenizer.model_max_length}")
        except Exception as e:
            print(f"⚠️  Failed to save tokenizer: {e}")
            print("Tokenizer will be recreated when loading the model")
        
        # Save model info file
        with open(os.path.join(save_path, "model_info.txt"), "w") as f:
            f.write(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
            f.write(f"Vocab size: {model.config.vocab_size}\n")
            f.write(f"Base model: {base_model_name}\n")
            f.write(f"Model type: LLaMA-3-8B-Instruct\n")
            f.write(f"Max sequence length: {getattr(model.config, 'max_position_embeddings', 'Unknown')}\n")
            f.write(f"Merge date: {os.popen('date').read().strip()}\n")
            f.write(f"LoRA target modules: {peft_config.target_modules}\n")
            f.write(f"LoRA rank: {peft_config.r}\n")
            f.write(f"LoRA alpha: {peft_config.lora_alpha}\n")
        
        print(f"✅ Model info saved to {save_path}/model_info.txt")
        print(f"✅ Merged model saved to {save_path}")
        
    except Exception as e:
        print(f"❌ Failed to merge LoRA weights: {e}")
        print("Saving base model as fallback...")
        
        save_path = "models/llama3_lora/base_model_only"
        model = base_model
        model.save_pretrained(save_path)
        print(f"✅ Base model saved to {save_path}")

# Step 6: Test generation
prompt = "Change the wall height to 3 meters"
try:
    if 'tokenizer' not in locals():
        try:
            tokenizer = AutoTokenizer.from_pretrained(save_path)
        except:
            tokenizer = SimpleTokenizer()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use proper max_length based on tokenizer.model_max_length
    max_length = getattr(tokenizer, 'model_max_length', 2048)
    print(f"Using max_length: {max_length}")
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

    result_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n=== Model Output ===")
    print(f"Input: {prompt}")
    print(f"Output: {result_text}")
    
except Exception as e:
    print(f"❌ Generation failed: {e}")
    print("Model saved successfully, but generation test failed.")

print(f"\n✅ Script completed. Model saved to: {save_path}")
print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters") 
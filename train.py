import os
import json
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from PIL import Image
from typing import Dict, List, Any
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Disable wandb and other logging services
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "true"

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
IMAGE_DIR = "./images"
JSONL_TRAIN = "./src/data/qa_train.jsonl"
JSONL_VALIDATION = "./src/data/qa_val.jsonl"
OUTPUT_DIR = "./llava-finetuned"
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 3
PER_IMAGE_CAP = 6
USE_LORA = True  # Recommended: prevents overfitting, preserves base capabilities
LEARNING_RATE = 2e-5 if USE_LORA else 5e-6
MAX_SEQ_LENGTH = 2048  # Maximum sequence length


# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================
def load_and_process_data() -> Dict[str, Dataset]:
    """Load and process dataset from JSONL file"""
    print("Loading dataset...")
    
    try:
        dataset = load_dataset(
            "json",
            data_files={"train": JSONL_TRAIN, "test": JSONL_VALIDATION},
        )
        print(f"Loaded {len(dataset['train'])} training and {len(dataset['test'])} validation examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    def format_conversations(conversations):
        """
        Format conversations for LLaVA
        CRITICAL: Don't add default text to prevent overfitting
        """
        conversation_text = ""
        current_user = ""
        has_image_token = False

        for turn in conversations:
            if turn["from"] == "human":
                current_user = turn['value'].strip()
                if '<image>' in current_user:
                    has_image_token = True
            elif turn["from"] == "gpt" and current_user:
                if conversation_text:
                    conversation_text += "\n"
                conversation_text += f"USER: {current_user}\nASSISTANT: {turn['value']}"
                current_user = ""

        # CRITICAL FIX: Return None for invalid examples instead of adding default text
        # This prevents the model from always talking about dermatology
        if not conversation_text:
            return None
        
        # Add image token if missing
        if not has_image_token:
            conversation_text = "<image>\n" + conversation_text

        return conversation_text

    def process_example(example):
        """Process single example"""
        try:
            conversations = example["conversations"]
            text = format_conversations(conversations)
            
            # Skip invalid examples
            if text is None:
                return None

            return {
                "image_path": os.path.join(IMAGE_DIR, os.path.basename(example["image"])),
                "text": text,
                "id": example["id"]
            }
        except Exception as e:
            print(f"Warning: Error processing {example.get('id', 'unknown')}: {e}")
            return None

    print("Processing conversations...")
    def capped_indices(split):
        counts = {}
        keep = []
        for i, image in enumerate(split["image"]):
            name = os.path.basename(image)
            if counts.get(name, 0) < PER_IMAGE_CAP:
                keep.append(i)
                counts[name] = counts.get(name, 0) + 1
        return keep

    dataset["train"] = dataset["train"].select(capped_indices(dataset["train"]))

    processed_dataset = dataset.map(
        process_example,
        remove_columns=dataset["train"].column_names,
        desc="Processing"
    )
    
    # Filter out None values (invalid examples)
    processed_dataset = processed_dataset.filter(lambda x: x["text"] is not None)
    print(f"Valid examples after filtering: {len(processed_dataset)}")
    
    if len(processed_dataset["train"]) == 0:
        raise ValueError("No valid examples found after processing!")

    print(f"Train: {len(processed_dataset['train'])}, Validation: {len(processed_dataset['test'])}")
    return processed_dataset


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model_and_processor():
    """Load model with optional LoRA and processor"""
    
    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    processor.tokenizer.padding_side = "right"  # Required for training
    
    # Determine dtype
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
        use_bf16 = True
        print("Using bfloat16 precision")
    else:
        model_dtype = torch.float16
        use_bf16 = False
        print("Using float16 precision")

    if USE_LORA:
        print("Loading model with LoRA (recommended - preserves base capabilities)...")
        
        # Quantization config for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_dtype,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model with quantization
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        print("\nTrainable parameters:")
        model.print_trainable_parameters()
        
    else:
        print("Loading model for full fine-tuning...")
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=model_dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    
    return model, processor, use_bf16


# ============================================================================
# DATA COLLATOR
# ============================================================================
def create_data_collator(processor, model):
    """Create data collator function"""
    
    def data_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of features into model inputs
        FIXED: No truncation to avoid image token mismatch
        """
        model_device = next(model.parameters()).device

        images = []
        texts = [feature["text"] for feature in features]

        # Load images
        for feature in features:
            try:
                image_path = feature["image_path"]
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    image = Image.new('RGB', (224, 224), color='white')
                else:
                    image = Image.open(image_path).convert("RGB")
                images.append(image)
            except Exception as e:
                print(f"Error loading image: {str(e)}")
                images.append(Image.new('RGB', (224, 224), color='white'))

        # Process inputs - CRITICAL: No truncation to avoid image token mismatch
        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=False,  # Don't truncate - prevents image token mismatch error
            return_tensors="pt"
        )

        # Create labels (copy of input_ids with padding masked)
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        # Move all tensors to model device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(model_device)

        return inputs
    
    return data_collator


# ============================================================================
# TRAINING
# ============================================================================
def train_model():
    """Main training function"""
    print("\n" + "=" * 60)
    print("LLAVA FINE-TUNING")
    print("=" * 60)
    print(f"Base model: {MODEL_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Training method: {'LoRA' if USE_LORA else 'Full fine-tuning'}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 60 + "\n")

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    # Load model and processor
    model, processor, use_bf16 = load_model_and_processor()
    
    # Load dataset
    dataset = load_and_process_data()
    
    # Create data collator
    data_collator = create_data_collator(processor, model)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        warmup_steps=50,
        weight_decay=0.01,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb, tensorboard, mlflow, etc.
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        push_to_hub=False,
        dataloader_num_workers=0,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )
    
    # Test single batch
    print("\nTesting single batch...")
    test_features = [dataset["train"][i] for i in range(min(2, len(dataset["train"])))]
    test_batch = data_collator(test_features)
    
    print(f"Batch keys: {test_batch.keys()}")
    print(f"Input shape: {test_batch['input_ids'].shape}")
    print(f"Pixel values shape: {test_batch['pixel_values'].shape if 'pixel_values' in test_batch else 'N/A'}")
    
    with torch.no_grad():
        test_outputs = model(**test_batch)
    
    print(f"Batch test successful - Loss: {test_outputs.loss:.4f}\n")
    
    # Train
    print("Starting training...")
    print("=" * 60)
    trainer.train()
    print("=" * 60)
    print("Training completed!\n")
    
    # Save model
    print("Saving model...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if USE_LORA:
        # Save LoRA adapters only
        model.save_pretrained(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        print(f"✓ LoRA adapters saved to {OUTPUT_DIR}")
        print("  To use: Load base model + merge these adapters with PEFT")
    else:
        # Save full model
        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        print(f"✓ Full model saved to {OUTPUT_DIR}")
    
    # Save training info
    model_info = {
        "model_type": "llava_with_lora" if USE_LORA else "llava_full_finetuned",
        "base_model": MODEL_NAME,
        "training_method": "LoRA" if USE_LORA else "Full fine-tuning",
        "training_samples": len(dataset["train"]),
        "validation_samples": len(dataset["test"]),
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05
        } if USE_LORA else None
    }
    
    with open(os.path.join(OUTPUT_DIR, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✓ Training info saved to {OUTPUT_DIR}/model_info.json")
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Verify paths
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")
    if not os.path.exists(JSONL_PATH):
        raise FileNotFoundError(f"JSONL file not found: {JSONL_PATH}")

    # Check PEFT installation if using LoRA
    if USE_LORA:
        try:
            import peft
            import bitsandbytes
        except ImportError:
            print("ERROR: LoRA requires 'peft' and 'bitsandbytes'")
            print("Install with: pip install peft bitsandbytes")
            exit(1)

    # Run training
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        import traceback
        traceback.print_exc()

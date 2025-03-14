# -*- coding: utf-8 -*-
"""Gemma2 9B Continued Pretraining with Domain Knowledge.ipynb"""

# Install Unsloth
# %%capture
# pip install unsloth
# # Get the latest nightly Unsloth
# pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

# # Install Flash Attention 2 for softcapping support
# import torch
# if torch.cuda.get_device_capability()[0] >= 8:
#     !pip install --no-deps packaging ninja einops "flash-attn>=2.6.3"

# Install Flash Attention 2 for softcapping support
import torch
import subprocess
import sys

if torch.cuda.get_device_capability()[0] >= 8:
    # Use subprocess to install packages when running as a script
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", 
                          "packaging", "ninja", "einops", "flash-attn>=2.6.3"])
    
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048  # You can adjust this based on your document length
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage

# Load Gemma2 9B model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-9b",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # Use if needed for gated models
)

# Add LoRA adapters including embed_tokens and lm_head for continued pretraining
model = FastLanguageModel.get_peft_model(
    model,
    r = 128,  # Higher rank for domain adaptation
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "embed_tokens", "lm_head"],  # Include embeddings for domain adaptation
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = True,  # Rank stabilized LoRA recommended for continued pretraining
    loftq_config = None,
)

# Code to load and preprocess the markdown files
import os
import re
from datasets import Dataset

def load_markdown_files(file_paths):
    """Load content from markdown files and return as a list of texts."""
    documents = []
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Remove any markdown artifacts if needed
            # This is a basic cleaning - you might need more specific processing
            # content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)  # Remove code blocks if desired
            documents.append(content)
    
    return documents

# File paths to your markdown documents
md_file_paths = [
    "../transciptions/summary_20250220.md", 
    "../transciptions/summary_20250225.md"
]

# Load the documents
documents = load_markdown_files(md_file_paths)

# Create a dataset format
def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into chunks with overlap for better learning."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 100:  # Only include substantial chunks
            chunks.append(chunk)
    return chunks

# Process all documents into chunks
all_chunks = []
for doc in documents:
    all_chunks.extend(chunk_text(doc))

# Format data as expected by the model
formatted_texts = []

# Simple document format
document_format = """Knowledge Document:
{}"""

for chunk in all_chunks:
    formatted_text = document_format.format(chunk)
    formatted_texts.append(formatted_text)

# Create a dataset
dataset = Dataset.from_dict({"text": formatted_texts})
print(f"Created dataset with {len(dataset)} examples")

# Sample an example to verify
print("\nSample example:")
print(dataset[0]["text"][:500] + "...")

# Set up the DataCollatorForLanguageModeling for continued pretraining
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal language modeling, not masked language modeling
)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=max_seq_length)

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=2,
    remove_columns=["text"]
)

# Set up the UnslothTrainer for continued pretraining
from transformers import Trainer, TrainingArguments
from unsloth import is_bfloat16_supported

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_steps=200,
    warmup_steps=20,
    learning_rate=5e-5,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Run the training
trainer_stats = trainer.train()

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory/max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Save the model
model.save_pretrained("gemma2_domain_model")
tokenizer.save_pretrained("gemma2_domain_model")

# Enable inference mode for testing
FastLanguageModel.for_inference(model)

# Test the model with a domain-specific query
test_prompt = """Knowledge Document:
Обясни процеса на съдебно събиране на вземания."""

inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=300, temperature=0.7)
#!pip install -q peft transformers accelerate bitsandbytes datasets

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from huggingface_hub import login
login("hf_O")

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    token=True
)

from peft import LoraConfig, get_peft_model, TaskType

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

import os
os.environ["WANDB_DISABLED"] = "true"



import json
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import Trainer, TrainingArguments

# Step 1: Load mimic_all.jsonl
with open("/content/mimic_all_fixed.jsonl", "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

# Step 2: Ensure output is a JSON object string (if not already)
for ex in raw_data:
    if isinstance(ex["output"], dict):
        ex["output"] = json.dumps(ex["output"])

# Step 3: Fix tokenizer padding
tokenizer.pad_token = tokenizer.eos_token

# Step 4: Tokenize each example
def tokenize_example(example):
    prompt = example["instruction"].strip()
    output = example["output"].strip()

    full_text = prompt + "\n" + output
    tokenized = tokenizer(full_text, truncation=True, padding="max_length", max_length=1024)

    # Compute label mask to ignore prompt tokens
    prompt_len = len(tokenizer(prompt, truncation=True, max_length=1024)["input_ids"])
    labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]

    # Pad labels if needed
    labels += [-100] * (1024 - len(labels))
    labels = labels[:1024]

    return {
        "input_ids": torch.tensor(tokenized["input_ids"]),
        "attention_mask": torch.tensor(tokenized["attention_mask"]),
        "labels": torch.tensor(labels)
    }


# Step 5: Custom PyTorch-compatible dataset
class CustomDataset(TorchDataset):
    def __init__(self, data):
        self.samples = [tokenize_example(ex) for ex in data]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

train_dataset = CustomDataset(raw_data)

# preview
train_dataset[0]



# Step 6: Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=30,
    logging_dir="./logs",
    report_to="none",  # disables wandb
    save_strategy="no",
    label_names=["labels"]
)

# Step 7: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Step 8: Train
trainer.train()


model.save_pretrained("./llama3-lora-final")
tokenizer.save_pretrained("./llama3-lora-final")

#!zip -r llama3-lora-final.zip llama3-lora-final

from google.colab import files
files.download("llama3-lora-final.zip")
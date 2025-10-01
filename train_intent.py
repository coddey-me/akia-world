# train_intent.py
# CPU-friendly DistilBERT intent classifier using Hugging Face Trainer

import os
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

def prepare_dataset(csv_path):
    # CSV expected: text,intent
    ds = load_dataset('csv', data_files={'train': csv_path})['train']
    # map label to id
    labels = sorted(list(set(ds['intent'])))
    label2id = {l:i for i,l in enumerate(labels)}
    ds = ds.map(lambda x: {'labels': label2id[x['intent']]}) # Renamed label_id to labels
    return ds, label2id

def tokenize_fn(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

# Define arguments directly in the notebook
train_csv = 'expanded_intent_data.csv'
output_dir = 'checkpoints/intent'

os.makedirs(output_dir, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
# load dataset
ds, label2id = prepare_dataset(train_csv)
num_labels = len(label2id)

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

# tokenize
ds = ds.map(lambda ex: tokenize_fn(ex, tokenizer), batched=True)
ds.set_format(type='torch', columns=['input_ids','attention_mask','labels']) # Renamed label_id to labels

# training args tuned for CPU
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=20,
    fp16=False,
    logging_steps=20,
    save_steps=100,
    save_total_limit=5,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    report_to=[],  # disable wandb / others
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = (preds == labels).mean()
    return {'accuracy': acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(output_dir)
# save tokenizer
tokenizer.save_pretrained(output_dir)

# save label mapping
import json
with open(os.path.join(output_dir, 'label2id.json'), 'w') as f:
    json.dump(label2id, f)

print("Training finished. Checkpoint saved to", output_dir)

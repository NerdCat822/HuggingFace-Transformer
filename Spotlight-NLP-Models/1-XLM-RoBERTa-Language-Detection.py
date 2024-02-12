import time
from pathlib import Path

import langid
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    pipeline,
    Trainer,
    TrainingArguments
)

output_path = Path('./')

dataset = load_dataset("papluca/language-identification")

ds_train = dataset['train']
ds_valid = dataset['validation']
ds_test = dataset['test']

model_ckpt = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize_text(sequence):
    """Tokenize input sequence."""
    return tokenizer(sequence["text"], truncation=True, max_length=128)

tok_train = ds_train.map(tokenize_text, batched=True)
tok_valid = ds_valid.map(tokenize_text, batched=True)
tok_test = ds_test.map(tokenize_text, batched=True)

amazon_languages = ['en', 'de', 'fr', 'es', 'ja', 'zh']
xnli_languages = ['ar', 'el', 'hi', 'ru', 'th', 'tr', 'vi', 'bg', 'sw', 'ur']
stsb_languages = ['it', 'nl', 'pl', 'pt']

all_langs = sorted(list(set(amazon_languages + xnli_languages + stsb_languages)))

id2label = {idx: all_langs[idx] for idx in range(len(all_langs))}
label2id = {v: k for k, v in id2label.items()}

def encode_labels(example):
    """Map string labels to integers."""
    example["labels"] = label2id[example["labels"]]
    return example

tok_train = tok_train.map(encode_labels, batched=False)
tok_valid = tok_valid.map(encode_labels, batched=False)
tok_test = tok_test.map(encode_labels, batched=False)

# Use dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
  model_ckpt, num_labels=len(all_langs), id2label=id2label, label2id=label2id
)

def compute_metrics(pred):
    """Custom metric to be used during training."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)  # Accuracy
    f1 = f1_score(labels, preds, average="weighted")  # F1-score
    return {
        "accuracy": acc,
        "f1": f1
        }

epochs = 2
lr = 2e-5
train_bs = 64
eval_bs = train_bs * 2

# Log training loss at each epoch
logging_steps = len(tok_train) // train_bs
# Out dir
output_dir = output_path / "xlm-roberta-base-finetuned-language-detection"

training_args = TrainingArguments(
  output_dir=output_dir,
  num_train_epochs=epochs,
  learning_rate=lr,
  per_device_train_batch_size=train_bs,
  per_device_eval_batch_size=eval_bs,
  evaluation_strategy="epoch",
  logging_steps=logging_steps,
  fp16=True,  # Remove if GPU doesn't support it
)

trainer = Trainer(
  model,
  training_args,
  compute_metrics=compute_metrics,
  train_dataset=tok_train,
  eval_dataset=tok_valid,
  data_collator=data_collator,
  tokenizer=tokenizer,
)

trainer.train()

ds_test = ds_test.to_pandas()
ds_test.head(3)

langid.set_languages(all_langs)

langid_preds = [langid.classify(s)[0] for s in ds_test.text.values.tolist()]

print(classification_report(ds_test.labels.values.tolist(), langid_preds, digits=3))
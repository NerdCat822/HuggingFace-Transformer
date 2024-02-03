from datasets import load_dataset
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch

# Load a subset of the IMDB dataset
dataset = load_dataset('imdb', split='train[:1000]')

# Load the FlauBERT tokenizer and model
tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')
model = FlaubertForSequenceClassification.from_pretrained('flaubert/flaubert_base_cased', num_labels=2)

# Preprocess the data
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=128, padding='max_length')

# Map the dataset with the preprocessing function
encoded_dataset = dataset.map(preprocess_function, batched=True)

# Define the labels
def compute_labels(examples):
    labels = examples['label']
    if isinstance(labels, list):  # handling both single and batched data
        return {'labels': [0 if label == 'neg' else 1 for label in labels]}
    else:
        return {'labels': [0 if labels == 'neg' else 1]}

# Map the dataset with the compute labels function
encoded_dataset = encoded_dataset.map(compute_labels)

# Define the training arguments
training_args = TrainingArguments(
    "test-trainer",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    disable_tqdm=True
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    eval_dataset=encoded_dataset,
)

# Train the model
trainer.train()

# Let's say you have the following sentences:
sentences = ["I love this movie!"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# You would preprocess these sentences using the same steps as during training:
inputs = tokenizer(sentences, truncation=True, padding=True, return_tensors="pt")

# Move the inputs tensors to the same device as the model
inputs = {key: value.to(device) for key, value in inputs.items()}

# You can then feed these inputs to the model:
outputs = model(**inputs)

# The model returns the logits, which are the raw output values from the last layer of the model
logits = outputs.logits

# To get the predicted class, you can apply a softmax to the logits and take the argmax:
probs = torch.nn.functional.softmax(logits, dim=-1)
predictions = torch.argmax(probs, dim=-1)

print(predictions) # tensor([1], device='cuda:0'))
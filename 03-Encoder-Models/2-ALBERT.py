import torch
from transformers import AlbertTokenizer, AlbertForMaskedLM

# Load the pretrained ALBERT model and tokenizer
model_name = 'albert-base-v2'
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertForMaskedLM.from_pretrained(model_name)

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Next word prediction function
def predict_next_word(text):
    # Tokenize input text
    tokenized_text = tokenizer.tokenize(text)
    masked_index = tokenized_text.index('[MASK]')
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Convert tokens to tensor
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(tokens_tensor)

    predictions = outputs[0][0, masked_index].topk(k=5).indices.tolist()

    predicted_tokens = []
    for token_index in predictions:
        predicted_token = tokenizer.convert_ids_to_tokens([token_index])[0]
        predicted_tokens.append(predicted_token)

    return predicted_tokens

# Example usage
text_with_mask = "I want to [MASK] a pizza for dinner."
predicted_tokens = predict_next_word(text_with_mask)

# Print the predicted tokens
print(predicted_tokens)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW

# Load the pretrained ALBERT model and tokenizer
model_name = 'albert-base-v2'
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your own dataset and dataloader
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        return text, label

    def __len__(self):
        return len(self.texts)

# Example training data
train_texts = ['This is the first sentence.', 'This is the second sentence.']
train_labels = [0, 1]

# Create the dataset and dataloader
train_dataset = MyDataset(train_texts, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Move the model to the device
model.to(device)

# Training settings
epochs = 10
lr = 2e-5
optimizer = AdamW(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    total_loss = 0
    model.train()

    for texts, labels in train_dataloader:
        # Tokenize input texts
        input_ids = []
        attention_masks = []
        for text in texts:
            encoded = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            input_ids.append(encoded['input_ids'].squeeze())
            attention_masks.append(encoded['attention_mask'].squeeze())

        input_ids = torch.stack(input_ids).to(device)
        attention_masks = torch.stack(attention_masks).to(device)
        labels = torch.tensor(labels).to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs.logits

        # Compute loss
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {average_loss:.4f}")

# Save the fine-tuned model
model.save_pretrained('path/to/save/model')
tokenizer.save_pretrained('path/to/save/tokenizer')
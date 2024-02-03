import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification

# Load XLNet tokenizer and model
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)  # Change num_labels according to your classification task

# Example text for classification
text = "The dog is really cute."

# Tokenize input text
inputs = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    padding='max_length',
    max_length=128,
    truncation=True,
    return_tensors='pt'
)

# Perform classification
outputs = model(**inputs)
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1).item()

# Get the predicted label
labels = ['Negative', 'Positive']  # Replace with your actual class labels
predicted_label = labels[predicted_class]

print("Text:", text)
print("Predicted Label:", predicted_label)
"""
Text: The dog is really cute.
Predicted Label: Positive
"""
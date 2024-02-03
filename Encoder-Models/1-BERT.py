import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the pretrained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Text classification function
def classify_text(text):
    # Tokenize input text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model.to('cuda')(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).squeeze(dim=0)
    predicted_class = torch.argmax(probabilities).item()

    return predicted_class, probabilities

# Example usage
text_to_classify = "This is an example sentence."
predicted_class, probabilities = classify_text(text_to_classify)

# Print the predicted class and probabilities
print(f"Predicted class: {predicted_class}")
print("Probabilities:")
for i, prob in enumerate(probabilities):
    print(f"Class {i}: {prob.item()}")
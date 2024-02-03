import torch
from transformers import AutoTokenizer, TransfoXLForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103")
model = TransfoXLForSequenceClassification.from_pretrained("transfo-xl-wt103")

input_str = input("")
inputs = tokenizer(input_str, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()

# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
num_labels = len(model.config.id2label)
model = TransfoXLForSequenceClassification.from_pretrained("transfo-xl-wt103", num_labels=num_labels)

labels = torch.tensor([1])
loss = model(**inputs, labels=labels).loss

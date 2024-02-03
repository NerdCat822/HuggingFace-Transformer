import torch
from transformers import AutoTokenizer, TransfoXLForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103")
model = TransfoXLForSequenceClassification.from_pretrained("transfo-xl-wt103", problem_type="multi_label_classification")

input_str = input("")

inputs = tokenizer(input_str, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

num_labels = len(model.config.id2label)
model = TransfoXLForSequenceClassification.from_pretrained("transfo-xl-wt103", num_labels=num_labels, problem_type="multi_label_classification")

labels = torch.sum(torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1).to(torch.float)

loss = model(**inputs, labels=labels).loss
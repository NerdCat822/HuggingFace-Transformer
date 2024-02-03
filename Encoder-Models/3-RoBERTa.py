import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM

# Load the pretrained RoBERTa model and tokenizer
model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForMaskedLM.from_pretrained(model_name)

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the model to evaluation mode
model.eval()

# Input text with a masked token
input_text = "The capital of Korea is <mask>."

# Tokenize input text
input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors='pt').to(device)

# Masked language modeling
with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs[0]

# Get the predicted token
masked_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
predicted_token_ids = torch.argmax(predictions[0, masked_index], dim=1).tolist()
predicted_tokens = tokenizer.batch_decode(predicted_token_ids)

# Print the predicted token
print("Predicted token:", predicted_tokens)
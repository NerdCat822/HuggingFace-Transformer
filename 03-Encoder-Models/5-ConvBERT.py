from transformers import ConvBertTokenizer, ConvBertForMaskedLM
import torch

def predict_next_word(sentence):
    # Load ConvBERT tokenizer and model
    tokenizer = ConvBertTokenizer.from_pretrained('YituTech/conv-bert-base')
    model = ConvBertForMaskedLM.from_pretrained('YituTech/conv-bert-base')

    # Tokenize input sentence
    tokens = tokenizer.encode(sentence, add_special_tokens=True, return_tensors='pt')

    # Find the masked token
    masked_index = torch.where(tokens == tokenizer.mask_token_id)[1].tolist()[0]

    # Generate predictions for the masked token
    with torch.no_grad():
        outputs = model(tokens)
        predictions = outputs.logits[0, masked_index]

    # Get the top-k predicted tokens and their probabilities
    top_predictions = torch.topk(predictions, k=5)
    predicted_tokens = tokenizer.convert_ids_to_tokens(top_predictions.indices.tolist())
    probabilities = top_predictions.values.exp().tolist()

    return predicted_tokens, probabilities

sentence = "I want to [MASK] a car"

predicted_tokens, probabilities = predict_next_word(sentence)
for token, prob in zip(predicted_tokens, probabilities):
    print(f"Token: {token}, Probability: {prob}")
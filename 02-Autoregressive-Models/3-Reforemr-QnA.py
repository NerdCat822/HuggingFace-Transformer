import torch
from transformers import AutoTokenizer, ReformerForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
model = ReformerForQuestionAnswering.from_pretrained("google/reformer-crime-and-punishment")

question = "What is the capital of Germany?"
text = "Germany is a country based in Europe and the captial name is Berlin."

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

"""
outputs : QuestionAnsweringModelOutput(loss=None, start_logits=tensor([[-0.3538, -0.1295, -0.4811, -0.1445, -0.3287, -0.0500, -0.2725,  0.0254,
          0.0020, -0.2791, -0.8788,  0.0760, -0.1713, -0.2015, -0.5512, -0.2112,
         -0.1154, -0.2549,  0.1483, -0.3111, -0.2290, -0.2806, -0.1326, -0.1399,
         -0.1924, -0.1691,  0.1851,  0.0780, -0.1407, -0.2611, -0.1789, -0.4415,
         -0.6127,  0.4565, -0.0589, -0.4990, -0.1128,  0.0796,  0.5344, -0.1095,
          0.1374,  0.2332, -0.1252,  0.0424, -0.0670, -0.1993, -0.4477,  0.3327,
         -0.1506, -0.3038,  0.8044, -0.0266, -0.1369,  0.1533, -0.4257, -0.6308]]), end_logits=tensor([[ 0.0217,  0.2129,  0.1305, -0.5226, -0.0407,  0.5836,  0.0063,  0.2467,
         -0.0516,  0.5053,  0.0290, -0.1077, -0.0156,  0.3999,  0.9891,  0.7713,
         -0.3043,  0.0972,  0.2414,  0.3405,  0.5636,  0.7653,  0.4877,  0.6189,
         -0.1989,  0.1969,  0.2315,  0.3782,  0.1472,  0.9732,  0.0204,  0.0952,
          1.0173, -0.4966,  0.2581,  0.5317, -0.0137,  0.6619,  0.7173,  0.2183,
          0.6740,  0.1177,  0.3691,  0.8983,  0.0347,  0.3106,  0.4395,  0.4362,
          0.0974, -0.2470,  0.4531, -0.3385,  0.5196, -0.5564, -0.5881,  0.3481]]), hidden_states=None, attentions=None)
"""

def decode_qa_output(question, context, start_logits, end_logits, tokenizer):

    start_logits = torch.tensor(start_logits)
    end_logits = torch.tensor(end_logits)

    # Get the start and end indices with the highest logits
    start_index = start_logits.argmax()
    end_index = end_logits.argmax()

    # Get the tokens corresponding to the answer span
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(context))
    answer_tokens = tokens[start_index: end_index + 1]

    # Decode the answer tokens into text
    answer_text = tokenizer.convert_tokens_to_string(answer_tokens)

    # Remove special tokens or unwanted characters
    answer_text = answer_text.replace('[CLS]', '').replace('[SEP]', '').strip()

    return answer_text

answer_start_index = outputs.start_logits.argmax() # tensor(0)
# answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax() # tensor(53)

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]

target_start_index = torch.tensor([12]) # tensor([12])
target_end_index = torch.tensor([12]) # tensor([12])

outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
loss = outputs.loss # tensor(4.2324, grad_fn=<DivBackward0>)

start_logits = outputs["start_logits"]
end_logits = outputs["end_logits"]

decoded_answer = decode_qa_output(question, text, start_logits, end_logits, tokenizer)

start_logits.argmax() # tensor(0)
end_logits.argmax() # tensor(0)

print(decoded_answer) # Germany is a country based in Europe and the captial name is Berlin.
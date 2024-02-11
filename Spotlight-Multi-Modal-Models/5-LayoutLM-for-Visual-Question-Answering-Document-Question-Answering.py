"""
!tar -xf /content/drive/MyDrive/val.tar.gz
!pip install datasets
!sudo apt install tesseract-ocr
!pip install pytesseract
!pip install pyyaml==5.1
!pip install torch torchvision
!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
!pip install transformers
"""

model_checkpoint = "microsoft/layoutlmv2-base-uncased"
batch_size = 16

import json

with open('/content/val/val_v1.0.json') as f:
  data = json.load(f)

print("Dataset name:", data['dataset_name'])
print("Dataset split:", data['dataset_split'])

import pandas as pd

df = pd.DataFrame(data['data'])
df.head()

# pick a random example
example = data['data'][10]
for k,v in example.items():
  print(k + ":", v)

from PIL import Image

root_dir = '/content/val/'
image = Image.open(root_dir + example['image'])

ocr_root_dir = root_dir + "ocr_results/"

with open(ocr_root_dir + example['ucsf_document_id'] + "_" + example['ucsf_document_page_no'] + ".json") as f:
  ocr = json.load(f)

words = ""
for item in ocr['recognitionResults'][0]['lines']:
  words += item['text']

print(words)

from datasets import Dataset

dataset = Dataset.from_pandas(df.iloc[:50])

from transformers import LayoutLMv2FeatureExtractor

feature_extractor = LayoutLMv2FeatureExtractor()

def get_ocr_words_and_boxes(examples):

  # get a batch of document images
  images = [Image.open(root_dir + image_file).convert("RGB") for image_file in examples['image']]

  # resize every image to 224x224 + apply tesseract to get words + normalized boxes
  encoded_inputs = feature_extractor(images)

  examples['image'] = encoded_inputs.pixel_values
  examples['words'] = encoded_inputs.words
  examples['boxes'] = encoded_inputs.boxes

  return examples

dataset_with_ocr = dataset.map(get_ocr_words_and_boxes, batched=True, batch_size=2)

# check whether words and normalized bounding boxes are added correctly
print(dataset_with_ocr[0]['words'])
print(dataset_with_ocr[0]['boxes'])
print("-----")
print(dataset_with_ocr[1]['words'])
print(dataset_with_ocr[1]['boxes'])

def subfinder(words_list, answer_list):
    matches = []
    start_indices = []
    end_indices = []
    for idx, i in enumerate(range(len(words_list))):
        if words_list[i] == answer_list[0] and words_list[i:i+len(answer_list)] == answer_list:
            matches.append(answer_list)
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)
    if matches:
      return matches[0], start_indices[0], end_indices[0]
    else:
      return None, 0, 0
    
question = "where is it located?"
words = ["this", "is", "located", "in", "the", "university", "of", "california", "in", "the", "US"]
boxes = [[1000,1000,1000,1000] for _ in range(len(words))]
answer = "university of california"

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

import transformers
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
encoding = tokenizer(question, words, boxes=boxes)

match, word_idx_start, word_idx_end = subfinder(words, answer.split())

print("Match:", match)
print("Word idx start:", word_idx_start)
print("Word idx end:", word_idx_end)

sequence_ids = encoding.sequence_ids()

sequence_ids = encoding.sequence_ids()

# Start token index of the current span in the text.
token_start_index = 0
while sequence_ids[token_start_index] != 1:
    token_start_index += 1

# End token index of the current span in the text.
token_end_index = len(encoding.input_ids) - 1
while sequence_ids[token_end_index] != 1:
    token_end_index -= 1

print("Token start index:", token_start_index)
print("Token end index:", token_end_index)
print(tokenizer.decode(encoding.input_ids[token_start_index:token_end_index+1]))

word_ids = encoding.word_ids()[token_start_index:token_end_index+1]
print("Word ids:", word_ids)
for id in word_ids:
  if id == word_idx_start:
    start_position = token_start_index
  else:
    token_start_index += 1

for id in word_ids[::-1]:
  if id == word_idx_end:
    end_position = token_end_index
  else:
    token_end_index -= 1

print(start_position)
print(end_position)
print("Reconstructed answer:", tokenizer.decode(encoding.input_ids[start_position:end_position+1]))

def encode_dataset(examples, max_length=512):
  # take a batch
  questions = examples['question']
  words = examples['words']
  boxes = examples['boxes']

  # encode it
  encoding = tokenizer(questions, words, boxes, max_length=max_length, padding="max_length", truncation=True)

  # next, add start_positions and end_positions
  start_positions = []
  end_positions = []
  answers = examples['answers']
  # for every example in the batch:
  for batch_index in range(len(answers)):
    print("Batch index:", batch_index)
    cls_index = encoding.input_ids[batch_index].index(tokenizer.cls_token_id)
    # try to find one of the answers in the context, return first match
    words_example = [word.lower() for word in words[batch_index]]
    for answer in answers[batch_index]:
      match, word_idx_start, word_idx_end = subfinder(words_example, answer.lower().split())
      if match:
        break
    # EXPERIMENT (to account for when OCR context and answer don't perfectly match):
    if not match:
      for answer in answers[batch_index]:
        for i in range(len(answer)):
          # drop the ith character from the answer
          answer_i = answer[:i] + answer[i+1:]
          # check if we can find this one in the context
          match, word_idx_start, word_idx_end = subfinder(words_example, answer_i.lower().split())
          if match:
            break
    # END OF EXPERIMENT

    if match:
      sequence_ids = encoding.sequence_ids(batch_index)
      # Start token index of the current span in the text.
      token_start_index = 0
      while sequence_ids[token_start_index] != 1:
          token_start_index += 1

      # End token index of the current span in the text.
      token_end_index = len(encoding.input_ids[batch_index]) - 1
      while sequence_ids[token_end_index] != 1:
          token_end_index -= 1

      word_ids = encoding.word_ids(batch_index)[token_start_index:token_end_index+1]
      for id in word_ids:
        if id == word_idx_start:
          start_positions.append(token_start_index)
          break
        else:
          token_start_index += 1

      for id in word_ids[::-1]:
        if id == word_idx_end:
          end_positions.append(token_end_index)
          break
        else:
          token_end_index -= 1

      print("Verifying start position and end position:")
      print("True answer:", answer)
      start_position = start_positions[batch_index]
      end_position = end_positions[batch_index]
      reconstructed_answer = tokenizer.decode(encoding.input_ids[batch_index][start_position:end_position+1])
      print("Reconstructed answer:", reconstructed_answer)
      print("-----------")

    else:
      print("Answer not found in context")
      print("-----------")
      start_positions.append(cls_index)
      end_positions.append(cls_index)

  encoding['image'] = examples['image']
  encoding['start_positions'] = start_positions
  encoding['end_positions'] = end_positions

  return encoding

from datasets import Features, Sequence, Value, Array2D, Array3D

# we need to define custom features
features = Features({
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'start_positions': Value(dtype='int64'),
    'end_positions': Value(dtype='int64'),
})

encoded_dataset = dataset_with_ocr.map(encode_dataset, batched=True, batch_size=2,
                                       remove_columns=dataset_with_ocr.column_names,
                                       features=features)

image = Image.open('/content/val/' + dataset['image'][idx])

start_position = encoded_dataset['start_positions'][idx]
end_position = encoded_dataset['end_positions'][idx]
if start_position != 0:
  print(tokenizer.decode(encoded_dataset['input_ids'][idx][start_position: end_position+1]))
else:
  print("Answer not found in context")

import torch

encoded_dataset.set_format(type="torch")
dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=4)
batch = next(iter(dataloader))

for k,v in batch.items():
  print(k, v.shape)

idx = 2

tokenizer.decode(batch['input_ids'][2])

start_position = batch['start_positions'][idx].item()
end_position = batch['end_positions'][idx].item()

tokenizer.decode(batch['input_ids'][idx][start_position:end_position+1])

import torch, detectron2
# !nvcc --version
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()
for epoch in range(20):  # loop over the dataset multiple times
   for idx, batch in enumerate(dataloader):
        # get the inputs;
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        bbox = batch["bbox"].to(device)
        image = batch["image"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                       bbox=bbox, image=image, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        print("Loss:", loss.item())
        loss.backward()
        optimizer.step()

# step 1: pick a random example
example = data['data'][2]
root_dir = '/content/val/'
question = example['question']
image = Image.open(root_dir + example['image']).convert("RGB")
print(question)

from transformers import LayoutLMv2Processor

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")

# prepare for the model
encoding = processor(image, question, return_tensors="pt")
print(encoding.keys())

# step 2: forward pass

for k,v in encoding.items():
  encoding[k] = v.to(model.device)

outputs = model(**encoding)

# step 3: get start_logits and end_logits
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# step 4: get largest logit for both
predicted_start_idx = start_logits.argmax(-1).item()
predicted_end_idx = end_logits.argmax(-1).item()
print("Predicted start idx:", predicted_start_idx)
print("Predicted end idx:", predicted_end_idx)

# step 5: decode the predicted answer
processor.tokenizer.decode(encoding.input_ids.squeeze()[predicted_start_idx:predicted_end_idx+1])
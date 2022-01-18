import pandas as pd
import numpy as np
import re
import pickle
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
import torch
import random
import csv
from rouge import Rouge
import wandb

wandb.login()


class BartDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels['input_ids'])  # len(self.labels)

def prepare_data(model_name,
                 train_texts, train_labels,
                 val_texts=None, val_labels=None,
                 test_texts=None, test_labels=None):
  """
  Prepare input data for model fine-tuning
  """
  tokenizer = BartTokenizer.from_pretrained(model_name)

  prepare_val = False if val_texts is None or val_labels is None else True
  prepare_test = False if test_texts is None or test_labels is None else True

  def tokenize_data(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True)
    decodings = tokenizer(labels, truncation=True, padding=True)
    dataset_tokenized = BartDataset(encodings, decodings)
    return dataset_tokenized

  train_dataset = tokenize_data(train_texts, train_labels)
  val_dataset = tokenize_data(val_texts, val_labels) if prepare_val else None
  test_dataset = tokenize_data(test_texts, test_labels) if prepare_test else None

  return train_dataset, val_dataset, test_dataset, tokenizer

def prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset=None, freeze_encoder=False, output_dir='./results'):
  """
  Prepare configurations and base model for fine-tuning
  """
  torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = BartForConditionalGeneration.from_pretrained(model_name).to(torch_device)

  if freeze_encoder:
    for param in model.model.encoder.parameters():
      param.requires_grad = False

  if val_dataset is not None:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=10,            # total number of training epochs
      per_device_train_batch_size=2,   # batch size per device during training, can increase if memory allows
      per_device_eval_batch_size=2,    # batch size for evaluation, can increase if memory allows
      save_steps=10000,                  # number of updates steps before checkpoint saves
      save_total_limit=5,              # limit the total amount of checkpoints and deletes the older checkpoints
      evaluation_strategy='steps',     # evaluation strategy to adopt during training
      eval_steps=100,                  # number of update steps before evaluation
      warmup_steps=100,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      logging_steps=10,
      report_to="wandb",
      run_name="augment1500"
    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      eval_dataset=val_dataset,            # evaluation dataset
      tokenizer=tokenizer
    )

  else:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=3,            # total number of training epochs
      per_device_train_batch_size=2,   # batch size per device during training, can increase if memory allows
      save_steps=10000,                # number of updates steps before checkpoint saves
      save_total_limit=5,              # limit the total amount of checkpoints and deletes the older checkpoints
      warmup_steps=100,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      logging_steps=10,
      report_to="wandb",
      run_name="augmented1500"
    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      tokenizer=tokenizer
    )

  return trainer


df = pd.read_csv("new_bart_augment.csv")
print(len(df))
df = df.dropna(subset=["Summary"])
document = df['Text']
summary = df['Summary']

train_text = list(document)
train_label = list(summary)
for i in range(len(df)):
    train_text[i] = " ".join(train_text[i].split())
    train_label[i] = str(train_label[i])
print(len(df))

df = pd.read_csv("test.csv")
print(len(df))
df = df.dropna(subset=["Summary"])
document = df['Text']
summary = df['Summary']

test_text = list(document)
test_label = list(summary)
for i in range(len(df)):
    test_text[i] = " ".join(test_text[i].split())
    test_label[i] = str(test_label[i])
print(len(df))

# test_text = train_text[350:]
# test_label = train_label[350:]
# train_text = train_text[:350]
# train_label = train_label[:350]



model_name = 'facebook/bart-large-cnn'
path = "./bart_cnn_final"
train_dataset, val_dataset, _, tokenizer = prepare_data(path, train_text, train_label, test_text, test_label)
trainer = prepare_fine_tuning(path, tokenizer, train_dataset, val_dataset)
trainer.train()
trainer.save_model()
wandb.finish()

src_text = test_text;
summ = test_label;
# model_name = 'facebook/bart-large-cnn'
path = './results'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# tokenizer = BartTokenizer.from_pretrained(path)
model = BartForConditionalGeneration.from_pretrained(path) #.to(device)
tab=[]

for i in range(len(src_text)):
  print(i)
  inputs = tokenizer(src_text[i], max_length=1024, return_tensors='pt')
  summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=1000, early_stopping=True)
  tab.append([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])



tabx=[]

for i in range(len(tab)):
  tabx.append(tab[i][0])

model_out = list(tabx)
reference = list(summ)

# model_ox=[]
# reference_ox=[]

# for i in range(len(model_out)):
#   if(len(reference[i])<=1):
#     x=1
#   else:
#     model_ox.append(model_out[i])
#     reference_ox.append(reference[i])

rouge = Rouge()
res = rouge.get_scores(model_out, reference)

header = ['Reference', 'Bart_summary', 'Rouge1_f', 'Rouge2_f', 'Rougel_f']
with open('augmented_bart_cnn_1500.csv', 'a', newline ='') as file:
	for i in range(len(tab)):
          try:
            writer = csv.DictWriter(file, fieldnames = header)
            writer.writerow({'Reference':summ[i], 'Bart_summary': tabx[i],
                             'Rouge1_f':res[i]['rouge-1']['f'], 'Rouge2_f':res[i]['rouge-2']['f'], "Rougel_f":res[i]['rouge-l']['f']})
            print(i)
            i+=1
          except IndexError as e:
            print(e)

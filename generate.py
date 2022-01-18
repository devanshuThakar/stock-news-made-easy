import torch
from rouge import Rouge
import pandas as pd
import numpy as np
import re
from transformers import BartForConditionalGeneration, BartTokenizer

df = pd.read_csv("100_days_clean.csv")
print(len(df))
df = df.dropna(subset=["Summary"])
document = df['Text']
summary = df['Summary']
# document = document.apply(lambda i: i.decode('utf-8'))
# summary = summary.apply(lambda i: i.decode('utf-8'))

test_text = list(document)
test_label = list(summary)
for i in range(len(df)):
# train_text[i] = str(train_text[i])
    test_text[i] = " ".join(test_text[i].split()[:300])
    test_label[i] = str(test_label[i])
print(len(df))

# test_text = train_text[350:]
# test_label = train_label[350:]
# train_text = train_text[:350]
# train_label = train_label[:350]

for i in range(len(test_text)):
    tweet = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0–9@:%.\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%\+.~#?&//=]*)', '', test_text[i], flags=re.MULTILINE) # to remove links that start with HTTP/HTTPS in the tweet
    tweet = re.sub(r'http\S+', '', tweet,flags=re.MULTILINE)
    tweet = re.sub(r'[-a-zA-Z0–9@:%.\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%\+.~#?&//=]*)', '', tweet, flags=re.MULTILINE) # to remove other url links

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    emoji_pattern.sub(r'', tweet)
    # tweet = ' '.join(re.sub('/[\u{1F600}-\u{1F6FF}]/'," ",tweet).split()) # for emojis

    tweet = re.sub(r"#(\w+)", ' ', tweet, flags=re.MULTILINE)
    tweet = re.sub(r"@(\w+)", ' ', tweet, flags=re.MULTILINE)
    test_text[i] = tweet

src_text = test_text
summ = test_label
# model_name = 'google/pegasus-cnn_dailymail'
# path = './results/'
path = './bart_cnn'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# tokenizer = PegasusTokenizer.from_pretrained(path)
# model = PegasusForConditionalGeneration.from_pretrained(path) #.to(device)
tokenizer = BartTokenizer.from_pretrained(path)
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

rouge = Rouge()
res = rouge.get_scores(model_out, reference)

import csv
header = ['Reference', 'Pegasus_summary', 'Rouge1_f', 'Rouge2_f', 'Rougel_f']
with open('bart_cnn_625.csv', 'a', newline ='') as file:
        for i in range(len(tab)):
          try:
            writer = csv.DictWriter(file, fieldnames = header)
            writer.writerow({'Reference':summ[i], 'Pegasus_summary': tabx[i],
                             'Rouge1_f':res[i]['rouge-1']['f'], 'Rouge2_f':res[i]['rouge-2']['f'], "Rougel_f":res[i]['rouge-l']['f']})
            print(i)
            i+=1
          except IndexError as e:
            print(e)

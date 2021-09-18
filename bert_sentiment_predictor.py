from transformers import BertModel, DistilBertModel, BertForSequenceClassification, AdamW, BertTokenizer, get_linear_schedule_with_warmup, Trainer, TrainingArguments
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import pandas as pd
from pathlib import Path
from torch.utils.data import TensorDataset, random_split
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from torch.nn import functional as F
from collections import defaultdict
import random
import os


if torch.cuda.is_available():
  print("\nUsing: ", torch.cuda.get_device_name(0))
  device = torch.device('cuda')
else:
  print("\nUsing: CPU")
  device = torch.device('cpu')

labeled_dataset = "data/news_headlines_sentiment.csv"
labeled_dataset_file = Path(labeled_dataset)
file_loaded = False
while not file_loaded:
  if labeled_dataset_file.exists():
    labeled_dataset = pd.read_csv(labeled_dataset_file)
    file_loaded = True
    print("Dataset Loaded")
  else:
    print("File not Found")
print(labeled_dataset)

additional_dataset = "data/more_news_data.csv"
phrase_bank_dataset_file = Path(additional_dataset)
file_loaded = False
while not file_loaded:
  if phrase_bank_dataset_file.exists():
    phrase_dataset = pd.read_csv(additional_dataset, encoding='latin-1', names=["sentiment", "news"])
    phrase_dataset = phrase_dataset[["news", "sentiment"]]
    phrase_dataset["sentiment"].replace(['positive', 'negative', 'neutral'], [0,1,2], inplace=True)
    file_loaded = True
    print("Dataset Loaded")
  else:
    print("File not Found")
print(phrase_dataset)

class NewsSentimentDataset(torch.utils.data.Dataset):
  def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

  def __getitem__(self, idx):
      item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
      item['labels'] = torch.tensor(self.labels[idx])
      return item

  def __len__(self):
      return len(self.labels)

def tokenize_headlines(df, tokenizer):
  encodings = tokenizer.batch_encode_plus(
      df["news"].tolist(),           # input the news headlines
      add_special_tokens = True,     # special tokens added to mark beginning & end of sentence
      truncation = True,             # make all sentences a fixed length
      padding = 'max_length',        # pad with zeros to max length
      return_attention_mask = True,  # include attention mask in encoding
      return_tensors = 'pt'          # return as pytorch tensor
  )

  dataset = NewsSentimentDataset(encodings, df["sentiment"].tolist())
  return dataset

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
train_data, val_data = train_test_split(phrase_dataset, test_size=.2)
merged_train_data, merged_val_data = train_test_split(merged_dataset, test_size=.2)
labeled_train_data, labeled_val_data = train_test_split(labeled_dataset, test_size=.2)


print("Train Dataset\n", train_data.reset_index(drop=True))
print("Validation Dataset\n", val_data.reset_index(drop=True))

train_dataset = tokenize_headlines(merged_train_data, tokenizer)
val_dataset = tokenize_headlines(merged_val_data, tokenizer)

# Training & Testing

model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3)
model = model.to(device)
# data loader
train_batch_size = 8
val_batch_size = 8

train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=RandomSampler(train_dataset))
val_data_loader = DataLoader(val_dataset, batch_size=val_batch_size, sampler=SequentialSampler(val_dataset))

# optimizer and scheduler
num_epochs = 1
num_steps = len(train_data_loader) * num_epochs
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_steps)

# training and evaluation
seed_val = 64
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

for epoch in range(num_epochs):

  print("\n###################################################")
  print("Epoch: {}/{}".format(epoch + 1, num_epochs))
  print("###################################################\n")

  # training phase
  average_train_loss = 0
  average_train_acc = 0
  for step, batch in enumerate(train_data_loader):

    model.train()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    loss, logits = model(input_ids,
                         token_type_ids=None,
                         attention_mask=attention_mask,
                         labels=labels, return_dict=False)

    print(loss)
    # loss is cross entropy loss by default
    average_train_loss += loss

    if step % 100 == 0:
      print("At Step {} Training Loss: {:.5f}".format(step, loss.item()))

    # backpropagation
    loss.backward()
    # maximum gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # update parameters
    optimizer.step()
    # update learning rate
    scheduler.step()

    logits_for_acc = logits.detach().cpu().numpy()
    label_for_acc = labels.to('cpu').numpy()
    average_train_acc += sklearn.metrics.accuracy_score(label_for_acc, np.argmax(logits_for_acc, axis=-1))

    # print out sentences + sentiment predictions + labels
    print(tokenizer.batch_decode(input_ids, skip_special_tokens=True))
    print("Predictions: ", np.argmax(logits_for_acc, axis=1))
    print("Labels:      ", label_for_acc)
    print("#############")

  average_train_loss = average_train_loss / len(train_data_loader)
  average_train_acc = average_train_acc / len(train_data_loader)
  print("======Average Training Loss: {:.5f}=========".format(average_train_loss))
  print("======Average Training Accuracy: {:.2f}%========".format(average_train_acc * 100))

  # validation phase
  average_val_loss = 0
  average_val_acc = 0

  for step, batch in enumerate(val_data_loader):
    model.eval()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
      loss, logits = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=labels, return_dict=False)

    # loss is cross entropy loss by default
    average_val_loss += loss.item()

    logits_for_acc = logits.detach().cpu().numpy()
    label_for_acc = labels.to('cpu').numpy()
    average_val_acc += sklearn.metrics.accuracy_score(label_for_acc, np.argmax(logits_for_acc, axis=-1))

    # print out sentences + sentiment predictions + labels
    # print(tokenizer.batch_decode(input_ids, skip_special_tokens=True))
    # print("Predictions: ",np.argmax(logits_for_acc, axis=1))
    # print("Labels:      ",label_for_acc)
    # print("#############")

  average_val_loss = average_val_loss / len(val_data_loader)
  average_val_acc = average_val_acc / len(val_data_loader)
  print("======Average Validation Loss: {:.5f}=========".format(average_val_loss))
  print("======Average Validation Accuracy: {:.2f}%======".format(average_val_acc * 100))


#test and prediction
def tokenize_headlines_test(df, tokenizer):
  encodings = tokenizer.batch_encode_plus(
      df["Title+Summary"].tolist(),           # input the news headlines
      max_length=512,
      add_special_tokens = True,     # special tokens added to mark beginning & end of sentence
      truncation = True,             # make all sentences a fixed length
      padding = 'max_length',        # pad with zeros to max length
      return_attention_mask = True,  # include attention mask in encoding
      return_tensors = 'pt'          # return as pytorch tensor
  )

  dataset = NewsSentimentDataset(encodings, np.zeros(len(df["Title+Summary"])))
  return dataset

def generate_test_data_for_bitcoin(fname):
    df = pd.read_excel(fname)
    df = df.dropna()
    df["Title+Summary"] = df["Title"].copy() + " " + df["Summary"].copy()

    test_dataset = tokenize_headlines_test(df, tokenizer)
    test_data_loader = DataLoader(test_dataset, batch_size=8)
    return df, test_data_loader

def bert_inference(test_dataloader):
    result = []
    for step,batch in enumerate(test_dataloader):
        model.eval()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids,
                                 token_type_ids=None,
                                 attention_mask=attention_mask
                                 ,return_dict=False)[0]
        logits_for_acc = logits.detach().cpu().numpy()
        result.extend(np.argmax(logits_for_acc, axis=-1).tolist())
    return result



if __name__ == "__main__":
    fname = "data/articles.xlsx"
    test_df, test_dataloader = generate_test_data_for_bitcoin(fname)
    test_data_pred_sentiments = bert_inference(test_dataloader)
    test_df["sentiments"] = test_data_pred_sentiments
    test_df.to_csv("data/bitcoin_articles_with_sentiments.csv")




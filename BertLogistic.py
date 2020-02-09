import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import transformers as ppb
import warnings
warnings.filterwarnings("ignore")

# load data
df = pd.read_csv("./data/train.tsv", delimiter="\t", header=None)
batch_1 = df[:2000]

# 1. for DistilBert
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, "distilbert-base-uncased")
# 2. for Bert
# model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, "bert-base-uncased")

# load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# step1: tokenize the sentence: split sentence into word or subword
tokenized = batch_1[0].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

# step2: padding; pad all list to same size
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

# step3: attention mask, to ignore the padding we've added 0
attention_mask = np.where(padded != 0, 1, 0)

# step4: Deep Learning: bert
input_ids = torch.tensor(padded, dtype=torch.long)
attention_mask = torch.tensor(attention_mask)
with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)
features = last_hidden_states[0][:, 0, :].numpy()
labels = batch_1[1]

# train test split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

parameters = {"C":np.linspace(0.0001, 100, 20)}
grid_search = GridSearchCV(LogisticRegression(), parameters)
grid_search.fit(train_features, train_labels)

print("best parameters:", grid_search.best_params_, "\tbest scores:", grid_search.best_score_)

# train logisticRegression model
lr_clf = LogisticRegression(C=grid_search.best_params_["C"])
lr_clf.fit(train_features, train_labels)


# evaluate model
print(lr_clf.score(test_features, test_labels))





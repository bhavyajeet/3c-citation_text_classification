from sklearn.ensemble import RandomForestClassifier
from transformers import AutoTokenizer, AutoModel
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from torch import cuda
import csv
import torch
import sys

device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)

file = open('train.csv')
cr = csv.reader(file)
lines = list(cr)
file.close()

X=[]
Y=[]

for i in lines:
	input_ids = torch.tensor(tokenizer.encode(i[-2])).unsqueeze(0)
	outputs = model(input_ids.to(device))
	last_hidden_states = outputs[0]
	X.append(last_hidden_states.tolist()[0][0])
	Y.append(int(i[-1]))

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, Y)

file = open('validation.csv')
cr = csv.reader(file)
lines = list(cr)
file.close()

corr = 0

preds = []
Y = []

for i in lines:
	input_ids = torch.tensor(tokenizer.encode(i[-2])).unsqueeze(0)
	outputs = model(input_ids.to(device))
	last_hidden_states = outputs[0]
	pred = clf.predict([last_hidden_states.tolist()[0][0]])
	preds.append(pred)
	Y.append(int(i[-1]))
	if pred == int(i[-1]):
		corr+=1

mf1 = f1_score(Y, preds, average='macro')
acc = corr/len(lines)
file = open("random_forest_{0}.txt".format(sys.argv[1]),'w')
file.write("Macro F1: {0}\nAccuracy: {1}\n".format(mf1,acc))
file.close()

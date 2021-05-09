import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import sys
import csv
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel


SciBERTTokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
SciBERTModel = AutoModel.from_pretrained(sys.argv[1])

device = 'cuda' if cuda.is_available() else 'cpu'

train_dataset = pd.read_csv('./combined.csv', sep=',', names=['CGT','CDT','CC','label'])
testing_dataset = pd.read_csv('./validation.csv', sep=',', names=['CGT','CDT','CC','label'])

MAX_LEN = 512
TRAIN_BATCH_SIZE = int(sys.argv[2])
VALID_BATCH_SIZE = int(sys.argv[2])
EPOCHS = 3
LEARNING_RATE = float(sys.argv[3])
tokenizer = SciBERTTokenizer

class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        CGT = str(self.data.CGT[index])
        CGT = " ".join(CGT.split())
        inputs = self.tokenizer.encode_plus(
            CGT,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        CGT_ids = inputs['input_ids']
        CGT_mask = inputs['attention_mask']


        CDT = str(self.data.CDT[index])
        CDT = " ".join(CDT.split())
        inputs = self.tokenizer.encode_plus(
            CDT,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        CDT_ids = inputs['input_ids']
        CDT_mask = inputs['attention_mask']


        CC = str(self.data.CC[index])
        CC = " ".join(CC.split())
        inputs = self.tokenizer.encode_plus(
            CC,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        CC_ids = inputs['input_ids']
        CC_mask = inputs['attention_mask']

        return {
            'CGT_ids': torch.tensor(CGT_ids, dtype=torch.long),
            'CGT_mask': torch.tensor(CGT_mask, dtype=torch.long),
            
            'CDT_ids': torch.tensor(CDT_ids, dtype=torch.long),
            'CDT_mask': torch.tensor(CDT_mask, dtype=torch.long),
            
            'CC_ids': torch.tensor(CC_ids, dtype=torch.long),
            'CC_mask': torch.tensor(CC_mask, dtype=torch.long),
            
            'targets': torch.tensor(self.data.label[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len


training_set = Triage(train_dataset, tokenizer, MAX_LEN)
testing_set = Triage(testing_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

class SciBERTClass(torch.nn.Module):
    def __init__(self):
        super(SciBERTClass, self).__init__()
        self.l1 = SciBERTModel
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(float(sys.argv[4]))
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, data):
        
        input_ids = data['CC_ids'].to(device, dtype = torch.long)
        attention_mask = data['CC_mask'].to(device, dtype = torch.long)
        
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = SciBERTClass()
model.to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr=LEARNING_RATE)

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct


def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    for _,data in enumerate(training_loader, 0):
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(data)
        # print(outputs.shape)
        # print(targets.shape)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}\n')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}\n")
    print(f"Training Accuracy Epoch: {epoch_accu}\n")
    print("\n")
    return

def inference(model):
    model.eval()
    file=open('SDP_test.csv','r')
    cr = csv.reader(file)
    lines = list(cr)
    file.close()

    to_write = []

    for line in lines:
        inputs = tokenizer.encode_plus(
            line[-1],
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        data={'CC_ids':torch.tensor([inputs['input_ids']], dtype=torch.long),'CC_mask':torch.tensor([inputs['attention_mask']], dtype=torch.long)}
        outputs = model(data).squeeze()
        # print(outputs)
        big_val, big_idx = torch.max(outputs.data, dim=0)
        # print(big_idx.tolist())
        to_write.append( [line[0],big_idx.tolist()] )

    file = open('submit.csv','w')
    cw = csv.writer(file)
    cw.writerows(to_write)
    file.close()

    

for epoch in range(EPOCHS):
    train(epoch)
    # acc = valid(model, testing_loader)
    print("\n")
    print("\n")

inference(model)

from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from torch import cuda
import csv
import torch

device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)

file = open('train.csv')
cr = csv.reader(file)
lines = list(cr)
file.close()

xtrain=[]
ytrain=[]

for i in lines:
	input_ids = torch.tensor(tokenizer.encode(i[-2])).unsqueeze(0)
	outputs = model(input_ids.to(device))
	last_hidden_states = outputs[0]
	xtrain.append(last_hidden_states.tolist()[0][0])
	ytrain.append(int(i[-1]))

xtest=[]
ytest=[]

file = open('validation.csv')
cr = csv.reader(file)
lines = list(cr)
file.close()

for i in lines:
	input_ids = torch.tensor(tokenizer.encode(i[-2])).unsqueeze(0)
	outputs = model(input_ids.to(device))
	last_hidden_states = outputs[0]
	xtest.append(last_hidden_states.tolist()[0][0])
	ytest.append(int(i[-1]))

class classifier(nn.Module):    
    def __init__(self):
        super().__init__()          
        self.lstm = nn.LSTM(768, 
                           768, 
                           bidirectional=True, 
                           dropout=0.1,
                           batch_first=True)
        
        self.fc = nn.Linear(768 * 2, 2)
        
        self.act = nn.Sigmoid()
        
    def forward(self,text):
        
        packed_output, (hidden, cell) = self.lstm(text)
        dense_outputs=self.fc(hidden)
        outputs=self.act(dense_outputs)
        return outputs

model = classifier()

optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)    
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc
    
model = model.to(device)
criterion = criterion.to(device)


def train(model, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0    
    model.train()  
    
    for i in len(range(xtrain)):
        optimizer.zero_grad()   
     
        predictions = model(xtrain[i]).squeeze()  
        loss = criterion(predictions, ytrain[i])        
        
        acc = binary_accuracy(predictions, ytrain[i])   
        
        loss.backward()       
        
        optimizer.step()      
        
        epoch_loss += loss.item()  
        epoch_acc += acc.item()    
        
    return epoch_loss / len(xtrain), epoch_acc / len(xtrain)

N_EPOCHS = 5
for epoch in range(N_EPOCHS):
     
    train_loss, train_acc = train(model, optimizer, criterion)
    
    # valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    # if valid_loss < best_valid_loss:
    #     best_valid_loss = valid_loss
    #     torch.save(model.state_dict(), 'saved_weights.pt')
    
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

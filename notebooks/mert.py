from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch, torch.optim
from torch import nn
import torchaudio.transforms as T
from datasets import load_dataset
import librosa
import pandas as pd
import sklearn.preprocessing
import numpy as np
import tqdm

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)
        self.resample_rate = self.processor.sampling_rate
        self.output = nn.Linear(9984, 13)

    # x represents our data
    def forward(self, x):
        batch_size = x.shape[0]
        inputs = self.processor(x, sampling_rate=self.resample_rate, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
        time_reduced_hidden_states = all_layer_hidden_states.mean(-2).reshape((batch_size, -1))
        return self.output(time_reduced_hidden_states)

df = pd.read_pickle('recordings.pkl')
# df = df[df['player'].isin([3, 7])]

train = df[df['player'].isin([3])]
# train = df[df['type'] == 'scale']
X_train = []
for index, row in tqdm.tqdm(train.iterrows()):
    x, sr = librosa.load(str(row['file']), sr=24000, duration=10)
    X_train.append(x)
X_train = np.stack(X_train)
y_train = sklearn.preprocessing.LabelBinarizer().fit_transform(train['violin'])
# y_train = sklearn.preprocessing.label_binarize(train['violin'], classes=range(1, 14))
y_train = torch.tensor(y_train, dtype=torch.float32)

test = df[df['player'].isin([10])]
# test = df[(df['type'] == 'free') & (df['player'] == 7)]
print(test)
X_test = []
for index, row in tqdm.tqdm(test.iterrows()):
    x, sr = librosa.load(str(row['file']), sr=24000, duration=10)
    X_test.append(x)
X_test = np.stack(X_test)
y_test = sklearn.preprocessing.LabelBinarizer().fit_transform(test['violin'])
# y_test = sklearn.preprocessing.label_binarize(test['violin'], classes=range(1, 14))
y_test = torch.tensor(y_test, dtype=torch.float32)

model = Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# batch_size = 10
batch_size = len(y_train)
batches_per_epoch = len(X_train) // batch_size

for epoch in range(30):
    epoch_loss = []
    epoch_acc = []
    
    model.train()
    for i in range(batches_per_epoch):
        # take a batch
        start = i * batch_size
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        # forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
        # compute and store metrics
        acc = (torch.argmax(y_pred, -1) == torch.argmax(y_batch, -1)).float().mean()
        print(acc)
        print(loss)
        epoch_loss.append(float(loss))
        epoch_acc.append(float(acc))

    print('EVALUATION')
    model.eval()
    # y_pred = model(X_train)
    # loss = loss_fn(y_pred, y_train)
    # acc = (torch.argmax(y_pred, -1) == torch.argmax(y_train, -1)).float().mean()
    # print(acc)
    # print(loss)
    y_pred = model(X_test)
    print(y_pred)
    loss = loss_fn(y_pred, y_test)
    acc = (torch.argmax(y_pred, -1) == torch.argmax(y_test, -1)).float().mean()
    print(acc)
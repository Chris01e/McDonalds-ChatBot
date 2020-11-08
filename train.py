import numpy as np
import random
import json
import nltk
nltk.download('punkt')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
#(lặp lại từng câu trong các mẫu ý định của chúng tôi)
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list (thêm vào danh sách thẻ)
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence (mã hóa từng từ trong câu)
        w = tokenize(pattern)
        # add to our words list (thêm vào danh sách từ của chúng tôi)
        all_words.extend(w)
        # add to xy pair (thêm vào cặp xy)
        xy.append((w, tag))

# stem and lower each word (gốc và hạ thấp từng từ)
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort (loại bỏ các bản sao và sắp xếp)
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data (tạo tệp dữ liệu train)
X_train = [] 
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence (túi từ cho mỗi câu mẫu)
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    # Thư viện Pytorch chỉ cần labels một lớp câu, không phải lable từng câu.
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters (Siêu tham số)
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    #hỗ trợ lập chỉ mục để tập dữ liệu [i] có thể được sử dụng để lấy mẫu thứ i
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    #chúng ta có thể gọi len (dataset) để trả về kích thước
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer (Mất mát và tối ưu hóa)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model (Đào tạo mô hình)
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass (Chuyển tiếp qua)
        outputs = model(words)
        # if y would be one-hot, we must apply 
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize (Lùi lại và tối ưu hóa)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')

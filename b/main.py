import torch
import torch.nn as nn
import numpy as np

x_data = ["hat ", "rat ", "cat ", "flat", "matt", "cap ", "son "]
y_data = ["🎩", "🐀", "🐈", "🏠", "🚶", "🧢", "🧒"]

characters = 27
classes = len(y_data)
characters_per_word = 4

char_encodings = np.eye(characters)


def string_to_array(string):
    if len(string) != characters_per_word:
        raise ValueError("Length of string must be 4")
    char_array = list(string)
    enc = np.zeros((len(char_array), characters))
    for c_i in range(len(char_array)):
        if char_array[c_i] == ' ':
            enc[c_i] = np.array(char_encodings[0])
        else:
            enc[c_i] = np.array(char_encodings[ord(char_array[c_i]) - 96])
    return enc


def index_to_char(i):
    if i == 0:
        return ' '
    else:
        return chr(i + 96)


amount_x_data = len(x_data)
x_train_values = np.zeros((amount_x_data, characters_per_word, characters))
for x_i in range(amount_x_data):
    x_train_values[x_i] = string_to_array(x_data[x_i])

y_train_values = np.eye(7)

x_train = torch.tensor(x_train_values, requires_grad=True, dtype=torch.float)
y_train = torch.tensor(y_train_values, dtype=torch.float)

x_test_data = ['rat ', 'rt  ']
x_test_values = np.zeros((len(x_test_data), characters_per_word, characters))
for i in range(len(x_test_data)):
    x_test_values[i] = string_to_array(x_test_data[i])

x_test = torch.tensor(x_test_values, dtype=torch.float)


class LSTM_ManyOne(nn.Module):
    def __init__(self, encoding_size, label_length):
        self.inner_size = 128
        super(LSTM_ManyOne, self).__init__()
        self.lstm = nn.LSTM(encoding_size, self.inner_size)
        self.dense = nn.Linear(self.inner_size, label_length)

    def logits(self, x):
        print(x.shape)
        out, _ = self.lstm(x)
        print(out[-1].shape)
        return self.dense(out[-1].reshape(-1, self.inner_size))

    def forward(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        x = self.logits(x)
        y = y.argmax(1)
        print(x.shape)
        print(y.shape)
        return nn.functional.cross_entropy(x, y)


model = LSTM_ManyOne(characters, 7)

lr = 0.05
epochs = 500
optimizer = torch.optim.RMSprop(model.parameters(), lr)

for epoch in range(epochs):
    loss = model.loss(x_train, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 100 == 99:
        print("Epoch %i: Loss: %s" % (epoch, loss.item()))

print("----------------")

y_values = model.forward(x_test).detach().numpy()


import torch
import torch.nn as nn
import numpy as np


class Dataset:
    def __init__(self, data):
        self.data = np.array(data)
        self.unique_chars = np.unique(list(''.join(self.data[:, 0])))
        self.char_to_index = dict()
        for element in self.unique_chars:
            self.char_to_index[element] = len(self.char_to_index)

    def encode_char(self, char):
        return np.eye(len(self.unique_chars))[self.char_to_index[char]]

    def encode_words(self):
        one_hots = []
        for word in self.data[:, 0]:
            one_hot = []
            for char in word:
                one_hot.append(self.encode_char(char))
            one_hots.append(one_hot)
        return one_hots

    def y_to_word(self, one_hot):
        return self.data[one_hot.argmax(), 1]


dataset = Dataset([['hat ', '🎩'],
                   ['rat ', '🐀'],
                   ['cat ', '🐈'],
                   ['flat', '🏢'],
                   ['matt', '👨'],
                   ['cap ', '🧢'],
                   ['son ', '👦']])
x_train = torch.tensor(dataset.encode_words(), dtype=torch.float).transpose(0, 1)
y_train = torch.tensor(np.eye(np.shape(x_train)[1]), dtype=torch.float)


class LSTMManyOne(nn.Module):
    def __init__(self, encoding_size, label_size):
        self.state_size = 128
        super(LSTMManyOne, self).__init__()
        self.lstm = nn.LSTM(encoding_size, self.state_size)
        self.dense = nn.Linear(self.state_size, label_size)

    def reset(self, batch_size):
        zero_state = torch.zeros(1, batch_size, self.state_size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out[-1].reshape(-1, self.state_size))

    def forward(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


model = LSTMManyOne(len(dataset.unique_chars), np.shape(y_train)[1])


def generate(text):
    model.reset(1)
    gen_x = []
    for char in text:
        gen_x.append(dataset.encode_char(char))
    y = model.forward(torch.tensor(gen_x, dtype=torch.float).detach().reshape(4, 1, -1))
    return dataset.y_to_word(y)


lr = 0.05
epochs = 500
optimizer = torch.optim.RMSprop(model.parameters(), lr)

for epoch in range(epochs):
    model.reset(x_train.size(1))
    loss = model.loss(x_train, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 100 == 9:
        print("Epoch %i: Loss: %s | 'rats'=>%s, 'rt  '=>%s, 'mt  '=>%s" % (epoch,
                                                                           loss.item(),
                                                                           generate('rats'),
                                                                           generate('rt  '),
                                                                           generate('mt  ')))


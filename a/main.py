import torch
import torch.nn as nn


# Result: "hello world    rld    rld    rld    rld    rld    rl"


class LSTM_ManyMany(nn.Module):
    def __init__(self, encoding_size):
        super(LSTM_ManyMany, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)
        self.dense = nn.Linear(128, encoding_size)

    def reset(self):
        zero_state = torch.zeros(1, 1, 128)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0.],  # ' '
    [0., 1., 0., 0., 0., 0., 0., 0.],  # 'h'
    [0., 0., 1., 0., 0., 0., 0., 0.],  # 'e'
    [0., 0., 0., 1., 0., 0., 0., 0.],  # 'l'
    [0., 0., 0., 0., 1., 0., 0., 0.],  # 'o'
    [0., 0., 0., 0., 0., 1., 0., 0.],  # 'w'
    [0., 0., 0., 0., 0., 0., 1., 0.],  # 'r'
    [0., 0., 0., 0., 0., 0., 0., 1.]   # 'd'
]

encoding_size = len(char_encodings)

index_to_char = [' ', 'h', 'e', 'l', 'o', 'w', 'r', 'd']

x_train = torch.tensor([[char_encodings[0]],   # _
                        [char_encodings[1]],     # h
                        [char_encodings[2]],     # e
                        [char_encodings[3]],     # l
                        [char_encodings[3]],     # l
                        [char_encodings[4]],     # o
                        [char_encodings[0]],     # _
                        [char_encodings[5]],     # w
                        [char_encodings[4]],     # o
                        [char_encodings[6]],     # r
                        [char_encodings[3]],     # l
                        [char_encodings[7]],     # d
                        [char_encodings[0]]])    # _

y_train = torch.tensor([char_encodings[1],     # h
                        char_encodings[2],     # e
                        char_encodings[3],     # l
                        char_encodings[3],     # l
                        char_encodings[4],     # o
                        char_encodings[0],     # _
                        char_encodings[5],     # w
                        char_encodings[4],     # o
                        char_encodings[6],     # r
                        char_encodings[3],     # l
                        char_encodings[7],     # d
                        char_encodings[0],
                        char_encodings[0]])

model = LSTM_ManyMany(encoding_size)


lr = 0.001
epochs = 5000
optimizer = torch.optim.RMSprop(model.parameters(), lr)

for epoch in range(epochs):
    model.reset()
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

model.reset()
text = ' h'
model.f(torch.tensor([[char_encodings[0]]]))
y = model.f(torch.tensor([[char_encodings[1]]]))
text += index_to_char[y.argmax(1)]
for c in range(50):
    y = model.f(torch.tensor([[char_encodings[y.argmax(1)]]]))
    text += index_to_char[y.argmax(1)]
print(text)




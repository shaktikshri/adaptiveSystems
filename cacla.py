import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable


class CACLA(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=12):
        super(CACLA, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        # self.layer2 = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU()
        # )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        # out = self.layer2(out)
        out = self.output_layer(out)
        return out


# In[89]:
network = CACLA(input_size=1, output_size=1)

# TODO : Need to make this part of the scheduler, when you are using
learning_rate = 0.001
criterion = nn.MSELoss()

# TODO : Check if you should use SGD here for the cacla
optimizer = optim.Adam(network.parameters(), lr=learning_rate)
X = np.arange(-1, 1, 0.001).reshape(-1, 1)
X = Variable(torch.from_numpy(X))
y = np.power(X, 2)
# y = torch.from_numpy(y)

for epoch in range(10000):
    outputs = network(X.float())
    loss = criterion(outputs, y.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print('Epoch : {}, Loss: {}'.format(epoch, loss.item()))

X_test = np.arange(-1,1,0.001).reshape(-1, 1)
y_test = np.power(X_test, 2)
y_pred = network(Variable(torch.from_numpy(X_test)).float()).data.numpy()
import matplotlib.pyplot as plt
plt.plot(X_test, y_pred)

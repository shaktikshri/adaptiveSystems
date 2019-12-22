import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
def main():
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

# In[]:

# This update is taken from PyTorch examples in
# https://github.com/jcjohnson/pytorch-examples/blob/master/README.md#pytorch-defining-new-autograd-functions
# We want to have a manual update scheme for the parameters where we update the parameters of critic only when the
# TD error is greater than 0. The update to the actor can be done with the same fucntion approxmiation that we wrote
# in the beginning.
# TODO : Need to make the actor and critic concurrent. Implement both of them either in Pytorch or in numpy
# TODO : but do remember to check for the correctness.

# import numpy as np
#
# # N is batch size; D_in is input dimension;
# # H is hidden dimension; D_out is output dimension.
# N, D_in, H, D_out = 64, 1, 12, 1
#
# # Create random input and output data
# x = np.random.randn(N, D_in)
#
# y = np.power(x, 2)
#
# # Randomly initialize weights
# w1 = np.random.randn(H, D_in)
# w2 = np.random.randn(D_out, H)
#
# learning_rate = 1e-6
# for t in range(500):
#     # Forward pass: compute predicted y
#     h = x.dot(w1)
#     h_relu = np.maximum(h, 0)
#     y_pred = h_relu.dot(w2)
#
#     loss = np.square(y_pred - y).sum()
#     print('Iteration : ', t, 'Loss : ', loss)
#
#     grad_y_pred = 2.0 * (y_pred - y)
#     grad_w2 = h_relu.T.dot(grad_y_pred)
#     grad_h_relu = grad_y_pred.dot(w2.T)
#     grad_h = grad_h_relu.copy()
#     grad_h[h < 0] = 0
#     grad_w1 = x.T.dot(grad_h)
#
#     # Update weights
#     w1 -= learning_rate * grad_w1
#     w2 -= learning_rate * grad_w2



# In[]:

import numpy as np
learning_rate = 0.01

X = np.arange(-1, 1, 0.001).reshape(-1, 1)
X_new = np.ones((X.shape[0], X.shape[1]+1))
X_new[:, 1] = np.squeeze(X)
X = X_new

t = np.power(X[:,1], 2).reshape(-1, 1)
input_dim = X.shape[1]
hidden_dim = 12
output_dim = 1

# TODO : Use autograd to find this out
W1 = np.random.randn(hidden_dim, input_dim)
W2 = np.random.randn(output_dim, hidden_dim)
# TODO : Can change the activation function as well
relu = lambda z : np.maximum(z, 0)
sigma = lambda z : 1/(1+np.exp(-z))
relu_dash = lambda z: np.where(z > 0, 1, 0)

loss_array = list()
for epoch in range(100000):
    # forward pass
    Z = np.dot(X, W1.T)
    H = relu(Z)
    y = np.dot(H, W2.T)
    L = (1/X.shape[0]) * np.sum(np.power(y-t, 2))
    # print('Epoch ', epoch, 'Loss : ', L)
    loss_array.append(L)
    # backward pass
    L_bar = 1
    y_bar = (1/X.shape[0]) * L_bar * (y-t)
    H_bar = np.dot(y_bar, W2)
    W2_bar = np.dot(y_bar.T, H)
    Z_bar = H_bar * relu_dash(Z)
    W1_bar = np.dot(Z_bar.T, X)

    # Update the Gradient
    W1 -= learning_rate * W1_bar
    W2 -= learning_rate * W2_bar

    if epoch % 1000 == 0:
        print('Epoch : ', epoch, ' Loss : ', L, 'Learning Rate : ', learning_rate)
        learning_rate = min(learning_rate, np.exp(-epoch//1000))
        learning_rate = max(1e-05, learning_rate)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
plt.plot(loss_array)

# In[]:

X = np.arange(-1, 1, 0.001).reshape(-1, 1)
X_new = np.ones((X.shape[0], X.shape[1]+1))
X_new[:, 1] = np.squeeze(X)
X = X_new

Z = np.dot(X, W1.T)
H = relu(Z)
y = np.dot(H, W2.T)

plt.plot(X[:, 1], y)
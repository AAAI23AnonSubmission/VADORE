"""
Utils for defining the neural network modules used in Vad.0 and Vad.1
"""

import torch
import torch.nn as nn

class ReluTanhEncoder(nn.Module):
    '''
    Corresponds to \phi and \psi in the Vad.0 module
    '''
    def __init__(self, n_features, lay1size, lay2size, lay3size):
        super(ReluTanhEncoder, self).__init__()
        self.n_features = n_features
        self.lay1size = lay1size
        self.lay2size = lay2size
        self.lay3size = lay3size
        self.fc1 = nn.Linear(self.n_features, self.lay1size)
        self.fc2 = nn.Linear(self.lay1size, self.lay2size)
        self.fc3 = nn.Linear(self.lay2size, self.lay3size)
    def forward(self, x):
        x = x.view(-1, self.n_features)
        x = torch.relu(self.fc1(x))
        x = x.view(-1, self.lay1size)
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class BilinearForm(nn.Module):
    '''
    The A matrix in the \phi(x)^T A \psi(y) Vad.0 module
    '''
    def __init__(self, n_features_x, n_features_y):
        super(BilinearForm, self).__init__()
        self.bilin = nn.Bilinear(n_features_x, n_features_y, 1)
    def forward(self, x, y):
        return self.bilin(x, y)

class SecondStageRanker(torch.nn.Module):
    '''
    Re-ranker corresponding to Vad.1
    '''
    def __init__(self, xshape, yshape, intermediary_dim):
        super(SecondStageRanker, self).__init__()
        self.linear_x = torch.nn.Linear(xshape, intermediary_dim)
        self.linear_y = torch.nn.Linear(yshape, intermediary_dim)
        self.linear2 = torch.nn.Linear(intermediary_dim * 3, intermediary_dim)
        self.linear3 = torch.nn.Linear(intermediary_dim, 1)

    def forward(self, x, y):
        '''
        x: user features
        y: item features (can include rank)
        '''
        x = self.linear_x(x)
        y = self.linear_y(y)
        z = torch.cat([x, y, x * y], axis=1)
        outputs = torch.relu(self.linear2(z))
        outputs = self.linear3(outputs)
        return outputs

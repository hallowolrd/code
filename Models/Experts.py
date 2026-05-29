import torch.nn.functional as F
from torch import nn
import torch


class Expert1(nn.Module):
    def __init__(self, output_dim):
        super(Expert1, self).__init__()
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(8 * 8 * 64, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 8 * 8 * 64)
        x = F.relu(self.dropout(self.fc1(x)))
        return x


class Expert2(nn.Module):
    def __init__(self, output_dim):
        super(Expert2, self).__init__()
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4 * 4 * 128, output_dim)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,4 * 4 * 128)
        x = F.relu(self.dropout(self.fc1(x)))
        return x

class Expert3(nn.Module):
    def __init__(self, output_dim):
        super(Expert3, self).__init__()
        self.dropout = nn.Dropout(0.6)
        self.fc1 = nn.Linear(16*16*32, output_dim)
    def forward(self, x):
        x = x.view(-1,16*16*32)
        x = F.relu(self.dropout(self.fc1(x)))
        return x

class Expert4(nn.Module):
    def __init__(self, output_dim):
        super(Expert4, self).__init__()
        self.dropout = nn.Dropout(0.6)
        self.fc1 = nn.Linear(output_dim, output_dim)
        self.out_dim = output_dim
    def forward(self, x):
        x = x.view(-1,self.out_dim)
        x = F.relu(self.dropout(self.fc1(x)))
        return x



# class Expert1(nn.Module):
#     def __init__(self, output_dim):
#         super(Expert1, self).__init__()
#         self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(16 * 16 * 64, output_dim)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = x.view(-1, 16 * 16 * 64)
#         x = F.relu(self.dropout(self.fc1(x)))
#         return x
#
#
# class Expert2(nn.Module):
#     def __init__(self, output_dim):
#         super(Expert2, self).__init__()
#         self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(8 * 8 * 128, output_dim)
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1,8 * 8 * 128)
#         x = F.relu(self.dropout(self.fc1(x)))
#         return x
#
# class Expert3(nn.Module):
#     def __init__(self, output_dim):
#         super(Expert3, self).__init__()
#         self.dropout = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(32*32*32, output_dim)
#     def forward(self, x):
#         x = x.view(-1,32*32*32)
#         x = F.relu(self.dropout(self.fc1(x)))
#         return x


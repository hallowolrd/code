import random
import torch.nn.functional as F
from torch import nn
import torch
from Models.MoE2 import *


class GlobalCNN(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.2,hidden_dim=256):
        super(GlobalCNN, self).__init__()
        self.top_k = 4
        self.num_experts = 4
        self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.experts = nn.ModuleDict({"Expert1":Expert1(output_dim=hidden_dim),"Expert2":Expert2(output_dim=hidden_dim),"Expert3":Expert3(output_dim=hidden_dim),"SubMoELayer":SubMoELayer(output_dim=hidden_dim)})

        self.header = nn.Linear(self.num_experts * hidden_dim, num_classes)
        # self.header = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        conv_out = self.pool(F.relu(self.conv(x)))
        feature = self.dropout(conv_out)
        expert_output = torch.cat([expert(feature) for k,expert in self.experts.items()], dim=1)
        x = self.header(self.dropout(expert_output))
        return x,feature




class CNNWithMoE(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.2,hidden_dim=256):
        super(CNNWithMoE, self).__init__()
        self.top_k = 2
        self.num_experts = 3
        self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.moe = MoELayer(output_dim=hidden_dim,top_k=self.top_k,num_experts=self.num_experts)

        self.header = nn.Linear(self.num_experts * hidden_dim, num_classes)
        # self.header = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        conv_out = self.pool(F.relu(self.conv(x)))
        feature = self.dropout(conv_out)
        moe_output,expert_activations,shared_activations = self.moe(feature)
        x = self.header(self.dropout(moe_output))
        return x,feature,expert_activations,shared_activations


class CNNWithMoE_1(CNNWithMoE):
    def __init__(self,num_classes,hidden_dim):
        super(CNNWithMoE_1,self).__init__(num_classes=num_classes, hidden_dim=hidden_dim,dropout_rate=0.1)
        self.top_k = 4
        self.num_experts = 5
        self.moe = MoELayer_1(output_dim=hidden_dim,top_k=self.top_k,num_experts=self.num_experts)
        self.header = nn.Linear(self.top_k * hidden_dim, num_classes)
        # self.header = nn.Linear(hidden_dim, num_classes)



class CNNWithMoE_2(CNNWithMoE):
    def __init__(self,num_classes,hidden_dim):
        super(CNNWithMoE_2,self).__init__(num_classes=num_classes, hidden_dim=hidden_dim,dropout_rate=0.1)
        self.top_k = 3
        self.num_experts = 5
        self.moe = MoELayer_2(output_dim=hidden_dim,top_k=self.top_k,num_experts=self.num_experts)
        self.header = nn.Linear(self.top_k * hidden_dim, num_classes)
        # self.header = nn.Linear(hidden_dim, num_classes)




class CNNWithMoE_3(CNNWithMoE):
    def __init__(self,num_classes,hidden_dim):
        super(CNNWithMoE_3,self).__init__(num_classes=num_classes, hidden_dim=hidden_dim,dropout_rate=0.1)
        self.top_k = 4
        self.num_experts = 6
        self.moe = MoELayer_3(output_dim=hidden_dim,top_k=self.top_k,num_experts=self.num_experts)
        self.header = nn.Linear(self.top_k * hidden_dim, num_classes)
        # self.header = nn.Linear(hidden_dim, num_classes)





class CNNWithMoE_4(CNNWithMoE):
    def __init__(self, num_classes, hidden_dim):
        super(CNNWithMoE_4, self).__init__(num_classes=num_classes, hidden_dim=hidden_dim, dropout_rate=0.1)
        self.top_k = 3
        self.num_experts = 5
        self.moe = MoELayer_4(output_dim=hidden_dim,top_k=self.top_k,num_experts=self.num_experts)
        self.header = nn.Linear(self.top_k * hidden_dim, num_classes)
        # self.header = nn.Linear(hidden_dim, num_classes)

class CNNWithMoE_5(CNNWithMoE):
    def __init__(self, num_classes, hidden_dim):
        super(CNNWithMoE_5, self).__init__(num_classes=num_classes, hidden_dim=hidden_dim, dropout_rate=0.1)
        self.top_k = 1
        self.num_experts = 2
        self.moe = MoELayer_5(output_dim=hidden_dim,top_k=self.top_k,num_experts=self.num_experts)
        self.header = nn.Linear(self.top_k * hidden_dim, num_classes)




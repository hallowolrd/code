from Models.Experts import *
import torch.nn.functional as F
from torch import nn
import torch


class MoELayer(nn.Module):
    def __init__(self, output_dim,top_k,num_experts):
        super(MoELayer, self).__init__()
        self.device = "cuda"
        self.top_k = top_k
        self.num_experts = num_experts
        self.experts = nn.ModuleList()
        self.gate = nn.Linear(16*16*32, self.num_experts)
        self.expert_activations = [0] * (self.num_experts - 1)

    def forward(self, x):
        gate_input = x.view(-1, 16*16*32)
        gate_scores = self.gate(gate_input)
        gate_probs = F.softmax(gate_scores, dim=-1)
        top_k_gate_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)



        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        temp = torch.take_along_dim(expert_outputs, top_k_indices.unsqueeze(-1), dim=1).squeeze(-1)
        output = temp.view(temp.shape[0], -1)
        # activation
        for i in range(top_k_indices.size(0)):
            for j in range(self.top_k):
                expert_idx = top_k_indices[i, j].item()
                if expert_idx != self.num_experts - 1:
                    self.expert_activations[expert_idx] += 1
        return output,torch.tensor(self.expert_activations,device=self.device),self.experts[-1].expert_activations
    def reset_activation_counts(self):
        self.expert_activations = [0] * (self.num_experts - 1)


class SubMoELayer(nn.Module):
    def __init__(self, output_dim,top_k=4,num_experts=6):
        super(SubMoELayer, self).__init__()
        self.device = device
        self.top_k = 2
        self.num_experts = 3
        self.output_dim = output_dim
        self.linear = nn.Linear(16*16*32,output_dim)
        self.experts = nn.ModuleList([Expert4(output_dim=output_dim) for i in range(self.num_experts)])
        # self.experts = nn.ModuleList([Expert4(output_dim=output_dim),Expert4(output_dim=output_dim),SubsubMoELayer(output_dim=output_dim)])
        self.gate = nn.Linear(self.output_dim, self.num_experts)
        self.expert_activations = [0] * self.num_experts

    def forward(self, x):
        x = self.linear(x.view(-1, 16*16*32))
        gate_scores = self.gate(x)
        gate_probs = F.softmax(gate_scores, dim=-1)
        top_k_gate_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_gate_probs = top_k_gate_probs / top_k_gate_probs.sum(dim=-1, keepdim=True)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        selected_expert_outputs = torch.gather(expert_outputs, 1,top_k_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)))
        output = (selected_expert_outputs * top_k_gate_probs.unsqueeze(-1)).sum(dim=1)
        return output

    def reset_activation_counts(self):
        self.expert_activations = [0] * self.num_experts

class SubsubMoELayer(nn.Module):
    def __init__(self, output_dim,top_k=4,num_experts=6):
        super(SubsubMoELayer, self).__init__()
        self.device = device
        self.top_k = 2
        self.num_experts = 3
        self.output_dim = output_dim
        self.experts = nn.ModuleList([Expert4(output_dim=output_dim) for i in range(self.num_experts)])
        self.gate = nn.Linear(output_dim, self.num_experts)
        self.expert_activations = [0] * self.num_experts

    def forward(self, x):
        gate_input = x.view(-1, self.output_dim)
        gate_scores = self.gate(gate_input)

        gate_probs = F.softmax(gate_scores, dim=-1)
        top_k_gate_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_gate_probs = top_k_gate_probs / top_k_gate_probs.sum(dim=-1, keepdim=True)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        selected_expert_outputs = torch.gather(expert_outputs, 1,top_k_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)))
        output = (selected_expert_outputs * top_k_gate_probs.unsqueeze(-1)).sum(dim=1)
        return output
    def reset_activation_counts(self):
        self.expert_activations = [0] * self.num_experts






# class SubMoELayer(nn.Module):
#     def __init__(self, output_dim,top_k=4,num_experts=6):
#         super(SubMoELayer, self).__init__()
#         self.device = "cuda"
#         self.top_k = 4
#         self.num_experts = 6
#         self.experts = nn.ModuleList([Expert3(output_dim=output_dim) for i in range(self.num_experts)])
#         self.gate = nn.Linear(32*32*32, self.num_experts)
#         self.expert_activations = [0] * self.num_experts
#
#     def forward(self, x):
#         gate_input = x.view(-1, 32*32*32)
#         gate_scores = self.gate(gate_input)
#         gate_probs = F.softmax(gate_scores, dim=-1)
#         top_k_gate_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
#         top_k_gate_probs = top_k_gate_probs / top_k_gate_probs.sum(dim=-1, keepdim=True)
#         expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
#         selected_expert_outputs = torch.gather(expert_outputs, 1,top_k_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)))
#         output = (selected_expert_outputs * top_k_gate_probs.unsqueeze(-1)).sum(dim=1)
#         for i in range(top_k_indices.size(0)):
#             for j in range(self.top_k):
#                 expert_idx = top_k_indices[i, j].item()
#                 self.expert_activations[expert_idx] += 1
#
#         return output,torch.tensor(self.expert_activations,device=self.device)
#     def reset_activation_counts(self):
#         self.expert_activations = [0] * self.num_experts
#
# class SubsubMoELayer(nn.Module):
#     def __init__(self, output_dim,top_k=4,num_experts=6):
#         super(SubsubMoELayer, self).__init__()
#         self.device = "cuda"
#         self.top_k = 4
#         self.num_experts = 6
#         self.experts = nn.ModuleList([Expert3(output_dim=output_dim) for i in range(self.num_experts)])
#         self.gate = nn.Linear(32*32*32, self.num_experts)
#         self.expert_activations = [0] * self.num_experts
#
#     def forward(self, x):
#         gate_input = x.view(-1, 32*32*32)
#         gate_scores = self.gate(gate_input)
#         gate_probs = F.softmax(gate_scores, dim=-1)
#         top_k_gate_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
#         top_k_gate_probs = top_k_gate_probs / top_k_gate_probs.sum(dim=-1, keepdim=True)
#         expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
#         selected_expert_outputs = torch.gather(expert_outputs, 1,top_k_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)))
#         output = (selected_expert_outputs * top_k_gate_probs.unsqueeze(-1)).sum(dim=1)
#         return output
#     def reset_activation_counts(self):
#         self.expert_activations = [0] * self.num_experts
#
# class MoELayer(nn.Module):
#     def __init__(self, output_dim,top_k,num_experts):
#         super(MoELayer, self).__init__()
#         self.device = "cuda"
#         self.top_k = top_k
#         self.num_experts = num_experts
#         self.experts = nn.ModuleList()
#         self.gate = nn.Linear(32*32*32, self.num_experts)
#         self.expert_activations = [0] * self.num_experts
#
#     def forward(self, x):
#         gate_input = x.view(-1, 32*32*32)
#         gate_scores = self.gate(gate_input)
#         gate_probs = F.softmax(gate_scores, dim=-1)
#         top_k_gate_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
#
#
#         personal_output = torch.cat([expert(x) for expert in self.experts[:-1]], dim=1)
#         shared_output = self.experts[-1](x)
#         output = torch.cat([personal_output,shared_output[0]],dim=-1)
#
#         # expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
#         # temp = torch.take_along_dim(expert_outputs, top_k_indices.unsqueeze(-1), dim=1).squeeze(-1)
#         # output = temp.view(temp.shape[0], -1)
#
#         # activation
#         for i in range(top_k_indices.size(0)):
#             for j in range(self.top_k):
#                 expert_idx = top_k_indices[i, j].item()
#                 self.expert_activations[expert_idx] += 1
#
#         return output,torch.tensor(self.expert_activations,device=self.device),shared_output[1]
#     def reset_activation_counts(self):
#         self.expert_activations = [0] * self.num_experts


class MoELayer_1(MoELayer):
    def __init__(self,output_dim,top_k,num_experts):
        super(MoELayer_1,self).__init__(output_dim,top_k,num_experts)

        self.experts = nn.ModuleList([Expert1(output_dim),
                                      Expert1(output_dim),
                                      Expert1(output_dim),
                                      Expert2(output_dim),
                                      SubMoELayer(output_dim)
                                      ])

class MoELayer_2(MoELayer):
    def __init__(self,output_dim,top_k,num_experts):
        super(MoELayer_2,self).__init__(output_dim,top_k,num_experts)

        self.experts = nn.ModuleList([Expert1(output_dim),
                                      Expert1(output_dim),
                                      Expert2(output_dim),
                                      Expert3(output_dim),
                                      SubMoELayer(output_dim)
                                      ])

class MoELayer_3(MoELayer):
    def __init__(self, output_dim, top_k, num_experts):
        super(MoELayer_3, self).__init__(output_dim, top_k, num_experts)

        self.experts = nn.ModuleList([Expert1(output_dim),
                                      Expert2(output_dim),
                                      Expert3(output_dim),
                                      Expert3(output_dim),
                                      Expert3(output_dim),
                                      SubMoELayer(output_dim)
                                      ])

class MoELayer_4(MoELayer):
    def __init__(self, output_dim,top_k,num_experts):
        super(MoELayer_4, self).__init__(output_dim,top_k,num_experts)

        self.experts = nn.ModuleList([Expert3(output_dim),
                                      Expert3(output_dim),
                                      Expert2(output_dim),
                                      Expert2(output_dim),
                                      SubMoELayer(output_dim)
                                      ])


class MoELayer_5(MoELayer):
    def __init__(self, output_dim,top_k,num_experts):
        super(MoELayer_5, self).__init__(output_dim,top_k,num_experts)

        self.experts = nn.ModuleList([Expert2(output_dim),
                                      SubMoELayer(output_dim)
                                      ])


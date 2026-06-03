import torch
import torch.nn as nn

from .resnet import ResNetBackbone


class ExpertFFN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TopKGating(nn.Module):
    def __init__(self, in_dim, num_experts, topk):
        super().__init__()
        self.topk = topk
        self.gate = nn.Linear(in_dim, num_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        probs = torch.softmax(logits.float(), dim=-1)
        topk_vals, topk_idx = probs.topk(self.topk, dim=-1)
        weights = torch.zeros_like(probs)
        weights.scatter_(1, topk_idx, topk_vals)
        weights = weights.to(x.dtype)
        return weights, topk_idx


class MoELayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_experts, topk):
        super().__init__()
        self.gating = TopKGating(in_dim, num_experts, topk)
        self.experts = nn.ModuleList(
            [ExpertFFN(in_dim, hidden_dim, out_dim) for _ in range(num_experts)]
        )

    def forward(self, x):
        weights, topk_idx = self.gating(x)
        B = x.size(0)
        C = self.experts[0].fc2.out_features
        out = torch.zeros(B, C, device=x.device, dtype=x.dtype)

        for i, expert in enumerate(self.experts):
            expert_mask = topk_idx == i
            token_mask = expert_mask.any(dim=-1)
            if not token_mask.any():
                continue
            expert_out = expert(x[token_mask])
            sel_weights = weights[token_mask, i]
            out[token_mask] += expert_out * sel_weights.unsqueeze(-1)

        return out


class MoEFedModel(nn.Module):
    def __init__(self, in_channels, num_classes, img_size, num_experts, topk):
        super().__init__()
        self.backbone = ResNetBackbone(in_channels, img_size)
        feat_dim = self.backbone.feat_dim
        self.moe_head = MoELayer(feat_dim, 512, num_classes, num_experts, topk)

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.moe_head(feat)
        return logits

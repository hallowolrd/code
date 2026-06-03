import math

import torch


def _safe_add_contribution(fisher_totals, expert_id, contribution):
    value = float(contribution.detach().cpu().item())
    if not math.isfinite(value):
        print(
            f"[FisherTotalHook][Warning] non-finite contribution for expert "
            f"{expert_id}; set to 0"
        )
        value = 0.0
    fisher_totals[expert_id] += value


def compute_expert_fisher_total_hook(model, loader, device):
    """
    Compute expert-level total empirical Fisher for MoE experts using Linear hooks.

    Returns:
        fisher_totals: list[float], length = num_experts
        expert_usage: list[int], length = num_experts
    """
    experts = model.moe_head.experts
    num_experts = len(experts)
    fisher_totals = [0.0 for _ in range(num_experts)]
    expert_usage = [0 for _ in range(num_experts)]
    activations = {}
    hooks = []

    module_training_states = [(module, module.training) for module in model.modules()]
    param_requires_grad = [(param, param.requires_grad) for param in model.parameters()]

    def make_forward_hook(expert_id, count_usage):
        def forward_hook(module, inputs, output):
            del output
            if not inputs or inputs[0] is None:
                activations.pop(module, None)
                return

            activation = inputs[0].detach()
            activations[module] = activation
            if count_usage:
                expert_usage[expert_id] += int(activation.shape[0])

        return forward_hook

    def make_backward_hook(expert_id):
        def backward_hook(module, grad_input, grad_output):
            del grad_input
            if not grad_output or grad_output[0] is None:
                activations.pop(module, None)
                return

            activation = activations.pop(module, None)
            if activation is None:
                return

            grad_out = grad_output[0].detach()
            if activation.shape[0] != grad_out.shape[0]:
                print(
                    f"[FisherTotalHook][Warning] activation/gradient batch "
                    f"mismatch for expert {expert_id}; skip contribution"
                )
                return

            x = activation.float().reshape(activation.shape[0], -1)
            go = grad_out.float().reshape(grad_out.shape[0], -1)
            grad_sq = go.square().sum(dim=1)
            input_sq = x.square().sum(dim=1)

            contribution = (grad_sq * input_sq).sum()
            if module.bias is not None:
                contribution = contribution + grad_sq.sum()
            _safe_add_contribution(fisher_totals, expert_id, contribution)

        return backward_hook

    try:
        model.eval()

        expert_param_ids = {
            id(param) for expert in experts for param in expert.parameters()
        }
        for param, _ in param_requires_grad:
            param.requires_grad_(id(param) in expert_param_ids)

        for expert_id, expert in enumerate(experts):
            hooks.append(
                expert.fc1.register_forward_hook(
                    make_forward_hook(expert_id, count_usage=True)
                )
            )
            hooks.append(expert.fc1.register_full_backward_hook(make_backward_hook(expert_id)))
            hooks.append(
                expert.fc2.register_forward_hook(
                    make_forward_hook(expert_id, count_usage=False)
                )
            )
            hooks.append(expert.fc2.register_full_backward_hook(make_backward_hook(expert_id)))

        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            model.zero_grad(set_to_none=True)
            activations.clear()

            with torch.no_grad():
                feat = model.backbone(x)
            logits = model.moe_head(feat)
            loss = criterion(logits, y)
            loss.backward()

    finally:
        for handle in hooks:
            handle.remove()
        activations.clear()
        for param, requires_grad in param_requires_grad:
            param.requires_grad_(requires_grad)
        for module, was_training in module_training_states:
            module.training = was_training
        model.zero_grad(set_to_none=True)

    return list(fisher_totals), list(expert_usage)

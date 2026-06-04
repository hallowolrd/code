import copy

import torch
import torch.nn as nn

from .fisher import compute_expert_fisher_total_hook


_EMPTY_ROUTER_STATS = {
    "expert_usage": [],
    "avg_router_probs": [],
    "router_count": 0,
}


def local_train(
    global_model,
    loader,
    device,
    local_epochs,
    lr,
    momentum,
    weight_decay,
    compute_fisher_total=False,
    fisher_loader=None,
    logger=None,
):
    model = copy.deepcopy(global_model).to(device)
    model.train()
    if hasattr(model, "reset_router_stats"):
        model.reset_router_stats()

    opt = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    total_loss, n_processed = 0.0, 0
    correct, total = 0, 0
    client_sample_count = len(loader.dataset)

    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_loss += loss.item() * x.size(0)
            n_processed += x.size(0)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())

    avg_loss = total_loss / max(n_processed, 1)
    train_acc = correct / max(total, 1)
    if hasattr(model, "get_router_stats"):
        train_router_stats = model.get_router_stats()
    else:
        train_router_stats = dict(_EMPTY_ROUTER_STATS)

    if compute_fisher_total:
        evidence_loader = fisher_loader if fisher_loader is not None else loader
        fisher_totals, fisher_expert_usage = compute_expert_fisher_total_hook(
            model, evidence_loader, device, logger=logger
        )
    else:
        fisher_totals, fisher_expert_usage = [], []

    state = {k: v.cpu() for k, v in model.state_dict().items()}

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        state,
        client_sample_count,
        avg_loss,
        train_acc,
        fisher_totals,
        fisher_expert_usage,
        train_router_stats,
    )

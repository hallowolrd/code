import copy

import torch
import torch.nn as nn

from .fisher import compute_expert_fisher_total_hook


def local_train(
    global_model,
    loader,
    device,
    local_epochs,
    lr,
    momentum,
    weight_decay,
    compute_fisher_total=False,
):
    model = copy.deepcopy(global_model).to(device)
    model.train()

    opt = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    total_loss, n_processed = 0.0, 0
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

    if compute_fisher_total:
        fisher_totals, expert_usage = compute_expert_fisher_total_hook(
            model, loader, device
        )
    else:
        fisher_totals, expert_usage = None, None

    state = {k: v.cpu() for k, v in model.state_dict().items()}
    avg_loss = total_loss / max(n_processed, 1)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return state, client_sample_count, avg_loss, fisher_totals, expert_usage

import numpy as np

from .aggregators import aggregate_split_model
from .client import local_train


def run_fl_round(
    global_model,
    client_loaders,
    chosen_clients,
    device,
    local_epochs,
    lr,
    momentum,
    weight_decay,
    non_expert_agg_method,
    expert_agg_method,
):
    client_states = []
    client_samples = []
    client_losses = []

    for cid in chosen_clients:
        state, sample_count, loss = local_train(
            global_model=global_model,
            loader=client_loaders[cid],
            device=device,
            local_epochs=local_epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        client_states.append(state)
        client_samples.append(sample_count)
        client_losses.append(loss)

    new_state = aggregate_split_model(
        global_model=global_model,
        client_states=client_states,
        client_samples=client_samples,
        non_expert_agg_method=non_expert_agg_method,
        expert_agg_method=expert_agg_method,
    )
    avg_loss = float(np.mean(client_losses))

    return new_state, avg_loss

import torch

from .param_groups import split_state_keys


def aggregate_keys_uniform(global_state, client_states, keys):
    if not client_states:
        raise ValueError("client_states must not be empty")

    num_clients = len(client_states)
    aggregated = {}

    for key in keys:
        target = global_state[key]
        if torch.is_floating_point(target):
            avg = torch.zeros_like(target, device="cpu", dtype=torch.float32)
            for client_state in client_states:
                avg += client_state[key].detach().cpu().float() / num_clients
            aggregated[key] = avg.to(device=target.device, dtype=target.dtype)
        else:
            aggregated[key] = client_states[0][key].to(
                device=target.device,
                dtype=target.dtype,
            )

    return aggregated


def aggregate_keys_sample_weighted(global_state, client_states, client_samples, keys):
    if not client_samples:
        raise ValueError("client_samples must not be empty")
    if len(client_samples) != len(client_states):
        raise ValueError("client_samples length must match client_states length")

    total_samples = sum(client_samples)
    if total_samples <= 0:
        raise ValueError("sum(client_samples) must be positive")

    weights = [sample / total_samples for sample in client_samples]
    aggregated = {}

    for key in keys:
        target = global_state[key]
        if torch.is_floating_point(target):
            avg = torch.zeros_like(target, device="cpu", dtype=torch.float32)
            for client_state, weight in zip(client_states, weights):
                avg += client_state[key].detach().cpu().float() * weight
            aggregated[key] = avg.to(device=target.device, dtype=target.dtype)
        else:
            aggregated[key] = client_states[0][key].to(
                device=target.device,
                dtype=target.dtype,
            )

    return aggregated


def build_key_aggregator(method):
    if method == "uniform":
        return aggregate_keys_uniform
    if method == "sample_weighted":
        return aggregate_keys_sample_weighted
    raise ValueError(f"Unsupported aggregation method: {method}")


def aggregate_split_model(
    global_model,
    client_states,
    client_samples,
    non_expert_agg_method="uniform",
    expert_agg_method="uniform",
):
    global_state = global_model.state_dict()
    expert_keys, non_expert_keys = split_state_keys(global_state)

    non_expert_agg = build_key_aggregator(non_expert_agg_method)
    expert_agg = build_key_aggregator(expert_agg_method)

    new_state = {}
    if non_expert_agg_method == "sample_weighted":
        new_state.update(
            non_expert_agg(global_state, client_states, client_samples, non_expert_keys)
        )
    else:
        new_state.update(non_expert_agg(global_state, client_states, non_expert_keys))

    if expert_agg_method == "sample_weighted":
        new_state.update(
            expert_agg(global_state, client_states, client_samples, expert_keys)
        )
    else:
        new_state.update(expert_agg(global_state, client_states, expert_keys))

    if set(new_state.keys()) != set(global_state.keys()):
        raise ValueError("Aggregated state keys do not match global_state keys")

    return {key: new_state[key] for key in global_state.keys()}

import math
import statistics

import torch

from .param_groups import get_expert_id_from_key, split_state_keys


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


def _sanitize_expert_fisher_weights(client_fisher_totals, expert_id):
    weights = []
    for client_idx, fisher_totals in enumerate(client_fisher_totals):
        if fisher_totals is None or expert_id >= len(fisher_totals):
            raise ValueError(
                f"Missing fisher_total for client {client_idx}, expert {expert_id}"
            )

        try:
            weight = float(fisher_totals[expert_id])
        except (TypeError, ValueError):
            weight = 0.0
        if not math.isfinite(weight) or weight < 0:
            weight = 0.0
        weights.append(weight)

    return weights


def _normalize_fisher_weights(weights, eps):
    positives = [weight for weight in weights if weight > 0]
    if not positives:
        return None

    scale = statistics.median(positives)
    if not math.isfinite(scale) or scale <= 0:
        scale = sum(positives) / len(positives)
    if not math.isfinite(scale) or scale <= 0:
        return None

    rel_weights = [weight / (scale + eps) for weight in weights]
    rel_sum = sum(rel_weights)
    if not math.isfinite(rel_sum) or rel_sum <= 0:
        return None

    return [weight / rel_sum for weight in rel_weights]


def _summarize_fisher_total_agg_weights(
    weights_by_expert, fallback_experts, tiny=1e-30
):
    num_experts = max(weights_by_expert.keys(), default=-1) + 1
    weight_max = [0.0 for _ in range(num_experts)]
    weight_min_pos = [0.0 for _ in range(num_experts)]
    weight_entropy = [0.0 for _ in range(num_experts)]

    for expert_id, weights in weights_by_expert.items():
        cleaned_weights = []
        for weight in weights:
            try:
                value = float(weight)
            except (TypeError, ValueError):
                value = 0.0
            if not math.isfinite(value) or value < 0:
                value = 0.0
            cleaned_weights.append(value)

        positives = [value for value in cleaned_weights if value > 0]
        if cleaned_weights:
            weight_max[expert_id] = max(cleaned_weights)
        if positives:
            weight_min_pos[expert_id] = min(positives)
            entropy = -sum(value * math.log(value + tiny) for value in positives)
            weight_entropy[expert_id] = entropy if math.isfinite(entropy) else 0.0

    return {
        "weight_max": weight_max,
        "weight_min_pos": weight_min_pos,
        "weight_entropy": weight_entropy,
        "fallback_experts": sorted(int(expert_id) for expert_id in fallback_experts),
    }


def aggregate_keys_fisher_total(
    global_state,
    client_states,
    client_fisher_totals,
    keys,
    eps=1e-30,
    return_stats=False,
):
    if not client_states:
        raise ValueError("client_states must not be empty")
    if client_fisher_totals is None:
        raise ValueError("client_fisher_totals must be provided for fisher_total")
    if len(client_fisher_totals) != len(client_states):
        raise ValueError(
            "client_fisher_totals length must match client_states length"
        )

    num_clients = len(client_states)
    weights_by_expert = {}
    fallback_experts = set()
    aggregated = {}

    for key in keys:
        target = global_state[key]
        if not torch.is_floating_point(target):
            aggregated[key] = client_states[0][key].to(
                device=target.device,
                dtype=target.dtype,
            )
            continue

        expert_id = get_expert_id_from_key(key)
        if expert_id is None:
            raise ValueError(f"fisher_total received non-expert key: {key}")

        if expert_id not in weights_by_expert:
            raw_weights = _sanitize_expert_fisher_weights(
                client_fisher_totals, expert_id
            )
            norm_weights = _normalize_fisher_weights(raw_weights, eps)
            if norm_weights is None:
                norm_weights = [1.0 / num_clients for _ in client_states]
                fallback_experts.add(expert_id)
            weights_by_expert[expert_id] = norm_weights

        avg = torch.zeros_like(target, device="cpu", dtype=torch.float32)
        for client_state, weight in zip(client_states, weights_by_expert[expert_id]):
            avg += client_state[key].detach().cpu().float() * weight
        aggregated[key] = avg.to(device=target.device, dtype=target.dtype)

    if return_stats:
        stats = _summarize_fisher_total_agg_weights(
            weights_by_expert, fallback_experts, tiny=eps
        )
        return aggregated, stats

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
    client_fisher_totals=None,
    return_stats=False,
):
    global_state = global_model.state_dict()
    expert_keys, non_expert_keys = split_state_keys(global_state)

    non_expert_agg = build_key_aggregator(non_expert_agg_method)

    new_state = {}
    agg_stats = {}
    if non_expert_agg_method == "sample_weighted":
        new_state.update(
            non_expert_agg(global_state, client_states, client_samples, non_expert_keys)
        )
    else:
        new_state.update(non_expert_agg(global_state, client_states, non_expert_keys))

    if expert_agg_method == "fisher_total":
        fisher_result = aggregate_keys_fisher_total(
            global_state,
            client_states,
            client_fisher_totals,
            expert_keys,
            return_stats=return_stats,
        )
        if return_stats:
            expert_state, agg_stats = fisher_result
        else:
            expert_state = fisher_result
        new_state.update(expert_state)
    else:
        expert_agg = build_key_aggregator(expert_agg_method)
        if expert_agg_method == "sample_weighted":
            new_state.update(
                expert_agg(global_state, client_states, client_samples, expert_keys)
            )
        else:
            new_state.update(expert_agg(global_state, client_states, expert_keys))

    if set(new_state.keys()) != set(global_state.keys()):
        raise ValueError("Aggregated state keys do not match global_state keys")

    ordered_state = {key: new_state[key] for key in global_state.keys()}
    if return_stats:
        return ordered_state, agg_stats
    return ordered_state

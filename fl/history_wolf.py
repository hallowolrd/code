import math
import statistics

import torch

from .param_groups import get_expert_id_from_key


_QUADRANTS = ("HH", "HL", "LH", "LL")


def _empty_quadrant_dict(value=0.0):
    return {quadrant: value for quadrant in _QUADRANTS}


def _init_history_wolf_state(history_wolf_state, num_clients, num_experts):
    if num_clients is None or num_experts is None:
        raise ValueError("num_clients and num_experts must be provided")

    expected_shape = (int(num_clients), int(num_experts))
    if expected_shape[0] <= 0 or expected_shape[1] <= 0:
        raise ValueError("num_clients and num_experts must be positive")

    if history_wolf_state is None:
        return torch.full(
            expected_shape,
            0.5,
            dtype=torch.float32,
            device="cpu",
        )

    if torch.is_tensor(history_wolf_state):
        state = history_wolf_state.detach().cpu().float().clone()
    else:
        state = torch.as_tensor(history_wolf_state, dtype=torch.float32).cpu().clone()

    if tuple(state.shape) != expected_shape:
        raise ValueError(
            "history_wolf_state shape must be "
            f"{expected_shape}, got {tuple(state.shape)}"
        )

    return state


def _group_expert_keys_by_id(expert_keys):
    keys_by_expert = {}
    for key in expert_keys:
        expert_id = get_expert_id_from_key(key)
        if expert_id is None:
            continue
        keys_by_expert.setdefault(expert_id, []).append(key)
    return keys_by_expert


def _flatten_expert_delta(global_state, client_state, expert_keys, eps):
    deltas = []
    for key in expert_keys:
        target = global_state[key]
        if not torch.is_floating_point(target):
            continue

        delta = client_state[key].detach().cpu().float() - target.detach().cpu().float()
        deltas.append(delta.reshape(-1))

    if not deltas:
        return None, 0.0

    flat_delta = torch.cat(deltas)
    delta_norm = float(torch.linalg.vector_norm(flat_delta).item())
    if not math.isfinite(delta_norm) or delta_norm <= eps:
        return None, delta_norm

    return flat_delta / (delta_norm + eps), delta_norm


def _mean(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _entropy(weights, eps):
    return -sum(weight * math.log(weight + eps) for weight in weights)


def _uniform_weights(count):
    if count <= 0:
        raise ValueError("count must be positive")
    return [1.0 / count for _ in range(count)]


def _weights_from_raw(raw_weights, eps):
    raw_sum = sum(raw_weights)
    if (
        any(not math.isfinite(weight) for weight in raw_weights)
        or not math.isfinite(raw_sum)
        or raw_sum <= eps
    ):
        return None
    return [weight / (raw_sum + eps) for weight in raw_weights]


def _aggregate_expert_keys(global_state, client_states, expert_keys, local_indices, weights):
    expert_state = {}
    for key in expert_keys:
        target = global_state[key]
        if torch.is_floating_point(target):
            avg = torch.zeros_like(target, device="cpu", dtype=torch.float32)
            for local_idx, weight in zip(local_indices, weights):
                avg += client_states[local_idx][key].detach().cpu().float() * weight
            expert_state[key] = avg.to(device=target.device, dtype=target.dtype)
        else:
            expert_state[key] = client_states[0][key].to(
                device=target.device,
                dtype=target.dtype,
            )
    return expert_state


def _read_client_fisher_lambda(client_fisher_totals, local_idx, expert_id, eps):
    if local_idx >= len(client_fisher_totals):
        return None

    fisher_totals = client_fisher_totals[local_idx]
    if fisher_totals is None:
        return None

    try:
        if expert_id >= len(fisher_totals):
            return None
        value = float(fisher_totals[expert_id])
    except (TypeError, ValueError, OverflowError, IndexError):
        return None

    if not math.isfinite(value) or value <= eps:
        return None
    return value


def _fisher_precision_values(lambdas, eps):
    positive_lambdas = [
        value
        for value in lambdas
        if value is not None and math.isfinite(value) and value > eps
    ]
    if not positive_lambdas:
        return [0.0 for _ in lambdas]

    median_positive_lambda = statistics.median(positive_lambdas)
    return [
        value / (value + median_positive_lambda + eps)
        if value is not None and math.isfinite(value) and value > eps
        else 0.0
        for value in lambdas
    ]


def _leave_one_out_q_values(directions, eps):
    if not directions:
        return []

    sum_dir = torch.zeros_like(directions[0])
    for direction in directions:
        sum_dir += direction

    q_values = []
    for direction in directions:
        ref = sum_dir - direction
        ref_norm = float(torch.linalg.vector_norm(ref).item())
        if not math.isfinite(ref_norm) or ref_norm <= eps:
            q_value = 0.5
        else:
            ref = ref / (ref_norm + eps)
            cosine = float(torch.dot(direction, ref).item())
            if not math.isfinite(cosine):
                cosine = 0.0
            cosine = max(-1.0, min(1.0, cosine))
            if cosine >= 0.0:
                q_value = 1.0
            else:
                q_value = 1.0 + cosine
            q_value = max(0.0, min(1.0, q_value))
        q_values.append(q_value)

    return q_values


def _quadrant_label(h_prev, q_value):
    history_good = h_prev >= 0.5
    current_good = q_value >= 0.5
    if history_good and current_good:
        return "HH"
    if history_good:
        return "HL"
    if current_good:
        return "LH"
    return "LL"


def _record_quadrant_stats(
    stats,
    expert_id,
    h_prev_values,
    q_values,
    raw_weights,
    weights,
    lambdas,
):
    counts = {quadrant: 0 for quadrant in _QUADRANTS}
    weight_values = {quadrant: [] for quadrant in _QUADRANTS}
    raw_weight_values = {quadrant: [] for quadrant in _QUADRANTS}
    fisher_values = {quadrant: [] for quadrant in _QUADRANTS}

    for h_prev, q_value, raw_weight, weight, lambda_value in zip(
        h_prev_values,
        q_values,
        raw_weights,
        weights,
        lambdas,
    ):
        quadrant = _quadrant_label(h_prev, q_value)
        counts[quadrant] += 1
        weight_values[quadrant].append(weight)
        raw_weight_values[quadrant].append(raw_weight)
        if lambda_value is not None and math.isfinite(lambda_value):
            fisher_values[quadrant].append(lambda_value)

    stats["quadrant_counts"][expert_id] = counts
    stats["quadrant_weight_mean"][expert_id] = {
        quadrant: _mean(weight_values[quadrant]) for quadrant in _QUADRANTS
    }
    stats["quadrant_raw_weight_mean"][expert_id] = {
        quadrant: _mean(raw_weight_values[quadrant]) for quadrant in _QUADRANTS
    }
    stats["quadrant_fisher_mean"][expert_id] = {
        quadrant: _mean(fisher_values[quadrant]) for quadrant in _QUADRANTS
    }


def aggregate_keys_history_wolf(
    global_state,
    client_states,
    selected_client_ids,
    keys,
    history_wolf_state,
    num_clients,
    num_experts,
    client_fisher_totals=None,
    use_fisher_precision=False,
    eps=1e-30,
    return_stats=False,
):
    if not client_states:
        raise ValueError("client_states must not be empty")
    if selected_client_ids is None:
        raise ValueError("selected_client_ids must be provided for history_wolf")
    if len(selected_client_ids) != len(client_states):
        raise ValueError("selected_client_ids length must match client_states length")
    if use_fisher_precision and client_fisher_totals is None:
        raise ValueError("client_fisher_totals must be provided for fisher_history_wolf")

    history_wolf_state = _init_history_wolf_state(
        history_wolf_state,
        num_clients,
        num_experts,
    )
    num_clients = int(num_clients)
    num_experts = int(num_experts)

    keys_by_expert = _group_expert_keys_by_id(keys)
    expert_state = {}
    fallback_experts = set()

    stats = {
        "q_mean": [0.0 for _ in range(num_experts)],
        "h_prev_mean": [0.0 for _ in range(num_experts)],
        "h_new_mean": [0.0 for _ in range(num_experts)],
        "rho_mean": [0.0 for _ in range(num_experts)],
        "p_mean": [0.0 for _ in range(num_experts)],
        "weight_entropy": [0.0 for _ in range(num_experts)],
        "fallback_experts": [],
        "valid_clients": [0 for _ in range(num_experts)],
        "quadrant_counts": [
            _empty_quadrant_dict(0) for _ in range(num_experts)
        ],
        "quadrant_weight_mean": [
            _empty_quadrant_dict() for _ in range(num_experts)
        ],
        "quadrant_raw_weight_mean": [
            _empty_quadrant_dict() for _ in range(num_experts)
        ],
        "quadrant_fisher_mean": [
            _empty_quadrant_dict() for _ in range(num_experts)
        ],
    }

    for expert_id, expert_keys in keys_by_expert.items():
        if expert_id < 0 or expert_id >= num_experts:
            raise ValueError(
                f"expert id {expert_id} is outside configured num_experts={num_experts}"
            )

        valid = []
        for local_idx, cid in enumerate(selected_client_ids):
            cid = int(cid)
            if cid < 0 or cid >= num_clients:
                raise ValueError(
                    f"client id {cid} is outside configured num_clients={num_clients}"
                )
            direction, delta_norm = _flatten_expert_delta(
                global_state,
                client_states[local_idx],
                expert_keys,
                eps,
            )
            if direction is None or delta_norm <= eps:
                continue

            lambda_value = None
            if use_fisher_precision:
                lambda_value = _read_client_fisher_lambda(
                    client_fisher_totals,
                    local_idx,
                    expert_id,
                    eps,
                )
                if lambda_value is None:
                    continue

            valid.append((local_idx, cid, direction, lambda_value))

        stats["valid_clients"][expert_id] = len(valid)
        directions = [item[2] for item in valid]
        q_values = _leave_one_out_q_values(directions, eps)
        h_prev_values = [
            float(history_wolf_state[cid, expert_id].item())
            for _, cid, _, _ in valid
        ]
        lambdas = [item[3] for item in valid]
        if use_fisher_precision:
            raw_weights = [
                lambda_value * q_value * h_prev
                for lambda_value, q_value, h_prev in zip(
                    lambdas,
                    q_values,
                    h_prev_values,
                )
            ]
        else:
            raw_weights = [
                q_value * h_prev
                for q_value, h_prev in zip(q_values, h_prev_values)
            ]

        if len(valid) < 2:
            fallback_experts.add(expert_id)
            local_indices = [item[0] for item in valid]

            if use_fisher_precision and local_indices:
                weights = _weights_from_raw(lambdas, eps)
                stats["p_mean"][expert_id] = _mean(
                    _fisher_precision_values(lambdas, eps)
                )
            else:
                weights = None

            if weights is None:
                if not local_indices:
                    local_indices = list(range(len(client_states)))
                weights = _uniform_weights(len(local_indices))

            expert_state.update(
                _aggregate_expert_keys(
                    global_state,
                    client_states,
                    expert_keys,
                    local_indices,
                    weights,
                )
            )
            h_values = [
                float(history_wolf_state[item[1], expert_id].item()) for item in valid
            ]
            stats["h_prev_mean"][expert_id] = _mean(h_values)
            stats["h_new_mean"][expert_id] = _mean(h_values)
            stats["weight_entropy"][expert_id] = _entropy(weights, eps)
            if valid:
                _record_quadrant_stats(
                    stats,
                    expert_id,
                    h_prev_values,
                    q_values,
                    raw_weights,
                    weights,
                    lambdas,
                )
            continue

        weights = _weights_from_raw(raw_weights, eps)
        if weights is None:
            fallback_experts.add(expert_id)
            if use_fisher_precision:
                weights = _weights_from_raw(lambdas, eps)
            if weights is None:
                weights = _uniform_weights(len(valid))

        local_indices = [item[0] for item in valid]
        expert_state.update(
            _aggregate_expert_keys(
                global_state,
                client_states,
                expert_keys,
                local_indices,
                weights,
            )
        )
        _record_quadrant_stats(
            stats,
            expert_id,
            h_prev_values,
            q_values,
            raw_weights,
            weights,
            lambdas,
        )

        p_values = (
            _fisher_precision_values(lambdas, eps)
            if use_fisher_precision
            else [0.0 for _ in valid]
        )
        residuals = [
            q_value - h_prev
            for q_value, h_prev in zip(q_values, h_prev_values)
        ]
        abs_residuals = [abs(residual) for residual in residuals]
        scale = statistics.median(abs_residuals) + _mean(abs_residuals) + eps
        rho_values = []
        h_new_values = []

        for (_, cid, _, _), q_value, h_prev, residual, p_value in zip(
            valid,
            q_values,
            h_prev_values,
            residuals,
            p_values,
        ):
            rho = 1.0 / (1.0 + residual * residual / (scale * scale + eps))
            if use_fisher_precision:
                gain_base = rho * p_value
                gain = gain_base / (1.0 + gain_base)
            else:
                gain = rho / (1.0 + rho)
            h_new = h_prev + gain * (q_value - h_prev)
            h_new = max(0.0, min(1.0, h_new))
            history_wolf_state[cid, expert_id] = h_new
            rho_values.append(rho)
            h_new_values.append(h_new)

        stats["q_mean"][expert_id] = _mean(q_values)
        stats["h_prev_mean"][expert_id] = _mean(h_prev_values)
        stats["h_new_mean"][expert_id] = _mean(h_new_values)
        stats["rho_mean"][expert_id] = _mean(rho_values)
        stats["p_mean"][expert_id] = _mean(p_values)
        stats["weight_entropy"][expert_id] = _entropy(weights, eps)

    stats["fallback_experts"] = sorted(int(expert_id) for expert_id in fallback_experts)
    if not return_stats:
        stats = {}

    return expert_state, history_wolf_state, stats

import numpy as np

from .aggregators import aggregate_split_model
from .client import local_train


def _finite_nonnegative(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value) or value < 0:
        return 0.0
    return value


def _int_nonnegative(value):
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(value, 0)


def _format_float_list(values):
    return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"


def _num_experts_from_client_stats(client_fisher_totals, client_expert_usages):
    client_fisher_totals = client_fisher_totals or []
    client_expert_usages = client_expert_usages or []
    fisher_lengths = [
        len(totals) for totals in client_fisher_totals if totals is not None
    ]
    usage_lengths = [
        len(usage) for usage in client_expert_usages if usage is not None
    ]
    return max(fisher_lengths + usage_lengths, default=0)


def _print_fisher_total_summary(client_fisher_totals, client_expert_usages):
    if not client_fisher_totals:
        return

    num_experts = _num_experts_from_client_stats(
        client_fisher_totals, client_expert_usages
    )
    if num_experts <= 0:
        return

    client_expert_usages = client_expert_usages or []

    usage_sum = []
    fisher_sum = []
    fisher_median_pos = []
    fisher_max = []
    zero_fisher_clients = []

    for expert_id in range(num_experts):
        usages = []
        for usage in client_expert_usages:
            if usage is not None and expert_id < len(usage):
                usages.append(_int_nonnegative(usage[expert_id]))

        fishers = []
        for totals in client_fisher_totals:
            if totals is not None and expert_id < len(totals):
                fishers.append(_finite_nonnegative(totals[expert_id]))
            else:
                fishers.append(0.0)

        positives = [value for value in fishers if value > 0]

        usage_sum.append(sum(usages))
        fisher_sum.append(sum(fishers))
        fisher_median_pos.append(float(np.median(positives)) if positives else 0.0)
        fisher_max.append(max(fishers) if fishers else 0.0)
        zero_fisher_clients.append(sum(value <= 0 for value in fishers))

    print(
        f"[FisherTotalHook] usage_sum={usage_sum} "
        f"fisher_sum={_format_float_list(fisher_sum)} "
        f"fisher_median_pos={_format_float_list(fisher_median_pos)} "
        f"fisher_max={_format_float_list(fisher_max)} "
        f"zero_fisher_clients={zero_fisher_clients}"
    )


def _stats_float_list(stats, key):
    if not isinstance(stats, dict):
        return []
    values = stats.get(key) or []
    return [_finite_nonnegative(value) for value in values]


def _stats_int_list(stats, key):
    if not isinstance(stats, dict):
        return []
    values = stats.get(key) or []
    return [_int_nonnegative(value) for value in values]


def _print_fisher_total_agg_summary(stats):
    print(
        f"[FisherTotalAgg] weight_max={_format_float_list(_stats_float_list(stats, 'weight_max'))} "
        f"weight_min_pos={_format_float_list(_stats_float_list(stats, 'weight_min_pos'))} "
        f"weight_entropy={_format_float_list(_stats_float_list(stats, 'weight_entropy'))} "
        f"fallback_experts={_stats_int_list(stats, 'fallback_experts')}"
    )


def _print_history_wolf_summary(stats, include_p_mean=False):
    print(f"[HistoryWoLF] q_mean={_format_float_list(_stats_float_list(stats, 'q_mean'))}")
    if include_p_mean:
        print(f"[HistoryWoLF] p_mean={_format_float_list(_stats_float_list(stats, 'p_mean'))}")
    print(
        f"[HistoryWoLF] h_prev_mean="
        f"{_format_float_list(_stats_float_list(stats, 'h_prev_mean'))}"
    )
    print(
        f"[HistoryWoLF] h_new_mean="
        f"{_format_float_list(_stats_float_list(stats, 'h_new_mean'))}"
    )
    print(
        f"[HistoryWoLF] rho_mean="
        f"{_format_float_list(_stats_float_list(stats, 'rho_mean'))}"
    )
    print(
        f"[HistoryWoLF] weight_entropy="
        f"{_format_float_list(_stats_float_list(stats, 'weight_entropy'))}"
    )
    print(f"[HistoryWoLF] fallback_experts={_stats_int_list(stats, 'fallback_experts')}")
    print(f"[HistoryWoLF] quadrant_counts={stats.get('quadrant_counts', {})}")
    print(
        f"[HistoryWoLF] quadrant_weight_mean="
        f"{stats.get('quadrant_weight_mean', {})}"
    )
    print(
        f"[HistoryWoLF] quadrant_raw_weight_mean="
        f"{stats.get('quadrant_raw_weight_mean', {})}"
    )
    if include_p_mean and "quadrant_fisher_mean" in stats:
        print(f"[HistoryWoLF] quadrant_fisher_mean={stats['quadrant_fisher_mean']}")


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
    history_wolf_state=None,
    num_clients=None,
    num_experts=None,
):
    compute_fisher_total = expert_agg_method in {"fisher_total", "fisher_history_wolf"}
    client_states = []
    client_samples = []
    client_losses = []
    client_fisher_totals = []
    client_expert_usages = []

    for cid in chosen_clients:
        state, sample_count, loss, fisher_totals, expert_usage = local_train(
            global_model=global_model,
            loader=client_loaders[cid],
            device=device,
            local_epochs=local_epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            compute_fisher_total=compute_fisher_total,
        )
        client_states.append(state)
        client_samples.append(sample_count)
        client_losses.append(loss)
        client_fisher_totals.append(fisher_totals)
        client_expert_usages.append(expert_usage)

    if compute_fisher_total:
        _print_fisher_total_summary(client_fisher_totals, client_expert_usages)

    aggregate_result = aggregate_split_model(
        global_model=global_model,
        client_states=client_states,
        client_samples=client_samples,
        non_expert_agg_method=non_expert_agg_method,
        expert_agg_method=expert_agg_method,
        client_fisher_totals=client_fisher_totals if compute_fisher_total else None,
        return_stats=(
            compute_fisher_total
            or expert_agg_method in {"history_wolf", "fisher_history_wolf"}
        ),
        selected_client_ids=chosen_clients,
        history_wolf_state=history_wolf_state,
        num_clients=num_clients,
        num_experts=num_experts,
    )
    new_state, agg_stats, history_wolf_state = aggregate_result
    if expert_agg_method == "fisher_total":
        _print_fisher_total_agg_summary(agg_stats)
    if expert_agg_method in {"history_wolf", "fisher_history_wolf"} and agg_stats:
        _print_history_wolf_summary(
            agg_stats,
            include_p_mean=expert_agg_method == "fisher_history_wolf",
        )

    avg_loss = float(np.mean(client_losses))

    return new_state, avg_loss, history_wolf_state

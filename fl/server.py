import numpy as np

from .aggregators import aggregate_split_model
from .client import local_train


def _emit(logger, message):
    if logger is not None:
        logger.info(message)
    else:
        print(message)


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


def _print_fisher_total_summary(client_fisher_totals, client_expert_usages, logger=None):
    num_experts = _num_experts_from_client_stats(
        client_fisher_totals, client_expert_usages
    )
    client_fisher_totals = client_fisher_totals or []
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

    _emit(logger, f"[FisherTotalHook] usage_sum={usage_sum}")
    _emit(logger, f"[FisherTotalHook] fisher_sum={_format_float_list(fisher_sum)}")
    _emit(
        logger,
        f"[FisherTotalHook] fisher_median_pos="
        f"{_format_float_list(fisher_median_pos)}",
    )
    _emit(logger, f"[FisherTotalHook] fisher_max={_format_float_list(fisher_max)}")
    _emit(logger, f"[FisherTotalHook] zero_fisher_clients={zero_fisher_clients}")


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


def _print_fisher_total_agg_summary(stats, logger=None):
    _emit(
        logger,
        f"[FisherTotalAgg] weight_max="
        f"{_format_float_list(_stats_float_list(stats, 'weight_max'))}",
    )
    _emit(
        logger,
        f"[FisherTotalAgg] weight_min_pos="
        f"{_format_float_list(_stats_float_list(stats, 'weight_min_pos'))}",
    )
    _emit(
        logger,
        f"[FisherTotalAgg] weight_entropy="
        f"{_format_float_list(_stats_float_list(stats, 'weight_entropy'))}",
    )
    _emit(
        logger,
        f"[FisherTotalAgg] fallback_experts="
        f"{_stats_int_list(stats, 'fallback_experts')}",
    )


def _print_history_wolf_summary(stats, include_p_mean=False, logger=None):
    _emit(logger, f"[HistoryWoLF] q_mean={_format_float_list(_stats_float_list(stats, 'q_mean'))}")
    if include_p_mean:
        _emit(logger, f"[HistoryWoLF] p_mean={_format_float_list(_stats_float_list(stats, 'p_mean'))}")
    _emit(
        logger,
        f"[HistoryWoLF] h_prev_mean="
        f"{_format_float_list(_stats_float_list(stats, 'h_prev_mean'))}",
    )
    _emit(
        logger,
        f"[HistoryWoLF] h_new_mean="
        f"{_format_float_list(_stats_float_list(stats, 'h_new_mean'))}",
    )
    _emit(
        logger,
        f"[HistoryWoLF] rho_mean="
        f"{_format_float_list(_stats_float_list(stats, 'rho_mean'))}",
    )
    _emit(
        logger,
        f"[HistoryWoLF] weight_entropy="
        f"{_format_float_list(_stats_float_list(stats, 'weight_entropy'))}",
    )
    _emit(logger, f"[HistoryWoLF] fallback_experts={_stats_int_list(stats, 'fallback_experts')}")
    _emit(logger, f"[HistoryWoLF] quadrant_counts={stats.get('quadrant_counts', {})}")
    _emit(
        logger,
        f"[HistoryWoLF] quadrant_weight_mean="
        f"{stats.get('quadrant_weight_mean', {})}",
    )
    _emit(
        logger,
        f"[HistoryWoLF] quadrant_raw_weight_mean="
        f"{stats.get('quadrant_raw_weight_mean', {})}",
    )
    if include_p_mean and "quadrant_fisher_mean" in stats:
        _emit(logger, f"[HistoryWoLF] quadrant_fisher_mean={stats['quadrant_fisher_mean']}")


def _sum_int_lists(value_lists):
    max_len = max((len(values) for values in value_lists if values), default=0)
    totals = [0 for _ in range(max_len)]
    for values in value_lists:
        if not values:
            continue
        for idx, value in enumerate(values[:max_len]):
            totals[idx] += _int_nonnegative(value)
    return totals


def _weighted_avg_router_probs(client_router_stats, client_samples):
    max_len = max(
        (
            len(stats.get("avg_router_probs") or [])
            for stats in client_router_stats
            if isinstance(stats, dict)
        ),
        default=0,
    )
    if max_len <= 0:
        return []

    weighted = [0.0 for _ in range(max_len)]
    total_weight = 0
    for stats, samples in zip(client_router_stats, client_samples):
        if not isinstance(stats, dict):
            continue
        probs = stats.get("avg_router_probs") or []
        if not probs or _int_nonnegative(stats.get("router_count", 0)) <= 0:
            continue
        weight = _int_nonnegative(samples)
        if weight <= 0:
            continue
        for idx, value in enumerate(probs[:max_len]):
            weighted[idx] += _finite_nonnegative(value) * weight
        total_weight += weight

    if total_weight <= 0:
        return []
    return [value / total_weight for value in weighted]


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _std(values):
    return float(np.std(values)) if values else 0.0


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
    round_id=None,
    logger=None,
    log_client_details=True,
):
    compute_fisher_total = expert_agg_method in {"fisher_total", "fisher_history_wolf"}
    chosen_clients = [int(cid) for cid in chosen_clients]
    client_states = []
    client_samples = []
    client_losses = []
    client_accs = []
    client_fisher_totals = []
    client_expert_usages = []
    client_router_stats = []
    client_records = []

    _emit(logger, f"[RoundClients] round={round_id} chosen_clients={chosen_clients}")

    for cid in chosen_clients:
        (
            state,
            sample_count,
            loss,
            train_acc,
            fisher_totals,
            fisher_expert_usage,
            train_router_stats,
        ) = local_train(
            global_model=global_model,
            loader=client_loaders[cid],
            device=device,
            local_epochs=local_epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            compute_fisher_total=compute_fisher_total,
            logger=logger,
        )
        client_states.append(state)
        client_samples.append(sample_count)
        client_losses.append(loss)
        client_accs.append(train_acc)
        client_fisher_totals.append(fisher_totals)
        client_expert_usages.append(fisher_expert_usage)
        client_router_stats.append(train_router_stats)

        expert_usage = train_router_stats.get("expert_usage", [])
        avg_router_probs = train_router_stats.get("avg_router_probs", [])
        client_record = {
            "round": round_id,
            "cid": cid,
            "samples": sample_count,
            "loss": loss,
            "acc": train_acc,
            "expert_usage": expert_usage,
            "avg_router_probs": avg_router_probs,
            "fisher_totals": fisher_totals,
            "fisher_expert_usage": fisher_expert_usage,
        }
        client_records.append(client_record)

        if log_client_details:
            _emit(
                logger,
                f"[ClientTrain] round={round_id} cid={cid} samples={sample_count} "
                f"loss={loss:.6g} acc={train_acc:.6g} "
                f"expert_usage={expert_usage} "
                f"avg_router_probs={_format_float_list(avg_router_probs)}",
            )

    if compute_fisher_total:
        _print_fisher_total_summary(
            client_fisher_totals, client_expert_usages, logger=logger
        )

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
        _print_fisher_total_agg_summary(agg_stats, logger=logger)
    if expert_agg_method in {"history_wolf", "fisher_history_wolf"} and agg_stats:
        _print_history_wolf_summary(
            agg_stats,
            include_p_mean=expert_agg_method == "fisher_history_wolf",
            logger=logger,
        )

    client_loss_mean = _mean(client_losses)
    client_loss_std = _std(client_losses)
    client_acc_mean = _mean(client_accs)
    client_acc_std = _std(client_accs)
    train_expert_usage_sum = _sum_int_lists(
        [stats.get("expert_usage", []) for stats in client_router_stats]
    )
    train_avg_router_probs = _weighted_avg_router_probs(
        client_router_stats, client_samples
    )
    avg_loss = client_loss_mean

    _emit(
        logger,
        f"[RoundTrain] round={round_id} "
        f"client_loss_mean={client_loss_mean:.6g} "
        f"client_loss_std={client_loss_std:.6g} "
        f"client_acc_mean={client_acc_mean:.6g} "
        f"client_acc_std={client_acc_std:.6g}",
    )
    _emit(
        logger,
        f"[RouterTrain] round={round_id} usage_sum={train_expert_usage_sum} "
        f"avg_router_probs_weighted={_format_float_list(train_avg_router_probs)}",
    )

    round_record = {
        "round": round_id,
        "chosen_clients": chosen_clients,
        "avg_loss": avg_loss,
        "client_loss_mean": client_loss_mean,
        "client_loss_std": client_loss_std,
        "client_acc_mean": client_acc_mean,
        "client_acc_std": client_acc_std,
        "train_expert_usage_sum": train_expert_usage_sum,
        "train_avg_router_probs": train_avg_router_probs,
    }

    return new_state, avg_loss, history_wolf_state, round_record, client_records

import numpy as np

from .aggregators import aggregate_split_model
from .client import local_train


def _finite_nonnegative(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        return 0.0
    return value


def _format_float_list(values):
    return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"


def _print_fisher_total_summary(client_fisher_totals, client_expert_usages):
    if not client_fisher_totals:
        return

    num_experts = len(client_fisher_totals[0])
    usage_sum = []
    fisher_sum = []
    fisher_median_pos = []
    zero_fisher_clients = []

    for expert_id in range(num_experts):
        usages = [
            int(usage[expert_id])
            for usage in client_expert_usages
            if usage is not None and expert_id < len(usage)
        ]
        fishers = [
            _finite_nonnegative(totals[expert_id])
            for totals in client_fisher_totals
            if totals is not None and expert_id < len(totals)
        ]
        positives = [value for value in fishers if value > 0]

        usage_sum.append(sum(usages))
        fisher_sum.append(sum(fishers))
        fisher_median_pos.append(float(np.median(positives)) if positives else 0.0)
        zero_fisher_clients.append(sum(value <= 0 for value in fishers))

    print(
        f"[FisherTotalHook] usage_sum={usage_sum} "
        f"fisher_sum={_format_float_list(fisher_sum)} "
        f"fisher_median_pos={_format_float_list(fisher_median_pos)} "
        f"zero_fisher_clients={zero_fisher_clients}"
    )


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
    compute_fisher_total = expert_agg_method == "fisher_total"
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

    new_state = aggregate_split_model(
        global_model=global_model,
        client_states=client_states,
        client_samples=client_samples,
        non_expert_agg_method=non_expert_agg_method,
        expert_agg_method=expert_agg_method,
        client_fisher_totals=client_fisher_totals if compute_fisher_total else None,
    )
    avg_loss = float(np.mean(client_losses))

    return new_state, avg_loss

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from data import (
    DATASET_CFG,
    get_dataset,
    get_deterministic_train_dataset,
    partition_dirichlet,
)
from fl import run_fl_round, summarize_param_groups
from model import MoEFedModel
from utils import evaluate, load_config, set_seed
from utils.logging_utils import append_jsonl, setup_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def resolve_device(device):
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def apply_config_defaults(cfg):
    cfg.setdefault("run_name", None)
    cfg.setdefault("output_dir", "outputs")
    cfg.setdefault("log_client_details", True)
    cfg.setdefault("save_jsonl", True)
    cfg.setdefault("non_expert_agg_method", "uniform")
    cfg.setdefault("expert_agg_method", "uniform")


def default_run_name(cfg):
    return (
        f"{cfg['dataset']}_clients{cfg['num_clients']}_"
        f"experts{cfg['num_experts']}_"
        f"nonexpert-{cfg['non_expert_agg_method']}_"
        f"expert-{cfg['expert_agg_method']}_seed{cfg['seed']}"
    )


def create_run_dir(cfg):
    base_run_name = cfg.get("run_name") or default_run_name(cfg)
    base_run_name = str(base_run_name)
    output_dir = Path(cfg.get("output_dir", "outputs"))

    actual_run_name = base_run_name
    run_dir = output_dir / actual_run_name
    if run_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_run_name = f"{base_run_name}_{timestamp}"
        run_dir = output_dir / actual_run_name
        counter = 2
        while run_dir.exists():
            actual_run_name = f"{base_run_name}_{timestamp}_{counter}"
            run_dir = output_dir / actual_run_name
            counter += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    cfg["run_name"] = base_run_name
    cfg["actual_run_name"] = actual_run_name
    cfg["run_dir"] = str(run_dir)
    return run_dir


def save_config_used(cfg, run_dir):
    config_path = run_dir / "config_used.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return config_path


def _json_default(value):
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _xlsx_safe_value(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=_json_default)
    return value


def _xlsx_safe_records(records):
    return [
        {key: _xlsx_safe_value(value) for key, value in record.items()}
        for record in records
    ]


def _ordered_frame(records, columns):
    frame = pd.DataFrame(_xlsx_safe_records(records))
    if frame.empty:
        return pd.DataFrame(columns=columns)
    extra_columns = [col for col in frame.columns if col not in columns]
    return frame.reindex(columns=columns + extra_columns)


def write_results_xlsx(path, round_records, client_records, cfg):
    round_columns = [
        "Round",
        "LR",
        "AvgLoss",
        "TestAcc",
        "BestAcc",
        "chosen_clients",
        "round_time_sec",
        "client_loss_mean",
        "client_loss_std",
        "client_acc_mean",
        "client_acc_std",
        "train_expert_usage_sum",
        "train_avg_router_probs",
    ]
    client_columns = [
        "round",
        "cid",
        "samples",
        "loss",
        "acc",
        "expert_usage",
        "avg_router_probs",
        "fisher_totals",
        "fisher_expert_usage",
    ]
    config_records = [
        {"key": key, "value": _xlsx_safe_value(value)}
        for key, value in cfg.items()
    ]

    with pd.ExcelWriter(path) as writer:
        _ordered_frame(round_records, round_columns).to_excel(
            writer, sheet_name="round_summary", index=False
        )
        _ordered_frame(client_records, client_columns).to_excel(
            writer, sheet_name="client_summary", index=False
        )
        pd.DataFrame(config_records, columns=["key", "value"]).to_excel(
            writer, sheet_name="config", index=False
        )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    apply_config_defaults(cfg)

    run_dir = create_run_dir(cfg)
    logger = setup_logger(run_dir)
    config_path = save_config_used(cfg, run_dir)
    train_log_path = run_dir / "train.log"
    results_path = run_dir / "results.xlsx"
    rounds_jsonl_path = run_dir / "rounds.jsonl"
    clients_jsonl_path = run_dir / "clients.jsonl"
    if cfg.get("save_jsonl", True):
        rounds_jsonl_path.touch()
        clients_jsonl_path.touch()

    set_seed(cfg["seed"])

    logger.info("[Export] run dir: %s", run_dir)
    logger.info("Config:\n%s", yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    device = resolve_device(cfg.get("device", "auto"))
    logger.info("Device: %s", device)

    train_ds, test_ds = get_dataset(cfg["dataset"], cfg["data_root"])
    client_indices = partition_dirichlet(
        train_ds, cfg["num_clients"], cfg["beta"], cfg["seed"]
    )
    client_loaders = [
        DataLoader(
            Subset(train_ds, idx),
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=True,
        )
        for idx in client_indices
    ]
    fisher_client_loaders = None
    if cfg["expert_agg_method"] in {
        "fisher_total",
        "fisher_history_wolf",
        "fisher_trace_per_active_sample",
    }:
        fisher_train_ds = get_deterministic_train_dataset(
            cfg["dataset"], cfg["data_root"]
        )
        fisher_client_loaders = [
            DataLoader(
                Subset(fisher_train_ds, idx),
                batch_size=cfg["batch_size"],
                shuffle=False,
                num_workers=cfg["num_workers"],
                pin_memory=True,
            )
            for idx in client_indices
        ]
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["test_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    logger.info("train dataset size: %s", len(train_ds))
    logger.info("test dataset size: %s", len(test_ds))
    logger.info("number of clients: %s", len(client_indices))
    logger.info("each client sample count: %s", [len(idx) for idx in client_indices])
    logger.info("number of client_loaders: %s", len(client_loaders))
    logger.info("test_loader batch size: %s", test_loader.batch_size)

    dataset_cfg = DATASET_CFG[cfg["dataset"]]
    global_model = MoEFedModel(
        in_channels=dataset_cfg["in_channels"],
        num_classes=dataset_cfg["num_classes"],
        img_size=dataset_cfg["img_size"],
        num_experts=cfg["num_experts"],
        topk=cfg["topk"],
    ).to(device)

    n_params = sum(p.numel() for p in global_model.parameters())
    logger.info("[Model] Total params: %s", f"{n_params:,}")
    summarize_param_groups(global_model.state_dict(), logger=logger)

    round_records = []
    client_records_all = []
    best_acc = 0.0
    m = max(1, int(cfg["num_clients"] * cfg["frac"]))
    history_wolf_state = None

    logger.info(
        f"{'Round':>5} | {'LR':>8} | {'AvgLoss':>9} | "
        f"{'TestAcc':>8} | {'BestAcc':>8}"
    )
    logger.info("-" * 52)

    for rnd in range(1, cfg["rounds"] + 1):
        round_start_time = time.time()
        current_lr = cfg["lr"]
        chosen = np.random.choice(cfg["num_clients"], m, replace=False).tolist()

        (
            new_state,
            avg_loss,
            history_wolf_state,
            server_round_record,
            client_records,
        ) = run_fl_round(
            global_model=global_model,
            client_loaders=client_loaders,
            chosen_clients=chosen,
            device=device,
            local_epochs=cfg["local_epochs"],
            lr=current_lr,
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"],
            non_expert_agg_method=cfg["non_expert_agg_method"],
            expert_agg_method=cfg["expert_agg_method"],
            fisher_client_loaders=fisher_client_loaders,
            history_wolf_state=history_wolf_state,
            num_clients=cfg["num_clients"],
            num_experts=cfg["num_experts"],
            round_id=rnd,
            logger=logger,
            log_client_details=cfg.get("log_client_details", True),
        )
        global_model.load_state_dict(new_state)

        acc = evaluate(global_model, test_loader, device)
        best_acc = max(best_acc, acc)
        round_time_sec = time.time() - round_start_time

        round_json_record = {
            "round": rnd,
            "lr": current_lr,
            "avg_loss": avg_loss,
            "test_acc": acc,
            "best_acc": best_acc,
            "chosen_clients": server_round_record.get("chosen_clients", chosen),
            "round_time_sec": round_time_sec,
            "client_loss_mean": server_round_record.get("client_loss_mean", avg_loss),
            "client_loss_std": server_round_record.get("client_loss_std", 0.0),
            "client_acc_mean": server_round_record.get("client_acc_mean", 0.0),
            "client_acc_std": server_round_record.get("client_acc_std", 0.0),
            "train_expert_usage_sum": server_round_record.get(
                "train_expert_usage_sum", []
            ),
            "train_avg_router_probs": server_round_record.get(
                "train_avg_router_probs", []
            ),
        }
        round_xlsx_record = {
            "Round": rnd,
            "LR": current_lr,
            "AvgLoss": avg_loss,
            "TestAcc": acc,
            "BestAcc": best_acc,
            "chosen_clients": round_json_record["chosen_clients"],
            "round_time_sec": round_time_sec,
            "client_loss_mean": round_json_record["client_loss_mean"],
            "client_loss_std": round_json_record["client_loss_std"],
            "client_acc_mean": round_json_record["client_acc_mean"],
            "client_acc_std": round_json_record["client_acc_std"],
            "train_expert_usage_sum": round_json_record["train_expert_usage_sum"],
            "train_avg_router_probs": round_json_record["train_avg_router_probs"],
        }
        round_records.append(round_xlsx_record)
        client_records_all.extend(client_records)

        if cfg.get("save_jsonl", True):
            append_jsonl(rounds_jsonl_path, round_json_record)
            for client_record in client_records:
                append_jsonl(clients_jsonl_path, client_record)

        logger.info(
            f"{rnd:5d} | {current_lr:8.4f} | {avg_loss:9.4f} | "
            f"{acc:8.2f} | {best_acc:8.2f}"
        )

        del new_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_results_xlsx(results_path, round_records, client_records_all, cfg)

    logger.info("Done. Best Acc: %.2f%%", best_acc)
    logger.info("[Export] run dir saved to: %s", run_dir)
    logger.info("[Export] xlsx saved to: %s", results_path)
    logger.info("[Export] log saved to: %s", train_log_path)
    logger.info("[Export] round jsonl saved to: %s", rounds_jsonl_path)
    logger.info("[Export] client jsonl saved to: %s", clients_jsonl_path)
    logger.info("[Export] config saved to: %s", config_path)


if __name__ == "__main__":
    main()

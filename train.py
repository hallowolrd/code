import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from data import DATASET_CFG, get_dataset, partition_dirichlet
from fl import run_fl_round, summarize_param_groups
from model import MoEFedModel
from utils import evaluate, load_config, set_seed


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


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    print("Config:")
    print(cfg)

    device = resolve_device(cfg.get("device", "auto"))
    print(f"Device: {device}")

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
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["test_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    print(f"train dataset size: {len(train_ds)}")
    print(f"test dataset size: {len(test_ds)}")
    print(f"number of clients: {len(client_indices)}")
    print(f"each client sample count: {[len(idx) for idx in client_indices]}")
    print(f"number of client_loaders: {len(client_loaders)}")
    print(f"test_loader batch size: {test_loader.batch_size}")

    dataset_cfg = DATASET_CFG[cfg["dataset"]]
    global_model = MoEFedModel(
        in_channels=dataset_cfg["in_channels"],
        num_classes=dataset_cfg["num_classes"],
        img_size=dataset_cfg["img_size"],
        num_experts=cfg["num_experts"],
        topk=cfg["topk"],
    ).to(device)

    n_params = sum(p.numel() for p in global_model.parameters())
    print(f"[Model] Total params: {n_params:,}")
    summarize_param_groups(global_model.state_dict())

    records = []
    best_acc = 0.0
    m = max(1, int(cfg["num_clients"] * cfg["frac"]))
    history_wolf_state = None

    print(f"{'Round':>5} | {'LR':>8} | {'AvgLoss':>9} | {'TestAcc':>8} | {'BestAcc':>8}")
    print("-" * 52)

    for rnd in range(1, cfg["rounds"] + 1):
        current_lr = cfg["lr"]
        chosen = np.random.choice(cfg["num_clients"], m, replace=False).tolist()

        new_state, avg_loss, history_wolf_state = run_fl_round(
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
            history_wolf_state=history_wolf_state,
            num_clients=cfg["num_clients"],
            num_experts=cfg["num_experts"],
        )
        global_model.load_state_dict(new_state)

        acc = evaluate(global_model, test_loader, device)
        best_acc = max(best_acc, acc)

        records.append(
            {
                "Round": rnd,
                "LR": current_lr,
                "AvgLoss": avg_loss,
                "TestAcc": acc,
                "BestAcc": best_acc,
            }
        )
        print(
            f"{rnd:5d} | {current_lr:8.4f} | {avg_loss:9.4f} | "
            f"{acc:8.2f} | {best_acc:8.2f}"
        )

        del new_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(
        cfg["output_dir"],
        f"MoEFed_results_{cfg['dataset']}_clients{cfg['num_clients']}_"
        f"experts{cfg['num_experts']}_nonexpert-{cfg['non_expert_agg_method']}_"
        f"expert-{cfg['expert_agg_method']}.xlsx",
    )
    pd.DataFrame(records, columns=["Round", "LR", "AvgLoss", "TestAcc", "BestAcc"]).to_excel(
        out_path,
        index=False,
    )

    print(f"Done. Best Acc: {best_acc:.2f}%")
    print(f"[Export] saved to: {out_path}")


if __name__ == "__main__":
    main()

"""
聚合逻辑:
  朴素普通平均 FedAvg (Uniform / Naive FedAvg)
  在 CPU 上安全累加，防止显存爆炸

架构改动 (Fast Top-2 Sparse MoE):
  1. backbone: ResNet (CIFAR 适配版)
  2. MoE 路由: 标准 Top-K 路由 (无乘法噪声，无负载均衡)
  3. MoE 计算: 真·稀疏按专家遍历，完美对齐 FedLPA 的架构
  4. 损失函数: 纯净交叉熵 (CrossEntropy)
  5. 学习率: 纯固定常数 LR (已移除所有余弦退火)
  6. 训练设置: 关闭 label smoothing，去掉 AutoAugment，仅保留基础增强
"""

import os, copy, argparse, zipfile, urllib.request, gc
import numpy as np
import pandas as pd  # 新增: 用于导出 Excel 表格
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder


# ════════════════════════════════════════════════════════
#  CLI 参数
# ════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',       default='cifar10',
                   choices=['cifar10','cifar100','tinyimagenet','femnist'])
    p.add_argument('--beta',          type=float, default=0.1)
    p.add_argument('--data_root',     default='./data')
    p.add_argument('--num_clients',   type=int,   default=10)
    p.add_argument('--num_experts',   type=int,   default=4)
    p.add_argument('--topk',          type=int,   default=2) # 统一使用 Top-2
    p.add_argument('--rounds',        type=int,   default=100)
    p.add_argument('--frac',          type=float, default=1)
    p.add_argument('--local_epochs',  type=int,   default=1)
    p.add_argument('--batch_size',    type=int,   default=64)
    p.add_argument('--lr',            type=float, default=0.01)
    p.add_argument('--lr_min',        type=float, default=1e-4) # 参数保留防止传参报错，但逻辑中不再使用
    p.add_argument('--warmup_rounds', type=int,   default=5)    # 参数保留防止传参报错，但逻辑中不再使用
    p.add_argument('--momentum',      type=float, default=0.9)
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    p.add_argument('--label_smooth',  type=float, default=0.0)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--device',        default='auto',
                   choices=['auto','cpu','cuda','mps'])
    return p.parse_args()


DATASET_CFG = {
    'cifar10':      {'num_classes': 10,  'in_channels': 3, 'img_size': 32},
    'cifar100':     {'num_classes': 100, 'in_channels': 3, 'img_size': 32},
    'tinyimagenet': {'num_classes': 200, 'in_channels': 3, 'img_size': 64},
    'femnist':      {'num_classes': 62,  'in_channels': 1, 'img_size': 28},
}

def _prepare_tinyimagenet(root):
    data_path = os.path.join(root, 'tiny-imagenet-200')
    if not os.path.exists(data_path):
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        zip_path = os.path.join(root, 'tiny-imagenet-200.zip')
        print('[Dataset] Downloading TinyImageNet...')
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(root)
        os.remove(zip_path)
    val_dir = os.path.join(data_path, 'val')
    anno    = os.path.join(val_dir, 'val_annotations.txt')
    img_dir = os.path.join(val_dir, 'images')
    if os.path.exists(anno):
        for line in open(anno):
            parts = line.strip().split('\t')
            cls_dir = os.path.join(val_dir, parts[1])
            os.makedirs(cls_dir, exist_ok=True)
            src = os.path.join(img_dir, parts[0])
            if os.path.exists(src):
                os.rename(src, os.path.join(cls_dir, parts[0]))
        os.remove(anno)
        if os.path.isdir(img_dir) and not os.listdir(img_dir):
            os.rmdir(img_dir)
    return data_path

def get_dataset(name, data_root):
    os.makedirs(data_root, exist_ok=True)
    if name == 'cifar10':
        mean, std = (0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)
        tr = transforms.Compose([
             transforms.RandomCrop(32, 4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return (torchvision.datasets.CIFAR10(data_root, True,  download=True, transform=tr),
                torchvision.datasets.CIFAR10(data_root, False, download=True, transform=te))
    elif name == 'cifar100':
        mean, std = (0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)
        tr = transforms.Compose([
             transforms.RandomCrop(32, 4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return (torchvision.datasets.CIFAR100(data_root, True,  download=True, transform=tr),
                torchvision.datasets.CIFAR100(data_root, False, download=True, transform=te))
    elif name == 'tinyimagenet':
        dp = _prepare_tinyimagenet(data_root)
        mean, std = (0.4802,0.4481,0.3975),(0.2302,0.2265,0.2262)
        tr = transforms.Compose([
             transforms.RandomCrop(64, 8),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return (ImageFolder(os.path.join(dp,'train'), transform=tr),
                ImageFolder(os.path.join(dp,'val'),   transform=te))
    else:
        tr = te = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.1307,),(0.3081,))])
        return (torchvision.datasets.EMNIST(data_root, split='byclass',
                    train=True,  download=True, transform=tr),
                torchvision.datasets.EMNIST(data_root, split='byclass',
                    train=False, download=True, transform=te))


def partition_dirichlet(dataset, num_clients, beta, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets if hasattr(dataset, 'targets') else dataset.labels)
    num_classes = int(labels.max()) + 1
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)

        props = rng.dirichlet(np.full(num_clients, beta))
        counts = rng.multinomial(len(idx_c), props)

        cur = 0
        for i, cnt in enumerate(counts):
            client_indices[i].extend(idx_c[cur:cur + cnt].tolist())
            cur += cnt

    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    print(f'\n[Partition] Dirichlet beta={beta}, {num_clients} clients')
    for i, idx in enumerate(client_indices):
        cnt = np.bincount(labels[idx], minlength=num_classes)
        print(f'  Client {i}: {len(idx):>6d} samples | {np.count_nonzero(cnt)}/{num_classes} classes')
    print()

    return client_indices


# ════════════════════════════════════════════════════════
#  ResNet Backbone
# ════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out

class ResNetBackbone(nn.Module):
    def __init__(self, in_channels, img_size):
        super().__init__()
        stem_stride = 1 if img_size <= 32 else 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64,  64,  stride=1)
        self.layer2 = self._make_layer(64,  128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 512

    @staticmethod
    def _make_layer(in_ch, out_ch, stride):
        return nn.Sequential(
            BasicBlock(in_ch,  out_ch, stride=stride),
            BasicBlock(out_ch, out_ch, stride=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.flatten(1)


# ════════════════════════════════════════════════════════
#  Fast Sparse MoE (Top-K) 模块
# ════════════════════════════════════════════════════════

class ExpertFFN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TopKGating(nn.Module):
    """标准的 Top-K 路由：无负载均衡损失，无乘法噪声"""
    def __init__(self, in_dim, num_experts, topk):
        super().__init__()
        self.topk = topk
        self.gate = nn.Linear(in_dim, num_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        probs = torch.softmax(logits.float(), dim=-1)
        
        topk_vals, topk_idx = probs.topk(self.topk, dim=-1)
        
        weights = torch.zeros_like(probs)
        weights.scatter_(1, topk_idx, topk_vals)
        weights = weights.to(x.dtype)
        
        return weights, topk_idx


class MoELayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_experts, topk):
        super().__init__()
        self.gating  = TopKGating(in_dim, num_experts, topk)
        self.experts = nn.ModuleList([
            ExpertFFN(in_dim, hidden_dim, out_dim) for _ in range(num_experts)
        ])

    def forward(self, x):
        weights, topk_idx = self.gating(x)
        B = x.size(0)
        C = self.experts[0].fc2.out_features
        out = torch.zeros(B, C, device=x.device, dtype=x.dtype)

        for i, expert in enumerate(self.experts):
            expert_mask = (topk_idx == i)
            token_mask  = expert_mask.any(dim=-1)
            if not token_mask.any():
                continue
            expert_out  = expert(x[token_mask])
            sel_weights = weights[token_mask, i]
            out[token_mask] += expert_out * sel_weights.unsqueeze(-1)

        return out


class MoEFedModel(nn.Module):
    def __init__(self, in_channels, num_classes, img_size, num_experts, topk):
        super().__init__()
        self.backbone = ResNetBackbone(in_channels, img_size)
        feat_dim = self.backbone.feat_dim
        # 接入 topk
        self.moe_head = MoELayer(feat_dim, 512, num_classes, num_experts, topk)

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.moe_head(feat)
        return logits


# ════════════════════════════════════════════════════════
#  客户端 & 服务端聚合
# ════════════════════════════════════════════════════════

def local_train(global_model, loader, device, local_epochs, lr,
                momentum, weight_decay, label_smooth):
    model = copy.deepcopy(global_model).to(device)
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr,
                    momentum=momentum, weight_decay=weight_decay)
    
    # 【已删除】此处的 optim.lr_scheduler.CosineAnnealingLR 已经被完全移除
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    total_loss, n_processed = 0.0, 0
    client_sample_count = len(loader.dataset) # 本地真实样本量

    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            
            # 纯净的 logits
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item() * x.size(0)
            n_processed += x.size(0)
        
        # 【已删除】此处的 scheduler.step() 已经被完全移除

    state = {k: v.cpu() for k, v in model.state_dict().items()}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return state, client_sample_count, total_loss / max(n_processed, 1)


def uniform_fedavg(global_model, client_states):
    """
    朴素 FedAvg：所有客户端权重完全相同，不按样本数加权。
    该版本更符合 naive MoE-FedAvg baseline。
    """
    new_state = {}
    num_clients = len(client_states)

    for key, g_param in global_model.state_dict().items():
        avg_param = torch.zeros_like(g_param, dtype=torch.float32, device='cpu')

        for state in client_states:
            avg_param += state[key].float() / num_clients

        new_state[key] = avg_param.to(g_param.device).to(g_param.dtype)

    return new_state


# ════════════════════════════════════════════════════════
#  评估 & 主流程
# ════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / max(total, 1)


def main():
    args = get_args()
    if args.device == 'auto':
        device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps'  if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = DATASET_CFG[args.dataset]

    print(f'\n{"="*66}')
    print(f'  MoE-FedAvg (Naive Uniform) | ResNet + Sparse MoE (Top-{args.topk})')
    print(f'  Dataset={args.dataset} | beta={args.beta} | '
          f'Clients={args.num_clients} | Experts={args.num_experts}')
    print(f'  Device={device} | Rounds={args.rounds} | LR={args.lr} (Constant)')
    print(f'{"="*66}\n')

    train_ds, test_ds = get_dataset(args.dataset, args.data_root)
    client_idx = partition_dirichlet(train_ds, args.num_clients, args.beta, args.seed)
    client_loaders = [
        DataLoader(Subset(train_ds, idx), batch_size=args.batch_size,
                   shuffle=True, num_workers=2, pin_memory=True)
        for idx in client_idx
    ]
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=True)

    global_model = MoEFedModel(
        in_channels=cfg['in_channels'],
        num_classes=cfg['num_classes'],
        img_size=cfg['img_size'],
        num_experts=args.num_experts,
        topk=args.topk,
    ).to(device)

    n_params = sum(p.numel() for p in global_model.parameters())
    print(f'[Model] Total params: {n_params:,}\n')

    m        = max(1, int(args.num_clients * args.frac))
    best_acc = 0.0
    print(f'{"Round":>5} | {"LR":>7} | {"AvgLoss":>8} | {"TestAcc":>8} | {"Best":>8}')
    print('-' * 50)

    # ================= 新增：初始化记录列表 =================
    history_records = []

    for rnd in range(1, args.rounds + 1):
        
        # 【已修改】当前学习率直接固定为 args.lr，不再使用 get_lr 函数进行衰减
        current_lr = args.lr
        
        chosen = np.random.choice(args.num_clients, m, replace=False).tolist()
        
        all_states, all_samples, all_losses = [], [], []

        for cid in chosen:
            state, n_samples, loss = local_train(
                global_model, client_loaders[cid], device,
                args.local_epochs, current_lr, args.momentum,
                args.weight_decay, args.label_smooth,
            )
            all_states.append(state)
            all_samples.append(n_samples)
            all_losses.append(loss)

        # 调用朴素普通平均 FedAvg 聚合函数
        new_state = uniform_fedavg(global_model, all_states)
        global_model.load_state_dict(new_state)

        avg_loss = float(np.mean(all_losses))
        
        # 释放内存
        del all_states, all_samples, all_losses, new_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        acc      = evaluate(global_model, test_loader, device)
        best_acc = max(best_acc, acc)
        print(f'{rnd:>5} | {current_lr:>7.5f} | {avg_loss:>8.4f} | '
              f'{acc:>7.2f}% | {best_acc:>7.2f}%')
        
        # ================= 新增：记录当前轮次数据 =================
        history_records.append({
            'Round': rnd,
            'TestAcc': acc
        })

    print(f'\nDone. Best Acc: {best_acc:.2f}%')

    # ================= 新增：代码跑完后生成 Excel =================
    excel_filename = f'NaiveFedAvg_MoE_results_{args.dataset}_clients{args.num_clients}_experts{args.num_experts}.xlsx'
    df = pd.DataFrame(history_records)
    df.to_excel(excel_filename, index=False)
    print(f'[Export] 训练数据已成功保存至 Excel 文件: {excel_filename}')


if __name__ == '__main__':
    main()
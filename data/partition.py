import numpy as np


def partition_dirichlet(dataset, num_clients, beta, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets if hasattr(dataset, "targets") else dataset.labels)
    num_classes = int(labels.max()) + 1
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)

        props = rng.dirichlet(np.full(num_clients, beta))
        counts = rng.multinomial(len(idx_c), props)

        cur = 0
        for i, cnt in enumerate(counts):
            client_indices[i].extend(idx_c[cur : cur + cnt].tolist())
            cur += cnt

    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    print(f"\n[Partition] Dirichlet beta={beta}, {num_clients} clients")
    for i, idx in enumerate(client_indices):
        cnt = np.bincount(labels[idx], minlength=num_classes)
        print(
            f"  Client {i}: {len(idx):>6d} samples | "
            f"{np.count_nonzero(cnt)}/{num_classes} classes"
        )
    print()

    return client_indices

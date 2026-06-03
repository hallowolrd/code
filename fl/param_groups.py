def is_expert_key(key: str) -> bool:
    return "moe_head.experts." in key


def get_expert_id_from_key(key: str):
    if not is_expert_key(key):
        return None

    parts = key.split(".")
    try:
        experts_idx = parts.index("experts")
        return int(parts[experts_idx + 1])
    except (ValueError, IndexError):
        return None


def split_state_keys(state_dict):
    expert_keys = []
    non_expert_keys = []

    for key in state_dict.keys():
        if is_expert_key(key):
            expert_keys.append(key)
        else:
            non_expert_keys.append(key)

    return expert_keys, non_expert_keys


def summarize_param_groups(state_dict):
    expert_keys, non_expert_keys = split_state_keys(state_dict)
    expert_key_counts = {}

    for key in expert_keys:
        expert_id = get_expert_id_from_key(key)
        if expert_id is None:
            continue
        expert_key_counts[expert_id] = expert_key_counts.get(expert_id, 0) + 1

    print(f"[ParamGroups] expert keys: {len(expert_keys)}")
    print(f"[ParamGroups] non-expert keys: {len(non_expert_keys)}")
    print(f"[ParamGroups] expert ids: {sorted(expert_key_counts)}")
    for expert_id in sorted(expert_key_counts):
        print(f"[ParamGroups] expert {expert_id} keys: {expert_key_counts[expert_id]}")

    return {
        "num_expert_keys": len(expert_keys),
        "num_non_expert_keys": len(non_expert_keys),
        "expert_key_counts": expert_key_counts,
    }

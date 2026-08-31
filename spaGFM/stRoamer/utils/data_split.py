import torch

def get_split(graph, setting):
    if setting == 'low':
        train_split = 0.1
        val_split = 0.1
    elif setting == 'median':
        train_split = 0.5
        val_split = 0.25
    elif setting == 'high':
        train_split = 0.6
        val_split = 0.2
    elif setting == 'very_high':
        train_split = 0.8
        val_split = 0.1
    else:
        raise ValueError("Split setting error!")

    num_nodes = graph.num_nodes
    idx = torch.randperm(num_nodes)

    train_idx = idx[:int(num_nodes * train_split)]
    val_idx = idx[int(num_nodes * train_split):int(num_nodes * (train_split + val_split))]
    test_idx = idx[int(num_nodes * (train_split + val_split)):]

    train_mask = idx2mask(train_idx, num_nodes)
    val_mask = idx2mask(val_idx, num_nodes)
    test_mask = idx2mask(test_idx, num_nodes)

    split = {'train': train_mask, 'val': val_mask, 'test': test_mask}

    return split

def idx2mask(idx, num_instances):
    mask = torch.zeros(num_instances, dtype=torch.bool)
    mask[idx] = 1
    return mask
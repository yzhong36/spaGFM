from torch_cluster import random_walk
import torch

def get_patterns(graph, params, batch_size=8192, nodes=None):
    num_patterns = params["n_total_rw"]
    pattern_size = params["walk_length"]
    p = 1
    q = 1
    bs = batch_size

    graph = graph.to(params["device"])

    row, col = graph.edge_index
    if nodes is None:
        nodes = torch.arange(graph.num_nodes).to(params["device"])
    else:
        nodes = nodes.to(params["device"])
    num_nodes_total = nodes.size(0)
    num_batches = (num_nodes_total + bs - 1) // bs

    patterns = []
    for i in range(num_batches):
        cur_nodes = nodes[i * bs: (i + 1) * bs]
        num_cur_nodes = cur_nodes.size(0)
        cur_nodes = cur_nodes.repeat(num_patterns)

        cur_patterns = random_walk(row, col, start=cur_nodes, walk_length=pattern_size, p=p, q=q,
                                   return_edge_indices=False)

        cur_patterns = cur_patterns.view(num_patterns, num_cur_nodes, pattern_size + 1)

        patterns.append(cur_patterns.detach().cpu())

    patterns = torch.concat(patterns, axis=1)

    return patterns
from scipy import stats
import numpy as np
import torch
from torch_geometric.data import Data
from itertools import combinations
from torch_geometric.utils import to_undirected
from torch_geometric.utils import coalesce

def filter_edge(delaunay_obj, threshold = 0.99):
    edge_lengths = []
    edge_ids = set()

    # First pass to collect unique edge lengths
    for triangle in delaunay_obj.simplices:
        for i in range(3):
            i0 = triangle[i]
            i1 = triangle[(i + 1) % 3]
            if (i1, i0) in edge_ids:
                continue
            length = np.linalg.norm(delaunay_obj.points[i0] - delaunay_obj.points[i1])
            edge_lengths.append(length)
            edge_ids.add((i0, i1))

    # Compute log-normal threshold
    log_mean = np.mean(np.log(edge_lengths))
    log_std = np.std(np.log(edge_lengths))
    threshold = stats.lognorm.ppf(threshold, s=log_std, scale=np.exp(log_mean))

    # Classify edges
    small_edges = set()
    large_edges = set()

    for edge in edge_ids:
        i0, i1 = edge
        length = np.linalg.norm(delaunay_obj.points[i0] - delaunay_obj.points[i1])
        if length < threshold:
            small_edges.add(edge)
        else:
            large_edges.add(edge)

    # Filter triangles
    small_triangles = []
    for triangle in delaunay_obj.simplices:
        all_small = True
        for i in range(3):
            i0 = triangle[i]
            i1 = triangle[(i + 1) % 3]
            if (i0, i1) in large_edges or (i1, i0) in large_edges:
                all_small = False
                break
        if all_small:
            small_triangles.append(triangle)
    triangles = small_triangles

    delaunay_obj.simplices = triangles
    return(delaunay_obj)

def edge_index_from_delaunay(delaunay_simplices, undirected=True):
    edge_index=[j for i in delaunay_simplices for j in combinations(i, 2)]
    edge_index1=[i[0] for i in edge_index]
    edge_index2=[i[1] for i in edge_index]
    edge_tensor=torch.tensor([edge_index1,edge_index2], dtype=torch.long)
    edge_tensor=to_undirected(coalesce(edge_tensor))

    return(edge_tensor)

def pyg_obj(x, edge_index):
    data = Data(x=x, edge_index=edge_index)
    
    return(data)
    
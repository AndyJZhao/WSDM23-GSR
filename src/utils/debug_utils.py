import torch as th
import numpy as np


def topk_uniques(adj, k=5):
    topk_neighbors = th.topk(adj, k).indices.flatten().cpu().numpy().tolist()
    d = {v: sum(topk_neighbors == v) / adj.shape[0] for v in np.unique(topk_neighbors)}
    return {k: round(d[k], 4) for k in sorted(d, key=d.get, reverse=True)}

import torch
from torch import device
import torch.nn.functional as F
import numpy as np
# import tensorflow as tf
import torch.distributions as dist
from torch.distributions import Uniform, Beta


def normalize(s):
    return (s.max() - s) / (s.max() - s.mean())


def permute(x: torch.Tensor) -> torch.Tensor:
    """
    Randomly permute node embeddings or features.

    Args:
        x: The latent embedding or node feature.

    Returns:
        torch.Tensor: Embeddings or features resulting from permutation.
    """
    return x[torch.randperm(x.size(0))]


def get_mixup_idx(x: torch.Tensor) -> torch.Tensor:
    """
    Generate node IDs randomly for mixup; avoid mixup the same node.

    Args:
        x: The latent embedding or node feature.

    Returns:
        torch.Tensor: Random node IDs.
    """
    mixup_idx = torch.randint(x.size(0) - 1, [x.size(0)])
    mixup_self_mask = mixup_idx - torch.arange(x.size(0))
    mixup_self_mask = (mixup_self_mask == 0)
    mixup_idx += torch.ones(x.size(0), dtype=torch.int) * mixup_self_mask
    return mixup_idx


def mixup(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Randomly mixup node embeddings or features with other nodes'.

    Args:
        x: The latent embedding or node feature.
        alpha: The hyperparameter controlling the mixup coefficient.

    Returns:
        torch.Tensor: Embeddings or features resulting from mixup.
    """
    device = x.device
    mixup_idx = get_mixup_idx(x).to(device)
    lambda_ = Uniform(alpha, 1.).sample([1]).to(device)
    x = (1 - lambda_) * x + lambda_ * x[mixup_idx]
    return x


def multiinstance_mixup(x1: torch.Tensor, x2: torch.Tensor,
                        alpha: float, shuffle=False) -> (torch.Tensor, torch.Tensor):
    """
    Randomly mixup node embeddings or features with nodes from other views.

    Args:
        x1: The latent embedding or node feature from one view.
        x2: The latent embedding or node feature from the other view.
        alpha: The mixup coefficient `\lambda` follows `Beta(\alpha, \alpha)`.
        shuffle: Whether to use fixed negative samples.

    Returns:
        (torch.Tensor, torch.Tensor): Spurious positive samples and the mixup coefficient.
    """
    device = x1.device
    lambda_ = Beta(alpha, alpha).sample([1]).to(device)
    if shuffle:
        mixup_idx = get_mixup_idx(x1).to(device)
    else:
        mixup_idx = x1.size(0) - torch.arange(x1.size(0)) - 1
    x_spurious = (1 - lambda_) * x1 + lambda_ * x2[mixup_idx]

    return x_spurious, lambda_


def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device)
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def add_random_gaussian_noise(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device
    noise_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    noise_mask = noise_mask.to(device)
    # generage noise of mean 0 and std 0.1
    noise = torch.randn_like(x) * 0.1
    x = x.clone()
    x[:, noise_mask] = x[:, noise_mask] + noise[:, noise_mask]

    return x

def drop_instance(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device
    drop_mask = torch.empty((x.size(0),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device)
    x = x.clone()
    x[drop_mask, :] = 0

    return x

def add_instance(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device
    n = int(x.size(0) * drop_prob)
    inst = torch.rand(n, x.size(1)).to(device)

    x = x.clone()
    x1 = torch.cat([x, inst], dim=0) 

    return x1

def rand_instance(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device

    drop_mask = torch.empty((x.size(0),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device)

    n = torch.sum(drop_mask)
    inst = torch.rand(n, x.size(1)).to(device)

    x1 = x.clone()
    x1[drop_mask, :] = inst 

    return x1


def replace_feature_np(x: torch.Tensor, p_m: float) -> torch.Tensor:
    device = x.device

    no, dim = x.shape
    m = torch.Tensor(np.random.binomial(1, p_m, x.shape)).to(device)

    x_bar = torch.zeros((no, dim), dtype=torch.float32).to(device)
    for i in range(dim):
        idx = torch.randperm(no)
        x_bar[:, i] = x[idx, i]
    x_tilde = x * (1-m) + x_bar * m
    x_tilde = x_tilde.to(device)

    return x_tilde


def replace_feature_np_ori(x: torch.Tensor, drop_prob: float) -> torch.Tensor:

    # Parameters
    no, dim = x.shape
    # Randomly (and column-wise) shuffle data
    x_bar = np.zeros([no, dim])
    for i in range(dim):
        idx = np.random.permutation(no)
        x_bar[:, i] = x[idx, i]

    m = np.random.berne(1, p_m, x.shape)

    # Corrupt samples
    x_tilde = x * (1-m) + x_bar * m
    # Define new mask matrix
    m_new = 1 * (x != x_tilde)

    return m_new, x_tilde


def shufflerow(tensor, axis):
    # device = tensor.device
    # get permutation indices
    row_perm = torch.rand(tensor.shape[:axis+1]).argsort(axis).cuda()
    for _ in range(tensor.ndim-axis-1):
        row_perm.unsqueeze_(-1)
    # reformat this for the gather operation
    row_perm = row_perm.repeat(
        *[1 for _ in range(axis+1)], *(tensor.shape[axis+1:]))
    return tensor.gather(axis, row_perm)


def replace_feature(x: torch.Tensor, p_r: float) -> torch.Tensor:

    # Randomly (and column-wise) shuffle data

    # fast
    # indexes = torch.randperm(no)
    # x_bar = x[indexes]

    # fast
    x_bar = shufflerow(x, 1)

    # low
    # x_bar=x
    no, dim = x.shape
    for i in range(dim):
        idx = torch.randperm(no).cuda()
        x_bar[:, i] = x[idx, i]

    p = torch.full(x.shape, p_r).cuda()

    m = torch.bernoulli(p)

    # Corrupt samples
    x_tilde = x * (1-m) + x_bar * m

    return x_tilde


def dropout_feature(x: torch.FloatTensor, drop_prob: float) -> torch.FloatTensor:
    return F.dropout(x, p=1. - drop_prob)



def get_feature_weights(x, centrality, sparse=True):
    if sparse:
        x = x.to(torch.bool).to(torch.float32)
    else:
        x = x.abs()
    w = x.t() @ centrality
    w = w.log()

    return normalize(w)


def drop_feature_by_weight(x, weights, drop_prob: float, threshold: float = 0.7):
    weights = weights / weights.mean() * drop_prob
    weights = weights.where(weights < threshold,
                            torch.ones_like(weights) * threshold)  # clip
    drop_mask = torch.bernoulli(weights).to(torch.bool)
    x = x.clone()
    x[:, drop_mask] = 0.
    return x





def drop_edge_by_weight(edge_index, weights, drop_prob: float, threshold: float = 0.7):
    weights = weights / weights.mean() * drop_prob
    weights = weights.where(weights < threshold,
                            torch.ones_like(weights) * threshold)
    drop_mask = torch.bernoulli(1. - weights).to(torch.bool)

    return edge_index[:, drop_mask]


class AdaptivelyAugmentTopologyAttributes(object):
    def __init__(self, edge_weights, feature_weights, pe=0.5, pf=0.5, threshold=0.7):
        self.edge_weights = edge_weights
        self.feature_weights = feature_weights
        self.pe = pe
        self.pf = pf
        self.threshold = threshold

    def __call__(self, x, edge_index):
        edge_index = drop_edge_by_weight(
            edge_index, self.edge_weights, self.pe, self.threshold)
        x = drop_feature_by_weight(
            x, self.feature_weights, self.pf, self.threshold)

        return x, edge_index





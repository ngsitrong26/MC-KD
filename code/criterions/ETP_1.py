import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn

class ETP_1(nn.Module):
    def __init__(self, sinkhorn_alpha=0.3, OT_max_iter=50):
        super(ETP_1, self).__init__()
        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.epsilon = 1e-8

    def forward(self, cost):
        # cost: (batch_size, M, N)
        batch_size, M, N = cost.shape
        device = cost.device
        dtype = cost.dtype

        # Initialize marginals
        a = torch.ones(batch_size, M, device=device, dtype=dtype) / M  # (batch_size, M)
        b = torch.ones(batch_size, N, device=device, dtype=dtype) / N  # (batch_size, N)

        # Sinkhorn iterations
        K = torch.exp(-cost / self.sinkhorn_alpha)  # (batch_size, M, N)
        u = torch.ones(batch_size, M, device=device, dtype=dtype)  # (batch_size, M)
        for _ in range(self.OT_max_iter):
            v = b / (torch.bmm(K.transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1) + self.epsilon)  # (batch_size, N)
            u = a / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + self.epsilon)  # (batch_size, M)

        # Transport plan
        P = u.unsqueeze(-1) * K * v.unsqueeze(-2)  # (batch_size, M, N)

        # Loss
        loss = (P * cost).sum(dim=(-1, -2))  # (batch_size,)
        return loss.mean(), P  # Scalar loss, transport plan
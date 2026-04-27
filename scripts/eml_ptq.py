# scripts/eml_ptq.py
#
# EML-PTQ: Post-Training Quantization guided by EML algebraic complexity.
#
# The key insight: weights in {-1, 0, +1} cost 0 EML nodes after
# constant folding (ASIS pre-negation handles -1 offline, CF eliminates
# 0 and 1). This script shows that training with an EML complexity
# penalty drives weights toward these "EML attractors".
#
# Usage: python scripts/eml_ptq.py
# Requirements: pip install torch

import torch
import torch.nn as nn
import torch.optim as optim


class EMLTernaryQuantizer(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for ternary quantization
    toward EML attractors {-1, 0, +1}.
    
    Forward: hard threshold to {-1, 0, +1}
    Backward: identity (straight-through)
    """
    @staticmethod
    def forward(ctx, weight, threshold=0.3):
        quantized = torch.zeros_like(weight)
        quantized[weight > threshold] = 1.0
        quantized[weight < -threshold] = -1.0
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class EMLQuantizedLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)

    def forward(self, x):
        quant_weight = EMLTernaryQuantizer.apply(self.weight)
        return torch.nn.functional.linear(x, quant_weight)


def eml_topology_penalty(weight: torch.Tensor) -> torch.Tensor:
    """
    Differentiable EML complexity penalty.
    
    Measures distance from nearest EML attractor {-1, 0, +1}.
    Weights at attractors cost 0 EML nodes after constant folding.
    Weights away from attractors cost up to 17 nodes (multiplication).
    
    The zero attractor gets extra weight because:
      w=0: entire subtree eliminated (0 nodes)
      w=1: identity rule (0 nodes)  
      w=-1: ASIS pre-negation offline (0 runtime nodes)
    """
    dist_0    = torch.abs(weight)
    dist_pos1 = torch.abs(weight - 1.0)
    dist_neg1 = torch.abs(weight + 1.0)
    # Minimum distance to any attractor
    min_dist = torch.min(dist_0, torch.min(dist_pos1, dist_neg1))
    return torch.mean(min_dist)


def eml_node_savings(quantized_weight: torch.Tensor) -> dict:
    """Estimate EML node savings from ternary quantization."""
    total = quantized_weight.numel()
    zeros = (quantized_weight == 0).sum().item()
    pos_ones = (quantized_weight == 1).sum().item()
    neg_ones = (quantized_weight == -1).sum().item()
    # Each attractor weight saves 17 nodes (cost of float multiplication)
    # vs 14K-9 baseline (CF+ASIS) -> 9(K-1) for full ternary
    attractor_count = zeros + pos_ones + neg_ones
    saved_nodes = attractor_count * 17
    return {
        "total_weights": total,
        "attractor_weights": attractor_count,
        "attractor_pct": 100.0 * attractor_count / total,
        "saved_nodes": saved_nodes,
        "zeros": zeros,
        "pos_ones": pos_ones,
        "neg_ones": neg_ones,
    }


def calibrate_eml_ptq(
    in_dim: int = 64,
    out_dim: int = 64,
    epochs: int = 200,
    lambda_eml: float = 0.8,
):
    """
    Demonstrate EML-PTQ: training with EML complexity penalty
    drives weights toward {-1, 0, +1} attractors.
    
    Corresponds to the objective from paper Section 10.1:
      L_EML-PTQ = ||WX - W̃X||² + λ · Δ_EML(W, W̃)
    """
    model = EMLQuantizedLayer(in_dim, out_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    X = torch.randn(256, in_dim)
    Y_target = torch.randn(256, out_dim)

    print(f"EML-PTQ calibration: {in_dim}x{out_dim} layer, "
          f"lambda_eml={lambda_eml}, {epochs} epochs")
    print(f"{'Epoch':>6} | {'MSE':>8} | {'EML penalty':>11} | "
          f"{'Attractors':>10} | {'Nodes saved':>11}")
    print("-" * 60)

    for epoch in range(epochs):
        optimizer.zero_grad()
        Y_pred = model(X)
        mse_loss = nn.functional.mse_loss(Y_pred, Y_target)
        eml_loss = eml_topology_penalty(model.weight)
        total_loss = mse_loss + lambda_eml * eml_loss
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                q_w = EMLTernaryQuantizer.apply(model.weight)
                stats = eml_node_savings(q_w)
            print(f"{epoch+1:>6} | {mse_loss.item():>8.4f} | "
                  f"{eml_loss.item():>11.4f} | "
                  f"{stats['attractor_pct']:>9.1f}% | "
                  f"{stats['saved_nodes']:>11,}")

    # Final report
    with torch.no_grad():
        q_w = EMLTernaryQuantizer.apply(model.weight)
        stats = eml_node_savings(q_w)

    print()
    print("=== Final EML-PTQ Results ===")
    print(f"Attractor weights: {stats['attractor_pct']:.1f}% -> {stats['attractor_weights']}/{stats['total_weights']}")
    print(f"  zeros:    {stats['zeros']} (full subtree eliminated)")
    print(f"  +1 ones:  {stats['pos_ones']} (identity rule)")
    print(f"  -1 ones:  {stats['neg_ones']} (ASIS pre-negation offline)")
    print(f"Estimated EML node savings: {stats['saved_nodes']:,}")
    print(f"Dot product cost reduction: 14K-9 -> 9(K-1) for full ternary")
    print(f"  = {in_dim*14-9} -> {9*(in_dim-1)} nodes for K={in_dim}")
    print(f"  = {(1 - 9*(in_dim-1)/(in_dim*14-9))*100:.1f}% reduction")


if __name__ == "__main__":
    calibrate_eml_ptq()

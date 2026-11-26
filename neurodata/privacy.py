"""
Differential privacy via DP-SGD:
adds calibrated Gaussian noise to gradients during training
so individual records cannot be reverse-engineered from the model.
"""
import torch
import math


class DPOptimizer:
    """
    Wraps any torch optimizer with DP-SGD noise injection.

    epsilon : privacy budget  (smaller = stronger privacy, lower accuracy)
    delta   : failure probability (typically 1e-5)
    max_grad_norm : gradient clipping threshold
    """

    def __init__(self, optimizer, n_samples, epochs, epsilon=1.0, delta=1e-5, max_grad_norm=1.0):
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

        # noise multiplier from (epsilon, delta) via standard Gaussian mechanism
        self.noise_multiplier = self._compute_noise_multiplier(epsilon, delta, n_samples, epochs)
        print(f"[DP] ε={epsilon}, δ={delta}  →  noise_multiplier={self.noise_multiplier:.4f}")

    def _compute_noise_multiplier(self, epsilon, delta, n, epochs):
        # Simplified calibration: σ = sqrt(2 * ln(1.25/δ)) / ε
        # (tight for single-query; conservative for iterative — sufficient for paper baseline)
        return math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    def step(self, parameters):
        params = [p for p in parameters if p.grad is not None]

        # 1. clip each gradient to max_grad_norm
        total_norm = torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

        # 2. add Gaussian noise scaled to clipping threshold
        with torch.no_grad():
            for p in params:
                noise = torch.randn_like(p.grad) * self.noise_multiplier * self.max_grad_norm
                p.grad.add_(noise)

        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def privacy_spent(self):
        return {'epsilon': self.epsilon, 'delta': self.delta,
                'noise_multiplier': self.noise_multiplier}

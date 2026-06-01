#
# GaussFluids: Spatio-Temporal Encoding
# Paper Section 3.2, Equation (4): ΔA_t = F(q, t), A_t = A_0 + ΔA_t
#

import torch
import torch.nn as nn
import numpy as np


class SpatioTemporalEncoder(nn.Module):
    """
    Per-particle transform features q + shared MLP F with frequency-domain
    time encoding (NeRF-style positional encoding, [RBA*19]).

    Input: transform_features (N, 64) + time PE (N, 2*L)
    Output: Δp (N, 3) + Δs (N, 3) + Δr (N, 4) = (N, 10)

    Architecture: 4 hidden layers × 256 neurons, ReLU
    Zero-initialized for identity mapping at t=0
    """

    def __init__(self, feature_dim=64, hidden_dim=256, num_layers=4,
                 time_pe_freqs=10, output_dim=10):
        super().__init__()

        self.feature_dim = feature_dim
        self.time_pe_freqs = time_pe_freqs
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Time positional encoding dimension: 2 * L (sin + cos per frequency)
        time_pe_dim = 2 * time_pe_freqs

        # Input: feature vector q || PE(t)
        input_dim = feature_dim + time_pe_dim

        # Build MLP layers
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

        # Output head: Δp (3) + Δs (3) + Δr (4) = 10
        self.output_head = nn.Linear(hidden_dim, output_dim)

        # Initialize for identity mapping (all outputs near zero)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

        # Small initialization for MLP layers
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=0.0, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def _positional_encoding(self, t):
        """
        NeRF-style sinusoidal positional encoding for scalar time.
        PE(t) = [sin(2^0 π t), cos(2^0 π t), ..., sin(2^{L-1} π t), cos(2^{L-1} π t)]

        Args:
            t: (N, 1) tensor of normalized times in [0, 1]
        Returns:
            (N, 2*L) tensor of positional encodings
        """
        N = t.shape[0]
        device = t.device

        # Frequency bands: 2^i * π for i = 0, ..., L-1
        freqs = (2.0 ** torch.arange(self.time_pe_freqs, device=device)) * np.pi
        # (L,)
        t_expanded = t.expand(-1, self.time_pe_freqs)  # (N, L)
        angles = t_expanded * freqs.unsqueeze(0)  # (N, L)

        pe_sin = torch.sin(angles)  # (N, L)
        pe_cos = torch.cos(angles)  # (N, L)

        return torch.cat([pe_sin, pe_cos], dim=-1)  # (N, 2L)

    def forward(self, transform_features, t):
        """
        Forward pass: decode transform features + time → pose changes.

        Args:
            transform_features: (N, feature_dim) — per-particle features q
            t: scalar or (N, 1) tensor of time values in [0, 1]

        Returns:
            delta_p: (N, 3) — position offset
            delta_s: (N, 3) — scale offset (applied to log-scale)
            delta_r: (N, 4) — rotation offset (quaternion, to be combined with base)
        """
        # Ensure t is a tensor of shape (N, 1)
        if isinstance(t, (int, float)):
            t = torch.tensor([[t]], dtype=torch.float32,
                             device=transform_features.device)
        if t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)

        # Expand time to match batch of particles
        if t.shape[0] == 1 and transform_features.shape[0] > 1:
            t = t.expand(transform_features.shape[0], -1)
        elif t.shape[0] != transform_features.shape[0]:
            t = t[:1].expand(transform_features.shape[0], -1)

        # Positional encode time
        time_pe = self._positional_encoding(t)  # (N, 2L)

        # Concatenate: [q || PE(t)]
        x = torch.cat([transform_features, time_pe], dim=-1)  # (N, 64 + 2L)

        # Pass through MLP
        h = self.mlp(x)  # (N, hidden_dim)
        output = self.output_head(h)  # (N, 10)

        # Split output into Δp, Δs, Δr
        delta_p = output[:, 0:3]    # (N, 3)
        delta_s = torch.tanh(output[:, 3:6]) * 10.0  # clamped to [-10,10], prevents exp overflow
        delta_r = output[:, 6:10]   # (N, 4)

        return delta_p, delta_s, delta_r

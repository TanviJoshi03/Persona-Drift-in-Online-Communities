"""
Change-Point Contrastive Persona Encoder (baseline model).

This was an earlier approach that uses a contrastive projection head to learn
drift-sensitive representations. Superseded by MultiViewDriftDetector which
fuses multiple signals for better performance.
"""

from typing import List
import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """MLP projection head for embedding transformation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2,
        activation: str = "relu",
        normalize_output: bool = True,
    ):
        super().__init__()

        self.normalize_output = normalize_output

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError("Unknown activation: {}".format(activation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.mlp(x)
        if self.normalize_output:
            z = nn.functional.normalize(z, p=2, dim=1)
        return z


class ChangePointContrastiveEncoder(nn.Module):
    """
    Change-Point Contrastive Persona Encoder.

    A lightweight neural projection model trained using weak supervision
    to learn drift-sensitive representations. Uses contrastive learning
    to pull stable window-pairs closer and push shifting pairs apart.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 128,
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()

        self.projection_head = ProjectionHead(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            normalize_output=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_head(x)

    def encode_pairs(
        self,
        x_t: torch.Tensor,
        x_t1: torch.Tensor,
    ) -> tuple:
        """Encode a pair of consecutive window embeddings."""
        z_t = self.forward(x_t)
        z_t1 = self.forward(x_t1)
        return z_t, z_t1

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "input_dim": self.projection_head.mlp[0].in_features,
                "output_dim": self.projection_head.mlp[-1].out_features,
            }
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["state_dict"])


def train_encoder(encoder, persona_vectors, weak_labels, cfg):
    """
    Train the contrastive encoder using weak supervision labels.

    This is the training loop for the baseline contrastive approach.
    Not used in the final pipeline (MultiViewDriftDetector is used instead).
    """
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.get("learning_rate", 0.001))

    epochs = cfg.get("epochs", 100)
    encoder.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_pairs = len(persona_vectors) - 1

        for i in range(n_pairs):
            x_t = torch.tensor(persona_vectors[i], dtype=torch.float32).unsqueeze(0).to(device)
            x_t1 = torch.tensor(persona_vectors[i + 1], dtype=torch.float32).unsqueeze(0).to(device)
            label = torch.tensor([weak_labels[i]], dtype=torch.float32).to(device)

            z_t, z_t1 = encoder.encode_pairs(x_t, x_t1)
            sim = F.cosine_similarity(z_t, z_t1)

            stable_loss = (1.0 - label) * (1.0 - sim)
            shift_loss = label * torch.clamp(sim, min=0.0)
            loss = (stable_loss + shift_loss).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    encoder.eval()
    return encoder

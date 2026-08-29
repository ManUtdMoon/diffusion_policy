import logging
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class VIBEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256, alpha=1.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.alpha = alpha

        logger.info(
            "number of parameters: %.2f M", sum(p.numel() for p in self.parameters()) / 1e6
        )

    def forward(self, x):
        features = self.network(x)
        mean = self.mean_head(features)
        base_logvar = self.logvar_head(features)
        final_logvar = base_logvar + 2 * torch.log(
            torch.tensor(self.alpha, device=base_logvar.device)
        )
        return mean, final_logvar


class AEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.latent_head = nn.Linear(hidden_dim, latent_dim)

        logger.info(
            "number of parameters: %.2f M", sum(p.numel() for p in self.parameters()) / 1e6
        )

    def forward(self, x):
        features = self.network(x)
        z = self.latent_head(features)
        z_logvar = torch.zeros_like(z)
        return z, z_logvar


class VIBDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        logger.info(
            "number of parameters: %.2f M", sum(p.numel() for p in self.parameters()) / 1e6
        )

    def forward(self, z):
        return self.network(z)

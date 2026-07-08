import torch
import torch.nn as nn
import torch.nn.functional as F


class AMPDiscriminator(nn.Module):
    def __init__(self, input_dim=90, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected AMP input dim {self.input_dim}, got {x.shape[-1]}"
            )

        return self.net(x).view(-1, 1)

    def predict_prob(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def amp_reward(self, x):
        """
        AMP style reward.

        The discriminator is trained with real motion as label 1 and policy motion
        as label 0, so higher logits mean more expert-like motion. This computes
        -log(1 - D(x)) using softplus(logit), which is numerically more stable
        than applying log to sigmoid probabilities directly.
        """
        logits = self.forward(x)
        reward = F.softplus(logits)
        return torch.clamp(reward, min=0.0, max=5.0)
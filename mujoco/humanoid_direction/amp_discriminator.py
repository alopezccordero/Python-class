import torch
import torch.nn as nn


class AMPDiscriminator(nn.Module):
    def __init__(self, input_dim=90, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)

    def predict_prob(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def amp_reward(self, x):
        """
        AMP style reward.
        Higher when discriminator thinks motion is real.
        """
        prob = self.predict_prob(x)
        reward = -torch.log(torch.clamp(1.0 - prob, min=1e-6))
        return reward
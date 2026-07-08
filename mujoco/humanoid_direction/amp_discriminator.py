import torch
import torch.nn as nn


class AMPDiscriminator(nn.Module):
    """Paper-style AMP discriminator.

    The AMP paper uses a least-squares discriminator that predicts +1 for
    reference motion transitions and -1 for policy transitions. The style reward
    is bounded in [0, 1]:

        r = max(0, 1 - 0.25 * (D(s, s') - 1)^2)

    This is different from BCE/GAIL rewards and usually gives a better-scaled
    motion-prior reward for PPO.
    """

    def __init__(self, input_dim=90, hidden_dim=512):
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

    def predict_score(self, x):
        return self.forward(x)

    def predict_prob(self, x):
        """Compatibility helper for old logging code.

        For paper-style AMP, the discriminator output is a score, not a true
        probability. This maps the score to [0, 1] only for rough diagnostics.
        """
        return torch.sigmoid(self.forward(x))

    def amp_reward(self, x):
        score = self.forward(x)
        score = torch.clamp(score, min=-10.0, max=10.0)
        reward = 1.0 - 0.25 * torch.square(score - 1.0)
        return torch.clamp(reward, min=0.0, max=1.0)
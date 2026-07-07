import os
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback


class AMPDiscriminatorCallback(BaseCallback):
    def __init__(
        self,
        motion_lib,
        discriminator,
        optimizer,
        batch_size=256,
        updates_per_call=4,
        train_freq=2048,
        save_freq=500_000,
        save_path="./models/checkpoints",
        device="cpu",
        verbose=1,
    ):
        super().__init__(verbose)

        self.motion_lib = motion_lib
        self.disc = discriminator
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.updates_per_call = updates_per_call
        self.train_freq = train_freq
        self.save_freq = save_freq
        self.save_path = save_path
        self.device = device
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.last_save_step = 0

        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.train_freq == 0:
            self.train_discriminator()

        if self.num_timesteps - self.last_save_step >= self.save_freq:
            self.save_discriminator()
            self.last_save_step = self.num_timesteps

        return True

    def train_discriminator(self):
        fake_lists = self.training_env.env_method("pop_fake_transitions")

        fake_transitions = []
        for lst in fake_lists:
            fake_transitions.extend(lst)

        if len(fake_transitions) < self.batch_size:
            return

        fake_transitions = np.array(fake_transitions, dtype=np.float32)

        for _ in range(self.updates_per_call):
            fake_idx = np.random.randint(0, len(fake_transitions), size=self.batch_size)
            fake_np = fake_transitions[fake_idx]

            real_np = np.array(
                [self.motion_lib.sample_amp_transition() for _ in range(self.batch_size)],
                dtype=np.float32,
            )

            x_np = np.concatenate([real_np, fake_np], axis=0)
            y_np = np.concatenate([
                np.ones((self.batch_size, 1), dtype=np.float32),
                np.zeros((self.batch_size, 1), dtype=np.float32),
            ], axis=0)

            idx = np.random.permutation(len(x_np))
            x_np = x_np[idx]
            y_np = y_np[idx]

            x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
            y = torch.tensor(y_np, dtype=torch.float32, device=self.device)

            logits = self.disc(x)
            loss = self.loss_fn(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            real_prob = torch.sigmoid(
                self.disc(torch.tensor(real_np, dtype=torch.float32, device=self.device))
            ).mean().item()

            fake_prob = torch.sigmoid(
                self.disc(torch.tensor(fake_np, dtype=torch.float32, device=self.device))
            ).mean().item()

        print(
            f"AMP Disc | loss={loss.item():.4f} "
            f"real_prob={real_prob:.3f} fake_prob={fake_prob:.3f}"
        )

        self.logger.record("amp/disc_loss", loss.item())
        self.logger.record("amp/real_prob", real_prob)
        self.logger.record("amp/fake_prob", fake_prob)

    def save_discriminator(self):
        path = os.path.join(
            self.save_path,
            f"amp_discriminator_{self.num_timesteps}.pt"
        )
        torch.save(self.disc.state_dict(), path)
        print("Saved discriminator checkpoint:", path)
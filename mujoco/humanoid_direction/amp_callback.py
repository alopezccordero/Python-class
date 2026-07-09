import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        amp_mean=None,
        amp_std=None,
        fake_replay_size=50000,
        gradient_penalty_weight=10.0,
        score_reg_weight=1e-4,
        max_grad_norm=1.0,
        real_label=1.0,
        fake_label=-1.0,
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
        self.loss_fn = nn.MSELoss()
        self.last_save_step = 0
        self.amp_mean = amp_mean
        self.amp_std = amp_std
        self.fake_replay = deque(maxlen=fake_replay_size)
        self.gradient_penalty_weight = gradient_penalty_weight
        self.score_reg_weight = score_reg_weight
        self.max_grad_norm = max_grad_norm
        self.real_label = real_label
        self.fake_label = fake_label

        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.train_freq == 0:
            self.train_discriminator()

        if self.num_timesteps - self.last_save_step >= self.save_freq:
            self.save_discriminator()
            self.last_save_step = self.num_timesteps

        return True

    def normalize_amp(self, x_np):
        if self.amp_mean is not None and self.amp_std is not None:
            return (x_np - self.amp_mean) / self.amp_std
        return x_np

    def train_discriminator(self):
        fake_lists = self.training_env.env_method("pop_fake_transitions")

        fake_transitions = []
        for lst in fake_lists:
            fake_transitions.extend(lst)

        if len(fake_transitions) == 0:
            return

        self.fake_replay.extend(fake_transitions)

        if len(self.fake_replay) < self.batch_size:
            return

        self.disc.train()

        last_loss = None
        last_real_np = None
        last_fake_np = None
        for _ in range(self.updates_per_call):
            fake_idx = np.random.randint(0, len(self.fake_replay), size=self.batch_size)
            fake_np = np.array(
                [self.fake_replay[i] for i in fake_idx],
                dtype=np.float32,
            )

            real_np = np.array(
                [self.motion_lib.sample_amp_transition() for _ in range(self.batch_size)],
                dtype=np.float32,
            )

            real_x_np = self.normalize_amp(real_np)
            fake_x_np = self.normalize_amp(fake_np)

            real_x = torch.tensor(real_x_np, dtype=torch.float32, device=self.device)
            fake_x = torch.tensor(fake_x_np, dtype=torch.float32, device=self.device)

            real_scores = self.disc(real_x)
            fake_scores = self.disc(fake_x)

            real_targets = torch.full_like(real_scores, self.real_label)
            fake_targets = torch.full_like(fake_scores, self.fake_label)

            # Use Huber/SmoothL1 on raw scores instead of hard-clamping before loss.
            # Hard clamp can create zero-gradient saturation when fake_score explodes.
            real_loss = F.smooth_l1_loss(real_scores, real_targets, beta=1.0)
            fake_loss = F.smooth_l1_loss(fake_scores, fake_targets, beta=1.0)

            gp_loss = self.gradient_penalty(real_x)

            score_reg = self.score_reg_weight * (
                real_scores.pow(2).mean() + fake_scores.pow(2).mean()
            )

            loss = real_loss + fake_loss + self.gradient_penalty_weight * gp_loss + score_reg

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.disc.parameters(), self.max_grad_norm)
            self.optimizer.step()

            last_loss = loss
            last_real_np = real_np
            last_fake_np = fake_np

        if last_loss is None or last_real_np is None or last_fake_np is None:
            return

        self.disc.eval()
        with torch.no_grad():
            real_eval_np = self.normalize_amp(last_real_np)
            fake_eval_np = self.normalize_amp(last_fake_np)

            real_scores = self.disc(
                torch.tensor(real_eval_np, dtype=torch.float32, device=self.device)
            )
            fake_scores = self.disc(
                torch.tensor(fake_eval_np, dtype=torch.float32, device=self.device)
            )

            real_score = real_scores.mean().item()
            fake_score = fake_scores.mean().item()
            real_score_clamped = torch.clamp(real_scores, -10.0, 10.0).mean().item()
            fake_score_clamped = torch.clamp(fake_scores, -10.0, 10.0).mean().item()

            real_reward = self.disc.amp_reward(
                torch.tensor(real_eval_np, dtype=torch.float32, device=self.device)
            ).mean().item()
            fake_rewards = self.disc.amp_reward(
                torch.tensor(fake_eval_np, dtype=torch.float32, device=self.device)
            )

            fake_reward = fake_rewards.mean().item()
            fake_reward_min = fake_rewards.min().item()
            fake_reward_max = fake_rewards.max().item()
            fake_reward_nonzero_frac = (fake_rewards > 1e-6).float().mean().item()

        print(
            f"AMP Disc | loss={last_loss.item():.4f} "
            f"real_score={real_score:.3f} fake_score={fake_score:.3f} "
            f"real_score_clamped={real_score_clamped:.3f} "
            f"fake_score_clamped={fake_score_clamped:.3f} "
            f"real_reward={real_reward:.3f} fake_reward={fake_reward:.3f} "
            f"fake_reward_min={fake_reward_min:.3g} "
            f"fake_reward_max={fake_reward_max:.3g} "
            f"fake_reward_nonzero_frac={fake_reward_nonzero_frac:.3f}"
        )

        self.logger.record("amp/real_score", real_score)
        self.logger.record("amp/fake_score", fake_score)
        self.logger.record("amp/real_score_clamped", real_score_clamped)
        self.logger.record("amp/fake_score_clamped", fake_score_clamped)
        self.logger.record("amp/real_reward", real_reward)
        self.logger.record("amp/fake_reward", fake_reward)
        self.logger.record("amp/fake_reward_min", fake_reward_min)
        self.logger.record("amp/fake_reward_max", fake_reward_max)
        self.logger.record("amp/fake_reward_nonzero_frac", fake_reward_nonzero_frac)

    def gradient_penalty(self, real):
        real = real.clone().detach().requires_grad_(True)
        scores = self.disc(real)

        gradients = torch.autograd.grad(
            outputs=scores.sum(),
            inputs=real,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return gradients.pow(2).sum(dim=1).mean()

    def save_discriminator(self):
        path = os.path.join(
            self.save_path,
            f"amp_discriminator_{self.num_timesteps}.pt"
        )
        torch.save(self.disc.state_dict(), path)
        print("Saved discriminator checkpoint:", path)

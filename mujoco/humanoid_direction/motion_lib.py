import pickle
import random
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np


class MotionLib:
    def __init__(
        self,
        motion_dir="retargeted_pkl",
        filter_bad_contacts=True,
        max_lowest_foot_min=0.12,
        transition_dt=None,
    ):
        self.motion_dir = Path(motion_dir)
        self.transition_dt = transition_dt
        self.motions = []

        files = sorted(self.motion_dir.rglob("*.pkl"))

        if len(files) == 0:
            raise FileNotFoundError(f"No .pkl files found in {motion_dir}")

        if filter_bad_contacts:
            env = gym.make("Humanoid-v5")
            model = env.unwrapped.model
            data = env.unwrapped.data
            left_id = model.body("left_foot").id
            right_id = model.body("right_foot").id
        else:
            env = None
            model = None
            data = None
            left_id = None
            right_id = None

        loaded = 0
        skipped = 0

        for file in files:
            with open(file, "rb") as f:
                motion = pickle.load(f)

            if "qpos" not in motion or "qvel" not in motion:
                print("Skipping invalid file:", file)
                skipped += 1
                continue

            qpos = motion["qpos"]
            qvel = motion["qvel"]
            fps = float(motion.get("fps", 30.0))
            frame_gap = self.compute_frame_gap(fps)

            if len(qpos) <= frame_gap:
                print("Skipping too-short file:", file)
                skipped += 1
                continue

            if filter_bad_contacts:
                lowest_foot_min = self.compute_lowest_foot_min(
                    qpos,
                    model,
                    data,
                    left_id,
                    right_id,
                )

                if lowest_foot_min >= max_lowest_foot_min:
                    print(
                        f"Skipping bad-contact motion: {file} "
                        f"lowest_foot_min={lowest_foot_min:.3f}"
                    )
                    skipped += 1
                    continue

            self.motions.append({
                "file": str(file),
                "fps": fps,
                "qpos": qpos,
                "qvel": qvel,
                "length": len(qpos),
                "frame_gap": frame_gap,
            })

            loaded += 1

        if env is not None:
            env.close()

        if len(self.motions) == 0:
            raise RuntimeError("No valid motions loaded after contact filtering.")

        gaps = sorted({m["frame_gap"] for m in self.motions})
        print(f"Loaded {loaded} motions")
        print(f"Skipped {skipped} motions")
        print(f"AMP transition frame gaps: {gaps}")

    def compute_frame_gap(self, fps):
        """Match expert transition duration to the environment control dt.

        The AMP paper compares policy and reference state transitions over a
        comparable time interval. If transition_dt is unavailable, fall back to
        adjacent frames.
        """
        if self.transition_dt is None:
            return 1
        return max(1, int(round(float(self.transition_dt) * float(fps))))

    def compute_lowest_foot_min(self, qpos_seq, model, data, left_id, right_id):
        lowest = float("inf")

        for qpos in qpos_seq:
            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            left_z = data.xpos[left_id][2]
            right_z = data.xpos[right_id][2]

            lowest = min(lowest, left_z, right_z)

        return lowest

    def sample_motion_id(self):
        return random.randint(0, len(self.motions) - 1)

    def sample_frame(self, motion_id=None):
        if motion_id is None:
            motion_id = self.sample_motion_id()

        motion = self.motions[motion_id]
        max_start = motion["length"] - motion["frame_gap"] - 1
        frame = random.randint(0, max_start)

        return motion_id, frame

    def get_state(self, motion_id, frame):
        motion = self.motions[motion_id]

        qpos = motion["qpos"][frame].copy()
        qvel = motion["qvel"][frame].copy()

        return qpos, qvel

    def get_transition(self, motion_id=None, frame=None):
        if motion_id is None or frame is None:
            motion_id, frame = self.sample_frame(motion_id)

        motion = self.motions[motion_id]
        frame_gap = motion["frame_gap"]

        qpos0, qvel0 = self.get_state(motion_id, frame)
        qpos1, qvel1 = self.get_state(motion_id, frame + frame_gap)

        return {
            "motion_id": motion_id,
            "frame": frame,
            "frame_gap": frame_gap,
            "qpos0": qpos0,
            "qvel0": qvel0,
            "qpos1": qpos1,
            "qvel1": qvel1,
        }

    def sample_reference_state(self):
        motion_id, frame = self.sample_frame()
        qpos, qvel = self.get_state(motion_id, frame)
        return qpos, qvel

    def get_amp_obs(self, qpos, qvel):
        return np.concatenate([
            qpos[2:],
            qvel,
        ]).astype(np.float32)

    def sample_amp_transition(self):
        transition = self.get_transition()

        amp_obs0 = self.get_amp_obs(
            transition["qpos0"],
            transition["qvel0"],
        )

        amp_obs1 = self.get_amp_obs(
            transition["qpos1"],
            transition["qvel1"],
        )

        return np.concatenate([amp_obs0, amp_obs1]).astype(np.float32)

    def compute_amp_stats(self, num_samples=10000):
        samples = np.array(
            [self.sample_amp_transition() for _ in range(num_samples)],
            dtype=np.float32,
        )

        mean = samples.mean(axis=0).astype(np.float32)
        std = samples.std(axis=0).astype(np.float32) + 1e-6

        return mean, std

import pickle
import random
from pathlib import Path
import numpy as np


class MotionLib:
    def __init__(self, motion_dir="retargeted_pkl"):
        self.motion_dir = Path(motion_dir)
        self.motions = []

        files = sorted(self.motion_dir.rglob("*.pkl"))

        if len(files) == 0:
            raise FileNotFoundError(f"No .pkl files found in {motion_dir}")

        for file in files:
            with open(file, "rb") as f:
                motion = pickle.load(f)

            if "qpos" not in motion or "qvel" not in motion:
                print("Skipping invalid file:", file)
                continue

            self.motions.append({
                "file": str(file),
                "fps": motion["fps"],
                "qpos": motion["qpos"],
                "qvel": motion["qvel"],
                "length": len(motion["qpos"]),
            })

        print(f"Loaded {len(self.motions)} motions")

    def sample_motion_id(self):
        return random.randint(0, len(self.motions) - 1)

    def sample_frame(self, motion_id=None):
        if motion_id is None:
            motion_id = self.sample_motion_id()

        motion = self.motions[motion_id]

        # avoid last frame so frame+1 is valid
        frame = random.randint(0, motion["length"] - 2)

        return motion_id, frame

    def get_state(self, motion_id, frame):
        motion = self.motions[motion_id]

        qpos = motion["qpos"][frame].copy()
        qvel = motion["qvel"][frame].copy()

        return qpos, qvel

    def get_transition(self, motion_id=None, frame=None):
        if motion_id is None or frame is None:
            motion_id, frame = self.sample_frame(motion_id)

        qpos0, qvel0 = self.get_state(motion_id, frame)
        qpos1, qvel1 = self.get_state(motion_id, frame + 1)

        return {
            "motion_id": motion_id,
            "frame": frame,
            "qpos0": qpos0,
            "qvel0": qvel0,
            "qpos1": qpos1,
            "qvel1": qvel1,
        }

    def get_amp_obs(self, qpos, qvel):
        return np.concatenate([
            qpos[2:],  # root height + root quat + joints
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
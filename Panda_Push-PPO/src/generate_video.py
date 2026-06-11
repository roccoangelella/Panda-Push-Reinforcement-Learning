import sys

import gymnasium as gym
import imageio
import numpy as np
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize

from src.config import (
    ENV_NAME,MAX_EPISODE_STEPS,MODEL_PATH,
    VEC_NORMALIZE_PATH,VIDEO_FPS,VIDEO_PATH,
)
from src.reward_wrapper import DenseRewardWrapper


def generate_video():
    if not MODEL_PATH.with_suffix(".zip").exists() and not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained model at {MODEL_PATH}.zip. "
            "Run `python main.py` from Panda_Push-PPO first, or copy best_model.zip into output/."
        )
    if not VEC_NORMALIZE_PATH.exists():
        raise FileNotFoundError(
            f"Missing VecNormalize stats at {VEC_NORMALIZE_PATH}. "
            "This PPO policy was trained with observation normalization, so video generation needs "
            "vec_normalize.pkl from the same training run. Re-run training with the updated code, "
            "or copy output/vec_normalize.pkl from the machine that trained best_model.zip."
        )

    def make_env():
        env_instance=gym.make(
            ENV_NAME,render_mode="rgb_array",reward_type="dense",max_episode_steps=MAX_EPISODE_STEPS
        )
        return DenseRewardWrapper(env_instance)

    env=DummyVecEnv([make_env])
    env=VecNormalize.load(str(VEC_NORMALIZE_PATH),env)
    env.training=False
    env.norm_reward=False

    model=PPO.load(str(MODEL_PATH))

    frames=[]#store frames before encoding

    print("Recording 10 episodes...")
    for episode_idx in range(10):
        obs=env.reset()

        for _ in range(MAX_EPISODE_STEPS):
            frame=env.render()
            if isinstance(frame,np.ndarray):
                frames.append(frame)

            action,_=model.predict(obs,deterministic=True)
            obs,_,dones,_=env.step(action)

            if dones[0]:
                break

        print(f"Recorded episode {episode_idx+1}/10")

    env.close()

    VIDEO_PATH.parent.mkdir(parents=True,exist_ok=True)
    if not frames:
        raise RuntimeError("No frames were recorded; cannot write video.")
    imageio.mimwrite(str(VIDEO_PATH),frames,fps=VIDEO_FPS)
    print(f"Video saved to {VIDEO_PATH}")


def main():
    try:
        generate_video()
    except (FileNotFoundError,RuntimeError) as exc:
        print(f"Error: {exc}",file=sys.stderr)
        sys.exit(1)


if __name__=="__main__":
    main()

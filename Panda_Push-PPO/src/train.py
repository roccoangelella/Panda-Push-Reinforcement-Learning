import csv
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import obs_as_tensor

from src.config import (
    CSV_PATH,ENV_NAME,EVAL_EPISODES,EVAL_FREQ,MAX_EPISODE_STEPS,
    MODEL_PATH,TOTAL_TIMESTEPS,VEC_NORMALIZE_PATH,
)
from src.reward_wrapper import DenseRewardWrapper
import gymnasium as gym


class SaveVecNormalizeCallback(BaseCallback):
    """Save normalization stats when EvalCallback writes a new best model."""

    def _on_step(self):
        vec_normalize_env=self.model.get_vec_normalize_env()
        if vec_normalize_env is None:
            raise RuntimeError("Expected the training environment to use VecNormalize.")

        VEC_NORMALIZE_PATH.parent.mkdir(parents=True,exist_ok=True)
        vec_normalize_env.save(str(VEC_NORMALIZE_PATH))
        return True


def _success_to_float(value):
    if value is None:
        return None
    return float(np.asarray(value,dtype=np.float32).mean())


class TrainingCsvCallback(BaseCallback):
    def __init__(
        self,
        csv_save_path=CSV_PATH,
    ):
        super().__init__()
        self.csv_save_path=Path(csv_save_path)
        self.episode_rewards=[]
        self.episode_durations=[]
        self.episode_successes=[]
        self.rollout_entropies=[]
        self.update_step=0
        self.fieldnames=[
            "update_step",
            "total_episodes",
            "total_env_steps",
            "average_reward",
            "average_episode_duration",
            "average_policy_entropy",
            "success_rate",
        ]

    def _on_training_start(self):
        self.csv_save_path.parent.mkdir(parents=True,exist_ok=True)
        with self.csv_save_path.open("w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(self.fieldnames)

    def _policy_entropy(self):
        obs=self.locals.get("obs_tensor")
        if obs is None:
            obs=self.locals.get("new_obs")
        if obs is None:
            return None

        obs_tensor=obs.to(self.model.device) if torch.is_tensor(obs) else obs_as_tensor(obs,self.model.device)
        distribution=self.model.policy.get_distribution(obs_tensor)
        entropy=distribution.distribution.entropy()
        if entropy.ndim>1:
            entropy=entropy.sum(dim=-1)
        return float(entropy.mean().item())

    def _on_step(self):
        for info in self.locals.get("infos",[]):
            episode=info.get("episode")
            if episode is not None:
                self.episode_rewards.append(float(episode["r"]))
                self.episode_durations.append(float(episode["l"]))
                success=_success_to_float(info.get("is_success"))
                if success is not None:
                    self.episode_successes.append(success)

        entropy=self._policy_entropy()
        if entropy is not None:
            self.rollout_entropies.append(entropy)

        return True

    def _on_rollout_end(self):
        self.update_step+=1
        avg_reward=np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
        avg_duration=np.mean(self.episode_durations[-100:]) if self.episode_durations else 0.0
        avg_entropy=np.mean(self.rollout_entropies) if self.rollout_entropies else 0.0
        success_rate=np.mean(self.episode_successes[-100:]) if self.episode_successes else 0.0

        with self.csv_save_path.open("a",newline="") as f:
            writer=csv.writer(f)
            writer.writerow([
                self.update_step,
                len(self.episode_rewards),
                self.num_timesteps,
                avg_reward,
                avg_duration,
                avg_entropy,
                success_rate,
            ])

        self.rollout_entropies.clear()


def train_ppo(model,env,csv_save_path=CSV_PATH):
    def make_eval_env():
        env_instance=gym.make(
            ENV_NAME,reward_type="dense",max_episode_steps=MAX_EPISODE_STEPS
        )
        return DenseRewardWrapper(env_instance)

    #separate eval env — EvalCallback auto-syncs VecNormalize stats
    eval_env=make_vec_env(
        make_eval_env,n_envs=1,
    )
    eval_env=VecNormalize(eval_env,norm_obs=True,norm_reward=False,training=False)

    eval_callback=EvalCallback(
        eval_env,
        callback_on_new_best=SaveVecNormalizeCallback(),
        best_model_save_path=str(MODEL_PATH.parent),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
    )

    csv_callback=TrainingCsvCallback(csv_save_path)#log one row per rollout

    print("Starting training with Stable Baselines3 (PPO)...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([csv_callback,eval_callback]),
    )

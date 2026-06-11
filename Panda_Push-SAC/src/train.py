import csv
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback,CallbackList,EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from src.config import (
    CSV_LOG_INTERVAL,
    CSV_PATH,
    ENV_NAME,
    EVAL_EPISODES,
    EVAL_FREQ,
    EVAL_LOG_DIR,
    MAX_EPISODE_STEPS,
    MODEL_PATH,
    REWARD_TYPE,
    SEED,
    TOTAL_TIMESTEPS,
)


def _success_to_float(value):
    if value is None:
        return None
    return float(np.asarray(value,dtype=np.float32).mean())


def _reset_env(env):
    if isinstance(env,VecEnv):
        return env.reset()
    obs,_=env.reset()
    return obs


def _step_env(env,action):
    if isinstance(env,VecEnv):
        obs,rewards,dones,infos=env.step(action)
        return obs,float(rewards[0]),bool(dones[0]),infos[0]

    obs,reward,terminated,truncated,info=env.step(action)
    return obs,float(reward),bool(terminated or truncated),info


def evaluate_policy_with_success(model,env,n_eval_episodes=EVAL_EPISODES,deterministic=True):
    episode_rewards=[]
    episode_lengths=[]
    episode_successes=[]

    for _ in range(n_eval_episodes):
        obs=_reset_env(env)
        done=False
        episode_reward=0.0
        episode_length=0
        last_info={}

        while not done:
            action,_=model.predict(obs,deterministic=deterministic)
            obs,reward,done,last_info=_step_env(env,action)
            episode_reward+=reward
            episode_length+=1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        success=_success_to_float(last_info.get("is_success"))
        if success is not None:
            episode_successes.append(success)

    success_rate=float(np.mean(episode_successes)) if episode_successes else 0.0

    return {
        "mean_reward":float(np.mean(episode_rewards)),
        "std_reward":float(np.std(episode_rewards)),
        "mean_episode_length":float(np.mean(episode_lengths)),
        "std_episode_length":float(np.std(episode_lengths)),
        "success_rate":success_rate,
    }


class SaveBestSuccessRateCallback(BaseCallback):
    """Save the policy that gets the best evaluation success rate."""

    def __init__(self,model_path:Path,verbose:int=1):
        super().__init__(verbose=verbose)
        self.model_path=model_path
        self.best_success_rate=-np.inf
        self.best_mean_reward=-np.inf

    def _on_step(self):
        successes=getattr(self.parent,"_is_success_buffer",[])
        if len(successes)==0:
            return True

        success_rate=float(np.mean(successes))
        mean_reward=float(getattr(self.parent,"last_mean_reward",-np.inf))
        success_improved=success_rate>self.best_success_rate
        reward_tiebreak=(
            np.isclose(success_rate,self.best_success_rate)
            and mean_reward>self.best_mean_reward
        )

        if success_improved or reward_tiebreak:
            self.model_path.parent.mkdir(parents=True,exist_ok=True)
            self.model.save(str(self.model_path))
            self.best_success_rate=success_rate
            self.best_mean_reward=mean_reward
            if self.verbose>=1:
                print(
                    "New best success rate: "
                    f"{100*success_rate:.2f}% "
                    f"(mean reward: {mean_reward:.2f})"
                )

        return True


class TrainingCsvCallback(BaseCallback):
    def __init__(
        self,
        csv_save_path=CSV_PATH,
        log_interval=CSV_LOG_INTERVAL,
    ):
        super().__init__()
        self.csv_save_path=Path(csv_save_path)
        self.log_interval=log_interval
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
        actor=self.model.policy.actor
        if getattr(actor,"use_sde",False):
            batch_size=next(iter(obs_tensor.values())).shape[0] if isinstance(obs_tensor,dict) else obs_tensor.shape[0]
            actor.reset_noise(batch_size=batch_size)
        _,log_prob=actor.action_log_prob(obs_tensor)
        return float((-log_prob).mean().item())

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

        if self.num_timesteps%self.log_interval==0:
            self.update_step+=1
            self._write_row()
            self.rollout_entropies.clear()

        return True

    def _write_row(self):
        avg_reward=np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
        avg_duration=np.mean(self.episode_durations[-100:]) if self.episode_durations else 0.0
        avg_entropy=np.mean(self.rollout_entropies) if self.rollout_entropies else 0.0
        success_rate=np.mean(self.episode_successes[-100:]) if self.episode_successes else 0.0

        row={
            "update_step":self.update_step,
            "total_episodes":len(self.episode_rewards),
            "total_env_steps":self.num_timesteps,
            "average_reward":avg_reward,
            "average_episode_duration":avg_duration,
            "average_policy_entropy":avg_entropy,
            "success_rate":success_rate,
        }

        with self.csv_save_path.open("a",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=self.fieldnames)
            writer.writerow(row)


def train_sac(model,env):
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    eval_freq=max(EVAL_FREQ//env.num_envs,1)

    eval_env=make_vec_env(
        ENV_NAME,
        n_envs=1,
        seed=SEED+1,
        env_kwargs={"reward_type":REWARD_TYPE,"max_episode_steps":MAX_EPISODE_STEPS},
    )
    eval_callback=EvalCallback(#select checkpoints by success rate
        eval_env,
        callback_after_eval=SaveBestSuccessRateCallback(MODEL_PATH),
        log_path=str(EVAL_LOG_DIR),
        eval_freq=eval_freq,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
    )
    csv_callback=TrainingCsvCallback(CSV_PATH)

    print("Starting training with Stable Baselines3 (SAC)...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([csv_callback,eval_callback]),
    )

    if not MODEL_PATH.with_suffix(".zip").exists() and not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing best model at {MODEL_PATH}.zip. "
            "No evaluation produced a success-rate metric to save."
        )

    best_model=SAC.load(str(MODEL_PATH),env=eval_env)
    metrics=evaluate_policy_with_success(
        best_model,
        eval_env,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
    )
    eval_env.close()
    print(
        "Mean reward: "
        f"{metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}; "
        f"success rate: {100*metrics['success_rate']:.2f}%"
    )

    return metrics["mean_reward"],metrics["std_reward"]

import csv
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback,CallbackList
from stable_baselines3.common.utils import obs_as_tensor

from src.config import (
    CSV_LOG_INTERVAL,
    CSV_PATH,
    EVAL_EPISODES,
    EVAL_FREQ,
    EVAL_LOG_DIR,
    FIXED_CUBE_MASS,
    MIN_CUBE_MASS,
    MAX_CUBE_MASS,
    MODEL_PATH,
    SEED,
    TOTAL_TIMESTEPS,
)
from src.domain_randomization import make_pandapush_vec_env
from src.evaluate_robustness import evaluate_model_robustness,evaluate_policy_with_success,_success_to_float

class RobustnessEvalCallback(BaseCallback):
    """Save the policy that gets the best average evaluation success rate across masses."""

    def __init__(
        self,
        eval_env,
        model_path:Path,
        eval_freq:int,
        n_eval_episodes:int,
        verbose:int=1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env=eval_env
        self.model_path=model_path
        self.eval_freq=eval_freq
        self.n_eval_episodes=n_eval_episodes
        self.best_success_rate=-np.inf
        self.best_mean_reward=-np.inf
        self.last_average_success_rate=None
        self.last_mean_reward=None

    def get_last_average_success_rate(self):
        return self.last_average_success_rate

    def _on_step(self):
        if self.eval_freq>0 and self.n_calls%self.eval_freq==0:
            masses=[1.0,2.0,3.0,4.0,5.0]
            avg_success_rate,avg_mean_reward=evaluate_model_robustness(
                self.model,self.eval_env,masses,self.n_eval_episodes
            )
            self.last_average_success_rate=avg_success_rate
            self.last_mean_reward=avg_mean_reward
            
            success_improved=avg_success_rate>self.best_success_rate
            reward_tiebreak=(
                np.isclose(avg_success_rate,self.best_success_rate)
                and avg_mean_reward>self.best_mean_reward
            )
            
            if success_improved or reward_tiebreak:
                self.model_path.parent.mkdir(parents=True,exist_ok=True)
                self.model.save(str(self.model_path))
                self.best_success_rate=avg_success_rate
                self.best_mean_reward=avg_mean_reward
                if self.verbose>=1:
                    print(
                        "New best robust success rate: "
                        f"{100*avg_success_rate:.2f}% "
                        f"(mean reward: {avg_mean_reward:.2f})"
                    )
            elif self.verbose>=1:
                print(
                    f"Robust evaluation - Success rate: {100*avg_success_rate:.2f}% "
                    f"(best: {100*self.best_success_rate:.2f}%)"
                )
        return True

class TrainingCsvCallback(BaseCallback):
    def __init__(
        self,
        csv_save_path=CSV_PATH,
        log_interval=CSV_LOG_INTERVAL,
        robust_average_success_rate_getter=None,
    ):
        super().__init__()
        self.csv_save_path=Path(csv_save_path)
        self.log_interval=log_interval
        self.robust_average_success_rate_getter=robust_average_success_rate_getter
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
            "robust_average_success_rate",
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
            "robust_average_success_rate":self._latest_robust_average_success_rate(),
        }

        with self.csv_save_path.open("a",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=self.fieldnames)
            writer.writerow(row)

    def _latest_robust_average_success_rate(self):
        if self.robust_average_success_rate_getter is None:
            return ""
        value=self.robust_average_success_rate_getter()
        return "" if value is None else value

def train_sac(model,env):
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    eval_freq=max(EVAL_FREQ//env.num_envs,1)

    eval_env=make_pandapush_vec_env(
        min_cube_mass=MIN_CUBE_MASS,
        max_cube_mass=MAX_CUBE_MASS,
        n_envs=1,
        seed=SEED+1,
    )
    eval_callback=RobustnessEvalCallback(#select checkpoints across fixed masses
        eval_env=eval_env,
        model_path=MODEL_PATH,
        eval_freq=eval_freq,
        n_eval_episodes=EVAL_EPISODES,
    )
    csv_callback=TrainingCsvCallback(
        CSV_PATH,
        robust_average_success_rate_getter=eval_callback.get_last_average_success_rate,
    )

    print("Starting training with Stable Baselines3 (SAC)...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([eval_callback,csv_callback]),
    )

    if not MODEL_PATH.with_suffix(".zip").exists() and not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing best model at {MODEL_PATH}.zip. "
            "No evaluation produced a success-rate metric to save."
        )

    best_model=SAC.load(str(MODEL_PATH),env=eval_env)
    
    #final evaluation of the best model across all masses
    masses=[1.0,2.0,3.0,4.0,5.0]
    avg_success_rate,avg_mean_reward=evaluate_model_robustness(
        best_model,eval_env,masses,n_episodes=EVAL_EPISODES
    )
    eval_env.close()
    
    print(
        "Final Best Model Robustness - Mean reward: "
        f"{avg_mean_reward:.2f}; "
        f"average success rate: {100*avg_success_rate:.2f}%"
    )

    return avg_mean_reward,0.0#returning 0.0 for std_reward as a placeholder since it's not strictly computed across all masses together here

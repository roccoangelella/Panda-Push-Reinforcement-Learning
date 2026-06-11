import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback,CallbackList
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
import torch

from src.config import (
    ADR_CURRICULUM_CSV_PATH,
    ADR_EVAL_EPISODES,
    ADR_MASS_INCREMENT,
    ADR_MAX_MASS,
    ADR_SUCCESS_THRESHOLD,
    ADR_UPDATE_FREQ,
    CSV_LOG_INTERVAL,
    CSV_PATH,
    EVAL_EPISODES,
    EVAL_FREQ,
    FIXED_CUBE_MASS,
    MODEL_PATH,
    SEED,
    TOTAL_TIMESTEPS,
)
from src.domain_randomization import make_pandapush_vec_env


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


def evaluate_model_robustness(model,env,masses,n_episodes):
    success_rates=[]
    mean_rewards=[]

    for mass in masses:
        env.env_method("set_fixed_cube_mass",mass)
        metrics=evaluate_policy_with_success(
            model,
            env,
            n_eval_episodes=n_episodes,
            deterministic=True,
        )
        success_rates.append(metrics["success_rate"])
        mean_rewards.append(metrics["mean_reward"])

    return float(np.mean(success_rates)),float(np.mean(mean_rewards))


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


class RobustnessEvalCallback(BaseCallback):
    """Save the policy with the best average success rate across fixed masses."""

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
                self.model,
                self.eval_env,
                masses,
                self.n_eval_episodes,
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


class AutomaticDomainRandomizationCallback(BaseCallback):
    """Expand the ADR mass range when the policy succeeds at the boundary."""

    def __init__(
        self,
        training_env,
        min_cube_mass,
        initial_max_cube_mass,
        max_cube_mass=ADR_MAX_MASS,
        update_freq=ADR_UPDATE_FREQ,
        eval_episodes=ADR_EVAL_EPISODES,
        success_threshold=ADR_SUCCESS_THRESHOLD,
        mass_increment=ADR_MASS_INCREMENT,
        seed=0,
        verbose=1,
    ):
        super().__init__(verbose=verbose)
        self.domain_randomization_env=training_env
        self.curriculum_csv_path=Path(ADR_CURRICULUM_CSV_PATH)
        self.min_cube_mass=float(min_cube_mass)
        self.current_max_cube_mass=float(initial_max_cube_mass)
        self.max_cube_mass=float(max_cube_mass)
        self.update_freq=update_freq
        self.eval_episodes=eval_episodes
        self.success_threshold=success_threshold
        self.mass_increment=mass_increment
        self.seed=seed
        self._last_update_timesteps=0

    def _on_training_start(self):
        self.curriculum_csv_path.parent.mkdir(parents=True,exist_ok=True)
        with self.curriculum_csv_path.open("w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow([
                "timesteps",
                "success_rate",
                "current_min_cube_mass",
                "current_max_cube_mass",
                "range_updated",
            ])

        self.domain_randomization_env.env_method(
            "set_current_max_cube_mass",
            self.current_max_cube_mass,
        )

    def _on_step(self):
        if self.num_timesteps-self._last_update_timesteps<self.update_freq:
            return True

        self._last_update_timesteps=self.num_timesteps
        success_rate=self._evaluate_current_boundary()#test the current mass limit
        range_updated=False

        if (
            success_rate>=self.success_threshold
            and self.current_max_cube_mass<self.max_cube_mass
        ):
            self.current_max_cube_mass=min(
                self.current_max_cube_mass+self.mass_increment,
                self.max_cube_mass,
            )
            self.domain_randomization_env.env_method(
                "set_current_max_cube_mass",
                self.current_max_cube_mass,
            )
            range_updated=True

        self._write_curriculum_row(success_rate,range_updated)
        if self.verbose>=1:
            print(
                "ADR update: "
                f"success={success_rate:.2f}, "
                f"mass_range=[{self.min_cube_mass:.2f}, "
                f"{self.current_max_cube_mass:.2f}]"
            )
        return True

    def _evaluate_current_boundary(self):
        eval_env=make_pandapush_vec_env(
            cube_mass_mode="fixed",
            fixed_cube_mass=self.current_max_cube_mass,
            n_envs=1,
            seed=self.seed+991,
        )
        try:
            metrics=evaluate_policy_with_success(
                self.model,
                eval_env,
                n_eval_episodes=self.eval_episodes,
                deterministic=True,
            )
        finally:
            eval_env.close()
        return metrics["success_rate"]

    def _write_curriculum_row(self,success_rate,range_updated):
        self.curriculum_csv_path.parent.mkdir(parents=True,exist_ok=True)
        with self.curriculum_csv_path.open("a",newline="") as f:
            writer=csv.writer(f)
            writer.writerow([
                self.num_timesteps,
                success_rate,
                self.min_cube_mass,
                self.current_max_cube_mass,
                int(range_updated),
            ])


def get_vec_env_mass_range(env):
    ranges=env.env_method("get_cube_mass_range")
    mins=[float(mass_range[0]) for mass_range in ranges]
    maxes=[float(mass_range[1]) for mass_range in ranges]
    return float(np.mean(mins)),float(np.mean(maxes))


class TrainingCsvCallback(BaseCallback):
    def __init__(
        self,
        csv_save_path=CSV_PATH,
        log_interval=CSV_LOG_INTERVAL,
        mass_range_getter=None,
        robust_average_success_rate_getter=None,
    ):
        super().__init__()
        self.csv_save_path=Path(csv_save_path)
        self.log_interval=log_interval
        self.mass_range_getter=mass_range_getter
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
            "current_min_cube_mass",
            "current_max_cube_mass",
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

        if self.mass_range_getter is not None:
            current_min,current_max=self.mass_range_getter()
            row["current_min_cube_mass"]=current_min
            row["current_max_cube_mass"]=current_max
        else:
            row["current_min_cube_mass"]=0.0
            row["current_max_cube_mass"]=0.0

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
        cube_mass_mode="fixed",
        fixed_cube_mass=FIXED_CUBE_MASS,
        n_envs=1,
        seed=SEED+1,
    )
    eval_callback=RobustnessEvalCallback(#select checkpoints across fixed masses
        eval_env=eval_env,
        model_path=MODEL_PATH,
        eval_freq=eval_freq,
        n_eval_episodes=EVAL_EPISODES,
    )
    
    mass_range_getter=lambda:get_vec_env_mass_range(env)
    
    csv_callback=TrainingCsvCallback(
        CSV_PATH,
        mass_range_getter=mass_range_getter,
        robust_average_success_rate_getter=eval_callback.get_last_average_success_rate,
    )
    
    adr_callback=AutomaticDomainRandomizationCallback(
        training_env=env,
        min_cube_mass=env.env_method("get_cube_mass_range")[0][0],#fetch min from env wrapper
        initial_max_cube_mass=env.env_method("get_cube_mass_range")[0][1],#fetch current max from env wrapper
        max_cube_mass=ADR_MAX_MASS,
        seed=SEED,
    )

    print("Starting training with Stable Baselines3 (SAC)...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([adr_callback,eval_callback,csv_callback]),
    )

    if not MODEL_PATH.with_suffix(".zip").exists() and not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing best model at {MODEL_PATH}.zip. "
            "No evaluation produced a success-rate metric to save."
        )

    best_model=SAC.load(str(MODEL_PATH),env=eval_env)
    masses=[1.0,2.0,3.0,4.0,5.0]
    avg_success_rate,avg_mean_reward=evaluate_model_robustness(
        best_model,
        eval_env,
        masses,
        n_episodes=EVAL_EPISODES,
    )
    eval_env.close()
    print(
        "Final Best Model Robustness - Mean reward: "
        f"{avg_mean_reward:.2f}; "
        f"average success rate: {100*avg_success_rate:.2f}%"
    )

    return avg_mean_reward,0.0

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from src.config import EVAL_EPISODES

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
    """
    Evaluates a given model across a list of masses.
    Returns the average success rate and average mean reward across all evaluated masses.
    """
    success_rates=[]
    mean_rewards=[]
    
    for mass in masses:#average performance across masses
        env.env_method("set_fixed_cube_mass",mass)
        metrics=evaluate_policy_with_success(model,env,n_eval_episodes=n_episodes,deterministic=True)
        success_rates.append(metrics["success_rate"])
        mean_rewards.append(metrics["mean_reward"])
        
    avg_success_rate=float(np.mean(success_rates))
    avg_mean_reward=float(np.mean(mean_rewards))
    return avg_success_rate,avg_mean_reward

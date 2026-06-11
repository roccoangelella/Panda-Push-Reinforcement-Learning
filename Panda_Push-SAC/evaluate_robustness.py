import argparse
import csv
from pathlib import Path
import numpy as np

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from src.config import ENV_NAME,MODEL_PATH,OUTPUT_DIR,REWARD_TYPE,MAX_EPISODE_STEPS,SEED
from src.train import evaluate_policy_with_success

class CubeMassRandomizationWrapper(gym.Wrapper):
    """Temporary wrapper to adjust the cube mass for robustness evaluation on the baseline."""
    def __init__(self,env,fixed_cube_mass=1.0,object_body_name="object"):
        super().__init__(env)
        self.fixed_cube_mass=float(fixed_cube_mass)
        self.object_body_name=object_body_name

    def reset(self,*,seed=None,options=None):
        obs,info=self.env.reset(seed=seed,options=options)
        self.apply_cube_mass(self.fixed_cube_mass)
        return obs,info

    def apply_cube_mass(self,cube_mass):
        sim=self.unwrapped.sim
        if hasattr(sim,"physics_client") and hasattr(sim,"_bodies_idx"):
            body_id=sim._bodies_idx[self.object_body_name]
            sim.physics_client.changeDynamics(body_id,-1,mass=cube_mass)
        elif hasattr(sim,"model") and hasattr(sim.model,"body_mass"):
            body_id=sim.model.body(self.object_body_name).id
            sim.model.body_mass[body_id]=cube_mass
            if hasattr(sim,"forward"):
                sim.forward()
        else:
            raise RuntimeError("Unsupported PandaPush simulator backend.")

def make_eval_env(mass):
    env=gym.make(ENV_NAME,reward_type=REWARD_TYPE,max_episode_steps=MAX_EPISODE_STEPS)
    return CubeMassRandomizationWrapper(env,fixed_cube_mass=mass)

def make_eval_vec_env(mass,seed=SEED):
    return make_vec_env(lambda:make_eval_env(mass),n_envs=1,seed=seed)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--episodes",type=int,default=50,help="Number of episodes per mass.")
    args=parser.parse_args()

    masses=[1.0,2.0,3.0,4.0,5.0]
    
    if not MODEL_PATH.with_suffix(".zip").exists() and not MODEL_PATH.exists():
        print(f"Error: Could not find model at {MODEL_PATH}")
        return

    results=[]
    model=None

    for mass in masses:#test each target mass
        print(f"Evaluating Baseline SAC on mass: {mass} kg...")
        env=make_eval_vec_env(mass)
        
        if model is None:
            model=SAC.load(str(MODEL_PATH),env=env)
        else:
            model.set_env(env)
            
        metrics=evaluate_policy_with_success(model,env,n_eval_episodes=args.episodes,deterministic=True)
        env.close()
        
        print(f"  Success Rate: {metrics['success_rate']*100:.1f}%")
        print(f"  Mean Reward: {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}\n")
        
        metrics["mass"]=mass
        results.append(metrics)
        
    csv_path=OUTPUT_DIR/"robustness_eval.csv"
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    with csv_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["mass","success_rate","mean_reward","std_reward","mean_episode_length","std_episode_length"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Results saved to {csv_path}")

if __name__=="__main__":
    main()

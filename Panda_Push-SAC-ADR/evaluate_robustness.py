import argparse
import csv

from stable_baselines3 import SAC

from src.config import MODEL_PATH,OUTPUT_DIR,SEED
from src.domain_randomization import make_pandapush_vec_env
from src.train import evaluate_policy_with_success

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
        print(f"Evaluating ADR Model on mass: {mass} kg...")
        env=make_pandapush_vec_env(cube_mass_mode="fixed",fixed_cube_mass=mass,n_envs=1,seed=SEED)
        
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

import argparse
import csv
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import panda_gym#noqa: F401 - importing registers PandaPush-v3 with gymnasium.
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnv


matplotlib.use("Agg")
import matplotlib.pyplot as plt


EVAL_DIR=Path(__file__).resolve().parent
REPO_ROOT=EVAL_DIR.parent
OUTPUT_DIR=EVAL_DIR/"output"


@dataclass(frozen=True)
class Experiment:
    name:str
    directory:Path
    kind:str


EXPERIMENTS=[
    Experiment("SAC",REPO_ROOT/"Panda_Push-SAC","baseline"),
    Experiment("SAC + UDR",REPO_ROOT/"Panda_Push-SAC-UDR","udr"),
    Experiment("SAC + ADR",REPO_ROOT/"Panda_Push-SAC-ADR","adr"),
]


def _success_to_float(value):
    if value is None:
        return None
    return float(np.asarray(value,dtype=np.float32).mean())


def _reset_env(env,seed=None):
    if isinstance(env,VecEnv):
        if seed is not None:
            env.seed(seed)
        return env.reset()
    obs,_=env.reset(seed=seed)
    return obs


def _step_env(env,action):
    if isinstance(env,VecEnv):
        obs,rewards,dones,infos=env.step(action)
        return obs,float(rewards[0]),bool(dones[0]),infos[0]

    obs,reward,terminated,truncated,info=env.step(action)
    return obs,float(reward),bool(terminated or truncated),info


def evaluate_policy_with_success(model,env,n_eval_episodes,seed,deterministic=True):
    episode_rewards=[]
    episode_lengths=[]
    episode_successes=[]

    for episode_idx in range(n_eval_episodes):
        episode_seed=seed+episode_idx
        obs=_reset_env(env,seed=episode_seed)
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
    success_count=int(np.sum(episode_successes)) if episode_successes else 0
    reward_std=float(np.std(episode_rewards))
    reward_sem=reward_std/np.sqrt(n_eval_episodes)
    length_std=float(np.std(episode_lengths))
    length_sem=length_std/np.sqrt(n_eval_episodes)

    return {
        "success_rate":success_rate,
        "success_count":success_count,
        "failure_count":n_eval_episodes-success_count,
        "mean_reward":float(np.mean(episode_rewards)),
        "std_reward":reward_std,
        "reward_sem":float(reward_sem),
        "reward_ci95":float(1.96*reward_sem),
        "min_reward":float(np.min(episode_rewards)),
        "max_reward":float(np.max(episode_rewards)),
        "mean_episode_length":float(np.mean(episode_lengths)),
        "std_episode_length":length_std,
        "episode_length_sem":float(length_sem),
        "episode_length_ci95":float(1.96*length_sem),
        "min_episode_length":float(np.min(episode_lengths)),
        "max_episode_length":float(np.max(episode_lengths)),
    }


def _clear_local_imports():
    for module_name in list(sys.modules):
        if (
            module_name=="src"
            or module_name.startswith("src.")
            or module_name=="evaluate_robustness"
        ):
            del sys.modules[module_name]


def _with_experiment_path(experiment_dir):
    experiment_dir=str(experiment_dir)
    if experiment_dir in sys.path:
        sys.path.remove(experiment_dir)
    sys.path.insert(0,experiment_dir)


def _load_config(experiment):
    _clear_local_imports()
    _with_experiment_path(experiment.directory)
    return importlib.import_module("src.config")


def _make_env(experiment,mass,seed):
    _clear_local_imports()
    _with_experiment_path(experiment.directory)

    if experiment.kind=="baseline":
        evaluate_robustness=importlib.import_module("evaluate_robustness")
        return evaluate_robustness.make_eval_vec_env(mass,seed=seed)

    domain_randomization=importlib.import_module("src.domain_randomization")
    if experiment.kind=="udr":
        return domain_randomization.make_pandapush_vec_env(
            min_cube_mass=mass,
            max_cube_mass=mass,
            n_envs=1,
            seed=seed,
        )
    if experiment.kind=="adr":
        return domain_randomization.make_pandapush_vec_env(
            cube_mass_mode="fixed",
            fixed_cube_mass=mass,
            n_envs=1,
            seed=seed,
        )

    raise ValueError(f"Unknown experiment kind: {experiment.kind}")


def _model_zip_path(model_path):
    model_path=Path(model_path)
    if model_path.suffix==".zip":
        return model_path
    return model_path.with_suffix(".zip")


def evaluate_experiment(experiment,mass,episodes,seed):
    config=_load_config(experiment)
    model_path=_model_zip_path(config.MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model for {experiment.name}: {model_path}")

    env=_make_env(experiment,mass,seed)
    try:
        model=SAC.load(str(config.MODEL_PATH),env=env)
        metrics=evaluate_policy_with_success(
            model,
            env,
            n_eval_episodes=episodes,
            seed=seed,
            deterministic=True,
        )
    finally:
        env.close()

    return {
        "model":experiment.name,
        "experiment_dir":experiment.directory.name,
        "model_path":str(model_path.relative_to(REPO_ROOT)),
        "mass_kg":mass,
        "episodes":episodes,
        **metrics,
    }


def _mass_label(mass):
    return f"{mass:g}".replace(".","p")


def _default_csv_path(mass,episodes):
    return OUTPUT_DIR/f"fixed_mass_{_mass_label(mass)}kg_{episodes}ep_eval.csv"


def _default_plot_path(mass,episodes):
    return OUTPUT_DIR/f"fixed_mass_{_mass_label(mass)}kg_{episodes}ep_performance.png"


def plot_results(results,plot_path,mass):
    labels=[row["model"] for row in results]
    colors=["#3B82F6","#10B981","#F59E0B"]

    success_rates=[100.0*row["success_rate"] for row in results]
    mean_rewards=[row["mean_reward"] for row in results]
    reward_errors=[row["reward_ci95"] for row in results]
    mean_lengths=[row["mean_episode_length"] for row in results]
    length_errors=[row["episode_length_ci95"] for row in results]

    fig,axes=plt.subplots(1,3,figsize=(13.5,4.2))
    fig.suptitle(
        f"Target Domain Evaluation - Cube Mass {mass:g} kg",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].bar(labels,success_rates,color=colors)
    axes[0].set_title("Success Rate")
    axes[0].set_ylabel("%")
    axes[0].set_ylim(0,105)
    for idx,value in enumerate(success_rates):
        axes[0].text(idx,value+1.5,f"{value:.1f}%",ha="center",fontsize=9)

    axes[1].bar(labels,mean_rewards,yerr=reward_errors,capsize=5,color=colors)
    axes[1].set_title("Mean Reward (95% CI)")
    axes[1].set_ylabel("reward")
    for idx,value in enumerate(mean_rewards):
        axes[1].text(idx,value-3.5,f"{value:.2f}",ha="center",fontsize=9)

    axes[2].bar(labels,mean_lengths,yerr=length_errors,capsize=5,color=colors)
    axes[2].set_title("Mean Episode Length (95% CI)")
    axes[2].set_ylabel("steps")
    axes[2].set_ylim(0,max(mean_lengths)+max(length_errors)+8)
    for idx,value in enumerate(mean_lengths):
        label_y=value+length_errors[idx]+1.5
        axes[2].text(idx,label_y,f"{value:.1f}",ha="center",fontsize=9)

    for ax in axes:
        ax.grid(axis="y",alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x",labelrotation=12)

    fig.tight_layout(rect=(0,0,1,0.92))
    plot_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(plot_path,dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(
        description="Evaluate PandaPush SAC checkpoints at one fixed cube mass."
    )
    parser.add_argument("--mass",type=float,default=5.0,help="Fixed cube mass in kg.")
    parser.add_argument("--episodes",type=int,default=100,help="Episodes per model.")
    parser.add_argument("--seed",type=int,default=42,help="Evaluation seed.")
    parser.add_argument("--output",type=Path,default=None,help="CSV output path.")
    parser.add_argument("--plot-output",type=Path,default=None,help="PNG plot output path.")
    args=parser.parse_args()

    output_path=args.output or _default_csv_path(args.mass,args.episodes)
    plot_path=args.plot_output or _default_plot_path(args.mass,args.episodes)
    output_path.parent.mkdir(parents=True,exist_ok=True)

    results=[]
    for experiment in EXPERIMENTS:#compare all trained variants
        print(
            f"Evaluating {experiment.name} at {args.mass:g} kg "
            f"for {args.episodes} episodes..."
        )
        row=evaluate_experiment(experiment,args.mass,args.episodes,args.seed)
        results.append(row)
        print(
            f"  success_rate={100*row['success_rate']:.1f}% "
            f"mean_reward={row['mean_reward']:.2f} "
            f"mean_length={row['mean_episode_length']:.1f}"
        )

    fieldnames=[
        "model",
        "experiment_dir",
        "model_path",
        "mass_kg",
        "episodes",
        "success_rate",
        "success_count",
        "failure_count",
        "mean_reward",
        "std_reward",
        "reward_sem",
        "reward_ci95",
        "min_reward",
        "max_reward",
        "mean_episode_length",
        "std_episode_length",
        "episode_length_sem",
        "episode_length_ci95",
        "min_episode_length",
        "max_episode_length",
    ]
    with output_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    plot_results(results,plot_path,args.mass)

    print(f"Results saved to {output_path}")
    print(f"Plot saved to {plot_path}")


if __name__=="__main__":
    main()

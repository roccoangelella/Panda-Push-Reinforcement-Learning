import panda_gym
from stable_baselines3.common.env_util import make_vec_env

from src.config import (
    ENV_NAME,MAX_EPISODE_STEPS,N_ENVS,OUTPUT_DIR,REWARD_TYPE,SEED,
)
from src.generate_video import generate_video
from src.model import create_model
from src.train import train_sac


def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    env=make_vec_env(
        ENV_NAME,
        n_envs=N_ENVS,
        seed=SEED,
        env_kwargs={"reward_type":REWARD_TYPE,"max_episode_steps":MAX_EPISODE_STEPS},
    )
    model=create_model(env)
    train_sac(model,env)
    env.close()

    generate_video()


if __name__=="__main__":
    main()

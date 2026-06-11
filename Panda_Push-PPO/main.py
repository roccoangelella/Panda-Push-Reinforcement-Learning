import gymnasium as gym
import panda_gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from src.config import (
    CLIP_OBS,ENV_NAME,MAX_EPISODE_STEPS,NORM_OBS,
    OUTPUT_DIR,VEC_NORMALIZE_PATH,
)
from src.generate_video import generate_video
from src.model import create_model
from src.reward_wrapper import DenseRewardWrapper
from src.train import train_ppo


def make_env():
    """Factory that creates a single env instance with the reward wrapper.

    make_vec_env calls this function once per parallel worker, so each
    of the 16 sub-environments gets its own DenseRewardWrapper.
    """
    env=gym.make(ENV_NAME,reward_type="dense",max_episode_steps=MAX_EPISODE_STEPS)
    return DenseRewardWrapper(env)


def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    raw_env=make_vec_env(make_env,n_envs=16)#collect parallel rollouts
    #norm_reward=False: the shaped reward already has a natural scale
    #(sum of two distances, typically in [0, 0.5] range). Normalizing it
    #would compress the useful gradient that reward shaping provides.
    env=VecNormalize(raw_env,norm_obs=NORM_OBS,norm_reward=False,clip_obs=CLIP_OBS)
    model=create_model(env)
    train_ppo(model,env)

    #save normalization stats so evaluation/video use the same scaling.
    env.save(str(VEC_NORMALIZE_PATH))
    env.close()

    generate_video()


if __name__=="__main__":
    main()

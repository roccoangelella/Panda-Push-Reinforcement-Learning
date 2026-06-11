import gymnasium as gym
import numpy as np
import panda_gym
from stable_baselines3.common.env_util import make_vec_env

from src.config import (
    ENV_NAME,
    FIXED_CUBE_MASS,
    MAX_CUBE_MASS,
    MAX_EPISODE_STEPS,
    MIN_CUBE_MASS,
    REWARD_TYPE,
    SEED,
)

class CubeMassRandomizationWrapper(gym.Wrapper):

    def __init__(
        self,
        env,
        min_cube_mass=MIN_CUBE_MASS,
        max_cube_mass=MAX_CUBE_MASS,
        object_body_name="object",
    ):
        super().__init__(env)
        self.cube_mass_mode="uniform"
        self.fixed_cube_mass=FIXED_CUBE_MASS
        self.min_cube_mass=min_cube_mass
        self.max_cube_mass=max_cube_mass

        self.object_body_name=object_body_name
        self.current_cube_mass=None
        self._rng=np.random.default_rng()

    def reset(self,*,seed=None,options=None):
        obs,info=self.env.reset(seed=seed,options=options)
        if seed is not None:
            self._rng=np.random.default_rng(seed)

        cube_mass=self.sample_cube_mass()#sample a new mass every episode
        self.apply_cube_mass(cube_mass)
        info=dict(info)
        info.update(self._mass_info())
        return obs,info

    def sample_cube_mass(self):
        if self.cube_mass_mode=="fixed":
            return self.fixed_cube_mass
        return self._uniform_mass(self.min_cube_mass,self.max_cube_mass)

    def _uniform_mass(self,low,high):
        if np.isclose(low,high):
            return float(low)
        return float(self._rng.uniform(low,high))

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

        self.current_cube_mass=cube_mass
        return cube_mass

    def get_cube_mass(self):
        sim=self.unwrapped.sim
        if hasattr(sim,"physics_client") and hasattr(sim,"_bodies_idx"):
            body_id=sim._bodies_idx[self.object_body_name]
            dynamics=sim.physics_client.getDynamicsInfo(body_id,-1)
            return float(dynamics[0])
        return self.current_cube_mass

    def set_fixed_cube_mass(self,cube_mass):
        self.fixed_cube_mass=cube_mass
        self.cube_mass_mode="fixed"
        return self.apply_cube_mass(self.fixed_cube_mass)

    def get_cube_mass_range(self):
        if self.cube_mass_mode=="fixed":
            return (self.fixed_cube_mass,self.fixed_cube_mass)
        return (self.min_cube_mass,self.max_cube_mass)

    def _mass_info(self):
        current_min,current_max=self.get_cube_mass_range()
        return {
            "cube_mass":float(self.current_cube_mass),
            "current_min_cube_mass":float(current_min),
            "current_max_cube_mass":float(current_max),
        }

def make_pandapush_env(
    min_cube_mass=MIN_CUBE_MASS,
    max_cube_mass=MAX_CUBE_MASS,
    env_name=ENV_NAME,
    reward_type=REWARD_TYPE,
    max_episode_steps=MAX_EPISODE_STEPS,
):
    env=gym.make(
        env_name,
        reward_type=reward_type,
        max_episode_steps=max_episode_steps,
    )
    return CubeMassRandomizationWrapper(
        env,
        min_cube_mass=min_cube_mass,
        max_cube_mass=max_cube_mass,
    )

def make_pandapush_vec_env(
    min_cube_mass=MIN_CUBE_MASS,
    max_cube_mass=MAX_CUBE_MASS,
    n_envs=1,
    seed=SEED,
):
    def make_env():
        return make_pandapush_env(
            min_cube_mass=min_cube_mass,
            max_cube_mass=max_cube_mass,
        )

    return make_vec_env(make_env,n_envs=n_envs,seed=seed)

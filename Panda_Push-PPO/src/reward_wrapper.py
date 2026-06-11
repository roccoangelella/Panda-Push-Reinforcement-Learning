#we shape a different goal for PPO, making the agent get reward also for touching the cube,
#pushing the cube, and moving the cube towards the goal.

import gymnasium as gym
import numpy as np


class DenseRewardWrapper(gym.Wrapper):
    def __init__(self,env,reach_weight=1.0,push_weight=5.0,progress_weight=10.0):
        super().__init__(env)
        self.reach_weight=reach_weight
        self.push_weight=push_weight
        self.progress_weight=progress_weight
        self._prev_obj_to_goal=None

    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)
        obj_pos=obs["achieved_goal"]
        goal_pos=obs["desired_goal"]
        self._prev_obj_to_goal=np.linalg.norm(obj_pos-goal_pos)
        return obs,info

    def step(self,action):
        obs,_reward,terminated,truncated,info=self.env.step(action)
        #extract positions from the observation dict.
        ee_pos=obs["observation"][:3]
        obj_pos=obs["achieved_goal"]
        goal_pos=obs["desired_goal"]

        #guide the gripper toward the object.
        dist_gripper_to_obj=np.linalg.norm(ee_pos-obj_pos)
        #guide the object toward the goal.
        dist_obj_to_goal=np.linalg.norm(obj_pos-goal_pos)
        #reward progress: positive when the object moves closer to the goal.
        progress=self._prev_obj_to_goal-dist_obj_to_goal
        self._prev_obj_to_goal=dist_obj_to_goal
        #combine all terms into the shaped reward.
        
        shaped_reward=(
            -self.reach_weight*dist_gripper_to_obj
            -self.push_weight*dist_obj_to_goal
            +self.progress_weight*progress
        )
        return obs,shaped_reward,terminated,truncated,info

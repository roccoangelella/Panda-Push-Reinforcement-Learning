import torch as th
from stable_baselines3 import PPO

from src.config import (
    BATCH_SIZE,CLIP_RANGE,ENT_COEF,GAE_LAMBDA,GAMMA,
    LEARNING_RATE,N_EPOCHS,N_STEPS,SDE_SAMPLE_FREQ,USE_SDE,
)


def create_model(env):
    policy_kwargs=dict(
        net_arch=dict(pi=[256,256],vf=[256,256]),#custom nn, the default one in PPO is shallower
        activation_fn=th.nn.Tanh,
    )
    return PPO(
        "MultiInputPolicy",env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        use_sde=USE_SDE,
        sde_sample_freq=SDE_SAMPLE_FREQ,
        policy_kwargs=policy_kwargs,
    )

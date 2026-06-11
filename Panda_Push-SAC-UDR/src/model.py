import torch as th
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from src.config import (
    BATCH_SIZE,BUFFER_SIZE,ENT_COEF,GAMMA,GOAL_SELECTION_STRATEGY,
    GRADIENT_STEPS,LEARNING_RATE,LEARNING_STARTS,N_SAMPLED_GOAL,
    SDE_SAMPLE_FREQ,SEED,TARGET_UPDATE_INTERVAL,TAU,TRAIN_FREQ,
    USE_SDE,USE_SDE_AT_WARMUP,
)

def create_model(env,seed=SEED):
    policy_kwargs=dict(net_arch=[256,256,256],activation_fn=th.nn.ReLU)
    replay_buffer_kwargs=dict(n_sampled_goal=N_SAMPLED_GOAL,goal_selection_strategy=GOAL_SELECTION_STRATEGY)
    return SAC(
        "MultiInputPolicy",env,verbose=1,
        learning_rate=LEARNING_RATE,buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,batch_size=BATCH_SIZE,
        tau=TAU,gamma=GAMMA,train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,ent_coef=ENT_COEF,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=replay_buffer_kwargs,
        use_sde=USE_SDE,sde_sample_freq=SDE_SAMPLE_FREQ,
        use_sde_at_warmup=USE_SDE_AT_WARMUP,
        policy_kwargs=policy_kwargs,seed=seed,
    )

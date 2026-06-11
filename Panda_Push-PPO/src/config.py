#configuration hyperparameters for Panda Push PPO
from pathlib import Path

#absolute paths — immune to working-directory issues
BASE_DIR=Path(__file__).resolve().parent.parent
OUTPUT_DIR=BASE_DIR/"output"

ENV_NAME="PandaPush-v3"
MAX_EPISODE_STEPS=200

#training parameters
TOTAL_TIMESTEPS=40000000

LEARNING_RATE=3e-4
N_STEPS=2048
BATCH_SIZE=64
N_EPOCHS=10
GAMMA=0.99
GAE_LAMBDA=0.95
CLIP_RANGE=0.2
ENT_COEF=0.02
USE_SDE=True
SDE_SAMPLE_FREQ=4

#observation and reward normalization
NORM_OBS=True
NORM_REWARD=True
CLIP_OBS=10.0

#logging and Saving
#evalCallback overwrites this file whenever a new best policy is found.
MODEL_PATH=OUTPUT_DIR/"best_model"
VEC_NORMALIZE_PATH=OUTPUT_DIR/"vec_normalize.pkl"
CSV_PATH=OUTPUT_DIR/"training.csv"
EVAL_FREQ=5000
EVAL_EPISODES=10
VIDEO_PATH=OUTPUT_DIR/"run.mp4"
VIDEO_FPS=30

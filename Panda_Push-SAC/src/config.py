#configuration hyperparameters for Panda Push SAC
from pathlib import Path

ENV_NAME="PandaPush-v3"
MAX_EPISODE_STEPS=200
REWARD_TYPE="sparse"
SEED=42

#absolute paths, immune to working-directory issues
BASE_DIR=Path(__file__).resolve().parent.parent
OUTPUT_DIR=BASE_DIR/"output"

#training parameters
TOTAL_TIMESTEPS=4000000
N_ENVS=1
LEARNING_RATE=3e-4
BUFFER_SIZE=1000000#past transitions kept for off-policy learning
LEARNING_STARTS=10000
BATCH_SIZE=256
TAU=0.005#soft update rate for the target networks
GAMMA=0.95
ENT_COEF="auto"
TARGET_UPDATE_INTERVAL=1#how often the target networks are updated
TRAIN_FREQ=1
GRADIENT_STEPS=1
USE_SDE=True
SDE_SAMPLE_FREQ=64
USE_SDE_AT_WARMUP=True

#hER parameters
N_SAMPLED_GOAL=4
GOAL_SELECTION_STRATEGY="future"

#logging and Saving
MODEL_PATH=OUTPUT_DIR/"best_model"
CSV_PATH=OUTPUT_DIR/"training.csv"
CSV_LOG_INTERVAL=32768
LOG_DIR=OUTPUT_DIR/"logs"
EVAL_LOG_DIR=OUTPUT_DIR/"eval"
EVAL_FREQ=10000
EVAL_EPISODES=50
VIDEO_PATH=OUTPUT_DIR/"run.mp4"
VIDEO_FPS=30

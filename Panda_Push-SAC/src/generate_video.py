import gymnasium as gym
import imageio
import panda_gym
from stable_baselines3 import SAC

from src.config import ENV_NAME,MAX_EPISODE_STEPS,MODEL_PATH,REWARD_TYPE,SEED,VIDEO_FPS,VIDEO_PATH


def generate_video():
    if not MODEL_PATH.with_suffix(".zip").exists() and not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained model at {MODEL_PATH}.zip. "
            "Run `python main.py` from Panda_Push-SAC first, or copy best_model.zip into output/."
        )

    render_env=gym.make(
        ENV_NAME,
        render_mode="rgb_array",
        reward_type=REWARD_TYPE,
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    model=SAC.load(str(MODEL_PATH),env=render_env)

    frames=[]
    obs,_=render_env.reset(seed=SEED)
    done=False

    print("Recording episode...")
    while not done:
        frames.append(render_env.render())
        action,_states=model.predict(obs,deterministic=True)
        obs,reward,terminated,truncated,info=render_env.step(action)
        done=terminated or truncated
        if done:
            frames.append(render_env.render())

    render_env.close()

    VIDEO_PATH.parent.mkdir(parents=True,exist_ok=True)
    if not frames:
        raise RuntimeError("No frames were recorded; cannot write video.")
    frames.extend([frames[-1]]*(2*VIDEO_FPS))
    imageio.mimwrite(str(VIDEO_PATH),frames,fps=VIDEO_FPS)
    print(f"Video saved to {VIDEO_PATH}")


def main():
    generate_video()


if __name__=="__main__":
    main()

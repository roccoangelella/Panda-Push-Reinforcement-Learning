import panda_gym

from src.config import N_ENVS,SEED,MIN_CUBE_MASS,ADR_MAX_MASS,ADR_INITIAL_MAX_MASS,OUTPUT_DIR
from src.domain_randomization import make_pandapush_vec_env
from src.generate_video import generate_video
from src.model import create_model
from src.train import train_sac


def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    env=make_pandapush_vec_env(
        cube_mass_mode="adr",
        min_cube_mass=MIN_CUBE_MASS,
        max_cube_mass=ADR_MAX_MASS,
        current_max_cube_mass=ADR_INITIAL_MAX_MASS,
        n_envs=N_ENVS,
        seed=SEED,
    )

    model=create_model(env)
    train_sac(model,env)
    env.close()

    generate_video()


if __name__=="__main__":
    main()

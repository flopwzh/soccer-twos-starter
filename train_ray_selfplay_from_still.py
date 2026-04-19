import os
import pickle

import ray
from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 1
CHECKPOINT_PATH = "./basic_agent/ray_results/PPO_SP/PPO_Soccer_5af09_00000_0_2026-04-19_00-29-29/checkpoint_000500/checkpoint-500"


def policy_mapping_fn(agent_id, *args, **kwargs):
    del args, kwargs
    del agent_id
    return "default_policy"


def load_checkpoint_config(checkpoint_path):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(checkpoint_dir, "..", "params.pkl")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "Could not find params.pkl in either the checkpoint dir or its parent directory."
        )

    with open(config_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    ray.init(include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)

    # Use the same spaces as the self-play environment.
    temp_env = create_rllib_env(
        {
            "variation": EnvType.multiagent_player,
            "multiagent": True,
            "single_player": False,
            "flatten_branched": True,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        }
    )
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    base_config = load_checkpoint_config(CHECKPOINT_PATH)

    # Keep checkpoint-compatible model settings, override env and rollout settings for self-play.
    base_config.update(
        {
            "num_gpus": 0,
            "num_workers": 1,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.multiagent_player,
                "multiagent": True,
                "single_player": False,
                "flatten_branched": True,
            },
            "multiagent": {
                "policies": {
                    "default_policy": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default_policy"],
            },
        }
    )

    # Remove opponent policy from vs-still training config if present.
    base_config["env_config"].pop("opponent_policy", None)

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_from_still",
        config=base_config,
        stop={
            "timesteps_total": 20000000,
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        restore=CHECKPOINT_PATH,
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")

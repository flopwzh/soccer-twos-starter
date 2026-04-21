# from logging import config
import pickle
import os
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls

from gym_unity.envs import ActionFlattener
from gym.spaces import Discrete
from utils import create_rllib_env

from soccer_twos import AgentInterface


ALGORITHM = "PPO"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "./ray_results/PPO_SP/PPO_Soccer_5af09_00000_0_2026-04-19_00-29-29/checkpoint_000500/checkpoint-500",
)
POLICY_NAME = "default"  # this may be useful when training with selfplay

class DummyEnv(gym.Env):
    def __init__(self, config):
        self.observation_space = config["observation_space"]
        self.action_space = config["action_space"]

    def reset(self):
        return self.observation_space.sample()
    
    def step(self, action):
        return self.observation_space.sample(), 0, True, {}
    
    def close(self):
        pass

class RayAgent(AgentInterface):
    """
    RayAgent is an agent that uses ray to train a model.
    """

    def __init__(self, env: gym.Env):
        """Initialize the RayAgent.
        Args:
            env: the competition environment.
        """
        super().__init__()
        ray.init(ignore_reinit_error=True)

        # Load configuration from checkpoint file.
        config_path = ""
        if CHECKPOINT_PATH:
            config_dir = os.path.dirname(CHECKPOINT_PATH)
            config_path = os.path.join(config_dir, "params.pkl")
            # Try parent directory.
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")

        # Load the config from pickled.
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        else:
            # If no config in given checkpoint -> Error.
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory!"
            )

        # no need for parallelism on evaluation
        config["num_workers"] = 0
        config["num_gpus"] = 0
        config["num_envs_per_worker"] = 1
        config["explore"] = False

        # create a dummy env since it's required but we only care about the policy
        env_name = "DummyEnv"
        tune.registry.register_env(env_name, lambda c: DummyEnv(c))
        config["env"] = env_name

        train_env_cfg = config.get("env_config", {})
        flattened = train_env_cfg.get("flatten_branched", False)

        if flattened:
            trainer_action_space = Discrete(int(np.prod(env.action_space.nvec)))
        else:
            trainer_action_space = env.action_space

        config["env_config"] = {
            "observation_space": env.observation_space,
            "action_space": trainer_action_space,
        }

        self.use_flattener = config.get("env_config", {}).get("flatten_branched", True)
        self.flattener = ActionFlattener(env.action_space.nvec)

        # create the Trainer from config
        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        # load state from checkpoint
        agent.restore(CHECKPOINT_PATH)
        # get policy for evaluation
        # self.policy = agent.get_policy(POLICY_NAME)
        self.policy = agent.get_policy()

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """The act method is called when the agent is asked to act.
        Args:
            observation: a dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
        Returns:
            action: a dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
        """
        actions = {}
        for player_id in observation:
            # compute_single_action returns a tuple of (action, action_info, ...)
            # as we only need the action, we discard the other elements
            # actions[player_id], *_ = self.policy.compute_single_action(
            #     observation[player_id]
            # )
            action, *_ = self.policy.compute_single_action(observation[player_id], explore=False)
            if np.isscalar(action):
                action = self.flattener.lookup_action(int(action))
            actions[player_id] = action
        return actions

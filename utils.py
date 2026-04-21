from random import uniform as randfloat

import gym
from ray.rllib import MultiAgentEnv
import soccer_twos

import numpy as np


class RewardsWrapper(gym.core.Wrapper):
    """
    Wrapper designed to reward agents for performing certain actions
    """
    def __init__(self, env):
        super(RewardsWrapper, self).__init__(env)
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        # self.observation_space = gym.spaces.Box(
        #     0, 1, dtype=np.float32, shape=(env.observation_space.shape[0] * 2,)
        # )
        # if isinstance(env.action_space, gym.spaces.Discrete):
        #     self.action_space = gym.spaces.Discrete(env.action_space.n ** 2)
        #     self.action_space_n = env.action_space.n
        # elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
        #     self.action_space = gym.spaces.MultiDiscrete(
        #         np.repeat(env.action_space.nvec, 2)
        #     )
        #     self.action_space_n = len(env.action_space.nvec)
        # else:
        #     raise ValueError("Unsupported action space type")
        
        # reward shaping parameters
        self.living_penalty = -0.001
        self.goal_reward_mult = 1.0
        self.progress_reward = 0.05
        self.ball_velocity_reward = 0.01
        self.centering_reward = 0.05
        self.concede_penalty = -5.0
        self.out_of_position_penalty = -0.05
        
        self.elapsed_time = 0

    
    def reset(self):
        obs = self.env.reset()
        self.elapsed_time = 0
        return obs

    def step(self, action):
        obs, rewards, dones, infos = self.env.step(action)
        # reward shaping
        shaped = dict(rewards)
        ball_pos, ball_vel = self._compute_ball_info(infos)
        for agent_id, base_reward in rewards.items():
            team_id = agent_id // 2
            team_val = 1 if agent_id < 2 else -1
            total_bonus = 0
            total_bonus += self.living_penalty

            # progress towards opponent goal
            if ball_pos is not None:
                progress = np.clip((ball_pos[0] - 0)/15, 0, 1) * team_val
                total_bonus += self.progress_reward * progress
            
            # ball velocity towards opponent goal
            if ball_vel is not None:
                vel_towards_goal = ball_vel[0] * team_val
                total_bonus += self.ball_velocity_reward * vel_towards_goal

            # once ball is on attacking side, reward centering
            if ball_pos is not None and ball_vel is not None and (ball_pos[0] * team_val) > 0:
                vel_towards_center = -np.sign(ball_pos[1]) * ball_vel[1]
                vel_towards_center_clipped = np.clip(vel_towards_center, -2.0, 2.0)
                total_bonus += self.centering_reward * vel_towards_center_clipped

            # penalty for being far from goal when opponent scores
            done_all = dones.get("__all__", False)
            if done_all and base_reward < 0 and ball_pos is not None:
                total_bonus += self.concede_penalty
                player_pos = infos.get(agent_id, {}).get("player_info", {}).get("position")
                if player_pos is not None:
                    dist_to_ball = np.linalg.norm(np.array(player_pos) - np.array(ball_pos))
                    total_bonus += self.out_of_position_penalty * np.clip(dist_to_ball, 0, 20)

            # clip and sum bonuses
            total_bonus = np.clip(total_bonus, -10.0, 10.0)
            shaped[agent_id] = self.goal_reward_mult * base_reward + total_bonus
        
        return obs, shaped, dones, infos

    def _compute_ball_info(self,infos):
        for agent_info in infos.values():
            ball_info = agent_info.get("ball_info", {})
            ball_pos = ball_info.get("position")
            ball_vel = ball_info.get("velocity")
            if ball_pos is not None and ball_vel is not None:
                return ball_pos, ball_vel
        return None, None


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    def __init__(self, env, wrapper_class=RewardsWrapper):
        super(RLLibWrapper, self).__init__(env)
        if wrapper_class is not None:
            self.env = wrapper_class(env)
        else:
            self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        obs = self.env.reset()
        return obs
    
    def step(self, action_dict):
        obs, reward, done, info = self.env.step(action_dict)
        return obs, reward, done, info


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env, RewardsWrapper)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s

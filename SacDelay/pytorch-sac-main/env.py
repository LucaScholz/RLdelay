import atexit
import os
from dataclasses import dataclass, InitVar
import gym
from gym.wrappers import TimeLimit

from wrappers import Float64ToFloat32, TimeLimitResetWrapper, NormalizeActionWrapper, RealTimeWrapper, TupleObservationWrapper, AffineObservationWrapper, AffineRewardWrapper, PreviousActionWrapper, FrameSkip, get_wrapper_by_class
from wrappers_rd import RandomDelayWrapper, WifiDelayWrapper1, WifiDelayWrapper2
import numpy as np
class RandomDelayEnv(Env):
    def __init__(self,
                 seed_val=0, id: str = "Pendulum-v0",
                 frame_skip: int = 0,
                 min_observation_delay: int = 0,
                 sup_observation_delay: int = 8,
                 min_action_delay: int = 0,  # this is equivalent to a MIN of 1 in the paper
                 sup_action_delay: int = 2,  # this is equivalent to a MAX of 2 in the paper
                 real_world_sampler: int = 0):  # 0 for uniform, 1 or 2 for simple wifi sampler
        env = gym.make(id)

        if frame_skip:
            original_frame_skip = getattr(env.unwrapped, 'frame_skip', 1)  # on many Mujoco environments this is 5
            # print("Original frame skip", original_frame_skip)

            # I think the two lines below were actually a mistake after all (at least for HalfCheetah)
            # if hasattr(env, 'dt'):
            #   env.dt = env.dt  # in case this is an attribute we fix it to its orignal value to not distort rewards (see
            #   halfcheetah.py)
            env.unwrapped.frame_skip = 1
            tl = get_wrapper_by_class(env, TimeLimit)
            tl._max_episode_steps = int(tl._max_episode_steps * original_frame_skip)
            # print("New max episode steps", env._max_episode_steps)
            env = FrameSkip(env, frame_skip, 1 / original_frame_skip)

        env = Float64ToFloat32(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        env = NormalizeActionWrapper(env)

        if real_world_sampler == 0:
            env = RandomDelayWrapper(env, range(min_observation_delay, sup_observation_delay), range(min_action_delay, sup_action_delay))
        elif real_world_sampler == 1:
            env = WifiDelayWrapper1(env)
        elif real_world_sampler == 2:
            env = WifiDelayWrapper2(env)
        else:
            assert False, f"invalid value for real_world_sampler:{real_world_sampler}"
        super().__init__(env)
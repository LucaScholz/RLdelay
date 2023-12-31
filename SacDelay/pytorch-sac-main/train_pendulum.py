import time
import gym
from gym.wrappers import TimeLimit
from replay_buffer import ReplayBuffer, Transition
from params_pool import ParamsPool
from action_wrappers import ScalingActionWrapper
from make_video_of_saved_actor import make_video_of_saved_actor
from env import RandomDelayEnv

import wandb
import argparse

# =================================================================================
# arguments

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=int)
args = parser.parse_args()

# =================================================================================
# logging

wandb.init(
    project='sacDelay',
    entity='lucascholz',
    group=f'rldelay',
    settings=wandb.Settings(_disable_stats=True),
    name=f'run_id={args.run_id}'
)

# =================================================================================
env_raw = RandomDelayEnv()
#env_raw = gym.make('Pendulum-v1')
env = ScalingActionWrapper(env_raw, scaling_factors=env_raw.action_space.high)
buf = ReplayBuffer(capacity=int(1e6))
param = ParamsPool(
    input_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0]
)

batch_size = 64
num_episodes = 1000

start_time = time.perf_counter()

for e in range(num_episodes):

    obs = env.reset()[0]

    total_reward = 0
    total_updates = 0

    while True:

        # ==================================================
        # getting the tuple (s, a, r, s', done)
        # ==================================================

        action = param.act(obs)
        next_obs, reward, done, _ , _= env.step(action)
        # no need to keep track of max time-steps, because the environment
        # is wrapped with TimeLimit automatically (timeout after 1000 steps)

        total_reward += reward

        # ==================================================
        # storing it to the buffer
        # ==================================================

        buf.push(Transition(obs, action, reward, next_obs, done))

        # ==================================================
        # update the parameters
        # ==================================================

        if buf.ready_for(batch_size):
            param.update_networks(buf.sample(batch_size))
            total_updates += 1

        # ==================================================
        # check done
        # ==================================================

        if done: break

        obs = next_obs

    # ==================================================
    # after each episode
    # ==================================================

    wandb.log({'return': total_reward})

    after_episode_time = time.perf_counter()
    time_elapsed = after_episode_time - start_time
    time_remaining = time_elapsed / (e + 1) * (num_episodes - (e + 1))

    print(f'Episode {e:4.0f} | Return {total_reward:9.3f} | Updates {total_updates:4.0f} | Remaining time {round(time_remaining / 3600, 2):5.2f} hours')

param.save_actor(
    save_dir='results/trained_policies_pth/',
    filename=f'{args.run_id}.pth'
)

make_video_of_saved_actor(
    save_dir='results/trained_policies_pth/',
    filename=f'{args.run_id}.pth'
)

env.close()
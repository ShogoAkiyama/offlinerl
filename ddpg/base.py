import os
import numpy as np
import gym
import pybullet_envs

from storage import ReplayBuffer
from ddpg.ddpg import DDPG


class Base:
    def __init__(self, env, device, model_dir, args):
        self.env = env
        self.env_name = args.env_name
        self.seed = args.seed
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.batch_size = args.batch_size
        self.max_timesteps = args.max_timesteps
        self.gaussian_std = args.gaussian_std
        self.start_timesteps = args.start_timesteps
        self.eval_freq = args.eval_freq
        self.rand_action_p = args.rand_action_p

        self.model_dir = os.path.join(model_dir, f"{args.env_name}_{args.seed}")

        self.algo = DDPG(
            self.state_dim, self.action_dim, self.max_action, device)

        self.storage = ReplayBuffer(
            self.state_dim, self.action_dim, device)

        self.eval_rewards = []

        self.total_steps = 0
        self.episodes = 0
        self.episode_steps = 0
        self.episode_rewards = 0

        self.state = None

    def iterate(self):
        assert self.state is not None

        self.episode_steps += 1

        if self.is_random_action():
            action = self.env.action_space.sample()
        else:
            action = (
                    self.algo.select_action(np.array(self.state))
                    + np.random.normal(
                        0, self.max_action * self.gaussian_std,
                        size=self.action_dim)
            ).clip(-self.max_action, self.max_action)

        next_state, reward, done, _ = self.env.step(action)
        done_bool = float(done) if self.episode_steps < self.env._max_episode_steps else 0

        self.storage.add(self.state, action, next_state, reward, done_bool)

        self.state = next_state
        self.episode_rewards += reward

        if done:
            print(
                f"Total T: {self.total_steps + 1} "
                f"Episode Num: {self.episodes + 1} "
                f"Episode T: {self.episode_steps} "
                f"Reward: {self.episode_rewards:.3f}")
            # Reset environment
            self.state = self.env.reset()
            self.episode_rewards = 0
            self.episode_steps = 0
            self.episodes += 1

        self.total_steps += 1

    def evaluate(self, eval_episodes=10):
        eval_env = gym.make(self.env_name)
        eval_env.seed(self.seed + 100)

        avg_reward = 0.
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action = self.algo.select_action(np.array(state))
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward

        avg_reward /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward

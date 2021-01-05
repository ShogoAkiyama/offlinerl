import os
import argparse
import gym
import numpy as np
import torch
from datetime import datetime
import pybullet_envs

from algo.drac import DRAC


def main(args):
    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.noise:
        is_noise = 'noise'
    else:
        is_noise = 'no_noise'

    buffer_dir = os.path.join(
        "./logs", "buffer", args.env_name, is_noise,
        "expert"+str(args.expert_number)+"_size"+str(args.data_size))
    summary_dir = os.path.join(
        "./logs", "summary", "bcq", is_noise,
        'expert' + str(args.expert_number) + '_size' + str(args.data_size),
        datetime.now().strftime("%Y%m%d-%H%M"))

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    bcq = DRAC(
        args.env_name, args.seed, buffer_dir, summary_dir, args.max_timesteps,
        args.eval_freq, args.batch_size, state_dim, action_dim, max_action,
        device, args.gamma, args.tau, args.lmbda, args.phi)

    bcq.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="HalfCheetahBulletEnv-v0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval_freq", default=1000, type=float)
    parser.add_argument("--max_timesteps", default=1e5, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.99)
    parser.add_argument("--tau", default=0.005)
    parser.add_argument("--lmbda", default=0.75)
    parser.add_argument("--phi", default=0.05)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--expert_number", default=1000000, type=int)
    parser.add_argument("--data_size", default=10000, type=int)
    args = parser.parse_args()

    main(args)

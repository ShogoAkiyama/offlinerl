import os
import argparse
import gym
import numpy as np
import torch
from datetime import datetime
import pybullet_envs

from algo.bear import BEAR


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
        "./logs", "summary", "bear", is_noise,
        'expert' + str(args.expert_number) + '_size' + str(args.data_size),
        datetime.now().strftime("%Y%m%d-%H%M"))

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    bear = BEAR(
        summary_dir, args.env_name, args.seed, device, args.max_timesteps, args.eval_freq,
        buffer_dir, 2, state_dim, action_dim, max_action,
        delta_conf=0.1, train_alpha=args.train_alpha, num_samples=args.num_samples, mmd_sigma=args.mmd_sigma,
        lagrange_thresh=args.lagrange_thresh, kernel_type=args.kernel_type)

    bear.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="HalfCheetahBulletEnv-v0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval_freq", default=1000, type=float)
    parser.add_argument("--max_timesteps", default=1e5, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--discount", default=0.99)
    parser.add_argument("--tau", default=0.005)
    parser.add_argument("--phi", default=0.05)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--expert_number", default=500000, type=int)
    parser.add_argument("--data_size", default=10000, type=int)

    parser.add_argument('--train_alpha', action='store_true')
    parser.add_argument('--num_samples', default=100, type=int)   # number of samples to do matching in MMD
    parser.add_argument('--mmd_sigma', default=20.0, type=float)        # The bandwidth of the MMD kernel parameter
    parser.add_argument('--kernel_type', default='gaussian', type=str)
    parser.add_argument('--lagrange_thresh', default=10.0, type=float)
    args = parser.parse_args()

    main(args)

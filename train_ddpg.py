import argparse
import gym
import numpy as np
import os
from datetime import datetime
import torch

from ddpg.trainer import Trainer
from ddpg.run_expert import Expert


def main(args):
    model_dir = os.path.join("./logs", "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        summary_dir = os.path.join(
            "./logs", "summary", "ddpg", datetime.now().strftime("%Y%m%d-%H%M"))
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

        trainer = Trainer(env, device, model_dir, summary_dir, args)
    else:
        if args.no_noise:
            args.rand_action_p = 0.0
            args.gaussian_std = 0.0
        trainer = Expert(env, device, model_dir, args)

    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="HalfCheetahBulletEnv-v0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval_freq", default=10000, type=float)
    parser.add_argument("--max_timesteps", default=10000, type=int)
    parser.add_argument("--start_timesteps", default=25e3, type=int)
    parser.add_argument("--rand_action_p", default=0.3, type=float)
    parser.add_argument("--gaussian_std", default=0.3, type=float)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--discount", default=0.99)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--expert_number", default=10000, type=str)
    parser.add_argument("--no_noise", action="store_true")
    args = parser.parse_args()

    main(args)

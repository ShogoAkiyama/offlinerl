import os
import glob
import torch
from torch import nn
import gym

import pybullet_envs
import shutil
gym.logger.set_level(40)


def wrap_monitor(env):
    return gym.wrappers.Monitor(
        env, '/tmp/monitor', video_callable=lambda x: True, force=True)


def play_mp4():
    path = glob.glob(os.path.join('/tmp/monitor', '*.mp4'))[0]
    shutil.move(path, './movies/outset.mp4')


class SACActor(nn.Module):

    def __init__(self, state_shape, action_shape, std=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2 * action_shape[0]),
        )
        self.std = std

    def forward(self, states):
        with torch.no_grad():
            actions = torch.tanh(self.net(states).chunk(2, dim=1)[0])
            return actions.add_(torch.randn_like(actions) * self.std).clamp_(-1.0, 1.0)

    def load(self, path):
        self.load_state_dict(torch.load(path))


def run_expert(env_id, weight_path, std=0.0, device=torch.device('cuda')):
    env = wrap_monitor(gym.make(env_id))

    actor = SACActor(env.observation_space.shape, env.action_space.shape, std).to(device)
    actor.load(weight_path)

    state = env.reset()
    done = False

    while (not done):
        action = actor(torch.tensor(state, dtype=torch.float, device=device).unsqueeze_(0)).cpu().numpy()[0]
        state, _, done, _ = env.step(action)

    del env
    del actor
    play_mp4()


if __name__ == "__main__":
    run_expert(
        'HalfCheetahBulletEnv-v0',
        os.path.join('./logs/expert', 'expert.pth'),
        std=0.1
    )

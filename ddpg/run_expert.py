import numpy as np
import os

from ddpg.base import Base


class Expert(Base):
    def __init__(self, env, device, model_dir, args):
        super().__init__(env, device, model_dir, args)
        model_path = os.path.join(self.model_dir, str(args.expert_number))
        self.algo.load(model_path)

        if args.no_noise:
            is_noise = 'no_noise'
        else:
            is_noise = 'noise'
        self.buffer_name = os.path.join(
            './logs', 'buffer', args.env_name, is_noise,
            'expert' + str(args.expert_number) + '_size' + str(args.max_timesteps))
        if not os.path.exists(self.buffer_name):
            os.makedirs(self.buffer_name)

    def run(self):
        self.state = self.env.reset()
        for _ in range(int(self.max_timesteps)):
            self.iterate()

        self.save()

    def is_random_action(self):
        return np.random.uniform(0, 1) < self.rand_action_p

    def save(self):
        self.storage.save(self.buffer_name)

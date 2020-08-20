import os
from torch.utils.tensorboard import SummaryWriter

from ddpg.base import Base


class Trainer(Base):
    def __init__(self, env, device, model_dir, summary_dir, args):
        super().__init__(env, device, model_dir, args)
        self.writer = SummaryWriter(log_dir=summary_dir)

    def run(self):
        self.state = self.env.reset()
        for _ in range(int(self.max_timesteps)):
            self.iterate()

            if self.total_steps >= self.start_timesteps:
                state, action, next_state, reward, not_done = self.storage.sample(self.batch_size)
                self.algo.train(state, action, next_state, reward, not_done)

            if self.total_steps % self.eval_freq == 0:
                reward = self.evaluate()
                self.writer.add_scalar(
                    'return/eval', reward, self.total_steps)
                self.eval_rewards.append(reward)
                self.algo.save(os.path.join(self.model_dir, str(self.total_steps)))

    def is_random_action(self):
        return self.total_steps < self.start_timesteps

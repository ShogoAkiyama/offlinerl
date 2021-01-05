import metaworld
import random

bench_name = 'ML1'

if bench_name == 'ML1':
    print(metaworld.ML1.ENV_NAMES)
    bench = metaworld.ML1('pick-place-v1')
elif bench_name == 'ML10':
    bench = metaworld.ML10()

training_envs = []
for name, env_cls in bench.train_classes.items():
    env = env_cls()
    task = random.choice([task for task in bench.train_tasks
                          if task.env_name == name])
    env.set_task(task)
    training_envs.append(env)

print(training_envs)

for env in training_envs:
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)


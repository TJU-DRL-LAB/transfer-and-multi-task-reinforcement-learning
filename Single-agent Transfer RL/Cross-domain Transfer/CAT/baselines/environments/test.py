import gym
import baselines.environments.register as register
import baselines.environments.init_path as init_path
init_path.bypass_frost_warning()

if __name__ == '__main__':
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        print('env_type:', env_type, 'env_id:', env.id)

    env = gym.make('CentipedeFour-v1')
    env.reset()
    for t in range(100):
        print(env.action_space)
        obs, _, _, _ = env.step(env.action_space.sample())
        if t == 0:
            print(obs.shape)
        env.render()



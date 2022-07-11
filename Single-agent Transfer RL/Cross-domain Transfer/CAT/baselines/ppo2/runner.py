import numpy as np
import pickle
from baselines.common.runners import AbstractEnvRunner
import baselines.common.encoder as encoder
import tensorflow as tf


class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, source_env, source_env1, model, nsteps, gamma, lam):
        super().__init__(env=env, source_env=source_env, source_env1=source_env1, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.source_env = source_env
        self.source_env1 = source_env1
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.sumreward = 0
        self.mapper = encoder.mlp(source_env.observation_space.shape[0])
        self.mapper1 = encoder.mlp(source_env1.observation_space.shape[0])
        self.state_scope = 'encoder'
        self.state_scope1 = 'encoder1'
        self.reverse_mapper = encoder.mlp(env.observation_space.shape[0])
        self.reverse_mapper1 = encoder.mlp(env.observation_space.shape[0])
        self.reverse_encoder_scope = 'reverse_encoder'
        self.reverse_encoder_scope1 = 'reverse_encoder1'
        self.action_mapper = encoder.mlp(env.action_space.shape[0])
        self.action_mapper1 = encoder.mlp(env.action_space.shape[0])
        self.action_encoder_scope = 'act_encoder'
        self.action_encoder_scope1 = 'act_encoder1'


    def run(self):
        # Here, we init the lists that will contain the minibatch of experiences
        mb_obs, mb_source_nextobs, mb_target_nextobs, mb_source1_nextobs, mb_target1_nextobs,\
        mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[],[],[],[],[]

        mb_states = self.states
        epinfos = []
        # evaluate source policy
        teacher_reward = np.zeros(2)
        for i in range(100):
            self.source_obs = self.mapper(tf.convert_to_tensor(self.obs), self.state_scope)
            source_actions, _, _, __ = self.model.teacher_model.step(self.source_obs)
            target_action = self.action_mapper(tf.convert_to_tensor(source_actions), self.action_encoder_scope)
            self.obs[:], rewards, self.dones, infos = self.env.step(target_action.eval())
            teacher_reward[0] += rewards

        for i in range(100):
            self.source1_obs = self.mapper1(tf.convert_to_tensor(self.obs), self.state_scope1)
            source_actions, _, _, __ = self.model.teacher_model1.step(self.source1_obs)
            target_action = self.action_mapper1(tf.convert_to_tensor(source_actions), self.action_encoder_scope1)
            self.obs[:], rewards, self.dones, infos = self.env.step(target_action.eval())
            teacher_reward[1] += rewards
        sum = np.exp(teacher_reward[0]) + np.exp(teacher_reward[1])
        teacher_reward[0] = np.exp(teacher_reward[0]) / sum
        teacher_reward[1] = np.exp(teacher_reward[1]) / sum


        # For n in range number of steps
        for i in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            #
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            # self.sumreward += rewards[0]
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos, teacher_reward)

    def run_correction(self):
        # Here, we init the lists that will contain the minibatch of experiences
        mb_source_nextobs, mb_target_nextobs, mb_source1_nextobs, mb_target1_nextobs = [],[],[],[]
        trajectory_index = np.random.choice(176, 1)[0]
        source_state = np.load(
            '.../.../teacher_buffer/centipedeFour/obs_{}.npy'.format(800 + trajectory_index), allow_pickle=True)
        source_action = np.load(
            '.../.../teacher_buffer/centipedeFour/actions_{}.npy'.format(800 + trajectory_index), allow_pickle=True)
        source_state1 = np.load(
            '.../.../teacher_buffer/centipedeSix\obs_{}.npy'.format(800 + trajectory_index),
            allow_pickle=True)
        source_action1 = np.load(
            '.../.../teacher_buffer/centipedeSix/actions_{}.npy'.format(800 + trajectory_index),
            allow_pickle=True)

        # For n in range number of steps
        for i in range(10):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            if i == 0:
                mb_target_nextobs.append(self.obs.copy())
                mb_target1_nextobs.append(self.obs.copy())
                mb_source_nextobs.append(source_state[i])
                mb_source1_nextobs.append(source_state1[i])
            else:
                target_action = self.action_mapper(tf.convert_to_tensor(source_action[i]), self.action_encoder_scope)
                self.obs[:], rewards, self.dones, infos = self.env.step(target_action.eval())
                mb_target_nextobs.append(self.obs.copy())
                #source_mapped_next_state = self.reverse_mapper(tf.convert_to_tensor(source_state[i]), self.reverse_encoder_scope)
                mb_source_nextobs.append(source_state[i])

                target_action1 = self.action_mapper1(tf.convert_to_tensor(source_action1[i]), self.action_encoder_scope1)
                self.obs[:], rewards, self.dones, infos = self.env.step(target_action1.eval())
                mb_target1_nextobs.append(self.obs.copy())
                #source_mapped_next_state1 = self.reverse_mapper1(tf.convert_to_tensor(source_state1[i]), self.reverse_encoder_scope1)
                mb_source1_nextobs.append(source_state1[i])

        mb_source_nextobs = np.asarray(mb_source_nextobs, dtype=self.obs.dtype)
        mb_target_nextobs = np.asarray(mb_target_nextobs, dtype=self.obs.dtype)
        mb_source1_nextobs = np.asarray(mb_source1_nextobs, dtype=self.obs.dtype)
        mb_target1_nextobs = np.asarray(mb_target1_nextobs, dtype=self.obs.dtype)


        return mb_source_nextobs, mb_target_nextobs, mb_source1_nextobs, mb_target1_nextobs

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])



import os
import time
import numpy as np
import tensorflow as tf
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.policies import build_policy
from baselines.common.teacher_policies import build_teacher_policy
from baselines.common.student_policies import build_student_policy
from baselines.common.schedules import LinearSchedule
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.ppo2.runner import Runner


def constfn(val):
    def f(_):
        return val
    return f

def learn(*, network, env, total_timesteps, pi_scope,
          vf_scope, eval_env=None, seed=None, nsteps=2048, ent_coef=0.0,
          lr=3e-4, vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          save_interval=0, load_path=None, model_fn=None, teacher_paths=None,
          save_path=None, log_path=None, teacher_network=None,
          value_network='copy', source_env=None, transfer=None,
          transfer_env=None, mapping=None, ps_coef=None, vf_ps_coef=None,
          mutual_info_coef=None, kl_coef=None, **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''

    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    logger.configure(dir=log_path)
    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    nupdates = total_timesteps//nbatch

    if teacher_paths is None:
        # train a teacher
        policy = build_teacher_policy(env, network, **network_kwargs)
        mode = 'teacher'
    else:
        teacher_networks = [teacher_network] * len(teacher_paths)
        policy = build_student_policy(env, network, teacher_networks,
                                      mapping, source_env=source_env, **network_kwargs)
        mode = 'student'
        teacher_frac = 0.9
        student_ps_coef_schedule = LinearSchedule(int(nupdates * teacher_frac / 2),
                                                  ps_coef, 0.0)
        vf_student_ps_coef_schedule = LinearSchedule(int(nupdates * teacher_frac / 2),
                                                     vf_ps_coef, 0.0)

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.ppo2.model import Model
        model_fn = Model

    if mode == 'teacher':
        # construct static graph for ppo
        model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space,
                         nbatch_act=nenvs, nbatch_train=nbatch_train,
                         nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                         max_grad_norm=max_grad_norm, mode=mode,
                         pi_scope=pi_scope, vf_scope=vf_scope)
    else:
        # construct static graph for ppo with transfer, and also construct an
        # independent version
        model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space,
                         nbatch_act=nenvs, nbatch_train=nbatch_train,
                         nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                         max_grad_norm=max_grad_norm, mode=mode,
                         pi_scope=pi_scope, vf_scope=vf_scope,
                         independent=False, mapping=mapping)
        # don't need to pass mapping into independent version
        independent_model = model_fn(policy=policy, ob_space=ob_space,
                                     ac_space=ac_space, nbatch_act=nenvs,
                                     nbatch_train=nbatch_train, nsteps=nsteps,
                                     ent_coef=ent_coef, vf_coef=vf_coef,
                                     max_grad_norm=max_grad_norm, mode=mode,
                                     pi_scope=pi_scope, vf_scope=vf_scope,
                                     independent=True, trainer=model.trainer,
                                     model=model)

    if load_path is not None:
        # load trained model
        variables = tf.trainable_variables(pi_scope) + \
                    tf.trainable_variables(vf_scope)
        model.load(load_path, variables=variables, transfer=transfer,
                   transfer_env=transfer_env)
    else:
        # load teachers if we're training a student
        if teacher_paths:
            teacher_pi_scopes = ['teacher{}'.format(i)
                                 for i in range(len(teacher_paths))]
            teacher_pi_vars = [tf.trainable_variables(s)
                               for s in teacher_pi_scopes]
            teacher_vf_scopes = ['vf_teacher{}'.format(i)
                                 for i in range(len(teacher_paths))]
            teacher_vf_vars = [tf.trainable_variables(s)
                               for s in teacher_vf_scopes]
            all_vars = [pi_vars + vf_vars for pi_vars, vf_vars in
                        zip(teacher_pi_vars, teacher_vf_vars)]
            print('loading from', teacher_paths)
            model.load_teachers(teacher_paths, all_vars)

    # Instantiate the runner object, which generates rollouts
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    if mode == 'student':
        independent_runner = Runner(env=env, model=independent_model,
                                    nsteps=nsteps, gamma=gamma, lam=lam)
    if eval_env is not None:
        eval_runner = Runner(env=eval_env, model=model, nstep =nsteps,
                             gamma=gamma, lam=lam)
        if mode == 'student':
            # create a runner that will use a student that is independent of its teacher
            independent_eval_runner = Runner(env=eval_env,
                                             model=independent_model,
                                             nsteps=nsteps, gamma=gamma,
                                             lam=lam)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    # Start total timer
    tfirststart = time.time()

    independent = False
    if save_interval and logger.get_dir() and (MPI is None or
                                               MPI.COMM_WORLD.Get_rank() == 0):
        checkdir = save_path
        os.makedirs(checkdir, exist_ok=True)
        savepath = osp.join(checkdir, '%.5i'%0)
        print('Saving to', savepath)
        # not sure this really matters
        if independent:
            independent_model.save(savepath)
        else:
            model.save(savepath)
        # save environment here
        env.save()
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)
        # generate rollouts
        if mode == 'student' and update > int(nupdates * teacher_frac):
            if not independent:
                independent = True
                print('Training independent of teachers')
                # don't mix results from when network depended on teachers
                epinfobuf.clear()
            obs, returns, masks, actions, values, neglogpacs, states, epinfos = independent_runner.run()
        else:
            # teacher or (student and not independent)
            obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
        if eval_env is not None:
            if mode == 'student' and update > int(nupdates * teacher_frac):
                eval_obs, eval_returns, eval_masks, eval_actions, eval_values, \
                eval_neglogpacs, eval_states, eval_epinfos = independent_eval_runner.run()
            else:
                eval_obs, eval_returns, eval_masks, eval_actions, eval_values, \
                eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run()

        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if mode == 'student':
            if update <= int(nupdates * teacher_frac):
                # don't include the coupling loss during first half of training
                if update <= int(nupdates * teacher_frac / 2):
                    student_ps_coef_now = 0.0
                    vf_student_ps_coef_now = 0.0
                    last_update = update
                else:
                    student_ps_coef_now = student_ps_coef_schedule.value(update -
                                                                         last_update - 1)
                    vf_student_ps_coef_now = \
                        vf_student_ps_coef_schedule.value(update - last_update - 1)
            else:
                student_ps_coef_now = None
                vf_student_ps_coef_now = None
        else:
            student_ps_coef_now = None
            vf_student_ps_coef_now = None
        # train on rollouts
        if states is None: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for noptepoch in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    if mode == 'student' and not independent:
                        state_batch_size = nbatch_train
                        train_res = model.train(lrnow, cliprangenow, *slices,
                                                student_ps_coef=student_ps_coef_now,
                                                vf_student_ps_coef=vf_student_ps_coef_now,
                                                mutual_info_coef=mutual_info_coef,
                                                kl_coef=kl_coef)
                        mblossvals.append(train_res)
                    elif mode == 'student':
                        # student and independent
                        train_res = independent_model.train(lrnow, cliprangenow,
                                                            kl_coef=kl_coef, *slices)
                        mblossvals.append(train_res)
                    else:
                        # teacher
                        train_res = model.train(lrnow, cliprangenow, *slices)
                        mblossvals.append(train_res)
        else: # recurrent version, ignore, mujoco doesn't need recurrent policies
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices,
                                                  mbstates, student_ps_coef_now))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.time()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfos]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfos]))
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfos]))
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfos]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals,
                                           independent_model.loss_names if
                                           independent else model.loss_names):
                logger.logkv(lossname, lossval)
            if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and (MPI is None or MPI.COMM_WORLD.Get_rank() == 0):
            # checkdir = osp.join(logger.get_dir(), 'checkpoints')
            checkdir = save_path
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            if mode == 'teacher':
                model.save(savepath)
            else:
                if independent:
                    independent_model.save(savepath)
                else:
                    model.save(savepath)
            env.save()
    return model
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

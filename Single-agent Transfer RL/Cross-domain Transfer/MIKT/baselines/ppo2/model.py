import tensorflow as tf
import functools

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize
import baselines.common.encoder as encoder

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm, mode, pi_scope,
                 vf_scope, microbatch_size=None, independent=False,
                 trainer=None, model=None, mapping=None):
        """
        :type mode: string indicating whether we are training a student or a
        teacher
        :type pi_scope: scope to use for the policy
        :type vf_scope: scope to use for the value function
        :type trainer: trainer (if independent) so we don't have duplicate slots
        or whatever
        :type model: model (if independent) so we don't have placeholders that
        aren't fed values
        """
        self.sess = sess = get_session()
        self.independent = independent

        self.mode = mode
        target_dim = ob_space.shape[0]
        if mode == 'student':
            act_model, _, _, mapped_state_act = policy(pi_scope, vf_scope,
                                                       nbatch_act, 1, sess,
                                                       independent=independent)
            train_model, student_ps, vf_student_ps, mapped_state_train = policy(pi_scope,
                                                                                vf_scope,
                                                                                nbatch_train,
                                                                                nsteps, sess,
                                                                                independent=independent)
            # will be None if independent
            self.mapped_state_act = mapped_state_act
            self.mapped_state_train = mapped_state_train
        else:
            # teacher
            act_model, _, _ = policy(pi_scope, vf_scope, nbatch_act, 1, sess)
            train_model, _, _ = policy(pi_scope, vf_scope, nbatch_train, nsteps, sess)

        # CREATE THE PLACEHOLDERS
        if model is None:
            self.A = A = train_model.pdtype.sample_placeholder([None])
            self.ADV = ADV = tf.placeholder(tf.float32, [None])
            self.R = R = tf.placeholder(tf.float32, [None])
            # Keep track of old actor
            self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
            # Keep track of old critic
            self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
            self.LR = LR = tf.placeholder(tf.float32, [])
            # Cliprange
            self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])
            self.KL_COEF = KL_COEF = tf.placeholder(tf.float32, [])
        else:
            # avoid duplicate placeholders that don't have feed values
            self.A = A = model.A
            self.ADV = ADV = model.ADV
            self.R = R = model.R
            # Keep track of old actor
            self.OLDNEGLOGPAC = OLDNEGLOGPAC = model.OLDNEGLOGPAC
            # Keep track of old critic
            self.OLDVPRED = OLDVPRED = model.OLDVPRED
            self.LR = LR = model.LR
            # Cliprange
            self.CLIPRANGE = CLIPRANGE = model.CLIPRANGE
            self.KL_COEF = KL_COEF = model.KL_COEF

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        if mode == 'student':
            params = tf.trainable_variables(pi_scope) + \
                     tf.trainable_variables(vf_scope)
            if independent:
                loss = loss + KL_COEF * approxkl
            else:
                # encoder isn't really the best name for the variational network
                # might change this later
                variational_net = encoder.mlp(target_dim)
                variational_mu = variational_net(mapped_state_train, 'variational')
                with tf.variable_scope('variational', reuse=tf.AUTO_REUSE):
                    variational_logstd = tf.get_variable(name='variational_logstd',
                                                      shape=[1, target_dim],
                                                      initializer=tf.zeros_initializer())
                encoder_params = tf.trainable_variables('encoder')
                variational_params = tf.trainable_variables('variational')
                params = params + encoder_params + variational_params
                mutual_info_loss = tf.reduce_mean(
                    variational_logstd + ((train_model.X - variational_mu)**2) /
                    (2 * (tf.exp(variational_logstd))**2))
                self.STUDENT_PS_COEF = STUDENT_PS_COEF = tf.placeholder(tf.float32, [])
                self.VF_STUDENT_PS_COEF = VF_STUDENT_PS_COEF = \
                    tf.placeholder(tf.float32, [])
                self.MUTUAL_INFO_COEF = MUTUAL_INFO_COEF = tf.placeholder(tf.float32,
                                                                          [])
                student_ps_mean = tf.reduce_mean(student_ps)
                student_ps_loss = -tf.reduce_mean(tf.log(student_ps))
                vf_student_ps_mean = tf.reduce_mean(vf_student_ps)
                vf_student_ps_loss = -tf.reduce_mean(tf.log(vf_student_ps))
                loss = (loss + STUDENT_PS_COEF * student_ps_loss + 
                        VF_STUDENT_PS_COEF * vf_student_ps_loss + KL_COEF * approxkl +
                        MUTUAL_INFO_COEF * mutual_info_loss)
        else:
            params = tf.trainable_variables(pi_scope) + \
                     tf.trainable_variables(vf_scope)
        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        # 2. Build our trainer
        if trainer is not None:
            self.trainer = trainer
        else:
            if MPI is not None:
                print('Using MpiAdamOptimizer')
                self.trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
            else:
                print('using tf AdamOptimizer')
                self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]
        if mode == 'student' and not independent:
            self.loss_names.append('student_ps_loss')
            self.loss_names.append('student_ps')
            self.loss_names.append('vf_student_ps_loss')
            self.loss_names.append('vf_student_ps')
            self.loss_names.append('mutual_info_loss')
            self.loss_names.append('mutual_info_coef')
            self.loss_names.append('kl_coef')
            self.loss_names.append('student_ps_coef')
            self.loss_names.append('vf_student_ps_coef')
            self.stats_list.append(student_ps_loss)
            self.stats_list.append(student_ps_mean)
            self.stats_list.append(vf_student_ps_loss)
            self.stats_list.append(vf_student_ps_mean)
            self.stats_list.append(mutual_info_loss)
            self.stats_list.append(MUTUAL_INFO_COEF)
            self.stats_list.append(KL_COEF)
            self.stats_list.append(STUDENT_PS_COEF)
            self.stats_list.append(VF_STUDENT_PS_COEF)
        elif mode == 'student' and independent:
            self.loss_names.append('student_ps')
            self.loss_names.append('vf_student_ps')
            self.stats_list.append(tf.constant(1.0))
            self.stats_list.append(tf.constant(1.0))

        def load_teachers(load_paths, variables):
            for p, v in zip(load_paths, variables):
                load_variables(p, v, sess)


        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)
        self.load_teachers = load_teachers

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables) #pylint: disable=E1101

    def train(self, lr, cliprange, obs, returns, masks, actions, values,
              neglogpacs, states=None, student_ps_coef=None,
              vf_student_ps_coef=None, mutual_info_coef=None, kl_coef=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values
        }
        if states is not None:
            # recurrent, i think?
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks
        td_map[self.KL_COEF] = kl_coef
        if student_ps_coef is not None:
            td_map[self.STUDENT_PS_COEF] = student_ps_coef
            td_map[self.VF_STUDENT_PS_COEF] = vf_student_ps_coef
            td_map[self.MUTUAL_INFO_COEF] = mutual_info_coef

        return self.sess.run(
            self.stats_list + [self._train_op], td_map)[:-1]

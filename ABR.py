import tensorflow._api.v2.compat.v1 as tf
import numpy as np
# from helper import show, searchIndex


ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
# NN_MODEL = "/home/team/就是这么皮/submit/results/nn_model_ep_72.ckpt"
# NN_MODEL = "/root/mmgc/team/The World/submit/results/nn_model_ep_225.ckpt"
NN_MODEL = './model/train/nn_model_ep_225.ckpt'
S_DIM = 16
A_DIM = 8
BIT_RATE = [500.0,850.0,1200.0,1850.0]
TARGET_BUFFER = [0.5, 1.0]
B = [0, 1, 2, 3, 0, 1, 2, 3]
T = [0, 0, 0, 0, 1, 1, 1, 1]


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=128,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            l2 = tf.layers.dense(
                inputs=l1,
                units=128,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l2,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int

    def get_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
        return np.argmax(probs[0])


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=128,  # number of hidden units
                activation=tf.nn.relu,  # None
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            l2 = tf.layers.dense(
                inputs=l1,
                units=128,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )

            self.v = tf.layers.dense(
                inputs=l2,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + 0.9 * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r})
        return td_error


def get_time(S_rebuf, S_decision_flag):
    rebuf_time_sum = S_rebuf[-1]
    n = 7499
    for i in range(n-1, -1, -1):
        if S_decision_flag[i]: break
        rebuf_time_sum += S_rebuf[i]
    return rebuf_time_sum


def get_tend(thr_record):
    ll_thr, l_thr, thr = thr_record[-3:]

    if l_thr >= ll_thr:
        if thr >= l_thr:
            return 0.264
        return 0.205
    if l_thr < ll_thr:
        if thr >= l_thr:
            return 0.370
        return 0.738


class Algorithm:
    def __init__(self):
        self.buffer_size = 0
        self.bit_rate = 0
        self.target_buffer = 0
        self.thr_record = np.zeros((4))

    def Initial(self):
        # Initail your session or something
        tf.reset_default_graph()
        with tf.Session().as_default() as sess:
            actor = Actor(sess, n_features=S_DIM, n_actions=A_DIM, lr=ACTOR_LR_RATE)
            critic = Critic(sess, n_features=S_DIM, lr=CRITIC_LR_RATE)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()  # save neural net parameters

            # restore neural net parameters
            if NN_MODEL is not None:  # NN_MODEL is the path to file
                saver.restore(sess, NN_MODEL)
                print("Testing model restored.")

            IntialVars = []
            IntialVars.append(actor)
            IntialVars.append(critic)
            return IntialVars

    def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len,
            S_end_delay, S_decision_flag, S_buffer_flag, S_cdn_flag, end_of_video, cdn_newest_id,
            download_id, cdn_has_frame, IntialVars):
        actor = IntialVars[0]
        critic = IntialVars[1]

        state = np.zeros((S_DIM))
        thr_record = self.thr_record
        last_thr = 0
        index = len(thr_record) - 1
        for i in range(7499, -1, -1):
            if not S_cdn_flag[i] and S_time_interval[i]:
                thr = S_send_data_size[i] / S_time_interval[i] / 1000000
                if abs(thr - last_thr) > 10**-4:
                    last_thr = thr
                    thr_record[index] = thr
                    index -= 1
                    if index < 0: break
        # print(searchIndex(S_decision_flag))

        buffer_size = S_buffer_size[-1]
        target_buffer = self.target_buffer
        buffer_flag = S_buffer_flag[-1]
        cdn_frame = cdn_newest_id - download_id

        last_thr = thr_record[-1]
        thr_mean = sum(thr_record) / len(thr_record)
        thr_tend = get_tend(thr_record)


        end_delay = S_end_delay[-1]
        rebuf_time_sum = get_time(S_rebuf, S_decision_flag)
        bit_rate = self.bit_rate
        next_chunk_size = [0] * 4
        for i in range(4):
            if len(cdn_has_frame[0]) < 25:
                next_chunk_size[i] = sum(cdn_has_frame[i]) + (25 - len(cdn_has_frame[i])) * BIT_RATE[i] * 40
            else:
                next_chunk_size[i] = sum(cdn_has_frame[i][:25])

        state[0] = buffer_size / 4.0
        state[1] = target_buffer / 10.0
        state[2] = buffer_flag / 2.0
        state[3] = max(TARGET_BUFFER[target_buffer] - buffer_size, 0)

        state[4] = cdn_frame / 40.0
        state[5] = next_chunk_size[0] / 10000000.0
        state[6] = next_chunk_size[1] / 10000000.0
        state[7] = next_chunk_size[2] / 10000000.0
        state[8] = next_chunk_size[3] / 10000000.0

        state[9] = last_thr / 10.0
        state[10] = thr_mean / 10.0
        state[11] = thr_tend

        state[12] = end_delay / 2.0
        state[13] = rebuf_time_sum / 10.0
        state[14] = bit_rate / 5.0

        # show(state)

        ac = actor.get_action(state)
        bit_rate = B[ac]
        target_buffer = T[ac]
        self.bit_rate = bit_rate
        self.target_buffer = target_buffer
        return bit_rate, target_buffer



import joblib
import keras

import train_net as ac
from LiveStreamingEnv import fixed_env
from LiveStreamingEnv import load_trace
import os

import numpy as np
import tensorflow._api.v2.compat.v1 as tf


def get_tend(input, model, scaler):
    if input[0] == 0:
        return 1.5
    mydata = [[x] for x in input]
    inputs = scaler.transform(mydata)
    X_test = [inputs]
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predict = model.predict(X_test)

    predict = scaler.inverse_transform(predict)
    return predict[0][0]


# basic parameters
last_bit_rate = 0
bit_rate = 0
target_buffer = 0
latency_limit = 2.0
random_seed = 42

# selection
BIT_RATE = [500.0, 850.0, 1200.0, 1850.0]
TARGET_BUFFER = [0.5, 1.0]
B = [0, 1, 2, 3]
T = [0, 0, 1, 1]
DEBUG = False

# qoe parameters, obviously we should use chunk_reward as our reward
reward_frame = 0
chunk_reward = 0
reward_all = 0
reward_all_sum = 0
frame_time_len = 0.04

# qoe calculation
SMOOTH_PENALTY = 0.02
REBUF_PENALTY = 1.85
LANTENCY_PENALTY = 0.005
SKIP_PENALTY = 0.5

# data settings
NETWORK_TRACES = './train_network/'
VIDEO_TRACES = './video_trace/AsianCup_China_Uzbekistan/frame_trace_'
LOG_FILE_PATH = './log'
MODEL_DIR = './model_with_lstm1/train/'
RESULT = './model_with_lstm1/train.csv'
LSTM_MODEL_DIR = './lstm_model/'
NN_MODEL = None

if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

with open(RESULT, 'a', encoding='utf-8') as f:
    f.write('turn,qoe\n')

all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(NETWORK_TRACES)

# neural network parameters
S_DIM = 10  # number of state
A_DIM = 4  # number of actionb
LR_A = 0.0001  # actor's learning rate
LR_C = 0.001  # critic's learning rate

# model
tf.reset_default_graph()
tf.disable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    actor = ac.Actor(sess, n_features=S_DIM, n_actions=A_DIM, lr=LR_A)
    critic = ac.Critic(sess, n_features=S_DIM, lr=LR_C)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=400)
    nn_model = NN_MODEL
    if nn_model is not None:
        saver.restore(sess, nn_model)
        print("Model restored.")

    # model training, just train 200 turn, avoid overfitting

    # ??????????????????

    model = keras.models.load_model('./lstm_model/lstm.h5')
    scaler = joblib.load('./lstm_model/scaler.save')

    for turn in range(400):
        net_env = fixed_env.Environment(all_cooked_time=all_cooked_time,
                                        all_cooked_bw=all_cooked_bw,
                                        random_seed=random_seed,
                                        logfile_path=LOG_FILE_PATH,
                                        VIDEO_SIZE_FILE=VIDEO_TRACES,
                                        Debug=DEBUG)

        learning_turn = 0
        video_count = 0
        thr_record = np.zeros(4)
        skip_time = []
        rebuf_time = []

        while True:
            reward_frame = 0
            time, time_interval, send_data_size, chunk_len, \
            rebuf, buffer_size, play_time_len, end_delay, \
            cdn_newest_id, download_id, cdn_has_frame, decision_flag, \
            buffer_flag, cdn_flag, end_of_video = net_env.get_video_frame(bit_rate, target_buffer)
            if end_delay <= 1.0:
                LANTENCY_PENALTY = 0.005
            else:
                LANTENCY_PENALTY = 0.01

            if not cdn_flag:
                reward_frame = frame_time_len * float(BIT_RATE[bit_rate]) / 1000 - \
                               REBUF_PENALTY * rebuf - LANTENCY_PENALTY * end_delay
            else:
                reward_frame = -(REBUF_PENALTY * rebuf)

            if not cdn_flag and time_interval:
                thr = send_data_size / time_interval / 1000000
                if abs(thr - thr_record[-1]) > 10 ** -4:
                    thr_record = np.roll(thr_record, -1, axis=0)
                    thr_record[-1] = thr

            rebuf_time.append(rebuf)

            # every time add reward_frame, until next decision_flag
            chunk_reward += reward_frame

            if decision_flag or end_of_video:

                reward_frame += -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
                last_bit_rate = bit_rate

                chunk_reward += reward_frame
                decision_reward = chunk_reward
                chunk_reward = 0

                # get some feature parameters here, which need to calculate
                predict_thr = get_tend(thr_record, model, scaler)
                rebuf_time_sum = sum(rebuf_time)
                rebuf_time = []

                last_thr = thr_record[-1]
                thr_mean = sum(thr_record) / len(thr_record)

                # end

                state = np.zeros(S_DIM)
                state[0] = buffer_size / 4.0
                state[1] = target_buffer / 10.0
                state[2] = buffer_flag / 2.0
                state[3] = max(TARGET_BUFFER[target_buffer] - buffer_size, 0)

                state[4] = last_thr / 10.0
                state[5] = thr_mean / 10.0
                state[6] = predict_thr / 10.0

                state[7] = end_delay / 2.0
                state[8] = rebuf_time_sum / 10.0
                state[9] = bit_rate / 5.0

                # show(state)

                action = actor.choose_action(state)
                bit_rate = B[action]
                target_buffer = T[action]

                if learning_turn > 0:
                    td_error = critic.learn(last_state, decision_reward, state)
                    actor.learn(last_state, last_action, td_error)
                last_state = state.copy()
                last_action = action
                learning_turn += 1

            reward_all += reward_frame
            if end_of_video:
                # just print first 10 turn to see detail effect
                # if the effect is too bad, train it again
                if turn <= 10:
                    print("video count: %d" % video_count, reward_all)
                reward_all_sum += reward_all
                if video_count >= len(all_file_names):
                    save_path = saver.save(sess, MODEL_DIR + "nn_model_ep_" + str(turn) + ".ckpt")
                    reward_all_sum = '%.2f' % (reward_all_sum / video_count)
                    with open(RESULT, 'a', encoding='utf-8') as f:
                        info = str(turn) + ',' + reward_all_sum + '\n'
                        f.write(info)
                    print('turn %d:' % turn, "reward_all_sum is:", reward_all_sum)
                    reward_all_sum = 0
                    break

                # reset the parameters
                reward_all = 0
                video_count += 1
                last_bit_rate = 0
                bit_rate = 0
                target_buffer = 0
                skip_time = []
                rebuf_time = []

import joblib

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow._api.v2.compat.v1 as tf
LSTM_MODEL_DIR = './lstm_model/'
tf.disable_eager_execution()
# 导入所需的包
file = "lstm_train.csv"
# 导入数据
data = pd.read_csv(file)
df = DataFrame(data)
new_data = pd.DataFrame(index=range(len(df)), columns=['time', 'through'])

for i in range(0, len(df)):
    new_data['time'][i] = df['time'][i]
    new_data['through'][i] = df['throughput'][i]
new_data.index = new_data.time
new_data.drop('time', axis=1, inplace=True)

dataset = new_data.values

train = dataset[0:4001]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

test = 4
x_train, y_train = [], []
for i in range(test, len(train)):
    x_train.append(scaled_data[i - test:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()  # 顺序模型，核心操作是添加layer（图层）
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))  # 全连接层

model.compile(loss='mean_squared_error', optimizer='adam')  # 选择优化器，并指定损失函数
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

model.save(LSTM_MODEL_DIR + 'lstm.h5')
joblib.dump(scaler, LSTM_MODEL_DIR + 'scaler.save')
ABR.py
    新的ABR算法，采用加载了LSTM的AC算法
        NN_MODEL = './model_with_lstm/train/nn_model_ep_180.ckpt'
         =>指定所以模型
ABR_.py
    默认的ABR算法
data_analysis.ipynb
    用于分析运行online所记录的码率，延时等数据
    file_path
        =>指定要分析数据所在的文件夹
lstm_train.csv
    用于训练LSTM模型的数据，原network_trace的0
network.ipynb
    分析网络流量并进行预测
online.py
    Data_Path = './data_lstm_ac/'
        =>指定保存运行收集数据的文件夹
    save_data = True
        =>是否要保存数据
    origin_abr = False
        =>是否启用默认算法
train.py
    训练模型
    NETWORK_TRACES = './train_network/'
        =>指定训练集目录
    VIDEO_TRACES = './video_trace/AsianCup_China_Uzbekistan/frame_trace_'
        =>指定视频
    LOG_FILE_PATH = './log'
        =>指定log文件夹
    MODEL_DIR = './model_with_lstm/train/'
        =>训练完成的AC模型保存位置
    RESULT = './model_with_lstm/train.csv'
        =>每轮训练完成的AC模型qoe的保存位置
    LSTM_MODEL_DIR = './lstm_model/'
        =>训练完成的LSTM模型保存位置
train_net.py
    AC模型代码
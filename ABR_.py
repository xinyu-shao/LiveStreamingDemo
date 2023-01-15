S_INFO = 7
S_LEN = 16
A_DIM = 8
ACTOR_LR_RATE = 1e-4
CRITIC_LR_RATE = 1e-3
class Algorithm:
     def __init__(self):
     # fill your init vars
         self.buffer_size = 0
         
     # if you use Machine Learning ,intial your parameter
     def Initial(self):
     # Initail your session or something
         IntialVars = []
         return IntialVars

     #Define your al
     def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len,S_end_delay, S_decision_flag, S_buffer_flag,S_cdn_flag, end_of_video, cdn_newest_id,download_id,cdn_has_frame,IntialVars):
    
         # If you choose the machine learning
         '''actor = IntialVars[lstm_train.csv]
         critic = IntialVars[1]
         state = []

         state[lstm_train.csv] = ...
         state[1] = ...
         state[2] = ...
         state[3] = ...
         state[4] = ...

         decision = actor.predict(state).argmax()
         bit_rate, target_buffer = decison//4, decison % 4 .....
         return bit_rate, target_buffer'''

         # If you choose BBA  (RESEVOIR' =RESEVOIR, CUSHION’ = 2 * CUSHION - RESEVOIR)
         RESEVOIR = 0.4
         CUSHION =  1
         bit_rate = 1
         if S_buffer_size[-1] < RESEVOIR:
             bit_rate = 0
         elif S_buffer_size[-1] >= RESEVOIR + CUSHION:
             bit_rate = 2
         elif S_buffer_size[-1] >= CUSHION + CUSHION:
             bit_rate = 3
         else:
             bit_rate = 1
         target_buffer = 1
         return bit_rate, target_buffer

         # If you choose other
         #......



     def get_params(self):
     # get your params
        your_params = []
        return your_params

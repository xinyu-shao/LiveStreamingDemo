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

def show(state):
    for i in range(len(state)):
        print(state[i])
    print()

def searchIndex(S_decision_flag):
    n = 7498
    for i in range(n, -1, -1):
        if S_decision_flag[i]:
            return 7499 - i
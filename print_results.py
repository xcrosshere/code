# Copyright (C) 2021-2023 Ye Xu, Xin Zhang, Chongpeng Huang, Xiaorong Qiu

import numpy as np
import os, joblib
import settings

def print_results(data_path):
    data = joblib.load(data_path)
    if 'cnn' not in data_path:
        print('accuracies=', data['accus'])
        print('std_accu=', np.std(np.array(data['accus'])))
        print('mean_accu=', np.max(np.mean(data['accus'], axis=0)))
    if 'cnn' in data_path:
        print('accuracies=', data['accus'])
        print('mean_accu=', np.mean(accus))
        print('std_accu=', np.std(accus))


print_results(r'/home/x/桌面/code/data/intermediate_data/obtain_cls_scores/obtain_scores_for_df/15-Scenes/hv/mnresnext50_fscp_trainsize100_dictsize8_gs2048_nc-1_st3.pkl')

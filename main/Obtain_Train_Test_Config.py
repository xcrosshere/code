# Copyright (C) 2021-2023 Ye Xu, Xin Zhang, Chongpeng Huang, Xiaorong Qiu

from sklearn.model_selection import train_test_split
from Obtain_Dataset_Info import obtain_dataset_info_by_name
import os, joblib, random

def obtain_train_test_config(dsname, times, split_setup, data_path, reset=False):
    saved_file = '%s/%s/%s_ts%d_%d.pkl' % (data_path, 'intermediate_data/split_datasets', dsname, split_setup['train_size'], times)
    if os.path.exists(saved_file) and not reset:
        splitting_ret = joblib.load(saved_file)
    else:
        x, y, y_labels, cnum = obtain_dataset_info_by_name(dsname, data_path=data_path)
        splitting_ret = {'y_labels': y_labels, 'num_class': cnum, 'split_setup': split_setup,
                         'split_times': times, 'data':{}}

        for t in range(times):
            dt = {}
            train_x = []; train_y = []
            test_x = []; test_y = []
            for c in range(cnum):
                c_x = [x[i] for i in range(len(x)) if y[i] == c]
                c_y = [y[i] for i in range(len(x)) if y[i] == c]
                c_num = len(c_x)

                rand_indices = list(range(c_num)); random.shuffle(rand_indices)
                if split_setup['train_size'] < 1:
                    end_train_index = round(c_num * split_setup['train_size'])
                else:
                    end_train_index = split_setup['train_size']

                c_train_x = [c_x[i] for i in rand_indices[0:end_train_index]]
                c_train_y = [c_y[i] for i in rand_indices[0:end_train_index]]
                if split_setup['test_size'] == None or split_setup['test_size'] + end_train_index > c_num:
                    c_test_x = [c_x[i] for i in rand_indices[end_train_index:]]
                    c_test_y = [c_y[i] for i in rand_indices[end_train_index:]]
                else:
                    c_test_x = [c_x[i] for i in rand_indices[end_train_index:end_train_index + split_setup['test_size']]]
                    c_test_y = [c_y[i] for i in rand_indices[end_train_index:end_train_index + split_setup['test_size']]]
                train_x.extend(c_train_x); test_x.extend(c_test_x)
                train_y.extend(c_train_y); test_y.extend(c_test_y)
            dt['train_x'] = train_x; dt['test_x'] = test_x; dt['train_y'] = train_y; dt['test_y'] = test_y
            splitting_ret['data'][t] = dt
        joblib.dump(splitting_ret, saved_file, compress=True)
    return splitting_ret

if __name__ == '__main__':
    splitting_ret = obtain_train_test_config('Caltech-256', 10, {'train_size': 60, 'test_size': 20}, reset=False)
    print(len(splitting_ret['data'][1]))
    print(len(splitting_ret['data'][1]['train_x']))
    print(len(splitting_ret['data'][1]['test_y']))
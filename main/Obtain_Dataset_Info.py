# Copyright (C) 2021-2023 Ye Xu, Xin Zhang, Chongpeng Huang, Xiaorong Qiu

import os

def obtain_dataset_info_by_name(dsname, data_path):
    ori_root_path = os.getcwd()
    os.chdir(data_path + "/datasets")
    x = []
    y = []
    y_labels = []
    label_index = 0
    for classname in os.listdir(dsname):
        y_labels.append(classname)
        for filename in os.listdir(dsname + '/' + classname):
            x.append(classname + '/' + filename)
            y.append(label_index)
        label_index += 1
    os.chdir(ori_root_path)
    return x, y, y_labels, len(y_labels)

if __name__ == '__main__':
    x, y, y_labels, cnum = obtain_dataset_info_by_name('15-Scenes')
    print(x[0:10]); print(y[0:10]); print(y_labels[0:10]); print(cnum)
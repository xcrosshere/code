# Copyright (C) 2021-2023 Ye Xu, Xin Zhang, Chongpeng Huang, Xiaorong Qiu

import numpy as np
import joblib, os, time
from Obtain_Feas import *
from Obtain_Train_Test_Config import obtain_train_test_config
from sklearn.cluster import MiniBatchKMeans
from sklearn import mixture
import torch
import torch.optim as optim

def obtain_dict_for_CPFC(dsname, group_size, dict_size, dict_method, fea_source, params_of, params_ottc, data_path, norm=False, reset=False):
    if fea_source == 'cp':
        feas_template_path = obtain_deep_features_by_cp(dsname, params_of['model_name'], params_of['target_layer'],
                                                        params_ottc, reset=params_of['reset'], skip=params_of['skip'], data_path=data_path)
    elif fea_source == 'fc':
        feas_template_path = obtain_deep_features_by_fc(dsname, params_of['model_name'], params_of['target_layer'],
                                                        params_of['sizes'], params_of['steps'],
                                                        params_ottc, reset=params_of['reset'], skip=params_of['skip'], data_path=data_path)
    tt_config = obtain_train_test_config(dsname, times=params_ottc['times'], split_setup=params_ottc['split_setup'],
                                         reset=params_ottc['reset'], data_path=data_path)

    saved_dir = data_path + '/intermediate_data/obtain_dicts/obtain_dict_for_cpfc/' + fea_source + '/st' + str(
        tt_config['split_times'])

    saved_path_nost = '%s/dict_%s_dm%s_ds%d_ts%d_gs%d_norm%d_mn%s' % (
        saved_dir, dsname, dict_method, dict_size, params_ottc['split_setup']['train_size'], group_size, norm,
        params_of['model_name'])

    for st in range(tt_config['split_times']):
        saved_path = saved_path_nost + '_st' + str(st) + '.pkl'
        if not os.path.exists(saved_path) or reset:
            config_info = tt_config['data'][st]
            train_files = config_info['train_x']

            if dict_method == 'GMM':
                if len(train_files) > 500:
                    selected = np.random.choice(list(range(len(train_files))), 500, replace=False)
                    train_files = [train_files[i] for i in selected]

            feas_all = []
            progress = 0
            for filename in train_files:
                print('obtain_dict_for_CPFC---->%d(dict_size=%d, dict_method=%s, group_size=%d):%s(%.3f)' % (
                    st, dict_size, dict_method, group_size, filename, progress / len(train_files)))
                if feas_template_path.find("refined") != -1:
                    feas_path = feas_template_path % (st, os.path.splitext(filename)[0])
                else:
                    feas_path = feas_template_path % os.path.splitext(filename)[0]
                feas = joblib.load(feas_path)
                if fea_source == 'cp':
                    fs = feas[params_of['target_layer']]
                    fs = np.reshape(fs, (fs.shape[0], -1))
                elif fea_source == 'fc':
                    fs = np.hstack(feas['feas'])
                feas_all.append(fs)
                progress += 1

            feas_all = np.hstack(feas_all).T
            num_feas_set = int(feas_all.shape[1] / group_size)
            feas_set = np.hsplit(feas_all, num_feas_set)
            dicts = []
            for i in range(num_feas_set):
                feas = feas_set[i]
                if norm:
                    norm_root = np.sqrt(np.sum(feas ** 2, axis=1))
                    norm_root[np.isnan(norm_root)] = 0
                    feas /= np.reshape(norm_root, (-1, 1))

                if dict_method == 'Kmeans':
                    kmeans = MiniBatchKMeans(n_clusters=dict_size, init='random')
                    feas[np.isnan(feas)] = 0
                    kmeans.fit(feas)
                    dic = kmeans.cluster_centers_.T
                    dic /= np.sqrt(np.sum(dic**2, 0))
                elif dict_method == 'GMM':
                    dic = mixture.GaussianMixture(n_components=dict_size, covariance_type='diag', max_iter=100, init_params='random')
                    dic.fit(feas)
                dicts.append(dic)

            if not os.path.exists(saved_dir):
                os.makedirs(saved_dir)
            joblib.dump(dicts, saved_path, compress=True)

    return saved_path_nost + '_st%d.pkl'

if __name__ == '__main__':
    params_obtain_tt_config_15Scenes = {'split_setup': {'train_size': 100, 'test_size': 100}, 'times': 5,
                                        'reset': False}

    pms_offc = dict(target_layer='avgpool', sizes=[96, 128, 160, 192, 224, 256], steps=[16, 16, 16, 16, 16, 16],
                    reset=False, skip=False)
    pms_dtree = dict(reset=False)

    pms_gen = dict(reset=False, h_num=6, w_num=6, ratio=0.2)
    params_of = dict(pms_offc=pms_offc, pms_gen=pms_gen, pms_dtree=pms_dtree, model_name='resnet50')


    # params_of = dict(model_name='resnet50',
    #                  target_layers=['layer4.0.relu', 'layer4.0.downsample.1', 'layer4.1.relu', 'layer4.2.relu'],
    #                  fdv_target_layers=['layer4.0.relu', 'layer4.1.relu', 'layer4.2.relu'], fdv_group_num=8,
    #                  reset_df=False, reset_fdv=False, skip=False)

    # obtain_dict_for_fdv('15-Scenes', 2048, 'Kmeans', params_of, params_obtain_tt_config_15Scenes, norm=True,
    #                     data_path='../../data',
    #                     reset=False)

    # s_time = time.time()
    # params_obtain_feas = dict(patch_sizes=[16, 24, 32], patch_steps=[8, 8, 8], reset=False, skip=True)
    # params_obtain_tt_config_TFFlowers = dict(times=10, split_setup=dict(train_size=0.7, test_size=None), reset=False)
    # dicts_path = obtain_dict_for_sift('TF-Flowers', 2048, 'Kmeans', params_obtain_feas, params_obtain_tt_config_TFFlowers, reset=False)
    # params_obtain_tt_config_NWPU = dict(times=10, split_setup=dict(train_size=150, test_size=60), reset=False)
    # dicts_path = obtain_dict_for_sift('NWPU', 2048, 'Kmeans', params_obtain_feas, params_obtain_tt_config_NWPU, reset=False)
    # params_obtain_tt_config_MITIndoor67 = dict(times=10, split_setup=dict(train_size=80, test_size=20), reset=False)
    # dicts_path = obtain_dict_for_sift('MIT Indoor-67', 256, 'Kmeans', params_obtain_feas, params_obtain_tt_config_MITIndoor67, reset=False)
    # dicts_path = obtain_dict_for_sift('MIT Indoor-67', 128, 'GMM', params_obtain_feas, params_obtain_tt_config_MITIndoor67, reset=False)
    #
    # params_obtain_tt_config_NWPU = dict(times=10, split_setup=dict(train_size=150, test_size=60), reset=False)
    # # dicts_path = obtain_dict_for_sift('NWPU', 256, 'Kmeans', params_obtain_feas, params_obtain_tt_config_NWPU, reset=False)
    # # dicts_path = obtain_dict_for_sift('NWPU', 128, 'GMM', params_obtain_feas, params_obtain_tt_config_NWPU, reset=False)
    # dicts_path = obtain_dict_for_sift('NWPU', 4096, 'Kmeans', params_obtain_feas, params_obtain_tt_config_NWPU,
    #                                   reset=False)
    #
    # params_obtain_tt_config_Caltech256 = dict(times=10, split_setup=dict(train_size=60, test_size=20), reset=False)
    # dicts_path = obtain_dict_for_sift('Caltech-256', 256, 'Kmeans', params_obtain_feas, params_obtain_tt_config_Caltech256, reset=False)
    # dicts_path = obtain_dict_for_sift('Caltech-256', 4096, 'Kmeans', params_obtain_feas,
    #                                   params_obtain_tt_config_Caltech256, reset=False)

    # params_obtain_tt_config_15Scenes = dict(times=10, split_setup=dict(train_size=100, test_size=None), reset=False)
    # dicts_path = obtain_dict_for_sift('15-Scenes', 256, 'Kmeans', params_obtain_feas, params_obtain_tt_config_15Scenes, reset=False)
    # dicts_path = obtain_dict_for_sift('15-Scenes', 4096, 'Kmeans', params_obtain_feas, params_obtain_tt_config_15Scenes, reset=False)
    #
    # params_obtain_tt_config_TFFlowers = dict(times=10, split_setup=dict(train_size=0.7, test_size=None), reset=False)
    # dicts_path = obtain_dict_for_sift('TF-Flowers', 256, 'Kmeans', params_obtain_feas, params_obtain_tt_config_TFFlowers, reset=False)
    # dicts_path = obtain_dict_for_sift('TF-Flowers', 128, 'GMM', params_obtain_feas, params_obtain_tt_config_TFFlowers, reset=False)
    # dicts_path = obtain_dict_for_sift('TF-Flowers', 4096, 'Kmeans', params_obtain_feas, params_obtain_tt_config_TFFlowers, reset=False)
    #
    # params_obtain_feas = dict(patch_sizes=[16, 24, 32], patch_steps=[8, 8, 8], reset=False, skip=False)
    # params_obtain_tt_config_COVID19 = dict(times=10, split_setup=dict(train_size=1000, test_size=300), reset=False)
    # dicts_path = obtain_dict_for_sift('COVID-19', 256, 'Kmeans', params_obtain_feas, params_obtain_tt_config_COVID19, reset=False)
    # dicts_path = obtain_dict_for_sift('COVID-19', 128, 'GMM', params_obtain_feas, params_obtain_tt_config_COVID19, reset=False)
    # dicts_path = obtain_dict_for_sift('COVID-19', 4096, 'Kmeans', params_obtain_feas, params_obtain_tt_config_COVID19, reset=False)
    #
    # # print(dicts_path % 3)
    # params_of = {}
    # params_of['model_name'] = 'resnet50'
    # params_of['target_layer'] = 'layer4.2.relu'
    # params_of['reset'] = False
    # params_of['skip'] = True
    # params_obtain_tt_config_TESTDS = dict(times=2, split_setup=dict(train_size=80, test_size=20), reset=False)
    # obtain_dict_for_CPFC('test_ds', 256, 64, 'Kmeans', 'cp', params_of, params_obtain_tt_config_TESTDS,
    #                      norm=True, reset=True)

    # params_of['model_name'] = 'vgg16bn'
    # params_of['target_layer'] = 'classifier.5'
    # params_of['sizes'] = [128, 160, 192]
    # params_of['steps'] = [32, 32, 32]
    # obtain_dict_for_CPFC('test_ds', 256, 8, 'GMM', 'classifier.5', 'fc', params_of, params_obtain_tt_config_TESTDS, norm=True, reset=False)

    # print('running time:%.5f' % (time.time() - s_time))

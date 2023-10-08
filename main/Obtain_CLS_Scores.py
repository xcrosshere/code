# Copyright (C) 2021-2023 Ye Xu, Xin Zhang, Chongpeng Huang, Xiaorong Qiu

from Obtain_Train_Test_Config import obtain_train_test_config
from Obtain_Image_Vectors import *
import numpy as np
import os, joblib, random, sys
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import *

def obtain_cls_scores_for_CPFC(dsname, classifier_config, params_oiv, params_odfdf, params_of, params_ottc,
                               data_path, reset=False, return_data=False):
    tt_config = obtain_train_test_config(dsname, times=params_ottc['times'], split_setup=params_ottc['split_setup'],
                                         reset=params_ottc['reset'], data_path=data_path)
    coding_method = params_oiv['coding_config']['coding_method']
    saved_path = data_path + '/intermediate_data/obtain_cls_scores/obtain_scores_for_df/' + dsname + '/' + coding_method
    saved_scores_path = saved_path + '/mn%s_fs%s_trainsize%d_dictsize%d_gs%d_nc%d_st%d.pkl' % (
        params_of['model_name'], params_of['fea_source'],
        params_ottc['split_setup']['train_size'], params_odfdf['dict_size'], params_of['group_size'],
        params_oiv['num_comp'],
        tt_config['split_times'])

    print(saved_scores_path)
    if not os.path.exists(saved_scores_path) or reset or return_data:
        img_vector_template_path = obtain_img_vectors_for_CPFC(dsname, params_oiv['num_comp'],
                                                               params_oiv['coding_config'], params_oiv['spr_config'],
                                                               params_odfdf, params_of, params_ottc,
                                                               reset=params_oiv['reset'], skip=params_oiv['skip'], data_path=data_path)
        return calculate_cls_score(classifier_config, img_vector_template_path, saved_scores_path, tt_config)
    return joblib.load(saved_scores_path)

def calculate_cls_score(classfier_config, img_vector_template_path, saved_scores_path, tt_config):
    accus = []
    cms = []
    crs = []
    recalls = []
    precs = []
    f1s = []

    train_X_list = []
    train_y_list = []
    test_X_list = []
    test_y_list = []
    for st in range(tt_config['split_times']):
    # for st in [2]:
        if classfier_config['type'] == 'poly':
            svm_clf = SVC(kernel='poly', degree=1, coef0=100, C=0.5)
        elif classfier_config['type'] == 'rbf':
            svm_clf = SVC(kernel='rbf', gamma=0.1, C=10)
        elif classfier_config['type'] == 'linear':
            svm_clf = LinearSVC(C=1)

        train_files = tt_config['data'][st]['train_x']
        train_y = tt_config['data'][st]['train_y']
        test_files = tt_config['data'][st]['test_x']
        test_y = tt_config['data'][st]['test_y']
        train_X = []
        test_X = []

        progress = 0
        for f in train_files:
            print(
                'obtain_cls_scores--->%d:%s(%.3f)' % (st, f, progress / (len(train_files) + len(test_files))))
            file_path = img_vector_template_path % (st, os.path.splitext(f)[0])
            img_vector = joblib.load(file_path)
            train_X.append(img_vector)

            progress += 1

        for f in test_files:
            print(
                'obtain_cls_scores--->%d:%s(%.3f)' % (st, f, progress / (len(train_files) + len(test_files))))
            file_path = img_vector_template_path % (st, os.path.splitext(f)[0])
            img_vector = joblib.load(file_path)
            test_X.append(img_vector)

            progress += 1

        train_X = np.vstack(train_X)
        test_X = np.vstack(test_X)
        train_X[np.isnan(train_X)] = 0
        test_X[np.isnan(test_X)] = 0
        svm_clf.fit(train_X, train_y)
        predicted_y = svm_clf.predict(test_X)

        accus.append(accuracy_score(test_y, predicted_y))
        recalls.append(recall_score(test_y, predicted_y, average='weighted'))
        precs.append(precision_score(test_y, predicted_y, average='weighted'))
        f1s.append(f1_score(test_y, predicted_y, average='weighted'))
        cms.append(confusion_matrix(test_y, predicted_y))
        crs.append(classification_report(test_y, predicted_y))

        print('accus=', accus)

        train_X_list.append(train_X)
        train_y_list.append(train_y)
        test_X_list.append(test_X)
        test_y_list.append(test_y)

        progress = 0
        for f in train_files + test_files:
            print(
                'delete--->%d:%s(%.3f)' % (st, f, progress / (len(train_files) + len(test_files))))
            file_path = img_vector_template_path % (st, os.path.splitext(f)[0])
            try:
                os.remove(file_path)
            except:
                pass
            progress += 1

    saved_scores_dir = os.path.dirname(saved_scores_path)
    if not os.path.exists(saved_scores_dir):
        os.makedirs(saved_scores_dir)
    joblib.dump(dict(avg_accu=np.mean(accus), accus=accus, cms=cms, crs=crs, recalls=recalls, precs=precs, f1s=f1s),
                saved_scores_path, compress=True)
    print("------------------finished--------------------")
    return train_X_list, train_y_list, test_X_list, test_y_list
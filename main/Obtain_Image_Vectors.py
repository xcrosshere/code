# Copyright (C) 2021-2023 Ye Xu, Xin Zhang, Chongpeng Huang, Xiaorong Qiu

import multiprocessing as mp

import joblib
import numpy as np
import os
import shutil
from scipy.spatial.distance import cdist

from Obtain_Dicts import *

def obtain_img_vectors_for_CPFC(dsname, num_comp, coding_config, spr_config, param_odfdf, params_of, params_ottc,
                                data_path, reset=False, skip=False):
    fea_source = params_of['fea_source']

    if fea_source == 'cp':
        feas_template_path = obtain_deep_features_by_cp(dsname, params_of['model_name'], params_of['target_layer'],
                                                        params_ottc, reset=params_of['reset'], skip=params_of['skip'], data_path=data_path)
    elif fea_source == 'fc':
        feas_template_path = obtain_deep_features_by_fc(dsname, params_of['model_name'], params_of['target_layer'],
                                                        params_of['sizes'], params_of['steps'],
                                                        params_ottc, reset=params_of['reset'], skip=params_of['skip'], data_path=data_path)
    tt_config = obtain_train_test_config(dsname, times=params_ottc['times'], split_setup=params_ottc['split_setup'],
                                         reset=params_ottc['reset'], data_path=data_path)

    group_size = params_of['group_size']
    if coding_config['do_coding']:
        dict_template_path = obtain_dict_for_CPFC(dsname, group_size, param_odfdf['dict_size'],
                                                  param_odfdf['dict_method'], fea_source, params_of, params_ottc, data_path,
                                                  param_odfdf['norm'], reset=param_odfdf['reset'])
    coding_method = coding_config['coding_method']

    saved_dir = data_path + '/intermediate_data/obtain_sprs/obtain_sprs_for_CPFC/' + dsname + '/train_size(' + str(
        params_ottc['split_setup']['train_size']) + ')/st' + str(tt_config['split_times'])

    if not skip:
        for st in range(tt_config['split_times']):
            saved_root_dir = saved_dir + '/' + str(st) + '/fs_' + fea_source + '/mn' + params_of[
                'model_name'] + '/gs_' + str(
                group_size) + '/' + coding_method + '/' + str(param_odfdf['dict_size']) + '/n_comp_' + str(num_comp)

            if reset and os.path.exists(saved_root_dir):
                shutil.rmtree(saved_root_dir)

            train_files = tt_config['data'][st]['train_x']
            test_files = tt_config['data'][st]['test_x']
            all_files = train_files + test_files

            if coding_config['do_coding']:
                dict_path = dict_template_path % st
                dicts = joblib.load(dict_path)
            else:
                dicts = None

            # for c in range(2):
            for c in [1]:
                if c == 0:
                    pool = mp.Pool(mp.cpu_count())
                print('---------------->generate sprs for training files')
                for idx, f in zip(range(len(train_files)), train_files):
                    if c == 0:
                        pool.apply_async(gen_sprs_for_CPFC, args=(
                            f, dicts, feas_template_path, fea_source, group_size, coding_config, params_of, param_odfdf,
                            spr_config, saved_root_dir, st, idx, len(train_files)))
                    else:
                        gen_sprs_for_CPFC(f, dicts, feas_template_path, fea_source, group_size, coding_config, params_of,
                                          param_odfdf, spr_config, saved_root_dir, st, idx, len(train_files))

                print('---------------->generate sprs for testing files')
                for idx, f in zip(range(len(test_files)), test_files):
                    if c == 0:
                        pool.apply_async(gen_sprs_for_CPFC, args=(
                            f, dicts, feas_template_path, fea_source, group_size, coding_config, params_of, param_odfdf,
                            spr_config, saved_root_dir, st, idx, len(test_files)))
                    else:
                        gen_sprs_for_CPFC(f, dicts, feas_template_path, fea_source, group_size, coding_config, params_of,
                                          param_odfdf, spr_config, saved_root_dir, st, idx, len(test_files))
                if c == 0:
                    pool.close()
                    pool.join()
                    pool.terminate()

            saved_weight_path = saved_root_dir + '/weights.pkl'
            sprs_template_path = saved_root_dir + '/%s' + '_sprs.pkl'
            if not os.path.exists(saved_weight_path):
                num_blocks = np.dot(np.array(spr_config['h_sp']), np.array(spr_config['w_sp']))
                if num_comp != -1:
                    train_sprs = None
                    for f in tt_config['data'][st]['train_x']:
                        sprs_list = joblib.load(sprs_template_path % os.path.splitext(f)[0])
                        for i in range(len(sprs_list)):
                            if train_sprs == None:
                                train_sprs = [[] for i in range(len(sprs_list))]
                            train_sprs[i].append(np.expand_dims(np.vstack(sprs_list[i]), 0))
                    weights_list = learn_compact_weights(train_sprs, tt_config['data'][st]['train_y'], tt_config['num_class'],
                                                         num_comp, num_blocks)
                else:
                    weights_list = []
                    weights_list.append(np.eye(num_blocks))
                joblib.dump(weights_list, saved_weight_path, compress=True)
            weights_list = joblib.load(saved_weight_path)

            progress = 0
            for f in all_files:
                saved_path = saved_root_dir + '/' + os.path.splitext(f)[0] + '.pkl'

                if not os.path.exists(saved_path):
                    print('obtain_img_vector_for_df---->%s(%.3f)' % (saved_path, progress / len(all_files)))

                    sprs_path = sprs_template_path % os.path.splitext(f)[0]
                    sprs_list = joblib.load(sprs_path); os.remove(sprs_path)

                    img_vector = []
                    for i in range(len(sprs_list)):
                        if len(weights_list) == 1:
                            img_sub_vector = np.dot(np.vstack(sprs_list[i]).T, weights_list[0]).flatten()
                        else:
                            img_sub_vector = np.dot(np.vstack(sprs_list[i]).T, weights_list[i]).flatten()
                        img_vector.append(img_sub_vector)
                    img_vector = np.hstack(img_vector)
                    if coding_method != 'fv':
                        img_vector = np.sqrt(np.abs(img_vector)) * np.sign(img_vector)
                        img_vector = img_vector / np.linalg.norm(img_vector)

                    saved_fv_dir = os.path.dirname(saved_path)
                    if not os.path.exists(saved_fv_dir):
                        os.makedirs(saved_fv_dir)
                    joblib.dump(img_vector.astype(np.float16), saved_path, compress=True)

                    progress += 1

    return saved_dir + '/%d/fs_' + fea_source + '/mn' + params_of['model_name'] + '/gs_' + str(
        group_size) + '/' + coding_method + '/' + str(
        param_odfdf['dict_size']) + '/n_comp_' + str(num_comp) + '/%s.pkl'

def gen_sprs_for_CPFC(f, dicts, feas_template_path, fea_source, group_size, coding_config, params_of, param_odfdf,
                      spr_config, saved_root_dir, st, idx, amount):
    coding_method = coding_config['coding_method']
    saved_sprs_path = saved_root_dir + '/' + os.path.splitext(f)[0] + '_sprs.pkl'
    saved_iv_path = saved_root_dir + '/' + os.path.splitext(f)[0] + '.pkl'
    if (not (os.path.exists(saved_sprs_path) or os.path.exists(saved_iv_path))) or (os.path.exists(saved_sprs_path) and os.path.getsize(saved_sprs_path) == 0):
        print('obtain_sprs_for_df--->%s(%.3f)' % (saved_sprs_path, idx / amount))
        if feas_template_path.find('refined') != -1:
            feas_path = feas_template_path % (st, os.path.splitext(f)[0])
        else:
            feas_path = feas_template_path % os.path.splitext(f)[0]

        feas = joblib.load(feas_path)
        if fea_source == 'cp':
            fs = feas[params_of['target_layer']]
            feas = np.reshape(fs, (fs.shape[0], -1))
            posis_0, posis_1 = np.meshgrid(range(fs.shape[1]), range(fs.shape[2]))
            posis_0 = posis_0.flatten()
            posis_1 = posis_1.flatten()
            posis_0 = posis_0 / np.max(posis_0)
            posis_1 = posis_1 / np.max(posis_1)
            posis = np.vstack((posis_0, posis_1))
        elif fea_source == 'fc':
            posis = [feas['patch_locs'][i]['pos'] for i in range(len(feas['feas']))]
            posis = np.hstack(posis)
            feas = np.hstack(feas['feas'])

        num_feas_set = int(feas.shape[0] / group_size)
        feas_set = np.hsplit(feas.T, num_feas_set)

        sprs_list = []
        for i in range(num_feas_set):
            sub_feas = feas_set[i]
            if param_odfdf['norm']:
                norm_root = np.sqrt(np.sum(sub_feas ** 2, axis=1))
                norm_root[np.isnan(norm_root)] = 0
                sub_feas /= np.reshape(norm_root, (-1, 1))

            fake_sift_feas = {}
            fake_sift_feas['sifts'] = sub_feas.T
            fake_sift_feas['posis'] = posis
            if coding_config['do_coding']:
                sprs = gen_sprs(fake_sift_feas, coding_config, coding_method, dicts[i], os.path.splitext(f)[0],
                                param_odfdf, saved_root_dir, spr_config, i)
            else:
                sprs = gen_sprs_for_other(fake_sift_feas['sifts'], fake_sift_feas['posis'], spr_config)
            sprs_list.append(sprs)

        saved_sprs_dir = os.path.dirname(saved_sprs_path)
        if not os.path.exists(saved_sprs_dir):
            os.makedirs(saved_sprs_dir)

        joblib.dump(sprs_list, saved_sprs_path, compress=True)

def gen_sprs(feas, coding_config, coding_method, dict, filename, params_odfs, saved_root_dir, spr_config, idx=None):
    if coding_method != 'fv':
        if idx == None:
            saved_path_kc = saved_root_dir + '/' + str(coding_config['K']) + '_clusters/' + str(
                params_odfs['dict_size']) + '/' + filename + '.pkl'
        else:
            saved_path_kc = saved_root_dir + '/' + str(coding_config['K']) + '_clusters/' + str(
                params_odfs['dict_size']) + '/' + filename + '_' + str(idx) + '.pkl'
        saved_dir_kc = os.path.dirname(saved_path_kc)
        if not os.path.exists(saved_dir_kc):
            os.makedirs(saved_dir_kc)

        K_clusters = obtain_K_nearset_clusters(feas['sifts'], dict, coding_config['K'], saved_path_kc,
                                               coding_config['K_reset'])
    if coding_method == 'sv':
        X = sv(feas['sifts'], K_clusters['indices'][:, :5], K_clusters['dists'][:, :5], params_odfs['dict_size'])
        sprs = gen_sprs_for_other(X, feas['posis'], spr_config)
    if coding_method == 'hv':
        X = hv(feas['sifts'], K_clusters['indices'][:, 0], params_odfs['dict_size'])
        sprs = gen_sprs_for_other(X, feas['posis'], spr_config)
    elif coding_method == 'llc':
        norm_dict = dict / np.reshape(np.sqrt(np.sum(dict ** 2, axis=0)), (1, -1))

        X = llc(feas['sifts'].T, K_clusters['indices'][:, :5], norm_dict.T, params_odfs['dict_size'])
        sprs = gen_sprs_for_other(X, feas['posis'], spr_config)
    elif coding_method == 'svc':
        sprs = gen_sprs_for_svc(feas['sifts'], feas['posis'], dict, K_clusters['indices'][:, :20],
                                K_clusters['dists'][:, :20], spr_config, params_odfs['dict_size'])
    elif coding_method == 'fv':
        sprs = gen_sprs_for_fv(feas['sifts'], feas['posis'], dict, spr_config, params_odfs['dict_size'])
    return sprs


def gen_sprs_for_svc(sifts, posis, dict, indices, dists, spr_config, dict_size):
    dim, num = sifts.shape
    W = np.exp(-10 * dists)
    W = W / np.reshape(np.sum(W, axis=1), (num, 1))

    i_inds = {}
    for i in range(dict_size):
        i_inds[i] = np.argwhere(indices == i)[:, 0]

    min_HW = np.min(posis, axis=1)
    max_HW = np.max(posis, axis=1) + 0.00000001

    sprs = []
    for i in range(len(spr_config['h_sp'])):
        h_splits = np.linspace(min_HW[0], max_HW[0], spr_config['h_sp'][i] + 1)
        w_splits = np.linspace(min_HW[1], max_HW[1], spr_config['w_sp'][i] + 1)
        for h in range(len(h_splits) - 1):
            for w in range(len(w_splits) - 1):
                spatial_ind = np.argwhere(
                    (posis[0, :] >= h_splits[h]) & (posis[0, :] < h_splits[h + 1]) & (posis[1, :] >= w_splits[w]) & (
                            posis[1, :] < w_splits[w + 1]))
                usk = []
                for i in range(dict_size):
                    idx = np.intersect1d(i_inds[i], spatial_ind)
                    if len(idx) > 0:
                        w = W[idx, :][indices[idx, :] == i]
                        sqrt_pk = np.sqrt(np.mean(w))
                        uk = np.sum((sifts[:, idx] - np.reshape(dict[:, i], (-1, 1))) * w, axis=1) / sqrt_pk
                        sk = 0.01 * sqrt_pk
                    else:
                        uk = np.zeros(dim)
                        sk = 0
                    usk.append([sk])
                    usk.append(uk)
                usk = np.hstack(usk)
                sprs.append(usk)
    return sprs


def gen_sprs_for_fv(sifts, posis, dict, spr_config, dict_size):
    dim, num = sifts.shape
    means = dict.means_.T
    covs = dict.covariances_.T
    weights = dict.weights_
    W = dict.predict_proba(sifts.T)

    min_HW = np.min(posis, axis=1)
    max_HW = np.max(posis, axis=1) + 0.00000001

    sprs = []
    for i in range(len(spr_config['h_sp'])):
        h_splits = np.linspace(min_HW[0], max_HW[0], spr_config['h_sp'][i] + 1)
        w_splits = np.linspace(min_HW[1], max_HW[1], spr_config['w_sp'][i] + 1)
        for h in range(len(h_splits) - 1):
            for w in range(len(w_splits) - 1):
                indices = np.argwhere(
                    (posis[0, :] >= h_splits[h]) & (posis[0, :] < h_splits[h + 1]) & (posis[1, :] >= w_splits[w]) & (
                            posis[1, :] < w_splits[w + 1])).T[0]
                sifts_sub = sifts[:, indices]
                sqrt_weights = np.sqrt(weights)
                diff_xu = np.sqrt(1 / np.expand_dims(covs.T, 2)) * (
                        np.tile(sifts_sub, (dict_size, 1, 1)) - np.expand_dims(means.T, 2))
                uk = np.sum(diff_xu * np.expand_dims(W[indices, :].T, 1), axis=2).T / (len(indices) * sqrt_weights)
                vk = np.sum((diff_xu ** 2 - 1) * np.expand_dims(W[indices, :].T, 1), axis=2).T / (
                        len(indices) * sqrt_weights * np.sqrt(2))
                fv = np.concatenate((uk.flatten(), vk.flatten()))

                # fv = []
                # for k in range(dict_size):
                #     diff_xu = np.dot(np.sqrt(np.linalg.inv(np.diag(covs[:, k]))), (sifts_sub - np.reshape(means[:, k], (dim, 1))))
                #     uk = np.sum(diff_xu * W[indices, k], axis=1) / (len(indices) * np.sqrt(weights[k]))
                #     vk = np.sum((diff_xu**2 - 1) * W[indices, k], axis=1) / (len(indices) * np.sqrt(2 * weights[k]))
                #     fv.append(uk); fv.append(vk)
                # fv = np.hstack(fv)

                fv = np.sqrt(np.abs(fv)) * np.sign(fv)
                fv = fv / np.sqrt(np.sum(fv ** 2))
                sprs.append(fv)
    return sprs

def gen_sprs_for_other(X, posis, spr_config):
    min_HW = np.min(posis, axis=1)
    max_HW = np.max(posis, axis=1) + 0.00000001

    X = np.abs(X)
    sprs = []
    for i in range(len(spr_config['h_sp'])):
        h_splits = np.linspace(min_HW[0], max_HW[0], spr_config['h_sp'][i] + 1)
        w_splits = np.linspace(min_HW[1], max_HW[1], spr_config['w_sp'][i] + 1)
        for h in range(len(h_splits) - 1):
            for w in range(len(w_splits) - 1):
                indices = np.argwhere(
                    (posis[0, :] >= h_splits[h]) & (posis[0, :] < h_splits[h + 1]) & (posis[1, :] >= w_splits[w]) & (
                            posis[1, :] < w_splits[w + 1]))
                if spr_config['pooling_method'] == 'avg':
                    spr = np.sum(X[:, indices], axis=1).T[0]
                    sprs.append((spr / sum(spr)))
                elif spr_config['pooling_method'] == 'max':
                    sprs.append((np.max(X[:, indices], axis=1)).T)
    return sprs


def llc(sifts, indices, dict, dict_size):
    k = np.min([5, dict_size])

    N = sifts.shape[0]
    II = np.eye(k)
    X = np.zeros((N, dict_size))  # Gammas
    ones = np.ones((k, 1))
    for i in range(N):
        idx = indices[i, :]
        z = dict[idx, :] - np.tile(sifts[i, :], (k, 1))  # shift ith pt to origin
        Z = np.dot(z, z.T)  # local covariance
        Z = Z + II * 1e-6 * np.trace(Z)  # regularlization (K>D)
        try:
            w = np.dot(np.linalg.pinv(Z), ones)
            # w = np.linalg.solve(Z, ones)  # np.dot(np.linalg.inv(Z), ones)
            w = w / np.sum(w)  # enforce sum(w)=1
        except:
            print('svd exception')
            w = np.zeros(5)
        X[i, idx] = w.ravel()

        # Z = dict[idx, :].T
        # X[i, idx] = np.dot(np.dot(np.linalg.pinv(np.dot(Z.T, Z)), Z.T), sifts[i, :].T).T
    return X.T


def hv(sifts, indices, dict_size):
    X = np.zeros((dict_size, sifts.shape[1]))
    X[indices, range(X.shape[1])] = 1
    return X


def sv(sifts, indices, dists, dict_size):
    num = sifts.shape[1]
    X = np.zeros((dict_size, num))
    W = np.exp(-10 * dists)
    W = W / np.reshape(np.sum(W, axis=1), (num, 1))
    for i in range(np.min([dict_size, 5])):
        X[indices[:, i], range(num)] = W[:, i]

    X[np.isnan(X)] = 0
    return X


def obtain_K_nearset_clusters(feas, dic, K, saved_path_kc, K_reset):
    if os.path.exists(saved_path_kc) and not K_reset and os.path.getsize(saved_path_kc) != 0:
        return joblib.load(saved_path_kc)
    else:
        K_clusters = {'K': K}
        if len(dic) == 2:
            all_dists = cdist(feas.T, dic[0].T, metric='euclidean')
        else:
            all_dists = cdist(feas.T, dic.T, metric='euclidean')
        K_clusters['indices'] = np.argsort(all_dists, axis=1)[:, 0:K]
        K_clusters['dists'] = np.sort(all_dists, axis=1)[:, 0:K]
        joblib.dump(K_clusters, saved_path_kc, compress=True)
        return K_clusters


if __name__ == '__main__':
    pass

# Copyright (C) 2021-2023 Ye Xu, Xin Zhang, Chongpeng Huang, Xiaorong Qiu

import DenseSIFT as DSIFT
import cv2, os, joblib, sklearn
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import copy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import entropy
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from PIL import Image
import multiprocessing as mp
import shutil
from scipy import spatial

from Obtain_Train_Test_Config import obtain_train_test_config
def cv_imread(file_path):
    root_dir, file_name = os.path.split(file_path)
    pwd = os.getcwd()
    if root_dir:
        os.chdir(root_dir)
    cv_img = cv2.imread(file_name)
    os.chdir(pwd)
    return cv_img

def obtain_cnn_model(model_name, data_path, dsname, ts, st, idx):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'vgg16bn':
        model = models.vgg16_bn(pretrained=True)
    elif model_name == 'resnext50':
        model = models.resnext50_32x4d(pretrained=True)
    elif model_name == 'swinb':
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet50(refined)':
        model_path = data_path + '/intermediate_data/obtain_cls_scores/obtain_scores_for_cnn/%s/%s_trainsize%d_splittimes%d_all.pkl' % (
            dsname, 'resnet50', ts, st)
        if not os.path.exists(model_path):
            raise Exception("no refined model, please run obtain_scores_for_cnn to get refined models")
        model = joblib.load(model_path)['models'][idx]
    elif model_name == 'vgg16bn(refined)':
        model_path = data_path + '/intermediate_data/obtain_cls_scores/obtain_scores_for_cnn/%s/%s_trainsize%d_splittimes%d_all.pkl' % (
            dsname, 'vgg16bn', ts, st)
        if not os.path.exists(model_path):
            raise Exception("no refined model, please run obtain_scores_for_cnn to get refined models")
        model = joblib.load(model_path)['models'][idx]
    elif model_name == 'resnext50(refined)':
        model_path = data_path + '/intermediate_data/obtain_cls_scores/obtain_scores_for_cnn/%s/%s_trainsize%d_splittimes%d_all.pkl' % (
            dsname, 'resnext50', ts, st)
        if not os.path.exists(model_path):
            raise Exception("no refined model, please run obtain_scores_for_cnn to get refined models")
        model = joblib.load(model_path)['models'][idx]
    elif model_name == 'swinb(refined)':
        model_path = data_path + '/intermediate_data/obtain_cls_scores/obtain_scores_for_cnn/%s/%s_trainsize%d_splittimes%d_all.pkl' % (
            dsname, 'swinb', ts, st)
        if not os.path.exists(model_path):
            raise Exception("no refined model, please run obtain_scores_for_cnn to get refined models")
        model = joblib.load(model_path)['models'][idx]

    # print(model)
    return model


def obtain_transform_for_cp(model_name):
    if model_name == 'resnet50' or model_name == 'resnet50(refined)':
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif model_name == 'vgg16bn' or model_name == 'vgg16bn(refined)':
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif model_name == 'resnext50' or model_name == 'resnext50(refined)':
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif model_name == 'swinb' or model_name == 'swinb(refined)':
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return tf

feas_cp = {}


def gen_hook_for_cp(module_name):
    def hook(model, input, output):
        feas_cp[module_name] = np.squeeze(output.detach().cpu().numpy(), axis=0)

    return hook

def obtain_deep_features_by_cp(dsname, model_name, target_layer, params_ottc, data_path, reset=False,
                               skip=False):
    train_test_config = obtain_train_test_config(dsname, times=params_ottc['times'],
                                                 split_setup=params_ottc['split_setup'],
                                                 reset=params_ottc['reset'], data_path=data_path)
    data = train_test_config['data']
    if not skip or reset:
        saved_root_dir = data_path + '/intermediate_data/ext_feas/ext_dfs/cp/' + model_name + '/' + dsname
        if reset and os.path.exists(saved_root_dir):
            shutil.rmtree(saved_root_dir)

        global feas_cp

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tf = obtain_transform_for_cp(model_name)

        for i in range(len(data)):
            model = obtain_cnn_model(model_name, data_path, dsname, train_test_config['split_setup']['train_size'],
                                     params_ottc['times'], i)
            model = model.to(device)
            model.eval()
            for name, module in model.named_modules():
                # print(name)
                if name == target_layer:
                    module.register_forward_hook(gen_hook_for_cp(name))

            train_test_data = data[i]['train_x'] + data[i]['test_x']

            d_len = len(train_test_data)
            progress = 0
            for filename in train_test_data:
                if model_name.find('refined') != -1:
                    saved_path = (saved_root_dir + '/ts' + str(train_test_config['split_setup']['train_size']) + '/'
                                  + str(i) + '/' + os.path.splitext(filename)[0] + '.pkl')
                else:
                    saved_path = saved_root_dir + '/' + os.path.splitext(filename)[0] + '.pkl'

                if not os.path.exists(saved_path) or reset:
                    print('obtain_deep_features_by_cp---->%s(%.3f)' % (filename, progress / d_len))

                    fpath = data_path + '/datasets/' + dsname + '/' + filename
                    image = Image.open(fpath)
                    image = image.convert('RGB')
                    image = tf(image).to(device)

                    feas_cp = {}
                    model(image[None, ...])

                    if model_name in ['swinb', 'swinb(refined)']:
                        feas_cp = {k:v.swapaxes(0, 2) for k, v in feas_cp.items()}

                    saved_dir = os.path.dirname(saved_path)
                    if not os.path.exists(saved_dir):
                        os.makedirs(saved_dir)
                    joblib.dump(feas_cp, saved_path, compress=True)
                progress += 1

    if model_name.find('refined') != -1:
        return saved_root_dir + '/ts' + str(train_test_config['split_setup']['train_size']) + '/%d/%s.pkl'
    else:
        return saved_root_dir + '/%s.pkl'


def extract_patch_locs(image, sizes=[64], steps=[8]):
    W = image.shape[1]
    H = image.shape[0]

    patch_locs = []
    for i in range(len(sizes)):
        size = sizes[i]
        step = steps[i]
        remH = np.mod(H - size, step)
        remW = np.mod(W - size, step)
        offsetH = remH / 2
        offsetW = remW / 2
        gridH, gridW = np.meshgrid(range(int(offsetH), H - size + 1, step), range(int(offsetW), W - size + 1, step))
        gridH = gridH.flatten()
        gridW = gridW.flatten()
        positions = np.vstack(((gridH + (size - 1) / 2) / np.double(H), (gridW + (size - 1) / 2) / np.double(W)))
        ret = dict(idx_0=gridH, idx_1=gridW, pos=positions, size=size, step=step, num=len(gridH))
        patch_locs.append(ret)
    return patch_locs


def obtain_transform_for_fc(model_name):
    if model_name == 'resnet50' or model_name == 'resnet50(refined)':
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif model_name == 'vgg16bn' or model_name == 'vgg16bn(refined)':
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif model_name == 'resnext50' or model_name == 'resnext50(refined)':
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif model_name == 'swinb' or model_name == 'swinb(refined)':
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return tf


feas_fc = []


def hook_for_fc(model, input, output):
    feas_fc.append(output.detach().cpu().numpy().flatten())


def obtain_deep_features_by_fc(dsname, model_name, target_layer, sizes, steps, params_ottc, data_path,
                               reset=False, skip=False):
    train_test_config = obtain_train_test_config(dsname, times=params_ottc['times'],
                                                 split_setup=params_ottc['split_setup'],
                                                 reset=params_ottc['reset'], data_path=data_path)
    data = train_test_config['data']
    saved_root_dir = data_path + '/intermediate_data/ext_feas/ext_dfs/fc/' + model_name + '/' + dsname
    if not skip or reset:
        if reset and os.path.exists(saved_root_dir):
            shutil.rmtree(saved_root_dir)

        global feas_fc

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tf = obtain_transform_for_fc(model_name)

        for i in range(len(data)):
            model = obtain_cnn_model(model_name, data_path, dsname, train_test_config['split_setup']['train_size'],
                                     params_ottc['times'], i)
            model = model.to(device)
            model.eval()
            for name, module in model.named_modules():
                # print(name)
                if name == target_layer:
                    module.register_forward_hook(hook_for_fc)

            train_test_data = data[i]['train_x'] + data[i]['test_x']
            train_test_labels = data[i]['train_y'] + data[i]['test_y']

            d_len = len(train_test_data)
            progress = 0
            for ii, filename in enumerate(train_test_data):
                if model_name.find('refined') != -1:
                    saved_path = saved_root_dir + '/ts' + str(train_test_config['split_setup']['train_size']) + '/' + str(i) + '/' + os.path.splitext(filename)[0] + '.pkl'
                else:
                    saved_path = saved_root_dir + '/' + os.path.splitext(filename)[0] + '.pkl'

                if not os.path.exists(saved_path):
                    print('obtain_deep_features_by_fc---->%s(%.3f)' % (filename, progress / d_len))

                    fpath = data_path + '/datasets/' + dsname + '/' + filename
                    image = Image.open(fpath)
                    image = image.convert('RGB')
                    ratio = 256 / min(image.size)
                    image = image.resize((int(np.round(image.size[0] * ratio)), int(np.round(image.size[1] * ratio))),
                                         Image.ANTIALIAS)
                    image = np.asarray(image)

                    patch_locs = extract_patch_locs(image, sizes, steps)
                    feas = []
                    for p_loc in patch_locs:
                        idx_0 = p_loc['idx_0']
                        idx_1 = p_loc['idx_1']
                        size = p_loc['size']
                        feas_fc = []
                        for n in range(p_loc['num']):
                            img_patch = Image.fromarray(image[idx_0[n]:idx_0[n] + size, idx_1[n]:idx_1[n] + size, :])
                            img_patch = tf(img_patch).to(device)
                            model(img_patch[None, ...])
                        feas.append(np.vstack(feas_fc).T)

                    saved_dir = os.path.dirname(saved_path)
                    if not os.path.exists(saved_dir):
                        os.makedirs(saved_dir)

                    ret = dict(feas=feas, patch_locs=patch_locs)
                    joblib.dump(ret, saved_path, compress=True)
                progress += 1

    if model_name.find('refined') != -1:
        return saved_root_dir + '/ts' + str(train_test_config['split_setup']['train_size']) + '/%d/%s.pkl'
    else:
        return saved_root_dir + '/%s.pkl'

if __name__ == '__main__':
    pass

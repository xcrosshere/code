# Copyright (C) 2021-2023 Ye Xu, Xin Zhang, Chongpeng Huang, Xiaorong Qiu

from Obtain_FT_Scores import obtain_scores_for_cnn
import settings, settings_15S

data_path = settings.data_path

params_ottc = settings_15S.params_ottc

def by_resnet50_part(num_epochs=25, da_type=None):
    print('->' * 25, 'cnn:by_resnet50_part', '<-' * 25)
    return obtain_scores_for_cnn('15-Scenes', 'resnet50', 'part', params_ottc, da_type=da_type, val_skip=True, num_epochs=num_epochs,
                                 data_path=data_path, reset=False)


def by_resnet50_all(num_epochs=25, da_type=None):
    print('->' * 25, 'cnn:by_resnet50_all', '<-' * 25)
    return obtain_scores_for_cnn('15-Scenes', 'resnet50', 'all', params_ottc, da_type=da_type, val_skip=True, num_epochs=num_epochs,
                                 data_path=data_path, reset=True)


def by_vgg16bn_part(num_epochs=25, da_type=None):
    print('->' * 25, 'cnn:by_vgg16bn_part', '<-' * 25)
    return obtain_scores_for_cnn('15-Scenes', 'vgg16bn', 'part', params_ottc, da_type=da_type, val_skip=True, num_epochs=num_epochs,
                                 data_path=data_path, reset=False)


def by_vgg16bn_all(num_epochs=25, da_type=None):
    print('->' * 25, 'cnn:by_vgg16bn_all', '<-' * 25)
    return obtain_scores_for_cnn('15-Scenes', 'vgg16bn', 'all', params_ottc, da_type=da_type, val_skip=True, num_epochs=num_epochs,
                                 data_path=data_path, reset=False)

def by_resnext50_part(num_epochs=25, da_type=None):
    print('->' * 25, 'cnn:by_resnext50_part', '<-' * 25)
    return obtain_scores_for_cnn('15-Scenes', 'resnext50', 'part', params_ottc, da_type=da_type, val_skip=True, num_epochs=num_epochs,
                                 data_path=data_path, reset=False)


def by_resnext50_all(num_epochs=25, da_type=None):
    print('->' * 25, 'cnn:by_resnext50_all', '<-' * 25)
    ret =  obtain_scores_for_cnn('15-Scenes', 'resnext50', 'all', params_ottc, da_type=da_type, val_skip=True, num_epochs=num_epochs,
                                 data_path=data_path, reset=True)
    return ret

def by_resnext101_all(num_epochs=25, da_type=None):
    print('->' * 25, 'cnn:by_resnext101_all', '<-' * 25)
    ret = obtain_scores_for_cnn('15-Scenes', 'resnext101', 'all', params_ottc, da_type=da_type, val_skip=True, num_epochs=num_epochs,
                                 data_path=data_path, reset=False)
    return ret

def by_resnet152_all(num_epochs=25, da_type=None):
    print('->' * 25, 'cnn:by_resnet152_all', '<-' * 25)
    return obtain_scores_for_cnn('15-Scenes', 'resnet152', 'all', params_ottc, da_type=da_type, val_skip=True, num_epochs=num_epochs,
                                 data_path=data_path, reset=False)

def by_swinb_all(num_epochs=25, da_type=None):
    print('->' * 25, 'cnn:by_swinb_all', '<-' * 25)
    return obtain_scores_for_cnn('15-Scenes', 'swinb', 'all', params_ottc, da_type=da_type, val_skip=True, num_epochs=num_epochs,
                                 data_path=data_path, reset=True)

def by_swinb_part(num_epochs=25, da_type=None):
    print('->' * 25, 'cnn:by_swinb_part', '<-' * 25)
    return obtain_scores_for_cnn('15-Scenes', 'swinb', 'part', params_ottc, da_type=da_type, val_skip=True, num_epochs=num_epochs,
                                 data_path=data_path, reset=True)

if __name__ == '__main__':
    # by_resnet50_part()
    by_resnet50_all()
    # by_vgg16bn_part()
    # by_vgg16bn_all()
    # by_resnext50_part()
    # by_resnext50_all()

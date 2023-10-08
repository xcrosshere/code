# Copyright (C) 2021-2023 Ye Xu, Xin Zhang, Chongpeng Huang, Xiaorong Qiu

def get_coding_config(coding_method):
    if coding_method == 'sv':
        coding_config = dict(do_coding=True, coding_method='sv', K=5)
    if coding_method == 'llc':
        coding_config = dict(do_coding=True, coding_method='llc', K=5)
    if coding_method == 'svc':
        coding_config = dict(do_coding=True, coding_method='svc', K=20)
    if coding_method == 'fv':
        coding_config = dict(do_coding=True, coding_method='fv')
    if coding_method == 'hv':
        coding_config = dict(do_coding=True, coding_method='hv', K=1)
    if coding_method == 'no_coding':
        coding_config = dict(do_coding=False)

    return coding_config

def get_params_odfdf(coding_method, dict_size):
    if coding_method == 'svc':
        params_odfdf = dict(dict_size=dict_size, dict_method='Kmeans', reset=False, norm=False)
    elif coding_method == 'fv':
        params_odfdf = dict(dict_size=dict_size, dict_method='GMM', reset=False, norm=False)
    elif coding_method == 'sv':
        params_odfdf = dict(dict_size=dict_size, dict_method='Kmeans', reset=False, norm=False)
    elif coding_method == 'llc':
        params_odfdf = dict(dict_size=dict_size, dict_method='Kmeans', reset=False, norm=False)
    elif coding_method == 'hv':
        params_odfdf = dict(dict_size=dict_size, dict_method='Kmeans', reset=False, norm=False)
    elif coding_method == 'no_coding':
        return None

    return params_odfdf

def get_spr_config(coding_method):
    if coding_method == 'hv':
        return dict(h_sp=[1, 3, 2], w_sp=[1, 1, 2], pooling_method='avg')
    else:
        return dict(h_sp=[1, 3, 2], w_sp=[1, 1, 2], pooling_method='max')

def get_classifier_config(coding_method):
    if coding_method == 'hv':
        return dict(type='rbf')
    else:
        return dict(type='linear')


from Obtain_CLS_Scores import obtain_cls_scores_for_CPFC
import settings, settings_MIT

params_ottc = settings_MIT.params_ottc
data_path = settings.data_path


def by_cp_resnet50(coding_method, dict_size, group_size=2048, num_comp=-1, return_data=False):
    print('->' * 25, 'df:by_cp_resnet50', '<-' * 25)

    params_of = dict(reset=False, skip=False, model_name='resnet50', target_layer='layer4.2.relu', group_size=group_size,
                     fea_source='cp')
    params_odfdf = get_params_odfdf(coding_method, dict_size)
    coding_config = get_coding_config(coding_method)
    coding_config['K_reset'] = params_odfdf['reset']
    params_oiv = dict(num_comp=num_comp, coding_config=coding_config, spr_config=get_spr_config(coding_method), reset=False, skip=False)

    return obtain_cls_scores_for_CPFC('MIT Indoor-67', get_classifier_config(coding_method), params_oiv, params_odfdf, params_of, params_ottc,
                                      reset=False, data_path=data_path, return_data=return_data)

def by_cp_resnet50_refined(coding_method, dict_size, group_size=2048, num_comp=-1, return_data=False):
    print('->' * 25, 'df:by_cp_resnet50_refined', '<-' * 25)

    params_of = dict(reset=False, skip=False, model_name='resnet50(refined)', target_layer='layer4.2.relu', group_size=group_size,
                     fea_source='cp')
    params_odfdf = get_params_odfdf(coding_method, dict_size)
    coding_config = get_coding_config(coding_method)
    coding_config['K_reset'] = params_odfdf['reset']
    params_oiv = dict(num_comp=num_comp, coding_config=coding_config, spr_config=get_spr_config(coding_method), reset=False, skip=False)

    return obtain_cls_scores_for_CPFC('MIT Indoor-67', get_classifier_config(coding_method), params_oiv, params_odfdf, params_of, params_ottc,
                                      reset=False, data_path=data_path, return_data=return_data)


def by_cp_vgg16bn(coding_method, dict_size, group_size=512, num_comp=-1, return_data=False):
    print('->' * 25, 'df:by_cp_vgg16bn', '<-' * 25)

    params_of = dict(reset=False, skip=False, model_name='vgg16bn', target_layer='features.42', group_size=group_size,
                     fea_source='cp')
    params_odfdf = get_params_odfdf(coding_method, dict_size)
    coding_config = get_coding_config(coding_method)
    coding_config['K_reset'] = params_odfdf['reset']
    params_oiv = dict(num_comp=num_comp, coding_config=coding_config, spr_config=get_spr_config(coding_method), reset=False, skip=False)

    return obtain_cls_scores_for_CPFC('MIT Indoor-67', get_classifier_config(coding_method), params_oiv, params_odfdf, params_of, params_ottc,
                                      reset=False, data_path=data_path, return_data=return_data)

def by_cp_vgg16bn_refined(coding_method, dict_size, group_size=512, num_comp=-1, return_data=False):
    print('->' * 25, 'df:by_cp_vgg16bn_refined', '<-' * 25)

    params_of = dict(reset=False, skip=False, model_name='vgg16bn(refined)', target_layer='features.42', group_size=group_size,
                     fea_source='cp')
    params_odfdf = get_params_odfdf(coding_method, dict_size)
    coding_config = get_coding_config(coding_method)
    coding_config['K_reset'] = params_odfdf['reset']
    params_oiv = dict(num_comp=num_comp, coding_config=coding_config, spr_config=get_spr_config(coding_method), reset=False, skip=False)

    return obtain_cls_scores_for_CPFC('MIT Indoor-67', get_classifier_config(coding_method), params_oiv, params_odfdf, params_of, params_ottc,
                                      reset=False, data_path=data_path, return_data=return_data)


def by_fc_resnet50(coding_method, dict_size, group_size=2048, num_comp=-1, return_data=False):
    print('->' * 25, 'df:by_fc_resnet50', '<-' * 25)

    params_of = dict(reset=False, skip=False, model_name='resnet50', target_layer='avgpool', group_size=group_size,
                     fea_source='fc', sizes=[96, 128, 160, 192, 224, 256], steps=[32, 32, 32, 32, 32, 32])
    params_odfdf = get_params_odfdf(coding_method, dict_size)
    coding_config = get_coding_config(coding_method)
    coding_config['K_reset'] = params_odfdf['reset']
    params_oiv = dict(num_comp=num_comp, coding_config=coding_config, spr_config=get_spr_config(coding_method), reset=False, skip=False)

    return obtain_cls_scores_for_CPFC('MIT Indoor-67', get_classifier_config(coding_method), params_oiv, params_odfdf, params_of, params_ottc,
                                      reset=False, data_path=data_path, return_data=return_data)


def by_fc_resnet50_refined(coding_method, dict_size, group_size=2048, num_comp=-1, return_data=False):
    print('->' * 25, 'df:by_fc_resnet50_refined', '<-' * 25)

    params_of = dict(reset=False, skip=True, model_name='resnet50(refined)', target_layer='avgpool', group_size=group_size,
                     fea_source='fc', sizes=[96, 128, 160, 192, 224, 256], steps=[32, 32, 32, 32, 32, 32])
    params_odfdf = get_params_odfdf(coding_method, dict_size)
    coding_config = get_coding_config(coding_method)
    coding_config['K_reset'] = params_odfdf['reset']
    params_oiv = dict(num_comp=num_comp, coding_config=coding_config, spr_config=get_spr_config(coding_method), reset=False, skip=False)
    return obtain_cls_scores_for_CPFC('MIT Indoor-67', get_classifier_config(coding_method), params_oiv, params_odfdf, params_of, params_ottc,
                                      reset=False, data_path=data_path, return_data=return_data)


def by_fc_vgg16bn(coding_method, dict_size, group_size=4096, num_comp=-1, return_data=False):
    print('->' * 25, 'df:by_fc_vgg16bn', '<-' * 25)

    params_of = dict(reset=False, skip=False, model_name='vgg16bn', target_layer='classifier.5', group_size=group_size,
                     fea_source='fc', sizes=[96, 128, 160, 192, 224, 256], steps=[32, 32, 32, 32, 32, 32])
    params_odfdf = get_params_odfdf(coding_method, dict_size)
    coding_config = get_coding_config(coding_method)
    coding_config['K_reset'] = params_odfdf['reset']
    params_oiv = dict(num_comp=num_comp, coding_config=coding_config, spr_config=get_spr_config(coding_method), reset=False, skip=False)

    return obtain_cls_scores_for_CPFC('MIT Indoor-67', get_classifier_config(coding_method), params_oiv, params_odfdf, params_of, params_ottc,
                                      reset=False, data_path=data_path, return_data=return_data)

def by_fc_vgg16bn_refined(coding_method, dict_size, group_size=4096, num_comp=-1, return_data=False):
    print('->' * 25, 'df:by_fc_vgg16bn(refined)', '<-' * 25)

    params_of = dict(reset=False, skip=False, model_name='vgg16bn(refined)', target_layer='classifier.5', group_size=group_size,
                     fea_source='fc', sizes=[96, 128, 160, 192, 224, 256], steps=[32, 32, 32, 32, 32, 32])
    params_odfdf = get_params_odfdf(coding_method, dict_size)
    coding_config = get_coding_config(coding_method)
    coding_config['K_reset'] = params_odfdf['reset']
    params_oiv = dict(num_comp=num_comp, coding_config=coding_config, spr_config=get_spr_config(coding_method), reset=False, skip=False)

    return obtain_cls_scores_for_CPFC('MIT Indoor-67', get_classifier_config(coding_method), params_oiv, params_odfdf, params_of, params_ottc,
                                      reset=False, data_path=data_path, return_data=return_data)

def by_fc_resnext50(coding_method, dict_size, group_size=2048, num_comp=-1, return_data=False):
    print('->' * 25, 'df:by_fc_resnext50', '<-' * 25)

    params_of = dict(reset=False, skip=False, model_name='resnext50', target_layer='avgpool', group_size=group_size,
                     fea_source='fc', sizes=[96, 128, 160, 192, 224, 256], steps=[32, 32, 32, 32, 32, 32])
    params_odfdf = get_params_odfdf(coding_method, dict_size)
    coding_config = get_coding_config(coding_method)
    coding_config['K_reset'] = params_odfdf['reset']
    params_oiv = dict(num_comp=num_comp, coding_config=coding_config, spr_config=get_spr_config(coding_method), reset=False, skip=False)

    return obtain_cls_scores_for_CPFC('MIT Indoor-67', get_classifier_config(coding_method), params_oiv, params_odfdf, params_of, params_ottc,
                                      reset=False, data_path=data_path, return_data=return_data)

def by_fc_resnext50_refined(coding_method, dict_size, group_size=2048, num_comp=-1, return_data=False):
    print('->' * 25, 'df:by_fc_resnext50', '<-' * 25)

    params_of = dict(reset=False, skip=True, model_name='resnext50(refined)', target_layer='avgpool', group_size=group_size,
                     fea_source='fc', sizes=[96, 128, 160, 192, 224, 256], steps=[32, 32, 32, 32, 32, 32])
    params_odfdf = get_params_odfdf(coding_method, dict_size)
    coding_config = get_coding_config(coding_method)
    coding_config['K_reset'] = params_odfdf['reset']
    params_oiv = dict(num_comp=num_comp, coding_config=coding_config, spr_config=get_spr_config(coding_method), reset=False, skip=False)

    return obtain_cls_scores_for_CPFC('MIT Indoor-67', get_classifier_config(coding_method), params_oiv, params_odfdf, params_of, params_ottc,
                                      reset=False, data_path=data_path, return_data=return_data)

def by_cp_resnext50(coding_method, dict_size, group_size=2048, num_comp=-1, return_data=False):
    print('->' * 25, 'df:by_cp_resnext50', '<-' * 25)

    params_of = dict(reset=False, skip=False, model_name='resnext50', target_layer='layer4.2.relu', group_size=group_size,
                     fea_source='cp')
    params_odfdf = get_params_odfdf(coding_method, dict_size)
    coding_config = get_coding_config(coding_method)
    coding_config['K_reset'] = params_odfdf['reset']
    params_oiv = dict(num_comp=num_comp, coding_config=coding_config, spr_config=get_spr_config(coding_method), reset=False, skip=False)

    return obtain_cls_scores_for_CPFC('MIT Indoor-67', get_classifier_config(coding_method), params_oiv, params_odfdf, params_of, params_ottc,
                                      reset=False, data_path=data_path, return_data=return_data)

def by_cp_resnext50_refined(coding_method, dict_size, group_size=2048, num_comp=-1, return_data=False):
    print('->' * 25, 'df:by_cp_resnext50_refined', '<-' * 25)

    params_of = dict(reset=False, skip=False, model_name='resnext50(refined)', target_layer='layer4.2.relu', group_size=group_size,
                     fea_source='cp')
    params_odfdf = get_params_odfdf(coding_method, dict_size)
    coding_config = get_coding_config(coding_method)
    coding_config['K_reset'] = params_odfdf['reset']
    params_oiv = dict(num_comp=num_comp, coding_config=coding_config, spr_config=get_spr_config(coding_method), reset=False, skip=False)

    return obtain_cls_scores_for_CPFC('MIT Indoor-67', get_classifier_config(coding_method), params_oiv, params_odfdf, params_of, params_ottc,
                                      reset=False, data_path=data_path, return_data=return_data)

def by_cp_swinb(coding_method, dict_size, group_size=1024, num_comp=-1, return_data=False):
    print('->' * 25, 'df:by_cp_swinb', '<-' * 25)

    params_of = dict(reset=False, skip=False, model_name='swinb', target_layer='norm', group_size=group_size,
                     fea_source='cp')
    params_odfdf = get_params_odfdf(coding_method, dict_size)
    coding_config = get_coding_config(coding_method)
    coding_config['K_reset'] = params_odfdf['reset']
    params_oiv = dict(num_comp=num_comp, coding_config=coding_config, spr_config=get_spr_config(coding_method), reset=False, skip=False)

    return obtain_cls_scores_for_CPFC('MIT Indoor-67', get_classifier_config(coding_method), params_oiv, params_odfdf, params_of, params_ottc,
                                      reset=False, data_path=data_path, return_data=return_data)

def by_cp_swinb_refined(coding_method, dict_size, group_size=1024, num_comp=-1, return_data=False):
    print('->' * 25, 'df:by_cp_swinb(refined)', '<-' * 25)

    params_of = dict(reset=False, skip=False, model_name='swinb(refined)', target_layer='norm', group_size=group_size,
                     fea_source='cp')
    params_odfdf = get_params_odfdf(coding_method, dict_size)
    coding_config = get_coding_config(coding_method)
    coding_config['K_reset'] = params_odfdf['reset']
    params_oiv = dict(num_comp=num_comp, coding_config=coding_config, spr_config=get_spr_config(coding_method), reset=False, skip=False)

    return obtain_cls_scores_for_CPFC('MIT Indoor-67', get_classifier_config(coding_method), params_oiv, params_odfdf, params_of, params_ottc,
                                      reset=False, data_path=data_path, return_data=return_data)


def by_fc_swinb(coding_method, dict_size, group_size=1024, num_comp=-1, return_data=False):
    print('->' * 25, 'df:by_fc_swinb', '<-' * 25)

    params_of = dict(reset=False, skip=False, model_name='swinb', target_layer='avgpool', group_size=group_size,
                     fea_source='fc', sizes=[96, 128, 160, 192, 224, 256], steps=[32, 32, 32, 32, 32, 32])
    params_odfdf = get_params_odfdf(coding_method, dict_size)
    coding_config = get_coding_config(coding_method)
    coding_config['K_reset'] = params_odfdf['reset']
    params_oiv = dict(num_comp=num_comp, coding_config=coding_config, spr_config=get_spr_config(coding_method), reset=False, skip=False)

    return obtain_cls_scores_for_CPFC('MIT Indoor-67', get_classifier_config(coding_method), params_oiv, params_odfdf, params_of, params_ottc,
                                      reset=False, data_path=data_path, return_data=return_data)

def by_fc_swinb_refined(coding_method, dict_size, group_size=1024, num_comp=-1, return_data=False):
    print('->' * 25, 'df:by_fc_swinb_refined', '<-' * 25)

    params_of = dict(reset=False, skip=False, model_name='swinb(refined)', target_layer='avgpool', group_size=group_size,
                     fea_source='fc', sizes=[96, 128, 160, 192, 224, 256], steps=[32, 32, 32, 32, 32, 32])
    params_odfdf = get_params_odfdf(coding_method, dict_size)
    coding_config = get_coding_config(coding_method)
    coding_config['K_reset'] = params_odfdf['reset']
    params_oiv = dict(num_comp=num_comp, coding_config=coding_config, spr_config=get_spr_config(coding_method), reset=False, skip=False)

    return obtain_cls_scores_for_CPFC('MIT Indoor-67', get_classifier_config(coding_method), params_oiv, params_odfdf, params_of, params_ottc,
                                      reset=False, data_path=data_path, return_data=return_data)


if __name__ == '__main__':
    # by_fc_resnet50('sv', 4096, group_size=2048, num_comp=-1)
    by_fc_resnet50_refined('sv', 4096, group_size=2048, num_comp=-1)
    # by_fc_vgg16bn('sv', -1)
    # by_cp_resnet50('sv', -1)
    # by_cp_resnet50_refined('sv', 3)
    # by_cp_vgg16bn('sv', -1)
    # by_fc_resnet50_refined('sv', 3)

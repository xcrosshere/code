# Copyright (C) 2021-2023 Ye Xu, Xin Zhang, Chongpeng Huang, Xiaorong Qiu

import run_df_15S, run_df_TFF, run_df_COV, run_df_MIT, run_df_NWP, run_df_CAL

if __name__ == '__main__':
    hv_dictsizes = [8, 64, 256]

    for ds in hv_dictsizes:
        run_df_15S.by_cp_resnext50('hv', ds)
        run_df_15S.by_fc_resnext50('hv', ds)
        run_df_15S.by_cp_resnext50_refined('hv', ds)
        run_df_15S.by_fc_resnext50_refined('hv', ds)

        run_df_15S.by_cp_vgg16bn('hv', ds)
        run_df_15S.by_fc_vgg16bn('hv', ds)
        run_df_15S.by_cp_vgg16bn_refined('hv', ds)
        run_df_15S.by_fc_vgg16bn_refined('hv', ds)
    #

    # for ds in hv_dictsizes:
    #     run_df_UIUC.by_cp_resnext50('hv', ds)
    #     run_df_UIUC.by_fc_resnext50('hv', ds)
    #     run_df_UIUC.by_cp_resnext50_refined('hv', ds)
    #     run_df_UIUC.by_fc_resnext50_refined('hv', ds)
    #
    #     run_df_UIUC.by_cp_vgg16bn('hv', ds)
    #     run_df_UIUC.by_fc_vgg16bn('hv', ds)
    #     run_df_UIUC.by_cp_vgg16bn_refined('hv', ds)
    #     run_df_UIUC.by_fc_vgg16bn_refined('hv', ds)
    #

    # for ds in hv_dictsizes:
    #     run_df_TFF.by_cp_resnext50('hv', ds)
    #     run_df_TFF.by_fc_resnext50('hv', ds)
    #     run_df_TFF.by_cp_resnext50_refined('hv', ds)
    #     run_df_TFF.by_fc_resnext50_refined('hv', ds)
    #
    #     run_df_TFF.by_cp_vgg16bn('hv', ds)
    #     run_df_TFF.by_fc_vgg16bn('hv', ds)
    #     run_df_TFF.by_cp_vgg16bn_refined('hv', ds)
    #     run_df_TFF.by_fc_vgg16bn_refined('hv', ds)
    #
    # for ds in hv_dictsizes:
    #     run_df_COV.by_cp_resnext50('hv', ds)
    #     run_df_COV.by_fc_resnext50('hv', ds)
    #     run_df_COV.by_cp_resnext50_refined('hv', ds)
    #     run_df_COV.by_fc_resnext50_refined('hv', ds)
    #
    #     run_df_COV.by_cp_vgg16bn('hv', ds)
    #     run_df_COV.by_fc_vgg16bn('hv', ds)
    #     run_df_COV.by_cp_vgg16bn_refined('hv', ds)
    #     run_df_COV.by_fc_vgg16bn_refined('hv', ds)

    for ds in hv_dictsizes:
        print('run_df_MIT')
        run_df_MIT.by_cp_resnext50('hv', ds)
        run_df_MIT.by_fc_resnext50('hv', ds)
        run_df_MIT.by_cp_resnext50_refined('hv', ds)
        run_df_MIT.by_fc_resnext50_refined('hv', ds)

        run_df_MIT.by_cp_vgg16bn('hv', ds)
        run_df_MIT.by_fc_vgg16bn('hv', ds)
        run_df_MIT.by_cp_vgg16bn_refined('hv', ds)
        run_df_MIT.by_fc_vgg16bn_refined('hv', ds)

    for ds in hv_dictsizes:
        print('run_df_NWP')
        run_df_NWP.by_cp_resnext50('hv', ds)
        run_df_NWP.by_fc_resnext50('hv', ds)
        run_df_NWP.by_cp_resnext50_refined('hv', ds)
        run_df_NWP.by_fc_resnext50_refined('hv', ds)

        run_df_NWP.by_cp_vgg16bn('hv', ds)
        run_df_NWP.by_fc_vgg16bn('hv', ds)
        run_df_NWP.by_cp_vgg16bn_refined('hv', ds)
        run_df_NWP.by_fc_vgg16bn_refined('hv', ds)

    for ds in hv_dictsizes:
        print('run_df_CAL')
        run_df_CAL.by_cp_resnext50('hv', ds)
        run_df_CAL.by_fc_resnext50('hv', ds)
        run_df_CAL.by_cp_resnext50_refined('hv', ds)
        run_df_CAL.by_fc_resnext50_refined('hv', ds)

        run_df_CAL.by_cp_vgg16bn('hv', ds)
        run_df_CAL.by_fc_vgg16bn('hv', ds)
        run_df_CAL.by_cp_vgg16bn_refined('hv', ds)
        run_df_CAL.by_fc_vgg16bn_refined('hv', ds)


# Copyright (C) 2021-2023 Ye Xu, Xin Zhang, Chongpeng Huang, Xiaorong Qiu

import run_df_15S, run_df_TFF, run_df_COV, run_df_MIT, run_df_NWP, run_df_CAL

if __name__ == '__main__':
    svc_dictsizes = [1, 2, 4, 8, 16, 32]

    for ds in svc_dictsizes:
        run_df_15S.by_cp_resnext50('svc', ds)
        run_df_15S.by_fc_resnext50('svc', ds)
        run_df_15S.by_cp_resnext50_refined('svc', ds)
        run_df_15S.by_fc_resnext50_refined('svc', ds)

        if ds <= 32:
            run_df_15S.by_cp_vgg16bn('svc', ds)
            run_df_15S.by_cp_vgg16bn_refined('svc', ds)
        if ds <= 16:
            run_df_15S.by_fc_vgg16bn('svc', ds)
            run_df_15S.by_fc_vgg16bn_refined('svc', ds)


    # for ds in svc_dictsizes:
    #     run_df_TFF.by_cp_resnext50('svc', ds)
    #     run_df_TFF.by_fc_resnext50('svc', ds)
    #     run_df_TFF.by_cp_resnext50_refined('svc', ds)
    #     run_df_TFF.by_fc_resnext50_refined('svc', ds)
    #
    #     if ds <= 32:
    #         run_df_TFF.by_cp_vgg16bn('svc', ds)
    #         run_df_TFF.by_cp_vgg16bn_refined('svc', ds)
    #     if ds <= 16:
    #         run_df_TFF.by_fc_vgg16bn('svc', ds)
    #         run_df_TFF.by_fc_vgg16bn_refined('svc', ds)
    #

    # for ds in svc_dictsizes:
    #     run_df_UIUC.by_cp_resnext50('svc', ds)
    #     run_df_UIUC.by_fc_resnext50('svc', ds)
    #     run_df_UIUC.by_cp_resnext50_refined('svc', ds)
    #     run_df_UIUC.by_fc_resnext50_refined('svc', ds)
    #
    #     if ds <= 32:
    #         run_df_UIUC.by_cp_vgg16bn('svc', ds)
    #         run_df_UIUC.by_cp_vgg16bn_refined('svc', ds)
    #     if ds <= 16:
    #         run_df_UIUC.by_fc_vgg16bn('svc', ds)
    #         run_df_UIUC.by_fc_vgg16bn_refined('svc', ds)
    #
    # for ds in svc_dictsizes:
    #     run_df_COV.by_cp_resnext50('svc', ds)
    #     run_df_COV.by_fc_resnext50('svc', ds)
    #     run_df_COV.by_cp_resnext50_refined('svc', ds)
    #     run_df_COV.by_fc_resnext50_refined('svc', ds)
    #
    #     if ds <= 32:
    #         run_df_COV.by_cp_vgg16bn('svc', ds)
    #         run_df_COV.by_cp_vgg16bn_refined('svc', ds)
    #     if ds <= 16:
    #         run_df_COV.by_fc_vgg16bn('svc', ds)
    #         run_df_COV.by_fc_vgg16bn_refined('svc', ds)

    for ds in svc_dictsizes:
        if ds <= 16:
            run_df_CAL.by_cp_resnext50('svc', ds)
            run_df_CAL.by_fc_resnext50('svc', ds)
            run_df_CAL.by_cp_resnext50_refined('svc', ds)
            run_df_CAL.by_fc_resnext50_refined('svc', ds)
            run_df_CAL.by_cp_vgg16bn('svc', ds)
            run_df_CAL.by_cp_vgg16bn_refined('svc', ds)
        if ds <= 8:
            run_df_CAL.by_fc_vgg16bn('svc', ds)
            run_df_CAL.by_fc_vgg16bn_refined('svc', ds)

    for ds in svc_dictsizes:
        if ds <= 16:
            run_df_MIT.by_cp_resnext50('svc', ds)
            run_df_MIT.by_fc_resnext50('svc', ds)
            run_df_MIT.by_cp_resnext50_refined('svc', ds)
            run_df_MIT.by_fc_resnext50_refined('svc', ds)
            run_df_MIT.by_cp_vgg16bn('svc', ds)
            run_df_MIT.by_cp_vgg16bn_refined('svc', ds)
        if ds <= 4:
            run_df_MIT.by_fc_vgg16bn('svc', ds)
            run_df_MIT.by_fc_vgg16bn_refined('svc', ds)

    for ds in svc_dictsizes:
        if ds <= 8:
            run_df_NWP.by_cp_resnext50('svc', ds)
            run_df_NWP.by_fc_resnext50('svc', ds)
            run_df_NWP.by_cp_resnext50_refined('svc', ds)
            run_df_NWP.by_fc_resnext50_refined('svc', ds)
            run_df_NWP.by_cp_vgg16bn('svc', ds)
            run_df_NWP.by_cp_vgg16bn_refined('svc', ds)
        if ds <= 4:
            run_df_NWP.by_fc_vgg16bn('svc', ds)
            run_df_NWP.by_fc_vgg16bn_refined('svc', ds)


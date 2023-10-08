# Copyright (C) 2021-2023 Ye Xu, Xin Zhang, Chongpeng Huang, Xiaorong Qiu

import run_df_15S, run_df_TFF, run_df_COV, run_df_MIT, run_df_NWP, run_df_CAL

if __name__ == '__main__':
    # sv_dictsizes = [2, 8, 32, 128, 512, 1024, 2048, 4096, 8192]
    sv_dictsizes = [16, 64, 256]

    for ds in sv_dictsizes:
        run_df_15S.by_cp_resnext50('sv', ds)
        run_df_15S.by_fc_resnext50('sv', ds)
        run_df_15S.by_cp_resnext50_refined('sv', ds)
        run_df_15S.by_fc_resnext50_refined('sv', ds)

        run_df_15S.by_cp_vgg16bn('sv', ds)
        run_df_15S.by_fc_vgg16bn('sv', ds)
        run_df_15S.by_cp_vgg16bn_refined('sv', ds)
        run_df_15S.by_fc_vgg16bn_refined('sv', ds)
    #
    # for ds in sv_dictsizes:
    #     run_df_TFF.by_cp_resnext50('sv', ds)
        # run_df_TFF.by_fc_resnext50('sv', ds)
        # run_df_TFF.by_cp_resnext50_refined('sv', ds)
        # run_df_TFF.by_fc_resnext50_refined('sv', ds)
        #
        # run_df_TFF.by_cp_vgg16bn('sv', ds)
        # run_df_TFF.by_fc_vgg16bn('sv', ds)
        # run_df_TFF.by_cp_vgg16bn_refined('sv', ds)
        # run_df_TFF.by_fc_vgg16bn_refined('sv', ds)

    # for ds in sv_dictsizes:
    #     run_df_COV.by_cp_resnext50('sv', ds)
        # run_df_COV.by_fc_resnext50('sv', ds)
        # run_df_COV.by_cp_resnext50_refined('sv', ds)
        # run_df_COV.by_fc_resnext50_refined('sv', ds)
        #
        # run_df_COV.by_cp_vgg16bn('sv', ds)
        # run_df_COV.by_fc_vgg16bn('sv', ds)
        # run_df_COV.by_cp_vgg16bn_refined('sv', ds)
        # run_df_COV.by_fc_vgg16bn_refined('sv', ds)
    #
    # for ds in sv_dictsizes:
    #     run_df_MIT.by_cp_resnext50('sv', ds)
    #     run_df_MIT.by_fc_resnext50('sv', ds)
    #     run_df_MIT.by_cp_resnext50_refined('sv', ds)
    #     run_df_MIT.by_fc_resnext50_refined('sv', ds)
    #
    #     run_df_MIT.by_cp_vgg16bn('sv', ds)
    #     run_df_MIT.by_fc_vgg16bn('sv', ds)
    #     run_df_MIT.by_cp_vgg16bn_refined('sv', ds)
    #     run_df_MIT.by_fc_vgg16bn_refined('sv', ds)
    # #
    # for ds in sv_dictsizes:
    #     run_df_NWP.by_cp_resnext50('sv', ds)

        # run_df_NWP.by_fc_resnext50('sv', ds)
        # run_df_NWP.by_cp_resnext50_refined('sv', ds)
        # run_df_NWP.by_fc_resnext50_refined('sv', ds)
        #
        # run_df_NWP.by_cp_vgg16bn('sv', ds)
        # run_df_NWP.by_fc_vgg16bn('sv', ds)
        # run_df_NWP.by_cp_vgg16bn_refined('sv', ds)
        # run_df_NWP.by_fc_vgg16bn_refined('sv', ds)

    # for ds in sv_dictsizes:
    #     run_df_CAL.by_cp_resnext50('sv', ds)
    #     run_df_CAL.by_fc_resnext50('sv', ds)
    #     run_df_CAL.by_cp_resnext50_refined('sv', ds)
    #     run_df_CAL.by_fc_resnext50_refined('sv', ds)
    #
    #     run_df_CAL.by_cp_vgg16bn('sv', ds)
    #     run_df_CAL.by_fc_vgg16bn('sv', ds)
    #     run_df_CAL.by_cp_vgg16bn_refined('sv', ds)
    #     run_df_CAL.by_fc_vgg16bn_refined('sv', ds)


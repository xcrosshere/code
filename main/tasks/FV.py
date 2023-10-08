# Copyright (C) 2021-2023 Ye Xu, Xin Zhang, Chongpeng Huang, Xiaorong Qiu

import run_df_15S, run_df_TFF, run_df_COV, run_df_MIT, run_df_NWP, run_df_CAL

if __name__ == '__main__':
    fv_dictsizes = [1,2, 4, 8, 16, 32, 64]

    for ds in fv_dictsizes:
        print('<-' * 25 + '15-Scenes' + '->' * 25)

        run_df_15S.by_cp_resnext50('fv', ds)
        run_df_15S.by_fc_resnext50('fv', ds)
        run_df_15S.by_cp_resnext50_refined('fv', ds)
        run_df_15S.by_fc_resnext50_refined('fv', ds)

        run_df_15S.by_cp_vgg16bn('fv', ds)
        run_df_15S.by_fc_vgg16bn('fv', ds)
        run_df_15S.by_cp_vgg16bn_refined('fv', ds)
        run_df_15S.by_fc_vgg16bn_refined('fv', ds)

        run_df_15S.by_cp_swinb('fv', ds)
        run_df_15S.by_fc_swinb('fv', ds)
        run_df_15S.by_cp_swinb_refined('fv', ds)
        run_df_15S.by_fc_swinb_refined('fv', ds)

    # for ds in fv_dictsizes:
    #     print('<-' * 25 + 'TF-Flowers' + '->' * 25)
    #
    #     run_df_TFF.by_cp_vgg16bn('fv', ds)
    #     run_df_TFF.by_fc_vgg16bn('fv', ds)
    #     run_df_TFF.by_cp_vgg16bn_refined('fv', ds)
    #     run_df_TFF.by_fc_vgg16bn_refined('fv', ds)
    #
    #     run_df_TFF.by_cp_resnext50('fv', ds)
    #     run_df_TFF.by_fc_resnext50('fv', ds)
    #     run_df_TFF.by_cp_resnext50_refined('fv', ds)
    #     run_df_TFF.by_fc_resnext50_refined('fv', ds)
    #
    #     run_df_TFF.by_cp_swinb('fv', ds)
    #     run_df_TFF.by_fc_swinb('fv', ds)
    #     run_df_TFF.by_cp_swinb_refined('fv', ds)
    #     run_df_TFF.by_fc_swinb_refined('fv', ds)

# for ds in fv_dictsizes:
    #     print('<-' * 25 + 'COVID' + '->' * 25)
    #
    #     run_df_COV.by_cp_vgg16bn('fv', ds)
    #     run_df_COV.by_fc_vgg16bn('fv', ds)
    #     run_df_COV.by_cp_vgg16bn_refined('fv', ds)
    #     run_df_COV.by_fc_vgg16bn_refined('fv', ds)
    #
    #     run_df_COV.by_cp_resnext50('fv', ds)
    #     run_df_COV.by_fc_resnext50('fv', ds)
    #     run_df_COV.by_cp_resnext50_refined('fv', ds)
    #     run_df_COV.by_fc_resnext50_refined('fv', ds)
    #
    #     run_df_COV.by_cp_swinb('fv', ds)
    #     run_df_COV.by_fc_swinb('fv', ds)
    #     run_df_COV.by_cp_swinb_refined('fv', ds)
    #     run_df_COV.by_fc_swinb_refined('fv', ds)

    # for ds in fv_dictsizes:
    #     print('<-' * 25 + 'MIT Indoor' + '->' * 25)
    #
    #     run_df_MIT.by_cp_vgg16bn('fv', ds)
    #     run_df_MIT.by_fc_vgg16bn('fv', ds)
    #     run_df_MIT.by_cp_vgg16bn_refined('fv', ds)
    #     run_df_MIT.by_fc_vgg16bn_refined('fv', ds)
    #
    #     run_df_MIT.by_cp_resnext50('fv', ds)
    #     run_df_MIT.by_fc_resnext50('fv', ds)
    #     run_df_MIT.by_cp_resnext50_refined('fv', ds)
    #     run_df_MIT.by_fc_resnext50_refined('fv', ds)
    #
    #     run_df_MIT.by_cp_swinb('fv', ds)
    #     run_df_MIT.by_fc_swinb('fv', ds)
    #     run_df_MIT.by_cp_swinb_refined('fv', ds)
    #     run_df_MIT.by_fc_swinb_refined('fv', ds)

# for ds in fv_dictsizes:
    #     print('<-' * 25 + 'NWPU' + '->' * 25)
    #
    #     run_df_NWP.by_cp_vgg16bn('fv', ds)
    #     run_df_NWP.by_fc_vgg16bn('fv', ds)
    #     run_df_NWP.by_cp_vgg16bn_refined('fv', ds)
    #     run_df_NWP.by_fc_vgg16bn_refined('fv', ds)
    #
    #     run_df_NWP.by_cp_resnext50('fv', ds)
    #     run_df_NWP.by_fc_resnext50('fv', ds)
    #     run_df_NWP.by_cp_resnext50_refined('fv', ds)
    #     run_df_NWP.by_fc_resnext50_refined('fv', ds)
    #
    #     run_df_NWP.by_cp_swinb('fv', ds)
    #     run_df_NWP.by_fc_swinb('fv', ds)
    #     run_df_NWP.by_cp_swinb_refined('fv', ds)
    #     run_df_NWP.by_fc_swinb_refined('fv', ds)

    # for ds in fv_dictsizes:
    #     print('<-' * 25 + 'Caltech 101' + '->' * 25)
    #
    #     run_df_CAL.by_cp_vgg16bn('fv', ds)
    #     run_df_CAL.by_fc_vgg16bn('fv', ds)
    #     run_df_CAL.by_cp_vgg16bn_refined('fv', ds)
    #     run_df_CAL.by_fc_vgg16bn_refined('fv', ds)
    #
    #     run_df_CAL.by_cp_resnext50('fv', ds)
    #     run_df_CAL.by_fc_resnext50('fv', ds)
    #     run_df_CAL.by_cp_resnext50_refined('fv', ds)
    #     run_df_CAL.by_fc_resnext50_refined('fv', ds)
    #
    #     run_df_CAL.by_cp_swinb('fv', ds)
    #     run_df_CAL.by_fc_swinb('fv', ds)
    #     run_df_CAL.by_cp_swinb_refined('fv', ds)
    #     run_df_CAL.by_fc_swinb_refined('fv', ds)
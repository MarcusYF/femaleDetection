import os
import scipy.sparse as sp
import numpy as np
import ETLUtil
import itertools


def trim_zero_features(sp_matrix):
    return sp_matrix[:, (sp_matrix.sum(0) > 0).A[0]]

working_dir = os.getcwd() + '/data/'
# posi_filename = 'jiufu_10+'
# nega_filename = 'daku2017'
posi_filename = 'gender1'
nega_filename = 'gender0'
posi_ratio = 1
nega_ratio = 1

posi_X, nega_X, train_X, train_y, test_X, test_y, test_X_, test_y_, posi_num, nega_num, feature_num \
    = ETLUtil.load_data(working_dir, posi_filename, nega_filename, posi_ratio, nega_ratio, selec_prominent_instance=0)


# table, fugai, sign_f = ETLUtil.gen_stats(posi_X, nega_X)

def comput_cross_feat(train_X):
    is_feat = (train_X.sum(0) > 10).tolist()[0]
    feat_index = []
    for i, v in enumerate(is_feat):
        if v:
            feat_index.append(i)

    # aoi_feat = [x for x in feat_index if x < 35]
    mod_feat = [x for x in feat_index if 35 < x < 100]
    app_feat = [x for x in feat_index if x > 100]

    instance_num = train_X.shape[0]

    cross_feature = sp.csc_matrix(np.ones((instance_num, 1)))
    for x in itertools.product(mod_feat, app_feat):
        print(x)
        cross_ind = list(set(train_X[:, x[0]].indices) & set(train_X[:, x[1]].indices))
        n = len(cross_ind)
        new_feature = sp.csc_matrix(([1] * n, (cross_ind, [0] * n)), (instance_num, 1))
        cross_feature = sp.hstack((cross_feature, new_feature))

    return cross_feature

cross_feature = comput_cross_feat(train_X)
train_ = sp.hstack((train_X, cross_feature))
test_ = sp.hstack((test_X, cross_feature))

train = trim_zero_features(train_)
test = trim_zero_features(test_)

x, lift, w = BenchMarks.ModelCollection.runLR_benchmark(train, train_y, test, test_y, C=.01)

x, lift, w = BenchMarks.ModelCollection.runLR_benchmark(train[:, :205], train_y, test[:, :205], test_y, C=.001)

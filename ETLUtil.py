# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 17:51:03 2017

@author: yaofan29597
"""

from sklearn.datasets import load_svmlight_file
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random


def load_data(working_dir, posi_filename, nega_filename, posi_ratio, nega_ratio, **param):

    # 读入数据集
    posi_X0, _ = load_svmlight_file(working_dir + posi_filename)
    nega_X, _ = load_svmlight_file(working_dir + nega_filename)
    posi_y = np.ones((posi_X0.shape[0],))
    nega_y = -np.ones((nega_X.shape[0],))
    if nega_X.shape[1] != posi_X0.shape[1]:
        posi_X = sp.hstack((posi_X0, np.zeros((posi_X0.shape[0], nega_X.shape[1] - posi_X0.shape[1]))))
    else:
        posi_X = posi_X0
    posi_num = posi_X.shape[0]
    nega_num = nega_X.shape[0]
    feature_num = posi_X.shape[1]

    selec_posi = np.array(random.sample(range(posi_num), int(posi_num * posi_ratio)))

    unselec_posi = np.array(list(set(range(posi_num)) - set(selec_posi)))
    selec_nega_from1 = np.array(random.sample(list(unselec_posi), int(len(unselec_posi) * nega_ratio)))
    selec_nega_from0 = np.array(random.sample(range(nega_num), int(nega_num * nega_ratio)))

    posi_X_visible = posi_X[selec_posi, :]
    posi_X_unvisible = posi_X[unselec_posi, :]
    nega_X_visible = sp.vstack((posi_X[selec_nega_from1, :], nega_X[selec_nega_from0, :]))

    # 全量样本作为测试集
    test_X = sp.vstack((posi_X, nega_X)).tocsc()
    test_y = np.concatenate((posi_y, nega_y))
    # 不包括训练样本的测试集
    test_X_ = sp.vstack((posi_X_unvisible, nega_X)).tocsc()
    test_y_ = np.concatenate((np.ones((posi_X_unvisible.shape[0],)), nega_y))
    # 可见正样本和不明样本作为训练集
    if param.get('selec_prominent_instance'):
        IND = (posi_X_visible.sum(1) > param.get('selec_prominent_instance')).A[:, 0]
        posi_X_visible = posi_X_visible[IND, :]

    train_X = sp.vstack((posi_X_visible, nega_X_visible)).tocsc()
    train_y = np.concatenate(
        (np.ones((posi_X_visible.shape[0],)), np.zeros((len(selec_nega_from0) + len(selec_nega_from1)), )))
    train_y[train_y == -1] = 0
    test_y[test_y == -1] = 0
    test_y_[test_y_ == -1] = 0
    return posi_X, nega_X, train_X, train_y, test_X, test_y, test_X_, test_y_, posi_num, nega_num, feature_num


def gen_stats(posi_X, nega_X):
    posi_num = posi_X.shape[0]
    nega_num = nega_X.shape[0]
    feature_num = posi_X.shape[1]
    feature_freq_posi = list(filter(lambda x: x[0] > 0, list(zip(posi_X.sum(0).A[0] / posi_num, range(feature_num)))))
    feature_freq_nega = list(filter(lambda x: x[0] > 0, list(zip(nega_X.sum(0).A[0] / nega_num, range(feature_num)))))

    posi_df = pd.DataFrame(feature_freq_posi, columns=['freq', 'index'])
    nega_df = pd.DataFrame(feature_freq_nega, columns=['freq_0', 'index'])
    posi_df['index'] = posi_df['index'].map(lambda x: x + 1)
    nega_df['index'] = nega_df['index'].map(lambda x: x + 1)
    posi_df = posi_df.sort_values(by='freq', axis=0, ascending=False)
    nega_df = nega_df.sort_values(by='freq_0', axis=0, ascending=False)

    a = pd.merge(posi_df, nega_df, how='inner', on=['index'])
    a['freq/freq_0'] = a.apply(lambda x: x[0] / x[2], axis=1)

    sqrt_n = np.sqrt(posi_num)
    a['signif'] = a.apply(lambda x: abs(x[3] - 1) * sqrt_n / np.sqrt(1 / x[2] - 1), axis=1)
    mapReadFromDF = genMap_Feat2Name()
    a['name'] = a['index'].map(lambda x: mapReadFromDF[x])
    stat_table = a[['index', 'name', 'freq_0', 'freq', 'freq/freq_0', 'signif']]
    # 选出超过5%且显著性水平>10的特征
    signif_features = (a[(a['freq'] > 0.01) & (a['signif'] > 20)]['index'] - 1).tolist()
    X_mini = posi_X.tocsc()[:, signif_features]
    #    X_mini_nega = nega_X.tocsc()[:, signif_features]
    #    X = sp.vstack((X_mini, X_mini_nega))

    # 特征覆盖率 
    #    X_mini_no0 = X_mini[X_mini.sum(1).A[:,0].nonzero()[0],:]
    #    X_mini_nega_no0 = X_mini_nega[X_mini_nega.sum(1).A[:,0].nonzero()[0],:]
    fugai = X_mini.sum(1).A[:, 0].nonzero()[0].shape[0] / posi_num
    return stat_table, fugai, signif_features


def mapFeat2Name(w):


    wZipInd = list(zip(w, range(1, len(w)+1)))
    wGeq1 = [x for x in wZipInd if abs(x[0]) > 0]
    mapReadFromDF = genMap_Feat2Name()
    res = [(x[0], mapReadFromDF[x[1]]) for x in wGeq1]
    res.sort()
    return res


def genMap_Feat2Name():
    path = '/Users/yaofan29597/Desktop/Princetechs/学习资料/research/Lookalike/app_label_category/'
    path_tag_name = path + 'tag_name.csv'
    path_loc_Ind = path + 'loc_Ind.csv'

    tag_name = pd.read_csv(path_tag_name)
    loc_ind = pd.read_csv(path_loc_Ind, header=None)
    loc_ind.columns = ['name', 'tagId']
    tagName = tag_name.loc[:, ['tagId', 'name']]

    a = dict(zip(tagName['tagId'], tagName['name']))
    b = dict(zip(loc_ind['tagId'], loc_ind['name']))
    return {**a, **b}

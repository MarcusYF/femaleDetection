# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 17:10:53 2017

@author: yaofan29597
"""
import os
import scipy.sparse as sp
import random
from sklearn.metrics import roc_auc_score
import ETLUtil

# import sys
# del sys.modules["ETLUtil"]
import BenchMarks

import numpy as np
import matplotlib.pyplot as plt


# 读入数据集

working_dir = os.getcwd() + '/data/'
# posi_filename = 'jiufu_tag_withAoi_libsvm'
# posi_filename = '10+_libsvm'
posi_filename = 'jiufu_10+'
# nega_filename = 'nega_tag_withAoi_libsvm'
# nega_filename = '10-20_libsvm'
nega_filename = 'daku2017'
posi_ratio = 1
nega_ratio = 1

posi_X, nega_X, train_X, train_y, test_X, test_y, test_X_, test_y_, posi_num, nega_num, feature_num \
    = ETLUtil.load_data(working_dir, posi_filename, nega_filename, posi_ratio, nega_ratio, selec_prominent_instance=0)

table, fugai, sign_f = ETLUtil.gen_stats(posi_X, nega_X)

# LR benchmark
x, lift, w = BenchMarks.ModelCollection.runLR_benchmark(train_X, train_y, test_X, test_y, C=100)
feat_score = ETLUtil.mapFeat2Name(w[0])
# NB benchmark
x1, lift1 = BenchMarks.ModelCollection.runNB_benchmark(train_X, train_y, test_X, test_y)
# Self-learning LR
x2, lift2 = BenchMarks.ModelCollection.runSelfLearning_LR(train_X, train_y, test_X, test_y, C=100, init_seed_size_n=10000, addp=1000, addn=1000)

plt.plot(x, lift)
# # plt.plot(x2, lift2, 'r')
plt.show(block=True)
plt.plot(x1, lift1, 'g')
plt.plot(x2, lift2, 'r')

lift2 = []
for i in range(len(x1)):
    lift2.append((lift1[i] * x1[i] + 5812) / (x1[i] + 5812))

#%% PCA
from sklearn.decomposition import PCA

#pca = PCA(n_components=2)
#pca.fit(X.todense())
#print(pca.explained_variance_ratio_)
#X_decomp = pca.fit_transform(X.todense())

#%% MDS
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform

mds = MDS(n_components=2,dissimilarity='precomputed')
k = 200
XX = sp.vstack((X_mini_no0[:k,:], X_mini_nega_no0[:800,:]))


#t = pdist(XX.todense(), 'wminkowski', p=1, w=np.ones((n_features, )))
t = pdist(XX.todense(), 'minkowski', p=2)
Y = squareform(t, force='no', checks=True)
#mds.fit(Y)
X_decomp = mds.fit_transform(Y)

import pylab as plt
plt.figure(figsize = (4,4))

plt.scatter(X_decomp[k:, 0], X_decomp[k:, 1], c='b', marker='o')
plt.scatter(X_decomp[:k, 0], X_decomp[:k, 1], c='r', marker='o', linewidths=1, edgecolors='r')

save_path = '/Users/yaofan29597/Desktop/Princetechs/学习资料/research/Lookalike/'
f = open(save_path + 'dist_Ouc' + job_code + 'withNega.txt', 'w')
for i in range(Y.shape[0]):
#    print(i)
    for j in range(Y.shape[0]):
        line = '%.3f' % Y[i,j] + ' '
        f.write(line)
    f.write('\n')
f.close()

#%% T-SNE
from sklearn.manifold import TSNE


digits_proj = TSNE(random_state=0).fit_transform(XX.todense())

plt.plot(x, lift)
plt.plot(x1, lift1, 'r')

k=200
plt.figure(figsize = (4,4))
plt.scatter(digits_proj[k:, 0], digits_proj[k:, 1], c='b', marker='o')
plt.scatter(digits_proj[:k, 0], digits_proj[:k, 1], c='r', marker='o')
plt.ylim((-0.2*1e15,0.2*1e15))
plt.xlim((-0.2*1e15,0.2*1e15))

#%% Isomap
from sklearn.manifold import Isomap
isomap = Isomap(n_neighbors=5, n_components=2, eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto')
X_decomp = isomap.fit_transform(XX.todense())

import pylab as plt
plt.figure(figsize = (4,4))
plt.scatter(X_decomp[k:, 0], X_decomp[k:, 1], c='b', marker='o')
plt.scatter(X_decomp[:k, 0], X_decomp[:k, 1], c='r', marker='o', linewidths=1, edgecolors='r')


#%% NB
from sklearn.naive_bayes import BernoulliNB
nb = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)

posi_ratio = .2

posi_XX = X_mini_no0
nega_XX = X_mini_nega_no0

#ind1 = random.sample(range(nega_XX.shape[0]), posi_XX.shape[0])
#posi_XX = X_mini_nega_no0[ind1, :]


non0 = nega_XX.sum(0).A[0]>0
posi_XX = posi_XX[:,non0]
nega_XX = nega_XX[:,non0]
train_XX = sp.vstack((posi_XX, nega_XX))
test_XX = sp.vstack((posi_XX, nega_XX))

ind0 = random.sample(range(posi_XX.shape[0]), int(posi_XX.shape[0]*posi_ratio))
ones = np.zeros((posi_XX.shape[0], ))
ones[np.array(ind0)] = 1
train_y = np.concatenate((ones, np.zeros((nega_XX.shape[0], ))))
test_y = np.concatenate((np.ones((posi_XX.shape[0], )), np.zeros((nega_XX.shape[0], ))))

nb.fit(train_XX, train_y)
prob = nb.predict_proba(test_XX)


res = list(zip(prob[:, 1].tolist(), test_y.tolist()))
res.sort()
#res1 = list(filter(lambda x:x[0] > thres, res))
res1 = res[-posi_num:]
hit_num = len(list(filter(lambda x:x[1] == 1, res1)))

acc = hit_num / len(res1)
rec = hit_num / len(posi_y)
auc = roc_auc_score(test_y, prob[:,1])
lift = acc / posi_num * len(test_y)
print(acc)
print(rec)
print(auc)




# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:04:11 2017

@author: yaofan29597
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import BernoulliNB


def comput_metric(prob, test_y):
    res = list(zip(prob[:, 1].tolist(), test_y.tolist()))
    res.sort()
    posi_num = int(test_y.sum())
    res1 = res[-posi_num:]
    hit_num = len(list(filter(lambda x: x[1] == 1, res1)))
    acc = hit_num / len(res1)
    # rec = hit_num / test_y.sum()
    auc = roc_auc_score(test_y, prob[:, 1])
    print(acc)
    # print(rec)
    print(auc)
    return res

def comput_lift(sorted_label):
    n = len(sorted_label)
    lift = []
    count_posi = 0
    for i in range(n):
        if sorted_label[i] == 1:
            count_posi += 1 
        lift.append( count_posi / (i+1) )
    return lift

class ModelCollection:

    @staticmethod
    def runLR_benchmark(train_X, train_y, test_X, test_y, **param):

        lr = LogisticRegression(C = param.get('C'), random_state=0)
        lr.fit(train_X, train_y)

        prob = lr.predict_proba(test_X)
        res = comput_metric(prob, test_y)
        sorted_label = list(map(lambda x:x[1], res))
        sorted_label.reverse()
        lift = comput_lift(sorted_label)

        return list(range(1, 1+len(lift))), lift, lr.coef_

    @staticmethod
    def runNB_benchmark(train_X, train_y, test_X, test_y, **param):
        
        
        
        nb = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
        train_y[train_y == -1] = 0
        

                
        
        non0 = test_X.sum(0).A[0]>0
        train_XX = train_X[:,non0]
        test_XX = test_X[:,non0]
        
        
        nb.fit(train_XX, train_y)
        prob = nb.predict_proba(test_XX)
        res = list(zip(prob[:, 1].tolist(), test_y.tolist()))
        res.sort()
        #res1 = list(filter(lambda x:x[0] > thres, res))
        posi_num = int(test_y.sum())
        res1 = res[-posi_num:]
        hit_num = len(list(filter(lambda x:x[1] == 1, res1)))  
        acc = hit_num / len(res1)
        # rec = hit_num / test_y.sum()
        auc = roc_auc_score(test_y, prob[:,1])
        print(acc)
        # print(rec)
        print(auc)
        
        a = list(map(lambda x:x[1], res1))
        a.reverse()
        lift = comput_lift(a)
        return list(range(1, 1+len(lift))), lift

    @staticmethod
    def runSelfLearning_LR(train_X, train_y, test_X, test_y, **param):    
        
        n_instance = test_y.shape[0]
        
        instance_added_per_round_p = param.get('addp')
        instance_added_per_round_n = param.get('addn')
        init_seed_size_n = param.get('init_seed_size_n')
        init_seed_size_p = int(train_y.sum())
        posi_num = int(test_y.sum())
        
        label = np.array([0] * n_instance)
        
        posi_X = test_X[:init_seed_size_p, :]
        label[:init_seed_size_p] = 1
        nega_X = test_X[-init_seed_size_n:, :]
        label[-init_seed_size_n:] = -1

        lr = LogisticRegression(C = param.get('C'), warm_start=True)

        
        x = []
        lift = []


        max_ite = int( (posi_num - init_seed_size_p) / instance_added_per_round_p)
        
        print(max_ite)
        for i in range(max_ite):    
            XX = sp.vstack((posi_X, nega_X))
            y = [1] * posi_X.shape[0] + [0] * nega_X.shape[0]
            lr.fit(XX, y)
            prob = lr.predict_proba(test_X)
        
#            if i == 0:
#                res = comput_metric(prob, test_y)
#                sorted_label = list(map(lambda x:x[1], res))
#                sorted_label.reverse()
#                lift_ = comput_lift(sorted_label)
        
            # 找出最有可能是+的100个下标posi_ind
            posi_dic = list(zip(prob[:, 1].tolist(), range(n_instance)))
            t = list(filter(lambda x:  label[x[1]]==0, posi_dic))
            t.sort()
            tt = t[-instance_added_per_round_p:]
            posi_ind = list(map(lambda x:x[1],tt))
            tt0 = t[:instance_added_per_round_n]
            nega_ind = list(map(lambda x:x[1],tt0))
            
            # 更新label, posi_X, nega_X
            label[posi_ind] = 1
            label[nega_ind] = -1
            posi_X = test_X[label == 1, :]
            nega_X = test_X[label == -1, :]
        
            # 计算指标
            hit = (label[init_seed_size_p:posi_num] == 1).sum()
            labeled_p = (label[init_seed_size_p:] == 1).sum()
        
            prec = float(hit) / (labeled_p)
            x.append(labeled_p)            
            lift.append(prec)
            
            print(i)            
            
        return x, lift#, lift_



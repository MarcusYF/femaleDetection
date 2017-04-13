import os
import scipy.sparse as sp
import ETLUtil


working_dir = os.getcwd() + '/data/'
posi_filename = 'jiufu_10+'
nega_filename = 'daku2017'
posi_ratio = 1
nega_ratio = 1

posi_X, nega_X, train_X, train_y, test_X, test_y, test_X_, test_y_, posi_num, nega_num, feature_num \
    = ETLUtil.load_data(working_dir, posi_filename, nega_filename, posi_ratio, nega_ratio, selec_prominent_instance=0)

train = train_X.tolil()
test = test_X.tolil()
# table, fugai, sign_f = ETLUtil.gen_stats(posi_X, nega_X)
is_feat = (train_X.sum(0) > 10).tolist()[0]
feat_index = []
for i, v in enumerate(is_feat):
    if v:
        feat_index.append(i)

aoi_feat = [x for x in feat_index if x < 35]
mod_feat = [x for x in feat_index if 35 < x < 100]
app_feat = [x for x in feat_index if x > 100]

for
train[:, 0]


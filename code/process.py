#coding:utf-8

import pandas as pd
import numpy as np
import psutil
import gensim
import time
import os

import xgboost as xgb
from xgboost import plot_importance
try:
    from glove import Glove
    from glove import Corpus
except:
    print('have not glove')

from scipy import stats
from tqdm import tqdm
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split


def reduce_memory(data, float_=True, int_=False):
    """
    input a dataframe
    """

    if float_:
        float64_ = data.dtypes[data.dtypes=='float64'].index
        if len(float64_):
            data[float64_] = data[float64_].astype('float32')
    if int_:
        int64_ = data.dtypes[data.dtypes=='int64'].index
        if len(int64_):
            data[int64_] = data[int64_].astype('int32')

    return data




def add_session_col(click_all, col="time", n_session=32):
    """
    n_session: 时间划分session个数，一个phase是4天，每天划分8个时区，一共32个,但是不同天的4个时区相同的汇总一起，共有8个session
    time_min: 时间开始
    time_max: 时间结束
    返回：
        click_all增加"session"列，"session"的划分根据click_all的"time"列。
    """
      
    # 将time扩大1000000倍以便划分session
    time_expand = 1000000*(click_all[col].astype(np.float32))

    time_min = time_expand.min()
    time_max = time_expand.max()
    ################### 增加session信息
    ###################################
    click_all["session"] = 0
    session_size = (time_max - time_min)/n_session
    
    #
    for i in range(1, n_session+1):
        tmp0 = time_min+((i-1)*session_size)
        tmp1 = time_min+(i*session_size)
        if i<n_session:
            click_all.loc[(time_expand>=tmp0) & (time_expand<tmp1), "session"] = i
        else:
            click_all.loc[(time_expand>=tmp0), "session"] = i
    # 将每天相同的时区归为一类
    for x in range(1, 1+int(n_session/4)):
        click_all.loc[click_all.session.isin([x+4*i for i in range(1, 4)]), "session"] = x
            
    return click_all



def load_click_data(phase, nrows=None):
    """
    加载phase的click数据拼接
    
    bug: 原始数据读入时，对time进行str，之后按照time排序，可能有问题。对策：先float排序好之后再str
    """
    
    print('================================== 加载click数据 ==================================')
    click_train = pd.read_csv('../data/underexpose_train/underexpose_train_click-{phase}.csv'.format(phase=phase), 
                             header=None, 
                             nrows=nrows,
                             names=['user_id', 'item_id', 'time'],
                             sep=',',
                             dtype={'user_id':np.str,
                                    'item_id':np.str,
                                    'time':np.float32}
                             )

    click_test = pd.read_csv('../data/underexpose_test/underexpose_test_click-{phase}.csv'.format(phase=phase), 
                             header=None, 
                             nrows=nrows,
                             names=['user_id', 'item_id', 'time'],
                             sep=',',
                             dtype={'user_id':np.str,
                                    'item_id':np.str,
                                    'time':np.float32}
                             )
    
    #拼接
    click_all = click_train.append(click_test) 
    
    #排序去重
    click_all = click_all.sort_values('time')
    click_all = click_all.drop_duplicates(['user_id', 'item_id', 'time'], keep='last')
    click_all['time'] = click_all['time'].astype(np.str)
    
    # test user_id
    test_user_set = set(click_test['user_id'])
    
    # 增加add_session_col
    click_all = add_session_col(click_all)
    
    return click_all, test_user_set


def data_generate(click_all, test_user_set):
    """
    数据生成
    """
    print('================================== 生成中间数据 ==================================')
    #训练测试用户
    set_pred = test_user_set
    set_train = set(click_all['user_id']) - set_pred
    
    #训练用户——最后一次点击项目字典（作为验证）
    temp_ = click_all
    temp_['pred'] = temp_['user_id'].map(lambda x: 'test' if x in set_pred else 'train')
    temp_ = temp_[temp_['pred']=='train'].drop_duplicates(['user_id'], keep='last')
    temp_['remove'] = 'remove'
    dict_label_user_item = dict(zip(temp_['user_id'], temp_['item_id']))
    
    #训练测试数据
    train_test = click_all
    train_test = train_test.merge(temp_, on=['user_id', 'item_id', 'time', 'pred'], how='left')
    train_test = train_test[train_test['remove']!='remove']

    #项目热度字典
    temp_ = train_test.groupby(['item_id'])['user_id'].count().reset_index()
    temp_.sort_values(['user_id'], ascending=False, inplace=True)
    item_hot_list = list(temp_['item_id'])


    #项目-用户数字典
    item_hot_dict = Counter(list(click_all['item_id']))
    
    
    #session用户量、项目量
    temp_ = click_all.groupby(['item_id'])['user_id'].count().reset_index()

    
    #用户点击时间特征
    click_all['time'] = click_all['time'].astype('float32')
    time_feat = click_all.groupby(['user_id'])[['item_id', 'time', "session"]].agg(
        {
        'item_id': lambda x: ','.join(list(x)[::-1]), 
        'time': lambda x: list(x)[::-1],
        'session': lambda x: list(x)[::-1]
        }).reset_index()
    time_feat.columns = ['user_id', 'item_id_list_str', 'time', 'session']
    time_feat['user_click_item_num'] = time_feat.item_id_list_str.map(lambda x: len(x.split(',')))
    time_feat['time_max_min_minus'] = time_feat['time'].map(lambda x: max(x)-min(x))
    time_feat['time_diff_mean'] = time_feat['time'].map(lambda x: pd.Series(x).diff().dropna().mean())
    time_feat['time_diff_mean'] = time_feat['time_diff_mean'].fillna(0)
    time_feat['time_mean'] = time_feat['time'].map(lambda x: pd.Series(x).mean())
    time_feat['time_std'] = time_feat['time'].map(lambda x: pd.Series(x).std())
    time_feat['time_max'] = time_feat['time'].map(lambda x: pd.Series(x).max())
    time_feat['time_min'] = time_feat['time'].map(lambda x: pd.Series(x).min())
    time_feat['different_session_num'] = time_feat['session'].map(lambda x: len(set(x)))
    time_feat['favorite_session_click'] = time_feat['session'].map(lambda x: stats.mode(x)[0][0])
    time_feat.drop(columns=['time', 'session'], axis = 1, inplace=True)

    # train_test = train_test.merge(time_feat, on=['user_id'], how='left')
    time_feat = reduce_memory(time_feat, float_=True, int_=True)
    return train_test, item_hot_list, dict_label_user_item, item_hot_dict, time_feat




def item_cluster_feat(click_all, item_cluster):
    """
    item聚类特征
    """
    data_ = click_all.merge(item_cluster, how='left', on=['item_id'])

    # 每一类的点击数量
    cluster_cnt = data_.groupby('item_cluster').size().reset_index()
    cluster_cnt.columns = ['item_cluster', 'item_cluster_cnt']
    data_ = data_.merge(cluster_cnt, how='left', on=['item_cluster'])

    # 每一类的用户数量
    cluster_user_cnt = data_.groupby('item_cluster')['user_id'].nunique().reset_index()
    cluster_user_cnt.columns = ['item_cluster', 'item_cluster_user_cnt']
    data_ = data_.merge(cluster_user_cnt, how='left', on=['item_cluster'])

    # 每一类的项目数量
    cluster_item_cnt = data_.groupby('item_cluster')['item_id'].nunique().reset_index()
    cluster_item_cnt.columns = ['item_cluster', 'item_cluster_item_cnt']
    data_ = data_.merge(cluster_item_cnt, how='left', on=['item_cluster'])

    #
    data_ = data_[['item_id', 'item_cluster', 'item_cluster_cnt', 'item_cluster_user_cnt', 'item_cluster_item_cnt']]
    # 去重
    data_.drop_duplicates(inplace=True)
    #
    data_[['item_cluster_cnt', 'item_cluster_user_cnt', 'item_cluster_item_cnt']] = data_[['item_cluster_cnt', 'item_cluster_user_cnt', 'item_cluster_item_cnt']].fillna(-1).astype(np.int32)
    return data_

def user_cluster_feat(click_all, user_cluster):
    """
    user聚类特征
    """
    data_ = click_all.merge(user_cluster, how='left', on=['user_id'])

    # 每一类的点击数量
    cluster_cnt = data_.groupby('user_cluster').size().reset_index()
    cluster_cnt.columns = ['user_cluster', 'user_cluster_cnt']
    data_ = data_.merge(cluster_cnt, how='left', on=['user_cluster'])

    # 每一类的用户数量
    cluster_user_cnt = data_.groupby('user_cluster')['user_id'].nunique().reset_index()
    cluster_user_cnt.columns = ['user_cluster', 'user_cluster_user_cnt']
    data_ = data_.merge(cluster_user_cnt, how='left', on=['user_cluster'])

    # 每一类的项目数量
    cluster_item_cnt = data_.groupby('user_cluster')['item_id'].nunique().reset_index()
    cluster_item_cnt.columns = ['user_cluster', 'user_cluster_item_cnt']
    data_ = data_.merge(cluster_item_cnt, how='left', on=['user_cluster'])
    
    #
    data_ = data_[['user_id', 'user_cluster', 'user_cluster_cnt', 'user_cluster_user_cnt', 'user_cluster_item_cnt']]
    # 去重
    data_.drop_duplicates(inplace=True)
    #
    data_[['user_cluster_cnt', 'user_cluster_user_cnt', 'user_cluster_item_cnt']] = data_[['user_cluster_cnt', 'user_cluster_user_cnt', 'user_cluster_item_cnt']].fillna(-1).astype(np.int32)
    
    return data_



def embedding_fea_pca(data, n_components=5):
    """
    reduce dimension with embedding feature.
    30->low dimension
    """
    #标准化
    scaler = StandardScaler()
    scaler.fit(data)
    X = scaler.transform(data)
        
    #进行降维
    pca = PCA(n_components)
    pca.fit(X)
    
    #打印所保留的n个成分各自的方差百分比
    print('------- 降成{n_components}维后的方差百分比: '.format(n_components=n_components), pca.explained_variance_ratio_)
    # print('降成{n_components}维后的方差: '.format(n_components=n_components), pca.explained_variance_)
    
    # 返回降维后的数据
    pca_data=pca.transform(X)
    
    return pca_data



def matrix_glove_embedding(click_all,flag,mode,threshold=0,dim=100,epochs=30,learning_rate=0.5):
    """
        glove 原理 + 矩阵分解：
            窗口内 加权统计 共线性词频
        
        四种向量化方式：
            flag='item' mode='all':
                sku1 sku2 sku3 sku4 sku5 user
            flag='user' mode='all':
                user1 user2 user3 user4 user5 sku
            flag='item',mode='only':
                item1 item2 item3 item4 item5
            flag='user' mode='only'
                user1 user2 user3 user4 user5
    """

    
    if flag == 'user':
        group_by_col, agg_col = 'item_id', 'user_id'
    if flag == 'item':
        group_by_col, agg_col = 'user_id', 'item_id'
    
    data_ = click_all.groupby([group_by_col])[agg_col].agg(lambda x:','.join(list(x))).reset_index()
    if mode == 'only':
        list_data = list(data_[agg_col].map(lambda x:x.split(',')))
    if mode == 'all':
        data_['concat'] = data_[agg_col] + ',' + data_[group_by_col].map(lambda x:'all_'+x)
        list_data = data_['concat'].map(lambda x:x.split(','))
    
    corpus_model = Corpus()
    corpus_model.fit(list_data, window=999999)
    
    glove = Glove(no_components=dim, learning_rate=learning_rate)
    glove.fit(corpus_model.matrix, epochs=epochs, no_threads=psutil.cpu_count(), verbose=True)
    glove.add_dictionary(corpus_model.dictionary)
    
    keys = glove.dictionary.keys()
    if mode == 'only':
        glove_embedding = {flag:{}}
    if mode == 'all':
        glove_embedding = {'user':{},'item':{}}
    for k in keys:
        if 'all' not in k:
            glove_embedding[flag][k] = glove.word_vectors[glove.dictionary[k]]
        if 'all' in k:
            flag_ = group_by_col.split('_')[0]
            k_ = k.split('_')[1]
            glove_embedding[flag_][k_] = glove.word_vectors[glove.dictionary[k]]
            
    return glove_embedding



def matrix_word2vec_embedding(click_all,flag,mode,threshold=0,dim=100,epochs=30,learning_rate=0.5):
    """
        word2vec 原理 skip bow：
            窗口内 预测
        # 注释：doc2vec 有bug，建议不使用
        
        四种向量化方式：
            flag='item' mode='all':
                sku1 sku2 sku3 sku4 sku5 user
            flag='user' mode='all':
                user1 user2 user3 user4 user5 sku
            flag='item',mode='only':
                item1 item2 item3 item4 item5
            flag='user' mode='only'
                user1 user2 user3 user4 user5
    """

    
    if flag == 'user':
        group_by_col, agg_col = 'item_id', 'user_id'
    if flag == 'item':
        group_by_col, agg_col = 'user_id', 'item_id'
    
    data_ = click_all.groupby([group_by_col])[agg_col].agg(lambda x:','.join(list(x))).reset_index()
    if mode == 'only':
        list_data = list(data_[agg_col].map(lambda x:x.split(',')))
    if mode == 'all':
        data_['concat'] = data_[agg_col] + ',' + data_[group_by_col].map(lambda x:'all_'+x)
        list_data = data_['concat'].map(lambda x:x.split(','))
    
    model = gensim.models.Word2Vec(
                    list_data,
                    size=dim,
                    alpha=learning_rate,
                    window=999999,
                    min_count=1,
                    workers=psutil.cpu_count(),
                    compute_loss=True,
                    iter=epochs,
                    hs=0,
                    sg=1,
                    seed=42
                )
    
    # model.build_vocab(list_data, update=True)
    # model.train(list_data, total_examples=model.corpus_count, epochs=model.iter)
    
    keys = model.wv.vocab.keys()
    if mode == 'only':
        word2vec_embedding = {flag:{}}
    if mode == 'all':
        word2vec_embedding = {'user':{},'item':{}}
    for k in keys:
        if 'all' not in k:
            word2vec_embedding[flag][k] = np.float32(model.wv[k])  # 2020-06-06转float32
        if 'all' in k:
            flag_ = group_by_col.split('_')[0]
            k_ = k.split('_')[1]
            word2vec_embedding[flag_][k_] = np.float32(model.wv[k])
            
    return word2vec_embedding





# @profile
def get_train_test_data(
                        topk_recall,
                        matrix_association_rules,
                        time_feat,
                        user_cluster_feat_,
                        item_cluster_feat_,
                        item_hot_dict,
                        txt_cosine_similarity_dict,
                        img_cosine_similarity_dict,
                        dict_embedding_all_ui_item,
                        dict_embedding_all_ui_user,
                        phase,
                        batch,
                        flag_test=False,
                        click_topn=5
                        ):
    """
    topk_recall:
                user_id     item_similar    score_similar   next_item_id    pred    label   keep
                   1            87837       0.657882        69359          train    0       0
    matrix_association_rules:
                cf计算的项目-项目得分字典，如matrix_association_rules['78142']['26646']=0.33891066477581644
    txt_cosine_similarity_dict:
                项目-项目文本向量余弦相似度字典
    img_cosine_similarity_dict:
                项目-项目文本向量余弦相似度字典
    dict_embedding...:
                各种Embedding数据
    user_cluster_feat_: ['user_id', 'user_cluster', 'user_cluster_cnt', 'user_cluster_user_cnt',
       'user_cluster_item_cnt']
    item_cluster_feat_: ['item_id', 'item_cluster', 'item_cluster_cnt', 'item_cluster_user_cnt',
       'item_cluster_item_cnt']

                
    增加各种交互特征：
        1.用户点击的项目数
        2.项目被点击数
        3.用户-项目的多种Embedding特征
        4.召回项目得分
        5.文本与图像余弦相似度
        
    """
    data_list = []
    
    print('------- 构建样本 -----------')

    # 将用户点击item序列加进来
    topk_recall = topk_recall.merge(time_feat[['user_id', 'item_id_list_str']], 
                                    on=['user_id'], how='left')

    temp_ = topk_recall


    """
        测试
    """
    if flag_test == True:
        len_temp = len(temp_)
        len_temp_2 = len_temp // 2
        temp_['label'] = [1] * len_temp_2 + [0] * (len_temp -  len_temp_2)
    if flag_test == False:
        temp_['label'] = [ 1 if next_item_id == item_similar else 0 for (next_item_id, item_similar) in zip(temp_['next_item_id'], temp_['item_similar'])]
    

    set_user_label_1 = set(temp_[temp_['label']==1]['user_id'])

    temp_['keep'] = temp_['user_id'].map(lambda x: 1 if x in set_user_label_1 else 0)
    
    # 节省内存
    temp_['score_similar'] = temp_['score_similar'].astype('float32')
    temp_['keep'] = temp_['keep'].astype('int16')
    temp_['label'] = temp_['label'].astype('int16')
    
    train_data = temp_[temp_['keep']==1][['user_id', 'item_similar', 
                                          'score_similar', 'label', 
                                          'item_id_list_str']]
    test_data = temp_[temp_['pred']=='test'][['user_id', 'item_similar', 
                                             'score_similar',
                                             'item_id_list_str']]


    # 加入用户行为序列 方便后续构建特征(2020-06-02新增)

    # train_data = train_data.merge(time_feat, on=['user_id'], how='left')
    # test_data = test_data.merge(time_feat, on=['user_id'], how='left')
    print('............... train len=', len(train_data))
    print('............... test  len=', len(test_data))
    list_train_test = [('train', train_data), ('test', test_data)]
    
    del train_data, test_data
    
    
    # item_cluster_feat_转为字典
    item_cluster_feat_cp = item_cluster_feat_.copy()
    item_cluster_feat_cp.index = item_cluster_feat_cp.item_id
    item_cluster_feat_cp.drop(columns=['item_id'], axis=1, inplace=True)
    item_cluster_feat_dict = item_cluster_feat_cp.to_dict('index')
    del item_cluster_feat_cp

    for flag, data in list_train_test:

        print('----------- 加入特征 {flag} -----------'.format(flag=flag))
        
        list_train_flag, list_user_id, list_item_similar, list_label, list_features = [], [], [], [], []
        feature_col_name = []
        for ith, row in tqdm(data.iterrows()):
            user_id, item_id, score_similar = row['user_id'], row['item_similar'], row['score_similar']
            # similarity(a,b) = a/|a| * b/|b|
            dim1_user = dict_embedding_all_ui_item['user'][user_id]
            dim1_item = dict_embedding_all_ui_item['item'][item_id]
            similarity_d1 =  np.sum(dim1_user/np.sqrt(np.sum(dim1_user**2)) * dim1_item/np.sqrt(np.sum(dim1_item**2)))

            dim2_user = dict_embedding_all_ui_user['user'][user_id]
            dim2_item = dict_embedding_all_ui_user['item'][item_id]
            similarity_d2 =  np.sum(dim2_user/np.sqrt(np.sum(dim2_user**2)) * dim2_item/np.sqrt(np.sum(dim2_item**2)))

            list_item_id = row['item_id_list_str'].split(',')
            user_click_item_num = np.int16(len(list_item_id))
            feature = [similarity_d1] + \
                      [similarity_d2] + \
                      [score_similar] + \
                      list(dim1_user) + \
                      list(dim1_item) + \
                      list(dim2_user) + \
                      list(dim2_item)
   
            for i in range(click_topn):
                if i < user_click_item_num:
                    item_i = list_item_id[i]
                
                    feature += [item_i, np.int16(item_hot_dict[item_i])]
                    
                    # 只有一个方向（已经不区分正方方向）
#                     if (item_i in matrix_association_rules) and (item_id in matrix_association_rules[item_i]):
#                         feature += [np.float32(matrix_association_rules[item_i][item_id])]
#                     else:
#                         feature += [np.float32(0)]
                    
                    # txt与img向量相似度对称，只写一个方向即可，且文本与图像都有相同的item，判断一次即可
                    if (item_i in txt_cosine_similarity_dict) and (item_id in txt_cosine_similarity_dict[item_i]):
                        feature += [np.float32(txt_cosine_similarity_dict[item_i][item_id]),
                                    np.float32(img_cosine_similarity_dict[item_i][item_id])]
                    else:
                        feature += [np.float32(0)] * 2


                    # 加入用户点击item统计信息
                    if (item_i in item_cluster_feat_dict):
                        feature += [item_cluster_feat_dict[item_i]['item_cluster'],
                                    item_cluster_feat_dict[item_i]['item_cluster_cnt'],
                                    item_cluster_feat_dict[item_i]['item_cluster_user_cnt'],
                                    item_cluster_feat_dict[item_i]['item_cluster_item_cnt']
                                    ]
                    else:
                        feature += [np.float32(0)] * 4
                  
            
                else:
                    feature += [np.float32(0)] * 8
                    
            list_features.append(feature)
            del feature

            list_train_flag.append(flag)
            list_user_id.append(user_id)
            list_item_similar.append(item_id)
            
            if flag == 'train':
                label = int(row['label'])
                list_label.append(label)

            if flag == 'test':  
                label = -1
                list_label.append(label)

        feature_all = pd.DataFrame(list_features)
        del list_features
        
        # 列名
        for i in range(click_topn):
            feature_col_name += ['clicked_item_' + str(i), 
                                 'clicked_item_' + str(i) + '_cnt',
#                                  'clicked_item_' + str(i) + '_to_item_' + str(i) + '_score',
                                 'item_' + str(i) + '_to_' + 'clicked_item_' + str(i) + '_txt_cosine_sim',
                                 'item_' + str(i) + '_to_' + 'clicked_item_' + str(i) + '_img_cosine_sim',
                                 'clicked_item_' + str(i) + '_item_cluster',
                                 'clicked_item_' + str(i) + '_item_cluster_cnt',
                                 'clicked_item_' + str(i) + '_item_cluster_user_cnt',
                                 'clicked_item_' + str(i) + '_item_cluster_item_cnt'
                                ]

        feature_col_name = ['similarity_d1',
                            'similarity_d2',
                            'score_similar'] + \
                    ['dim1_user_' + str(i) for i in range(len(list(dim1_user)))] + \
                    ['dim1_item_' + str(i) for i in range(len(list(dim1_item)))] + \
                    ['dim2_user_' + str(i) for i in range(len(list(dim2_user)))] + \
                    ['dim2_item_' + str(i) for i in range(len(list(dim2_item)))] + \
                    feature_col_name



        # 列名
        # feature_all.columns = ['f_'+str(i) for i in range(len(feature_all.columns))]
        feature_all.columns = feature_col_name
        
        # 其他列
        feature_all['train_flag'] = list_train_flag
        feature_all['user_id'] = list_user_id
        feature_all['item_similar'] = list_item_similar
        feature_all['label'] = list_label


        del list_train_flag, list_user_id, list_item_similar, list_label
        data_list.append(feature_all)
        del feature_all


        
    pool_feature_part = pd.concat(data_list)
    del data_list
    
    # # 加入时间相关统计特征
    # pool_feature_part = pool_feature_part.merge(time_feat[['user_id', 
    #                                                        'user_click_item_num',
    #                                                        'time_max_min_minus', 
    #                                                        'time_diff_mean', 
    #                                                        'time_mean', 
    #                                                        'time_std',
    #                                                        'time_max', 
    #                                                        'time_min']], 
    #                                             on=['user_id'], how='left')

    # # 加入用户聚类统计特征
    # pool_feature_part = pool_feature_part.merge(user_cluster_feat_, 
    #                                             on=['user_id'], how='left')

    # # 加入项目特征（召回得到的item_similar的特征）
    # item_feat_ = item_cluster_feat_.copy()
    # item_feat_.columns = ['item_similar', 'item_cluster', 'item_cluster_cnt', 
    #                       'item_cluster_user_cnt', 'item_cluster_item_cnt']
    # pool_feature_part = pool_feature_part.merge(item_feat_, 
    #                                             on=['item_similar'], how='left') 

    

    # # 采用进程方式在函数外整体输出即可
    # print('--------------------------- 特征数据 ---------------------')
    # len_f = len(pool_feature_part)
    # len_train = len(pool_feature_part[pool_feature_part['train_flag']=='train'])
    # len_test = len(pool_feature_part[pool_feature_part['train_flag']=='test'])
    # len_train_1 = len(pool_feature_part[(pool_feature_part['train_flag']=='train') & (pool_feature_part['label']== 1)]) 
    # print('所有数据条数', len_f)
    # print('训练数据 : ', len_train)
    # print('训练数据 label 1 : ', len_train_1)
    # print('训练数据 1 / 0 rate : ', len_train_1 * 1.0 / len_f)
    # print('测试数据 : ' , len_test)
    # print('flag : ', set(pool_feature_part['train_flag']))
    # print('--------------------------- 特征数据 ---------------------')
    
    # 数据类型转化以节省内存
    float64_ = pool_feature_part.dtypes[pool_feature_part.dtypes=='float64'].index
    int64_ = pool_feature_part.dtypes[pool_feature_part.dtypes=='int64'].index
    if len(float64_):
        pool_feature_part[float64_] = pool_feature_part[float64_].astype('float32')
    if len(int64_):
        pool_feature_part[int64_] = pool_feature_part[int64_].astype('int32')
    
    pool_feature_part.to_pickle('../cache/pool_feature_part_phase{phase}_pool{batch}.pkl'.format(phase=phase, batch=batch))
    return pool_feature_part

def load_pool_feature(phase, batchs_n):
    feature_all = pd.DataFrame()
    for batch in range(batchs_n):
        time.sleep(10)
        part = pd.read_pickle('../cache/pool_feature_part_phase{phase}_pool{batch}.pkl'.format(phase=phase, batch=batch))
        feature_all = pd.concat([feature_all, part])
    return feature_all



def remove_pool_feature(phase, batchs_n):
    for batch in range(batchs_n):
        file = '../cache/pool_feature_part_phase{phase}_pool{batch}.pkl'.format(phase=phase, batch=batch)
        if os.path.exists(file):
            os.remove(file)
        else:
            print("The file {file} does not exist".format(file=file))
    print("Removed all files of '../cache/pool_feature_part_phase{phase}...'".format(phase=phase))


    
def add_user_profile_info(feature_all, user_profile_feat):
    # 加入时间相关统计特征
    feature_all = feature_all.merge(user_profile_feat,
                          on=['user_id'], 
                          how='left')

    return feature_all


def add_time_statistics(feature_all, time_feat):
    # 加入时间相关统计特征
    feature_all = feature_all.merge(time_feat[['user_id', 
                                               'user_click_item_num',
                                               'time_max_min_minus', 
                                               'time_diff_mean', 
                                               'time_mean', 
                                               'time_std',
                                               'time_max', 
                                               'time_min',
                                               'different_session_num', 
                                               'favorite_session_click']], 
                                    on=['user_id'], how='left')

    return feature_all





def add_user_statistics(feature_all, user_cluster_feat_):
    # 加入用户聚类统计特征
    feature_all = feature_all.merge(user_cluster_feat_, 
                                    on=['user_id'], 
                                    how='left')

    return feature_all





def add_item_statistics(feature_all, item_cluster_feat_):
    # 加入项目聚类统计特征（召回得到的item_similar的特征）

    """
    feature_all中item_similar列与item_cluster_feat_中列item_id对应拼接数据

    """
    item_cluster_feat_.columns = ['item_similar', 'item_cluster', 'item_cluster_cnt', 
                          'item_cluster_user_cnt', 'item_cluster_item_cnt']
    feature_all = feature_all.merge(item_cluster_feat_, 
                                                on=['item_similar'], how='left') 

    return feature_all





def Pool_feature_concat(pool_feature_part):
    """
    进程数据拼接
    """
    for i, j in enumerate(pool_feature_part):
        print('------ part={i}'.format(i=i))
        
        # 数据类型转化以节省内存
#         float64_ = part_data.dtypes[part_data.dtypes=='float64'].index
#         int64_ = part_data.dtypes[part_data.dtypes=='int64'].index
#         if len(float64_):
#             part_data[float64_] = part_data[float64_].astype('float32')
#         if len(int64_):
#             part_data[int64_] = part_data[int64_].astype('int16')   
        
        if i==0:
            feature_all = j.get()
        else:
            feature_all = pd.concat([feature_all, j.get()])
    return feature_all



def user_feat_data(dict_embedding_user_only, user_cluster, time_feat):
    """
    dict_embedding_user_only: 用户embedding的字典数据
    user_cluster: 用户聚类数据
    time_feat: 用户的时间特征数据

    将所有的跟用户有关的数据连接以备预测缺失的用户性别、年龄、城市等信息
    """
    
    user_embedding = pd.DataFrame(dict_embedding_user_only['user']).T.reset_index()
    user_embedding.columns = ['user_id'] + ['user_emb_'+str(i) for i in range(30)]
    user_feat = user_embedding.merge(user_cluster, on=['user_id'], how='left')
    user_feat = user_feat.merge(time_feat[[i for i in time_feat.columns if i!='item_id_list_str']], on=['user_id'], how='left')
    underexpose_user_feat = pd.read_csv('../data/underexpose_train/underexpose_user_feat.csv', 
                                        header=None,
                                       names=['user_id', 'age', 'sex', 'city'],
                                       dtype={'user_id': np.str}
                                       )
    # 性别数值化
    sex = underexpose_user_feat.sex
    underexpose_user_feat['sex'] = 1
    underexpose_user_feat.loc[sex=='F', 'sex'] = 0
    # 删除重复用户
    underexpose_user_feat.drop_duplicates(['user_id'])
    user_feat = user_feat.merge(underexpose_user_feat, on=['user_id'], how='left')
    
    return user_feat





def user_feat_data(dict_embedding_user_only, user_cluster, time_feat):
    """
    dict_embedding_user_only: 用户embedding的字典数据
    user_cluster: 用户聚类数据
    time_feat: 用户的时间特征数据

    将所有的跟用户有关的数据连接以备预测缺失的用户性别、年龄、城市等信息
    """
    
    user_embedding = pd.DataFrame(dict_embedding_user_only['user']).T.reset_index()
    user_embedding.columns = ['user_id'] + ['user_emb_'+str(i) for i in range(30)]
    user_feat = user_embedding.merge(user_cluster, on=['user_id'], how='left')
    user_feat = user_feat.merge(time_feat[[i for i in time_feat.columns if i!='item_id_list_str']], on=['user_id'], how='left')
    underexpose_user_feat = pd.read_csv('../data/underexpose_train/underexpose_user_feat.csv', 
                                        header=None,
                                       names=['user_id', 'age', 'sex', 'city'],
                                       dtype={'user_id': np.str}
                                       )
    # 性别数值化
    sex = underexpose_user_feat.sex
    underexpose_user_feat['sex'] = 1
    underexpose_user_feat.loc[sex=='F', 'sex'] = 0
    # 删除重复用户
    underexpose_user_feat.drop_duplicates(['user_id'])
    user_feat = user_feat.merge(underexpose_user_feat, on=['user_id'], how='left')
    
    return user_feat



def age_sex_city_data(user_feat, col='age'):
    """
    col: 可供选择的参数'age'、'sex'、'city'
    提取年龄、性别、城市数据以备预测缺失
    """
    tmp = user_feat.iloc[:, 1:-3]
    tmp[col] = user_feat[col]
    
    X_test = tmp[tmp[col].isna()].iloc[:, :-1]
    
    
    X_ = tmp[~tmp[col].isna()].iloc[:, :-1]
    y_ = tmp[~tmp[col].isna()].iloc[:, -1]
    
    lbl = LabelEncoder()
    y_ = lbl.fit_transform(y_)
    return X_, y_, X_test, user_feat[['user_id', col]]




def null_xgb_predict(X_, y_, X_test, age_sex_city_data, col='age'):
    """
    X_: 训练集X
    y_: 训练集y
    X_test: 缺失预测特征X
    age_sex_city_data: 带user_id以及col的数据
    col: 要预测并填充age_sex_city_data缺失的列
    """
    print('-------------- {col}特征缺失预测 --------------'.format(col=col))
    # split
    X_train, X_vali, y_train, y_vali = train_test_split(X_, y_, random_state=2020, test_size=0.2)
    
    if len(set(y_)) == 2:
        objective='binary:logitraw'
        eval_metric = "auc"
    elif len(set(y_))>2:
        objective='multi:softmax'
        eval_metric = "mlogloss"
    else:
        print('分类的类别数少于2')
    
    # fit model for train data
    model = xgb.XGBClassifier(
                              booster='gbtree',
                              learning_rate=0.05,
                              n_estimators=1500,         # 树的个数
                              max_depth=8,               # 树的深度
                              min_child_weight = 3,      # 叶子节点最小权重
                              gamma=0.1,                 # 惩罚项中叶子结点个数前的参数
                              reg_lambda = 2,            # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                              subsample=0.8,             # 随机选择80%样本建立决策树
                              colsample_bytree=0.8,       # 随机选择80%特征建立决策树
                              objective=objective, # 指定损失函数
                              random_state=2020,           # 随机数
                              )
    model.n_classes_ = len(set(y_))
    xgb_model = model.fit(X_train,
                          y_train,
                          eval_set = [(X_vali, y_vali)],
                          eval_metric = eval_metric,
                          early_stopping_rounds = 200,
                          verbose = 200
                          )
    
    pre = model.predict(X_test)
    assert len(X_test)==age_sex_city_data[age_sex_city_data[col].isna()].shape[0]
    age_sex_city_data.loc[age_sex_city_data[col].isna(), col] = pre

    return age_sex_city_data




def phase_submit_save(submit, phase, topk):
    """
    保存阶段的模型预测数据，以待拼接所有phase提交答案
    """
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    file_name = '../result/submit-wanping7-{phase}-{time_str}.csv'.format(phase=phase, time_str=time_str)
    with open(file_name, 'w') as f:
        for i, row in submit.iterrows():  
            user_id = str(row['user_id'])
            item_list = str(row['item_similar']).split(',')[:topk]
            assert len(set(item_list)) == topk
            line = user_id + ',' + ','.join(item_list) + '\n'
            assert len(line.strip().split(',')) == (topk+1)
            f.write(line)